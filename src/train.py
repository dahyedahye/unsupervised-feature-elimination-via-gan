from __future__ import print_function
import argparse
import os
import random
import datetime
import math
import sys
import csv
from time import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.autograd as autograd
import torch.utils.data

from torch.utils.data import DataLoader
from torch.autograd import Variable

from torchvision import datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils

import numpy as np
import matplotlib.pyplot as plt


# ===================================
#         Import custom codes
# ===================================
import config
import utils
from models import discriminator
from models import generator
from datasets import melanoma_dataset

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
# ===================================
#             Get config
# ===================================
train_config = config.get_config()
config = train_config

# config - experiment
file_prefix = config.file_prefix
experiment_count = config.experiment_count
dir_output = config.dir_output
sample_interval = config.sample_interval
num_plot_img = config.num_plot_img

# config - data
dir_train_data_image = config.dir_train_data_image
dir_data_csv_hair = config.dir_data_csv_hair
dir_data_csv_non_hair = config.dir_data_csv_non_hair
image_format = config.image_format
height = config.height
width = config.width

# config - networks
trained_ckpt_path = config.trained_ckpt_path
num_in_channel = config.num_in_channel
num_out_channel = config.num_out_channel
network_d = config.network_d
network_g = config.network_g

# config - training env
num_workers = config.num_workers
multi_gpu = config.multi_gpu
num_gpu = config.num_gpu
cuda_id = config.cuda_id

# config - coefficient
lambda_gp = config.lambda_gp
lambda_gan = config.lambda_gan
lambda_distance = config.lambda_distance 
lambda_r1 = config.lambda_r1

# config - optimization
num_epoch = config.num_epoch
train_batch_size = config.train_batch_size
test_batch_size = config.test_batch_size
lr_d = config.lr_d
lr_g = config.lr_g
num_epoch = config.num_epoch
beta1_d = config.beta1_d
beta1_g = config.beta1_g
beta2_d = config.beta2_d
beta2_g = config.beta2_g
num_discriminator = config.num_discriminator


# ================================================
#     Set Path & Files to Save Training Result
# ================================================


# Set file names or dirs to debug or save training result
experiment_name = '{}_{}_h{}_w{}_lamgan_{}_lamdis{}_lamr1{}_lrd{}_lrg{}_bs{}_ndisc{}_nep{}_ex{}'.format(
                                datetime.datetime.now().strftime('%Y%m%d'),
                                file_prefix,
                                height,
                                width,
                                lambda_gan,
                                lambda_distance,
                                lambda_r1,
                                lr_d,
                                lr_g,
                                train_batch_size,
                                num_discriminator,
                                num_epoch,
                                experiment_count
                                )
experiment_name = experiment_name.replace(".","")

# dirs to save result files
dir_output = '{}/output'.format(dir_output)
dir_image = '{}/image'.format(dir_output)
dir_model = '{}/model'.format(dir_output)
dir_train_info = '{}/train_info'.format(dir_output)

# dirs to save result images and learning curves
dir_save_output_monitor_train = '{}/{}_monitor_train'.format(dir_image, experiment_name) # image to monitor train

directories = [dir_output, dir_image, dir_model, dir_train_info, dir_save_output_monitor_train]

# create dirs to save learning result if they don't exist.
for directory in directories:
    try:
        os.mkdir(directory)
        print("Directory " , directory,  " Created ") 
    except FileExistsError:
        print("Directory " , directory,  " already exists")


# file path to save .txt file which contains training configuration
txt_train_info = '{}/{}_train_info.txt'.format(dir_train_info, experiment_name)


# ===================================
#  define functions to help training
# ===================================

# weights initialization function
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# a function to compute gradient penalties for wgan
torch.cuda.set_device(cuda_id)
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    # fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False).to(device)

    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty




# ========================================
#  define functions for parameters update
# ========================================
# https://github.com/LMescheder/GAN_stability/blob/c1f64c9efeac371453065e5ce71860f4c2b97357/gan_training/train.py
def compute_loss(output_critic, target):
    targets = output_critic.new_full(size=output_critic.size(), fill_value=target)
    loss = F.binary_cross_entropy_with_logits(output_critic, targets)
    return loss

def toggle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)

def compute_grad2(d_out, x_in):
    batch_size = x_in.size(0)
    grad_dout = autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())
    reg = grad_dout2.view(batch_size, -1).sum(1)
    return reg




# ===================================
#             Load Data
# ===================================



transform_non_hair = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.ToPILImage(),
                            transforms.Resize((height, width)),
                            transforms.ToTensor()
])


transform_train_hair = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.ToPILImage(),
                            transforms.Resize((height, width)),
                            transforms.ToTensor()
])

dataset_non_hair = melanoma_dataset.MelanomaDataset(dir_data_csv_non_hair, dir_train_data_image, 'hair', image_format, transform=transform_non_hair)
dataset_train_hair = melanoma_dataset.MelanomaDataset(dir_data_csv_hair, dir_train_data_image, 'hair', image_format, transform=transform_train_hair)


loader_non_hair = torch.utils.data.DataLoader(
    dataset = dataset_non_hair,
    batch_size=train_batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=True)

loader_train_hair = torch.utils.data.DataLoader(
    dataset = dataset_train_hair,
    batch_size=train_batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=True)


# ===================================
#           Set Train Env
# ===================================
if multi_gpu == True:
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    ngpu = num_gpu # should be modified to specify multiple gpu ids to be used
elif cuda_id:
    device = torch.device(cuda_id)
    ngpu = 1
else:
    device = torch.device('cpu')
    ngpu = 0



# ===================================
#              Set Model
# ===================================
# init discriminator model instance
if network_d == 'resnet':
    pass # to be implemented
elif network_d =='vgg':
    model_d = discriminator.CNNDiscriminator(height, width).to(device)
else:
    raise ValueError('There is no such discriminator model')

# init segmentation model instance
if network_g == 'stylegan':
    pass # to be implemented
elif network_g =='resnet':
    pass # to be implemented
elif network_g =='vgg':
    model_g = generator.ResizeCNNGenerator(num_in_channel, num_out_channel).to(device)
elif network_g =='unet':
    pass # to be implemented
else:
    raise ValueError('There is no such generator model')

# decide whether to use multiple gpus
if multi_gpu == True:
    model_d = nn.DataParallel(model_d, list(range(num_gpu)))
    model_g = nn.DataParallel(model_g, list(range(num_gpu)))

# init model parameters
model_d.apply(weights_init)
if network_g =='vgg':
    model_g.apply(weights_init)

# optimizers
optimizer_d = torch.optim.Adam(model_d.parameters(), lr=lr_d, betas=(beta1_d, beta2_d))
optimizer_g = torch.optim.Adam(model_g.parameters(), lr=lr_g, betas=(beta1_g, beta2_g))


# ===================================
#           Save Train Info
# ===================================
with open(txt_train_info, 'a') as t:
    t.write('[Data]     Hair: {}'.format(dir_data_csv_hair) + os.linesep)
    t.write('[Data]     Non Hair: {}'.format(dir_data_csv_non_hair) + os.linesep)

    t.write('[Optim]    Total Epoch: {}'.format(num_epoch) + os.linesep)
    t.write('[Optim]    Train Batch Size: {}'.format(train_batch_size) + os.linesep)
    t.write('[Optim]    Test Batch Size: {}'.format(test_batch_size) + os.linesep)
    t.write('[Optim]    Learning Rate (Discriminator): {}'.format(lr_d) + os.linesep)
    t.write('[Optim]    Learning Rate (Generator): {}'.format(lr_g) + os.linesep)
    t.write('[Optim]    Beta1 (Discriminator): {}'.format(beta1_d) + os.linesep)
    t.write('[Optim]    Beta1 (Generator): {}'.format(beta1_g) + os.linesep)
    t.write('[Optim]    Beta2 (Discriminator): {}'.format(beta2_d) + os.linesep)
    t.write('[Optim]    Beta2 (Generator): {}'.format(beta2_g) + os.linesep)
    t.write('[Optim]    Num Discriminator: {}'.format(num_discriminator) + os.linesep)

    t.write('[Coeff]    [Discriminator] Lambda GP: {}'.format(lambda_gp) + os.linesep)
    t.write('[Coeff]    [Generator] Lambda Distance: {}'.format(lambda_distance) + os.linesep)

    t.write('[Network]  In Channel: {}'.format(num_in_channel) + os.linesep)
    t.write('[Network]  Out Channel: {}'.format(num_out_channel) + os.linesep)
    t.write('[Network]  Discriminator Model: {}'.format(network_d) + os.linesep)
    t.write('[Network]  Generator Model: {}'.format(network_g) + os.linesep)

    t.write('**************** Discriminator Structure ****************' + os.linesep)
    t.write('[*] num of params: {}'.format(utils.count_parameters(model_d)) + os.linesep)
    t.write('[*] model structure: {}'.format(model_d) + os.linesep)

    t.write('**************** Generator Model Structure ****************' + os.linesep)
    t.write('[*] num of params: {}'.format(utils.count_parameters(model_g)) + os.linesep)
    t.write('[*] model structure: {}'.format(model_g) + os.linesep)



# ==================================================
#      Init variables to save training results
# ==================================================
losses_d = []
losses_g = []
losses_g_gan = []
losses_distance = []

# ==============================
#         Start Training
# ==============================
total_iter = 0
time_total_start = time() # set a start time to monitor training time
for epoch in range(num_epoch):
    time_train_epoch_start = time() # set an epoch start time to monitor training time per epoch
    if epoch == 0:
        print('device:', device)
    print("start train epoch{}:".format(epoch))
    dataloader_iterator = iter(loader_non_hair)

    num_iters_d = 0
    num_iters_g = 0

    # ================================================
    #                     Train
    # ================================================
    model_d.train()
    model_g.train()

    for i, data_hair in enumerate(loader_train_hair, 0):


        # iterate dataloader_input at the same time
        try:
            data_non_hair = next(dataloader_iterator)
        except StopIteration:
            dataloader_iterator = iter(loader_non_hair)
            data_non_hair = next(dataloader_iterator)

        imgs_non_hair = Variable(data_non_hair['image'].type(Tensor)).to(device)
        imgs_hair = Variable(data_hair['image'].type(Tensor)).to(device)


        # -------------------------------
        #       Train Discriminator
        # ------------------------------- 

        toggle_grad(model_d, True)
        toggle_grad(model_g, False)

        # remove gradients on the computational graph of the discriminator
        optimizer_d.zero_grad()

        # make the gradients of non hair images computable
        imgs_non_hair.requires_grad_()
        # get scores of non-hair images from the discriminator
        validity_non_hair = model_d(imgs_non_hair)
        # compute loss
        loss_d_non_hair = compute_loss(validity_non_hair, 1)
        loss_d_non_hair.backward(retain_graph=True)
        reg = lambda_r1 * compute_grad2(validity_non_hair, imgs_non_hair).mean()
        reg.backward()

        with torch.no_grad():
            # get hair-removed output from the generator model
            output_hair_removed = model_g(imgs_hair)

        # get scores of hair-removed images from the discriminator
        validity_hair_removed = model_d(output_hair_removed)

        # compute a loss for the discriminator
        # GAN loss with R1 regularizer
        loss_d_non_hair_fake = compute_loss(validity_hair_removed, 0)

        loss_d_non_hair_fake.backward()

        # full loss
        loss_d = loss_d_non_hair + reg + loss_d_non_hair_fake
        losses_d.append(loss_d.item())

        # optimize discriminator
        optimizer_d.step()
        num_iters_d += 1

        if i % num_discriminator == 0:
            model_g.train()
            toggle_grad(model_d, False)
            toggle_grad(model_g, True)
            # ---------------------
            #    Train Generator
            # ---------------------
            # remove gradients on the computational graph of the generator
            optimizer_g.zero_grad()
            # get hair-removed output from the generator model
            output_hair_removed = model_g(imgs_hair)
            validity_fake = model_d(output_hair_removed)

            # compute a loss for the generator
            # compute distance between the original hair images and their hair-removed outputs of generator
            distance = F.l1_loss(imgs_hair, output_hair_removed)
            # Final loss = Adversarial loss using WGAN-GP + distance
            targets = validity_fake.new_full(size=validity_fake.size(), fill_value=1)
            loss_g_gan = F.binary_cross_entropy_with_logits(validity_fake, targets)
            loss_g = (lambda_gan * loss_g_gan) + (lambda_distance * distance)
            losses_g.append(loss_g.item())
            losses_distance.append(distance.item())
            losses_g_gan.append(loss_g_gan.item())
            loss_g.backward()

            # optimize generator
            optimizer_g.step()
            num_iters_g += 1

            # ---------------------
            #     Get Results
            # ---------------------
            print(
                "#-TRAIN-# [Epoch %d/%d] [Batch %d/%d] [G loss: %f] [D loss: %f] [Distance: %f]"
                % (epoch+1, num_epoch, i+1, len(loader_train_hair), loss_g.item(), loss_d.item(), distance.item())
            )

            total_time = time() - time_total_start
            total_hours, _total_rest = divmod(total_time, 3600)
            total_mins, total_secs = divmod(_total_rest, 60)
            print('#-TRAIN-# Total Running Time: {}h {}m {}s'.format(total_hours, total_mins, total_secs))
            train_epoch_time = time() - time_train_epoch_start
            train_epoch_hours, _train_epoch_rest = divmod(train_epoch_time, 3600)
            train_epoch_mins, train_epoch_secs = divmod(_train_epoch_rest, 60)
            print('#-TRAIN-# Epoch Time: {}h {}m {}s'.format(train_epoch_hours, train_epoch_mins, train_epoch_secs))


            # -----------------------------------------
            #      Plot images to monitor training
            # -----------------------------------------
            if num_iters_g in [0, 1] or (num_iters_g * num_discriminator) % sample_interval == 0:

                plot_hair_removed = output_hair_removed.cpu().detach().numpy()
                plot_hair = imgs_hair.cpu().detach().numpy()
                plot_non_hair = imgs_non_hair.cpu().detach().numpy()

                plot_hair_removed = vutils.make_grid(torch.from_numpy(plot_hair_removed[:num_plot_img]), padding=2, pad_value=1)
                plot_hair = vutils.make_grid(torch.from_numpy(plot_hair[:num_plot_img]), padding=2, pad_value=1)
                plot_non_hair = vutils.make_grid(torch.from_numpy(plot_non_hair[:num_plot_img]), padding=2, pad_value=1)

                imgs = [[plot_hair_removed, plot_hair, plot_non_hair]]
                imgs_list = [plot_hair_removed, plot_hair, plot_non_hair]
                imgs_names = ['hair removed', 'hair input', 'non-hair']
                fig, axes = plt.subplots(len(imgs), len(imgs[0]), figsize=(18,18))
                for plot_i, ax in enumerate(axes.flat):
                    ax.axis("off")
                    ax.set_title(imgs_names[plot_i])
                    ax.imshow(np.transpose(imgs_list[plot_i],(1,2,0)), vmin=0.0, vmax=1.0)
                    if plot_i + 1 == len(imgs_list):
                        break
                plt.show()
                file_name = '{}/results_{}_{}'.format(dir_save_output_monitor_train, epoch+1, i+1)
                fig.savefig(file_name, bbox_inches='tight', pad_inches=0.1)
                plt.clf()
                plt.close()

                # Plot loss & metric score curves
                curve_titles = [
                    "Discriminator Loss",
                    "Generator Loss",
                    "Distance",
                    "Generator GAN Loss"
                                ]
                curve_data = [[losses_d], [losses_g], [losses_distance], [losses_g_gan]]
                curve_labels = [["loss_d"], ["loss_g"], ["distance"], "loss_g_gan"]
                curve_xlabels = ["iterations", "iterations", "iterations", "iterations"]
                curve_ylabels = ["loss", "loss", "distance", "loss"]
                curve_filenames = ["learn-curve-loss-d", "learn-curve-loss-g", "learn-curve-distance", "learn-curve-loss-gan"]

                
                for i_curve, curve_data in enumerate(curve_data):
                    plt.figure(figsize=(10,5))
                    plt.title(curve_titles[i_curve])
                    for i_curve_data, curve_data_item in enumerate(curve_data):
                        plt.plot(curve_data_item,label=curve_labels[i_curve][i_curve_data])
                    plt.xlabel(curve_xlabels[i_curve])
                    plt.ylabel(curve_ylabels[i_curve])
                    plt.legend()
                    file_name = '{}/{}'.format(dir_save_output_monitor_train, curve_filenames[i_curve])
                    plt.show()
                    plt.savefig(file_name, bbox_inches='tight', pad_inches=0.1)
                    plt.clf()
                    plt.close()

                plt.figure(figsize=(10,5))
                plt.title("GAN Loss")
                plt.plot(losses_d,label="loss_d")
                plt.plot(losses_g_gan,label="loss_g_gan")
                plt.xlabel("iterations")
                plt.ylabel("loss")
                plt.legend()
                file_name = '{}/learn-curve-loss-gan'.format(dir_save_output_monitor_train)
                plt.show()
                plt.savefig(file_name, bbox_inches='tight', pad_inches=0.1)
                plt.clf()
                plt.close()

                         


        total_iter += 1
        if total_iter % 1000 == 0:
            torch.save({
                'epoch': epoch,
                'total_iter': total_iter,
                'discriminator': model_d,
                'generator': model_g,
            }, '{}/{}.pth'.format(dir_model, experiment_name))
            print('[*] model is saved at iteration {}'.format(total_iter))


            