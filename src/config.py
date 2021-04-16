import argparse
import os
import numpy as np

import torch

def str2bool(v):
    return v.lower() in ('true', '1')

d = os.path.dirname
parser = argparse.ArgumentParser(description='Unsupervised Segmentation incorporating Shape Prior via WGAN')
path_arg = parser.add_argument_group('Experiment Config')
path_arg = parser.add_argument_group('Data Config')
path_arg = parser.add_argument_group('Networks Config')
path_arg = parser.add_argument_group('Training Environment Config')
path_arg = parser.add_argument_group('Coefficient Config')
path_arg = parser.add_argument_group('Optimization Config')


path_arg = parser.add_argument_group('Experiment Config')
path_arg.add_argument('--file_prefix', type=str, default='experiment_name',
    help='Path of model checkpoint to be save')
path_arg.add_argument('--experiment_count', type=int, default=1,
    help='Experiment count number to be used in filename')
path_arg.add_argument('--dir_output', type=str, default='./csv',
    help='Directory where output.csv will be stored')
path_arg.add_argument('--sample_interval', type=int, default=200,
    help='An interval to check training results from a sample')
path_arg.add_argument('--num_plot_img', type=int, default=64,
    help='The number of images to plot per batch')


path_arg = parser.add_argument_group('Data Config')
path_arg.add_argument('--dir_train_data_image', type=str, default='/home01/kaggle/train',
    help='Directory of image data to be used')
path_arg.add_argument('--dir_data_csv_hair', type=str, default='/home01/kaggle/train_hair.csv',
    help='Directory of hair csv file')
path_arg.add_argument('--dir_data_csv_non_hair', type=str, default='/home01/kaggle/train_non_hair.csv',
    help='Directory of non hair csv file')
path_arg.add_argument('--image_format', type=str, default='jpg',
    help='Image format to be used')
path_arg.add_argument('--height', type=int, default=64,
    help='training image height to be resized by')
path_arg.add_argument('--width', type=int, default=64,
    help='training image width to be resized by')



path_arg = parser.add_argument_group('Networks Config')
path_arg.add_argument('--trained_ckpt_path', type=str, default=None,
    help='Path of trained model checkpoint to be loaded')
path_arg.add_argument('--num_in_channel', type=int, default=1,
    help='Number of channel of input')
path_arg.add_argument('--num_out_channel', type=int, default=1,
    help='Number of channel of output')
path_arg.add_argument('--network_d', type=str, default='vgg',
    help='Network architecture to be used as a discriminator')
path_arg.add_argument('--network_g', type=str, default='vgg',
    help='Network architecture to be used as a generator')



path_arg = parser.add_argument_group('Training Environment Config')
path_arg.add_argument('--num_workers', type=int, default=16,
    help='# of subprocesses to use for data loading for training')
path_arg.add_argument('--multi_gpu', type=str2bool, default=True,
    help='Decide whether to use multiple numbers of gpus or not')
path_arg.add_argument('--num_gpu', type=int, default=4,
    help='# of GPU to be used')
path_arg.add_argument('--cuda_id', type=str, default='cuda:0',
    help='GPU to be used')


path_arg = parser.add_argument_group('Coefficient Config')
path_arg.add_argument('--lambda_gp', type=int, default=10,
    help='The size of gradient penalty')
path_arg.add_argument('--lambda_distance', type=float, default=10,
    help='The weight of a distance term in the generator loss')

path_arg = parser.add_argument_group('Optimization Config')
path_arg.add_argument('--num_epoch', type=int, default=5,
    help='# of epochs to train for')
path_arg.add_argument('--train_batch_size', type=int, default=64,
    help='Batch size for training')
path_arg.add_argument('--test_batch_size', type=int, default=64,
    help='Batch size for testing')
path_arg.add_argument('--lr_d', type=float, default=1e-3,
    help='Fixed learning rate value for discriminator')
path_arg.add_argument('--lr_g', type=float, default=1e-3,
    help='Fixed learning rate value for generator')
path_arg.add_argument('--beta1_d', type=float, default=0.5,
    help='Beta1 hyperparam for Adam optimizers for discriminator')
path_arg.add_argument('--beta1_g', type=float, default=0.5,
    help='Beta1 hyperparam for Adam optimizers for generator')
path_arg.add_argument('--beta2_d', type=float, default=0.999,
    help='Beta2 hyperparam for Adam optimizers for discriminator')
path_arg.add_argument('--beta2_g', type=float, default=0.999,
    help='Beta2 hyperparam for Adam optimizers for generator')
path_arg.add_argument('--num_discriminator', type=int, default=5,
    help='The number of discriminator steps before one generator step')


def get_config():
    config = parser.parse_args()
    print('[*] Configuration')
    print(config)
    
    return config