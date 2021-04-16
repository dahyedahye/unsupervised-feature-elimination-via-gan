import numpy as np
import torch


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# to load dataset written in npy
def npy_loader(path):
    sample = torch.from_numpy(np.load(path))
    return sample

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag