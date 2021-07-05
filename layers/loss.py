import torch
import torch.nn as nn

def make_loss(cfg):
    return nn.MSELoss()
