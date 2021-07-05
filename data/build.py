# encoding: utf-8
"""
@author:  Minye Wu
@GITHUB: wuminye
"""

from torch.utils import data
import numpy as np
from .datasets.ray_dataset import Ray_Dataset, Ray_Dataset_View, Ray_Dataset_Render, Ray_Frame_Layer_Dataset
from .transforms import build_transforms, build_layered_transforms


def make_ray_data_loader(cfg, is_train=True):

    batch_size = cfg.SOLVER.IMS_PER_BATCH

    transforms_bkgd = build_layered_transforms(cfg, is_train=is_train, is_layer=False)
    transforms_layer = build_layered_transforms(cfg, is_train=is_train, is_layer=True)

    datasets = Ray_Dataset(cfg, transforms_bkgd, transforms_layer)

    num_workers = cfg.DATALOADER.NUM_WORKERS
    data_loader = data.DataLoader(
        datasets, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    return data_loader, datasets

def make_ray_data_loader_view(cfg, is_train=False):

    batch_size = cfg.SOLVER.IMS_PER_BATCH

    transforms = build_transforms(cfg, is_train)

    datasets = Ray_Dataset_View(cfg, transforms)

    num_workers = cfg.DATALOADER.NUM_WORKERS
    data_loader = data.DataLoader(
        datasets, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    return data_loader, datasets

def make_ray_data_loader_render(cfg, is_train=False):

    batch_size = cfg.SOLVER.IMS_PER_BATCH 
        
    
    transforms = build_transforms(cfg, is_train)

    datasets = Ray_Dataset_Render(cfg, transforms)

    num_workers = cfg.DATALOADER.NUM_WORKERS
    data_loader = data.DataLoader(
        datasets, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return data_loader, datasets