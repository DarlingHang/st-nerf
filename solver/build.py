# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import torch


def make_optimizer(cfg, model,active_list=[],frozen_list=[]):

    weight_decay = cfg.SOLVER.WEIGHT_DECAY
    lr = cfg.SOLVER.BASE_LR


    if cfg.SOLVER.OPTIMIZER_NAME == 'Adam':
        if active_list == [] and frozen_list == []:
            optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, betas=(0.9, 0.999),weight_decay =weight_decay )
        else:
            optimizer = torch.optim.Adam([
                     {'params': frozen_list, 'lr': 0.0},
                     {'params': active_list, 'lr': lr}], lr=lr, betas=(0.9, 0.999),weight_decay =weight_decay )
    else:
        pass

    return optimizer
