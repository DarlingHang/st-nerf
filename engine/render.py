import torch

from utils import batchify_ray, vis_density, ray_sampling
import numpy as np
import os
import torch


'''
Sample rays from views (and images) with/without masks

--------------------------
INPUT Tensors
K: intrinsics of camera (3,3)
T: extrinsic of camera (4,4)
image_size: the size of image [H,W]

ROI:  2D ROI bboxes  (4) left up corner(x,y) followed the height and width  (h,w)

masks:(M,H,W)
-------------------
OUPUT:
list of rays:  (N,6)  dirs(3) + pos(3)
RGB:  (N,C)
'''




def render(model, K,T,img_size,ROI = None, bboxes = None,only_coarse = False,near_far=None):
    model.eval()
    assert not (bboxes is None and near_far is None), ' either bbox or near_far should not be None.'
    mask = torch.ones(img_size[0],img_size[1])
    if ROI is not None:
        mask = torch.zeros(img_size[0],img_size[1])
        mask[ROI[0]:ROI[0]+ROI[2], ROI[1]:ROI[1]+ROI[3]] = 1.0
    rays,_ = ray_sampling(K.unsqueeze(0), T.unsqueeze(0), img_size, masks=mask.unsqueeze(0))

    if bboxes is not None:
        bboxes = bboxes.unsqueeze(0).repeat(rays.size(0),1,1)

    with torch.no_grad():
        stage2, stage1,_ = batchify_ray(model, rays, bboxes,near_far = near_far)


    rgb = torch.zeros(img_size[0],img_size[1], 3, device = stage2[0].device)
    rgb[mask>0.5,:] = stage2[0]

    depth = torch.zeros(img_size[0],img_size[1],1, device = stage2[1].device)
    depth[mask>0.5,:] = stage2[1]

    alpha = torch.zeros(img_size[0],img_size[1],1, device = stage2[2].device)
    alpha[mask>0.5,:] = stage2[2]
    
    stage2_final = [None]*3
    stage2_final[0] = rgb.reshape(img_size[0],img_size[1], 3)
    stage2_final[1] = depth.reshape(img_size[0],img_size[1])
    stage2_final[2] = alpha.reshape(img_size[0],img_size[1])


    rgb = torch.zeros(img_size[0],img_size[1], 3, device = stage1[0].device)
    rgb[mask>0.5,:] = stage1[0]

    depth = torch.zeros(img_size[0],img_size[1],1, device = stage1[1].device)
    depth[mask>0.5,:] = stage1[1]

    alpha = torch.zeros(img_size[0],img_size[1],1, device = stage1[2].device)
    alpha[mask>0.5,:] = stage1[2]

    stage1_final = [None]*3
    stage1_final[0] = rgb.reshape(img_size[0],img_size[1], 3)
    stage1_final[1] = depth.reshape(img_size[0],img_size[1])
    stage1_final[2] = alpha.reshape(img_size[0],img_size[1])


    
    return stage2_final, stage1_final
