import argparse
import os
import sys
from os import mkdir
import shutil

import torch.nn.functional as F

sys.path.append('..')

from config import cfg
from data import get_iteration_path
from engine.layered_trainer import do_train
from modeling import build_layered_model
from solver import make_optimizer, WarmupMultiStepLR,build_scheduler
from layers import make_loss

from utils.logger import setup_logger

import torch
from layers.RaySamplePoint import RaySamplePoint
import random

from utils import batchify_ray, vis_density, lookat
from torchvision import utils as vutils
import numpy as np
import imageio
import matplotlib.pyplot as plt

def load_dataset_model(cfg):
    para_file = get_iteration_path(cfg.OUTPUT_DIR)
    print(para_file)
    if para_file is None:
        assert 'training model does not exist'
    
    _, dataset = make_ray_data_loader_render(cfg)
    model = build_layered_model(cfg, dataset.camera_num)
    model.set_bkgd_bbox(dataset.datasets[0][0].layer_bbox)
    
    dict_0 = torch.load(os.path.join(output_dir,para_file),map_location='cuda')

    model_old_dict = dict_0['model']
    model_new_dict = model.state_dict()
    model_dict = {k: v for k, v in model_old_dict.items() if k in model_new_dict}

    model.load_state_dict(model_dict)

    model.cuda()

    return dataset, model

def load_dataset_model_frame_layer(cfg,frame,layer):
    para_file = get_iteration_path_frame_layer(cfg.OUTPUT_DIR,frame,layer)
    print(para_file)
    if para_file == '':
        return None, None

    _, dataset = make_data_loader_frame_layer_render(cfg, frame, layer)

    model = build_model(cfg).cuda()
    optimizer = make_optimizer(cfg, model)
    scheduler = build_scheduler(optimizer, cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.START_ITERS, cfg.SOLVER.END_ITERS, cfg.SOLVER.LR_SCALE)

    dict_0 = torch.load(para_file,map_location='cuda')

    model_old_dict = dict_0['model']
    model_new_dict = model.state_dict()
    model_dict = {k: v for k, v in model_old_dict.items() if k in model_new_dict}

    model.load_state_dict(model_dict)

    loss_fn = make_loss(cfg)

    model.cuda()

    return dataset, model


# def render_spherical_frame_layer(model,dataset,radius,ThetaStart,ThetaStep,ThetaEnd,
#                                                                       PhiStart,PhiStep,PhiEnd,offsets,up,
#                                                                       inverse_y_axis,density_threshold):
#     rgbs = []
#     depths = []
#     #rgbs_coarse = []
#     H = dataset.height
#     W = dataset.width
#     for theta in np.arange(ThetaStart,ThetaEnd,ThetaStep):
#         for phi in np.arange(PhiStart,PhiEnd,PhiStep):

#             rays,bbox, mask = dataset.get_rays_by_spherical(theta, phi, radius, offsets,up)

#             rays = rays.cuda()
#             bbox = bbox.cuda()
#             mask = mask.cuda()
            
#             uv_list = (mask).squeeze().nonzero()
#             u_list = uv_list[:,0]
#             v_list = uv_list[:,1]
            
#             with torch.no_grad():
#                 stage2, stage1, ray_mask = batchify_ray(model, rays, bbox, density_threshold=density_threshold)

#                 color = stage2[0]
#                 #color_coarse = stage1[0]
#                 depth = stage2[1]
#                 alpha = stage2[2]
#                 depth[alpha < 0.5] = 50

#                 color_img = torch.zeros((H,W,3)).cuda()
#                 #color_coarse_img = torch.zeros((H,W,3)).cuda()
#                 detph_img = 50 * torch.ones((H,W,1)).cuda()
        
#                 color_img = color_img.index_put_((u_list,v_list), color).cpu()
#                 #color_coarse_img = color_coarse_img.index_put_((u_list,v_list), color_coarse).cpu()
#                 depth_img = depth_img.index_put_((u_list,v_list), depth).cpu()
                
#                 if inverse_y_axis:
#                     color_img = torch.flip(color_img,[0])
#                     depth_img = torch.flip(depth_img,[0])
#                     #color_coarse_img = torch.flip(color_coarse_img,[0])
                    
#                 plt.imshow(color_img)
#                 plt.show()
#                 rgbs.append(color_img)
#                 #rgbs_coarse.append(color_coarse_img)
#                 depths.append(depth_img)
#                 print(theta,phi)
                
#     return rgbs, depths

def render_frame_layer(model, dataset, pose,inverse_y_axis=False,density_threshold=0):
    H = dataset.height
    W = dataset.width
    rays, bbox, near_far, mask = dataset.get_rays_by_pose(pose)
        
    rays = rays.cuda()
    if bbox is not None:
        bbox = bbox.cuda()
    mask = mask.cuda()
    near_far = near_far.cuda()

    uv_list = (mask).squeeze().nonzero()
    u_list = uv_list[:,0]
    v_list = uv_list[:,1]

    with torch.no_grad():
        stage2, stage1, ray_mask = batchify_ray(model, rays, bbox, density_threshold=density_threshold, near_far=near_far)

        color = stage2[0]
        #color_coarse = stage1[0]
        depth = stage2[1]
        alpha = stage2[2]
        depth[alpha < 0.5] = 50

        color_img = torch.zeros((H,W,3)).cuda()
        depth_img = 50 * torch.ones((H,W,1)).cuda()
        #color_coarse_img = torch.zeros((H,W, 3)).cuda()

        color_img = color_img.index_put_((u_list,v_list), color).cpu()
        depth_img = depth_img.index_put_((u_list,v_list), depth).cpu()
        #color_coarse_img = color_coarse_img.index_put_((u_list,v_list), color_coarse).cpu()

        if inverse_y_axis:
            color_img = torch.flip(color_img,[0])
            depth_img = torch.flip(depth_img,[0])
            #color_coarse_img = torch.flip(color_coarse_img,[0])

        plt.imshow(color_img)
        plt.show()
        
        return color_img, depth_img
    
def render_path_frame_layer(model, dataset,poses,inverse_y_axis=False,density_threshold=0,save_dir = ''):
    rgbs = []
    depths = []
    for i in range(len(poses)):
        pose = poses[i]
        print('Rendering image %d / %d' % (i+1,len(poses)))
        color_img, depth_img = render_frame_layer(model, dataset, pose, inverse_y_axis, density_threshold)

        rgbs.append(color_img)
        depths.append(depth_img)

        if save_dir != '':
            imageio.imwrite(os.path.join(save_dir,'color','%d.jpg'%i), color_img)
            imageio.imwrite(os.path.join(save_dir,'depth','%d.png'%i), depth_img)

    return rgbs, depths

def generate_poses_by_path(start,end,step_num,center,up):

    poses = []
    for i in range(step_num):
        pos = start + i * (end-start) / step_num
        pose = lookat(pos,center,up)
        poses.append(pose)

    print('Generated poses for rendering images')
    return poses

def generate_poses_by_path_center(start,end,step_num,centers,up):

    poses = []
    for i in range(step_num):
        pos = start + i * (end-start) / step_num
        pose = lookat(pos,centers[i],up)
        poses.append(pose)

    print('Generated poses for rendering images')
    return poses

def generate_poses_by_spherical(dataset,radius,ThetaStart,ThetaStep,ThetaEnd,
                                                PhiStart,PhiStep,PhiEnd,offsets,up):
    poses = []
    for theta in np.arange(ThetaStart,ThetaEnd,ThetaStep):
        for phi in np.arange(PhiStart,PhiEnd,PhiStep):
            pose = dataset.get_pose_by_spherical(theta, phi, radius, offsets, up)
            poses.append(pose)

    return poses