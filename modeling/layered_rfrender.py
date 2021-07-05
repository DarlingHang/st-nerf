
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
import math

from utils import Trigonometric_kernel, sample_pdf
from layers.RaySamplePoint import RaySamplePoint, RaySamplePoint_Near_Far
from .spacenet import SpaceNet
from .motion_net import MotionNet

from layers.render_layer import VolumeRenderer, gen_weight
import time

import copy

import pdb
class LayeredRFRender(nn.Module):

    def __init__(self, cfg, camera_num, scale = None,shift=None):
        super(LayeredRFRender, self).__init__()
        boarder_weight = cfg.MODEL.BOARDER_WEIGHT
        sample_method = cfg.MODEL.SAMPLE_METHOD
        layer_num=cfg.DATASETS.LAYER_NUM
        same_space_net = cfg.MODEL.SAME_SPACENET
        TriKernel_include_input = cfg.MODEL.TKERNEL_INC_RAW
        pose_refinement=cfg.MODEL.POSE_REFINEMENT
        use_dir = cfg.MODEL.USE_DIR
        use_deform_view=cfg.MODEL.USE_DEFORM_VIEW
        use_deform_time = cfg.MODEL.USE_DEFORM_TIME
        use_space_time = cfg.MODEL.USE_SPACE_TIME
        bkgd_use_deform_time = cfg.MODEL.BKGD_USE_DEFORM_TIME
        bkgd_use_space_time = cfg.MODEL.BKGD_USE_SPACE_TIME
        deep_rgb = (cfg.MODEL.DEEP_RGB and cfg.MODEL.USE_SPACE_TIME)
        self.coarse_ray_sample = cfg.MODEL.COARSE_RAY_SAMPLING
        self.fine_ray_sample = cfg.MODEL.FINE_RAY_SAMPLING
        self.sample_method = sample_method
        self.scale = scale
        self.shift = shift
        self.near = 0
        self.alpha = 1


        #Pose refinement part
        self.pose_refinement = pose_refinement
        if pose_refinement:
            from layers.camera_transform import CameraTransformer
            self.cam_pose=CameraTransformer(camera_num,True)

        if self.sample_method == 'NEAR_FAR':
            self.rsp_coarse = RaySamplePoint_Near_Far(self.coarse_ray_sample)   # use near far to sample points on rays
        else:
            self.rsp_coarse = RaySamplePoint(self.coarse_ray_sample)            # use bounding box to define point sampling ranges on rays

        self.layer_num = layer_num
        self.camera_num = camera_num

        self.spacenets = nn.ModuleList([])
        self.spacenets_fine = nn.ModuleList([])
        #TODO: False use space time for bkgd
        self.bkgd_spacenet = SpaceNet(include_input = TriKernel_include_input, use_dir = use_dir, use_time=bkgd_use_space_time, deep_rgb = deep_rgb)
        self.bkgd_spacenet_fine = copy.deepcopy(self.bkgd_spacenet)
        # 1,2,...,layer_num
        for i in range(layer_num):
            if i == 0:
                self.spacenets.append(SpaceNet(include_input = TriKernel_include_input, use_dir = use_dir, use_time=use_space_time, deep_rgb = deep_rgb))
            else:
                self.spacenets.append(copy.deepcopy(self.spacenets[0]))
            if same_space_net:
                spacenet_fine = self.spacenets[i]
            else:
                spacenet_fine = copy.deepcopy(self.spacenets[i])
            self.spacenets_fine.append(spacenet_fine)

        self.volume_render = VolumeRenderer(boarder_weight = boarder_weight)

        self.use_deform_view = use_deform_view
        self.use_deform_time = use_deform_time
        self.bkgd_use_deform_time = bkgd_use_deform_time
        self.use_space_time = use_space_time

        if use_deform_view:
            self.view_deform_net = MotionNet(include_input = TriKernel_include_input,c_input=4)

        self.time_deform_nets = nn.ModuleList([])

        if use_deform_time:
            for i in range(layer_num):
                self.time_deform_nets.append(MotionNet(include_input = TriKernel_include_input, c_input=4, input_time=True))
        
        if bkgd_use_deform_time:
            self.bkgd_time_deform_net = MotionNet(include_input = TriKernel_include_input, c_input=4)
        

        self.maxs = None
        self.mins = None

        self.display_layers = {}
        for layer_id in range(layer_num+1):
            print('set layer %d to be visible' % layer_id)
            self.display_layers[layer_id] = 1

    def hide_layer(self, layer_id):
        self.display_layers[layer_id] = 0

    def show_layer(self, layer_id):
        self.display_layers[layer_id] = 1

    def is_shown_layer(self,layer_id):
        # print(self.display_layers[layer_id])
        return self.display_layers[layer_id] == 1

    def set_bkgd_bbox(self,bbox):
        self.bkgd_bbox = bbox

    def set_bboxes(self,bboxes):
        self.bboxes = bboxes

    def set_bkgd_near_far(self,near_far):
        self.bkgd_near_torchfar = near_far

    def bbox_interpolation(self, float_frame_id, layer_id):
        start = self.bboxes[math.floor(float_frame_id),layer_id]
        end = self.bboxes[math.ceil(float_frame_id),layer_id]
        weight = float_frame_id - math.floor(float_frame_id)
        return torch.lerp(start, end, weight)

    '''
    INPUT

    rays: rays  (N,6)
    bboxes: bounding boxes (N,L,8,3)

    OUTPUT

    rgbs: color of each ray (N,3) 
    depths:  depth of each ray (N,1) 

    '''
    def forward(self, rays, labels, bboxes=None, only_coarse = False,near_far=None, near_far_points=[], density_threshold=0.0001,bkgd_density_threshold=0):

        near = self.near
        
        rays_frame_id = None
        rays_camera_id = None

        have_camera_id = self.use_deform_view
        have_frame_id = self.use_deform_time or self.use_space_time

        ray_size = 6
        if have_camera_id:
            ray_size += 1
        if have_frame_id:
            ray_size += 1
        
        if ray_size == rays.size(-1):
            self.retiming = False
        elif ray_size + self.layer_num == rays.size(-1):
            self.retiming = True
        else:
            print('undefined ray format in LayeredRFRender, ray dimension is ', rays.size(-1))
            exit(-1)

        # [x,y,z,dx,dy,dz,camera_id]
        if have_camera_id and not have_frame_id:
            rays_camera_id = rays[:,-1]
        # [x,y,z,dx,dy,dz,frame_id]
        elif not have_camera_id and have_frame_id:
            if not self.retiming:
                rays_frame_id = rays[:,-1]
            else:
                rays_frame_id = rays[:,6:]
        # [x,y,z,dx,dy,dz,camera_id, frame_id]
        elif have_camera_id and have_frame_id:
            if not self.retiming:
                rays_camera_id = rays[:,-2]
                rays_frame_id = rays[:,-1]
            else:
                rays_camera_id = rays[:,7]
                rays_frame_id = rays[:,7:]


        if self.pose_refinement:
            rays_o, rays_d=rays[:,:4],rays[:,4:8]
            rays_o, rays_d=self.cam_pose.forward(rays_o, rays_d)
            rays = torch.cat([rays_o, rays_d], dim=1)


        # self.bboxes (frame_num,layer_num,8,3)
        self.bboxes = self.bboxes.cuda()
        if not self.retiming:
            bboxes = self.bboxes.index_select(0,rays_frame_id.type(torch.int64)-1)
        else:
            bboxes = torch.zeros(rays.size(0),self.layer_num,8,3).cuda()
            for i in range(self.layer_num):
                # print('set layer %d frame %d bbox' % (i+1,rays_frame_id[0,i+1].type(torch.int64)-1))
                #TODO: Linear interpolation for float frame id
                #pdb.set_trace()
                bboxes[:,i] = self.bbox_interpolation(rays_frame_id[0,i+1]-1, i)
                #bboxes[:,i] = self.bboxes[math.floor(rays_frame_id[0,i+1]-1), i]


        #pdb.set_trace()

        # bboxes = torch.repeat_interleave(bboxes, repeats=rays.shape[0], dim=0)
        self.bkgd_bbox = self.bkgd_bbox.cuda()
        bboxes = torch.cat((self.bkgd_bbox.unsqueeze(0).repeat(rays.size(0),1,1,1), bboxes), 1)
        l = bboxes.size(1)
        #if self.maxs is None:
        #    print('please set max_min before use.')
        #    return None
        
        #print(self.bboxes.shape)
        #pdb.set_trace()
        bboxes1 = self.bboxes[0, :]
        #print(self.bboxes.shape)
        #pdb.set_trace()
        bboxes1 = bboxes1.unsqueeze(0).repeat(rays.size(0),1,1,1)
        bboxes1 = torch.cat((self.bkgd_bbox.unsqueeze(0).repeat(rays.size(0),1,1,1), bboxes1), 1)
        bboxes_center = torch.mean(bboxes1, 2).unsqueeze(2) #(N,L,1,3)

        #bboxes_center[:,:,0,1]=bboxes1[:,:,2,1]   synthetic
        #bboxes_center[:,:,0,0]=bboxes1[:,:,1,0]   #boxing
        #print(bboxes1.shape)
        bboxes_center[:,:,0,2]=bboxes1[:,:,1,2]   #spider
        bboxes_center_repeat = bboxes_center.repeat(1,1,8,1)         
        

        if self.scale != None:
            for i in range(len(self.scale)):
                bboxes[:,i] = (bboxes[:,i]-(bboxes_center_repeat[:,2]+bboxes_center_repeat[:,1])/2)*self.scale[i]+(bboxes_center_repeat[:,2]+bboxes_center_repeat[:,1])/2


        
        
        if self.shift != None:
            for i in range(len(self.shift)):
                if self.shift[i] == None:
                    continue
                shift_repeat =  torch.tensor( self.shift[i]).unsqueeze(0).unsqueeze(0).repeat(rays.size(0),8,1).cuda()
                bboxes[:,i] += shift_repeat



        


        ray_mask = None
        #beg = time.time()
        if self.sample_method == 'NEAR_FAR':
            assert near_far is not None, 'require near_far as input '
            sampled_rays_coarse_t, sampled_rays_coarse_xyz  = self.rsp_coarse.forward(rays , near_far = near_far)
            bkgd_sampled_rays_coarse_t, bkgd_sampled_rays_coarse_xyz = self.rsp_coarse.forward(rays , near_far = self.bkgd_near_far.repeat(rays.shape[0],1))
            rays_t = rays
            bkgd_rays_t = rays
            labels = labels.reshape(-1)
        else:
            sampled_rays_coarse_t, sampled_rays_coarse_xyz, ray_mask  = self.rsp_coarse.forward(rays, bboxes)
            #bkgd_sampled_rays_coarse_t = sampled_rays_coarse_t[0]
            #bkgd_sampled_rays_coarse_xyz = sampled_rays_coarse_xyz[0]
            bkgd_ray_mask  = ray_mask[0]

            #---------------------------------------------------
            # label_mask = (labels != 0).squeeze()
            # mask = bkgd_ray_mask & label_mask
            # mask_out = bkgd_ray_mask & (~label_mask)
            mask = bkgd_ray_mask
            #---------------------------------------------------
            rays_t = []
            for i in range(l):
                rays_t.append(rays.detach())


            #---------------------------------------------------
            # bkgd_sampled_rays_coarse_t_out = temp1[mask_out]
            # bkgd_sampled_rays_coarse_xyz_out = temp2[mask_out]
            # bkgd_rays_t_out = rays[mask_out].detach()
            #---------------------------------------------------
            
            labels = labels.reshape(-1)

            if rays_camera_id is not None:
                temp = rays_camera_id
                rays_camera_id = rays_camera_id.detach()
                # rays_camera_id_out = temp[mask_out].detach()
            if rays_frame_id is not None:
                temp = rays_frame_id
                rays_frame_id = rays_frame_id.detach()
                # rays_frame_id_out = temp[mask_out].detach()

            
            if self.shift != None:
                for i in range(len(self.shift)):
                    if self.shift[i] == None:
                        continue
                    shift_repeat =  torch.tensor( self.shift[i]).unsqueeze(0).unsqueeze(0).repeat(sampled_rays_coarse_xyz[0].shape[0],sampled_rays_coarse_xyz[0].shape[1],1).cuda()
                    sampled_rays_coarse_xyz[i] -= shift_repeat
            
            if self.scale != None:
                bboxes_center_repeat = bboxes_center.repeat(1,1,sampled_rays_coarse_xyz[0].shape[1],1)
                for i in range(len(self.scale)):
                    sampled_rays_coarse_xyz[i]=(sampled_rays_coarse_xyz[i]-(bboxes_center_repeat[:,2]+bboxes_center_repeat[:,1])/2)/self.scale[i]+(bboxes_center_repeat[:,2]+bboxes_center_repeat[:,1])/2


        rays_sum = 0
        for i in range(l):
            rays_sum += rays_t[i].size(0)
        if rays_sum > 1:
            N1 = self.coarse_ray_sample
            N2 = self.fine_ray_sample

            for i in range(l):
                sampled_rays_coarse_t[i] = sampled_rays_coarse_t[i].detach()
                sampled_rays_coarse_xyz[i] = sampled_rays_coarse_xyz[i].detach()
            #---------------------------------------------------
            # bkgd_sampled_rays_coarse_t_out = bkgd_sampled_rays_coarse_t_out.detach()
            # bkgd_sampled_rays_coarse_xyz_out = bkgd_sampled_rays_coarse_xyz_out.detach()
            # sampled_rays_coarse_xyz = NDC(sampled_rays_coarse_xyz, near_far_points)
            #---------------------------------------------------
            # self.maxs and self.mins are all None now
            # TODO: Add bkgd ray and bkgd out ray
            if self.use_deform_view:
                samples_coarse_input = sampled_rays_coarse_xyz
                scene_flow = []
                for i in range(l):
                    samples_coarse_input[i] = torch.cat([samples_coarse_input[i],rays_camera_id.view(-1,1,1).repeat(1,N1,1)], -1)
                    scene_flow.append(self.view_deform_net(samples_coarse_input[i]))
                    sampled_rays_coarse_xyz[i] = sampled_rays_coarse_xyz[i] + scene_flow[i]



                #---------------------------------------------------
                # bkgd_samples_coarse_input_out = bkgd_sampled_rays_coarse_xyz_out
                # bkgd_samples_coarse_input_out = torch.cat([bkgd_samples_coarse_input_out,rays_camera_id_out.view(-1,1,1).repeat(1,N1,1)], -1)
                # bkgd_scene_flow_out = self.view_deform_net(bkgd_samples_coarse_input_out)
                # bkgd_sampled_rays_coarse_xyz_out = bkgd_sampled_rays_coarse_xyz_out + bkgd_scene_flow_out
                #---------------------------------------------------

            if self.use_deform_time:
                
                for i in range(self.layer_num):
                    idx = ray_mask[i+1]
                    if torch.sum(idx) == 0:
                        continue
                    if not self.retiming:
                        temp_xyz = sampled_rays_coarse_xyz[i+1][idx]
                        temp_id = rays_frame_id[idx]
                        temp = torch.cat([temp_xyz,temp_id.view(-1,1,1).repeat(1,N1,1)], -1)
                    else:
                        temp_xyz = sampled_rays_coarse_xyz[i+1][idx]
                        temp_id = rays_frame_id[:,i+1][idx]
                        temp = torch.cat([temp_xyz,temp_id.view(-1,1,1).repeat(1,N1,1)], -1)

                    scene_flow = self.time_deform_nets[i](temp)
                    sampled_rays_coarse_xyz[i+1][idx] = sampled_rays_coarse_xyz[i+1][idx] + scene_flow

            if self.bkgd_use_deform_time and not self.retiming:
                samples_coarse_input = sampled_rays_coarse_xyz
                temp = torch.cat([samples_coarse_input[0],rays_frame_id.view(-1,1,1).repeat(1,N1,1)], -1)
                bkgd_scene_flow = self.bkgd_time_deform_net(temp)
                #pdb.set_trace()
                sampled_rays_coarse_xyz[0] = sampled_rays_coarse_xyz[0] + bkgd_scene_flow
            elif self.bkgd_use_deform_time and self.retiming:
                samples_coarse_input = sampled_rays_coarse_xyz
                temp = torch.cat([samples_coarse_input[0],rays_frame_id[:,0].view(-1,1,1).repeat(1,N1,1)], -1)
                bkgd_scene_flow = self.bkgd_time_deform_net(temp)
                sampled_rays_coarse_xyz[0] = sampled_rays_coarse_xyz[0] + bkgd_scene_flow

                #---------------------------------------------------
                # bkgd_samples_coarse_input_out = bkgd_sampled_rays_coarse_xyz_out
                # bkgd_samples_coarse_input_out = torch.cat([bkgd_samples_coarse_input_out,rays_frame_id_out.view(-1,1,1).repeat(1,N1,1)], -1)
                # bkgd_scene_flow_out = self.view_deform_net(bkgd_samples_coarse_input_out)
                # bkgd_sampled_rays_coarse_xyz_out = bkgd_sampled_rays_coarse_xyz_out + bkgd_scene_flow_out
                #---------------------------------------------------

            

            rgbs = []
            density = []

            if self.use_space_time and not self.retiming:
                bkgd_rgbs, bkgd_density = self.bkgd_spacenet(sampled_rays_coarse_xyz[0], rays_t[0], rays_frame_id)
                rgbs.append(bkgd_rgbs)
                density.append(bkgd_density)
                # bkgd_rgbs_out, bkgd_density_out = self.bkgd_spacenet(bkgd_sampled_rays_coarse_xyz_out, bkgd_rays_t_out, rays_frame_id_out)
            elif self.use_space_time and self.retiming:
                bkgd_rgbs, bkgd_density = self.bkgd_spacenet(sampled_rays_coarse_xyz[0], rays_t[0], rays_frame_id[:,0])
                rgbs.append(bkgd_rgbs)
                density.append(bkgd_density)
            else:
                bkgd_rgbs, bkgd_density = self.bkgd_spacenet(sampled_rays_coarse_xyz[0], rays_t[0])
                rgbs.append(bkgd_rgbs)
                density.append(bkgd_density)
                # bkgd_rgbs_out, bkgd_density_out = self.bkgd_spacenet(bkgd_sampled_rays_coarse_xyz_out, bkgd_rays_t_out)

            for i in range(1,l):
                rgbs.append(torch.zeros(sampled_rays_coarse_xyz[0].size(0),N1,3,device = rays.device))
                density.append(torch.zeros(sampled_rays_coarse_xyz[0].size(0),N1,1,device = rays.device))
                idx = ray_mask[i]
                if torch.sum(idx) == 0 or not self.is_shown_layer(i):
                    continue
                if not self.retiming and self.use_space_time:
                    temp1, temp2 = self.spacenets[i-1](sampled_rays_coarse_xyz[i][idx], rays_t[i][idx], rays_frame_id[idx].reshape(-1,1))
                elif self.retiming and self.use_space_time:
                    temp1, temp2 = self.spacenets[i-1](sampled_rays_coarse_xyz[i][idx], rays_t[i][idx], rays_frame_id[:,i][idx].reshape(-1,1))
                else:
                    temp1, temp2 = self.spacenets[i-1](sampled_rays_coarse_xyz[i][idx], rays_t[i][idx])


                
                rgbs[i][idx] = temp1
                density[i][idx] = temp2
                density[i][sampled_rays_coarse_t[i][:,:,0]<0,:] = 0.0

                if self.retiming:
                    sigma_threshold_idx = density[i] < density_threshold
                    density[i][sigma_threshold_idx] = 0
                
            
           
            density[0][sampled_rays_coarse_t[0][:,:,0] < near,:] = 0.0
            # bkgd_density_out[bkgd_sampled_rays_coarse_t_out[:,:,0]<density_threshold,:] = 0.0
            # hide bkgd
            mixed_sampled_rays_coarse_t, mixed_index = torch.sort(torch.cat(sampled_rays_coarse_t, -2), -2) # (N, l*N1,1)
            # mixed_sampled_rays_coarse_t = mixed_sampled_rays_coarse_t.unsqueeze(-1) # (N, l*N1,1)
            # pdb.set_trace()
            mixed_rgbs = torch.cat(rgbs, -2).gather(dim=1,index=mixed_index.repeat(1,1,3))
            mixed_density = torch.cat(density, -2).gather(dim=1,index=mixed_index)

            layer_color_0 = []
            layer_depth_0 = []
            layer_acc_map_0 = []
            layer_weights_0 = []
            for i in range(l):
                # print(i)
                # print(len(sampled_rays_coarse_t))
                # print(len(rgbs))
                # print(len(density))
                temp1, temp2, temp3, temp4 = self.volume_render(sampled_rays_coarse_t[i], rgbs[i], density[i])
                layer_color_0.append(temp1)
                layer_depth_0.append(temp2)
                layer_acc_map_0.append(temp3)
                layer_weights_0.append(temp4)

            # color_0_out, depth_0_out, acc_map_0_out, weights_0_out = self.volume_render(bkgd_sampled_rays_coarse_t_out, bkgd_rgbs_out, bkgd_density_out)
            # pdb.set_trace()
            mixed_color_0, mixed_depth_0, mixed_acc_map_0, mixed_weights_0 = self.volume_render(mixed_sampled_rays_coarse_t, mixed_rgbs, mixed_density)
            # pdb.set_trace()
            #torch.cuda.synchronize()
            #print('render coarse:',time.time()-beg)

            if not only_coarse:
                
                z_samples = []
                z_vals_fine = []
                samples_fine_xyz = []

                for i in range(l):
                    z_samples.append(sample_pdf(sampled_rays_coarse_t[i].squeeze(), layer_weights_0[i].squeeze()[...,1:-1], N_samples = self.fine_ray_sample))
                    z_samples[i] = z_samples[i].detach()   # (N,L)
                    temp, _ = torch.sort(torch.cat([sampled_rays_coarse_t[i].squeeze(), z_samples[i]], -1), -1) #(N, L1+L2)
                    z_vals_fine.append(temp)
                    bboxes_center_repeat = bboxes_center.repeat(1,1,3,1)
                    samples_fine_xyz.append( z_vals_fine[i].unsqueeze(-1)*rays_t[i][:,3:6].unsqueeze(1) + rays_t[i][:,:3].unsqueeze(1) ) # (N,L1+L2,3)

                    if self.shift != None:
                        if self.shift[i] == None:
                            continue
                        shift_repeat =  torch.tensor( self.shift[i]).unsqueeze(0).unsqueeze(0).repeat(samples_fine_xyz[i].shape[0],samples_fine_xyz[0].shape[1],1).cuda()
                        samples_fine_xyz[i] -= shift_repeat

                    if self.scale != None:
                        bboxes_center_repeat = bboxes_center.repeat(1,1,samples_fine_xyz[0].shape[1],1)
                        samples_fine_xyz[i]=(samples_fine_xyz[i]-(bboxes_center_repeat[:,2]+bboxes_center_repeat[:,1])/2)/self.scale[i]+(bboxes_center_repeat[:,2]+bboxes_center_repeat[:,1])/2


                    
                
                
                if self.use_deform_view:
                    samples_fine_input = samples_fine_xyz
                    for i in range(l):
                        samples_fine_input[i] = torch.cat([samples_fine_input[i],rays_camera_id.view(-1,1,1).repeat(1,N1+N2,1)], -1)
                        scene_flow = self.view_deform_net(samples_fine_input[i])
                        samples_fine_xyz[i] = samples_fine_xyz[i] + scene_flow



                    # bkgd_samples_fine_input_out = samples_fine_xyz_out
                    # bkgd_samples_fine_input_out = torch.cat([bkgd_samples_fine_input_out,rays_camera_id_out.view(-1,1,1)], -1)
                    # bkgd_scene_flow_out = self.view_deform_net(bkgd_samples_fine_input_out)
                    # samples_fine_xyz_out = samples_fine_xyz_out + bkgd_scene_flow_out

                if self.use_deform_time:
                    for i in range(1,l):
                        idx = ray_mask[i]
                        if torch.sum(idx) == 0:
                            continue
                        #pdb.set_trace()
                        temp_xyz = samples_fine_xyz[i][idx]
                        if not self.retiming:
                            temp_id = rays_frame_id[idx]
                        else:
                            temp_id = rays_frame_id[:,i][idx]

                        temp = torch.cat([temp_xyz,temp_id.view(-1,1,1).repeat(1,N1+N2,1)], -1)

                        scene_flow = self.time_deform_nets[i-1](temp)
                        samples_fine_xyz[i][idx] = samples_fine_xyz[i][idx] + scene_flow

                if self.bkgd_use_deform_time and not self.retiming:

                    samples_fine_input = samples_fine_xyz
                    temp = torch.cat([samples_fine_input[0],rays_frame_id.view(-1,1,1).repeat(1,N1+N2,1)], -1)
                    bkgd_scene_flow = self.bkgd_time_deform_net(temp)
                    samples_fine_xyz[0] = samples_fine_xyz[0] + bkgd_scene_flow
                
                elif self.bkgd_use_deform_time and self.retiming:

                    samples_fine_input = samples_fine_xyz
                    temp = torch.cat([samples_fine_input[0],rays_frame_id[:,0].view(-1,1,1).repeat(1,N1+N2,1)], -1)
                    bkgd_scene_flow = self.bkgd_time_deform_net(temp)
                    samples_fine_xyz[0] = samples_fine_xyz[0] + bkgd_scene_flow

                #beg = time.time()
                # samples_fine_xyz = NDC(samples_fine_xyz, near_far_points)
                rgbs = []
                density = []

                if self.use_space_time and not self.retiming:
                    bkgd_rgbs, bkgd_density = self.bkgd_spacenet_fine(samples_fine_xyz[0], rays_t[0], rays_frame_id)
                    rgbs.append(bkgd_rgbs)
                    density.append(bkgd_density)
                    # bkgd_rgbs_out, bkgd_density_out = self.bkgd_spacenet_fine(samples_fine_xyz_out, bkgd_rays_t_out, rays_frame_id_out)
                elif self.use_space_time and self.retiming:
                    bkgd_rgbs, bkgd_density = self.bkgd_spacenet_fine(samples_fine_xyz[0], rays_t[0], rays_frame_id[:,0])
                    if self.retiming:
                        sigma_threshold_idx = bkgd_density < bkgd_density_threshold
                        bkgd_density[sigma_threshold_idx] = 0
                    rgbs.append(bkgd_rgbs)
                    density.append(bkgd_density)
                else:
                    bkgd_rgbs, bkgd_density = self.bkgd_spacenet_fine(samples_fine_xyz[0], rays_t[0])
                    if self.retiming:
                        sigma_threshold_idx = bkgd_density < bkgd_density_threshold
                        bkgd_density[sigma_threshold_idx] = 0
                    rgbs.append(bkgd_rgbs)
                    density.append(bkgd_density)
                    # bkgd_rgbs_out, bkgd_density_out = self.bkgd_spacenet_fine(samples_fine_xyz_out, bkgd_rays_t_out)

                for i in range(1,l):
                    rgbs.append(torch.zeros(samples_fine_xyz[0].size(0),N1+N2,3,device = rays.device))
                    density.append(torch.zeros(samples_fine_xyz[0].size(0),N1+N2,1,device = rays.device))
                    idx = ray_mask[i]
                    if torch.sum(idx) == 0 or not self.is_shown_layer(i):
                        continue
                    if self.use_space_time and not self.retiming:
                        rgbs[i][idx], density[i][idx] = self.spacenets_fine[i-1](samples_fine_xyz[i][idx], rays_t[i][idx], rays_frame_id[idx].reshape(-1,1))
                    elif self.use_space_time and self.retiming:
                        rgbs[i][idx], density[i][idx] = self.spacenets_fine[i-1](samples_fine_xyz[i][idx], rays_t[i][idx], rays_frame_id[:,i][idx].reshape(-1,1))
                    else:
                        rgbs[i][idx], density[i][idx] = self.spacenets_fine[i-1](samples_fine_xyz[i][idx], rays_t[i][idx])
                    if self.retiming:
                        sigma_threshold_idx = density[i] < density_threshold
                        density[i][sigma_threshold_idx] = 0
                    #for spiderman result
                    #if i == 1:
                    #    density[i] *= 0.2
                    # For luning transpanrency 
                    #if i == 1 or i == 2:
                    #    density[i] *= 0.1
                    # teaser violin
                    #print(self.alpha)
                    if i == 2:
                        density[i] *= self.alpha




                
                    
                # bkgd_rgbs, bkgd_density = self.bkgd_spacenet_fine(bkgd_samples_fine_xyz, bkgd_rays_t, self.maxs, self.mins)
                # bkgd_rgbs_out, bkgd_density_out = self.bkgd_spacenet_fine(samples_fine_xyz_out, bkgd_rays_t_out, self.maxs, self.mins)


                mixed_z_vals_fine, mixed_index = torch.sort(torch.cat(z_vals_fine, -1), -1) #(N, L1+L2)
                mixed_z_vals_fine = mixed_z_vals_fine.unsqueeze(-1)
                #pdb.set_trace()
                mixed_rgbs = torch.cat(rgbs, -2).gather(dim=1,index=mixed_index.unsqueeze(-1).repeat(1,1,3))

                mixed_density = torch.cat(density, -2).gather(dim=1,index=mixed_index.unsqueeze(-1))

                layer_color = []
                layer_depth = []
                layer_acc_map = []
                layer_weights = []
                for i in range(l):
                    temp1,temp2,temp3,temp4 = self.volume_render(z_vals_fine[i].unsqueeze(-1), rgbs[i], density[i])
                    layer_color.append(temp1)
                    layer_depth.append(temp2)
                    layer_acc_map.append(temp3)
                    layer_weights.append(temp4)

                mixed_density[mixed_z_vals_fine < near] = 0
                color, depth, acc_map, weights = self.volume_render(mixed_z_vals_fine, mixed_rgbs, mixed_density)
                # color_out, depth_out, acc_map_out, weights_out = self.volume_render(z_vals_fine_out.unsqueeze(-1), bkgd_rgbs_out, bkgd_density_out)

                #These two lines are for visualizing the ray pdf
                self.density_fine_temp = density
                self.xyz_fine_temp = samples_fine_xyz

                if not self.sample_method == 'NEAR_FAR':
                    color_final_0 = torch.zeros(rays.size(0),3,device = rays.device)
                    color_final_0 = mixed_color_0
                    #color_final_0[mask_out] = color_0_out

                    depth_final_0 = torch.zeros(rays.size(0),1,device = rays.device)
                    depth_final_0 = mixed_depth_0
                    #depth_final_0[mask_out] = depth_0_out

                    acc_map_final_0 = torch.zeros(rays.size(0),1,device = rays.device)
                    acc_map_final_0 = mixed_acc_map_0
                    #acc_map_final_0[mask_out] = acc_map_0_out

                    layer_color_final_0 = []
                    layer_depth_final_0 = []
                    layer_acc_map_final_0 = []
                    for i in range(l):
                        layer_color_final_0.append(torch.zeros(rays.size(0),3,device = rays.device))
                        layer_color_final_0[i] = layer_color_0[i]
                        # color_final_0[mask_out] = color_0_out

                        layer_depth_final_0.append(torch.zeros(rays.size(0),1,device = rays.device))
                        layer_depth_final_0[i] = layer_depth_0[i]
                        # depth_final_0[mask_out] = depth_0_out

                        layer_acc_map_final_0.append(torch.zeros(rays.size(0),1,device = rays.device))
                        layer_acc_map_final_0[i] = layer_acc_map_0[i]
                        # acc_map_final_0[mask_out] = acc_map_0_out
                    
                else:
                    color_final_0, depth_final_0, acc_map_final_0 = mixed_color_0, mixed_depth_0, mixed_acc_map_0


                if not self.sample_method == 'NEAR_FAR':
                    color_final = torch.zeros(rays.size(0),3,device = rays.device)
                    color_final = color
                    #color_final[mask_out] = color_out

                    depth_final = torch.zeros(rays.size(0),1,device = rays.device)
                    depth_final = depth
                    #depth_final[mask_out] = depth_out

                    acc_map_final = torch.zeros(rays.size(0),1,device = rays.device)
                    acc_map_final = acc_map
                    #acc_map_final[mask_out] = acc_map_out
                    layer_color_final = []
                    layer_depth_final = []
                    layer_acc_map_final = []

                    for i in range(l):
                        layer_color_final.append(torch.zeros(rays.size(0),3,device = rays.device))
                        layer_color_final[i] = layer_color[i]
                        # color_final_0[mask_out] = color_0_out

                        layer_depth_final.append(torch.zeros(rays.size(0),1,device = rays.device))
                        layer_depth_final[i] = layer_depth[i]
                        # depth_final_0[mask_out] = depth_0_out

                        layer_acc_map_final.append(torch.zeros(rays.size(0),1,device = rays.device))
                        layer_acc_map_final[i] = layer_acc_map[i]
                        # acc_map_final_0[mask_out] = acc_map_0_out

                else:
                    color_final_0 = torch.zeros(rays.size(0),3,device = rays.device).requires_grad_()
                    depth_final_0 = torch.zeros(rays.size(0),1,device = rays.device).requires_grad_()
                    acc_map_final_0 = torch.zeros(rays.size(0),1,device = rays.device).requires_grad_()
                    color_final, depth_final, acc_map_final = color_final_0, depth_final_0, acc_map_final_0
                    layer_color_final, layer_depth_final, layer_acc_map_final = color_final_0, depth_final_0, acc_map_final_0
                    layer_color_final_0, layer_depth_final_0, layer_acc_map_final_0 = color_final_0, depth_final_0, acc_map_final_0
                    bkgd_color_final, bkgd_depth_final, bkgd_acc_map_final = color_final_0, depth_final_0, acc_map_final_0
                    bkgd_color_final_0, bkgd_depth_final_0, bkgd_acc_map_final_0 = color_final_0, depth_final_0, acc_map_final_0
            # Only coarse
            else:
                if not self.sample_method == 'NEAR_FAR':
                    color_final_0 = torch.zeros(rays.size(0),3,device = rays.device)
                    color_final_0 = mixed_color_0
                    #color_final_0[mask_out] = color_0_out

                    depth_final_0 = torch.zeros(rays.size(0),1,device = rays.device)
                    depth_final_0 = mixed_depth_0
                    #depth_final_0[mask_out] = depth_0_out

                    acc_map_final_0 = torch.zeros(rays.size(0),1,device = rays.device)
                    acc_map_final_0 = mixed_acc_map_0
                    #acc_map_final_0[mask_out] = acc_map_0_out

                    layer_color_final_0 = []
                    layer_depth_final_0 = []
                    layer_acc_map_final_0 = []


                    for i in range(l):
                        layer_color_final_0.append(torch.zeros(rays.size(0),3,device = rays.device))
                        layer_color_final_0[i] = layer_color_0[i]
                        
                        # color_final_0[mask_out] = color_0_out

                        layer_depth_final_0.append(torch.zeros(rays.size(0),1,device = rays.device))
                        layer_depth_final_0[i] = layer_depth_0[i]
                        # depth_final_0[mask_out] = depth_0_out

                        layer_acc_map_final_0.append(torch.zeros(rays.size(0),1,device = rays.device)) 
                        layer_acc_map_final_0[i] = layer_acc_map_0[i]
                        # acc_map_final_0[mask_out] = acc_map_0_out

                else:
                    color_final_0, depth_final_0, acc_map_final_0 = mixed_color_0, mixed_depth_0, mixed_acc_map_0

                color_final, depth_final, acc_map_final = mixed_color_0, mixed_depth_0, mixed_acc_map_0
                layer_color_final , layer_depth_final, layer_acc_map_final = layer_color_final_0, layer_depth_final_0, layer_acc_map_final_0
                #color_final, depth_final, acc_map_final = layer_color_final_0[2], layer_depth_final_0[2], layer_acc_map_final_0[2]

        fine_mixed = (color_final, depth_final, acc_map_final)
        coarse_mixed = (color_final_0, depth_final_0, acc_map_final_0)
        fine_layer = []
        coarse_layer = []
        for i in range(l):
            fine_layer.append((layer_color_final[i], layer_depth_final[i], layer_acc_map_final[i])) 
            coarse_layer.append((layer_color_final_0[i], layer_depth_final_0[i], layer_acc_map_final_0[i]))

        
        return fine_mixed , coarse_mixed, fine_layer, coarse_layer, ray_mask
        #return fine_layer[1] , coarse_layer[1], fine_layer, coarse_layer, ray_mask



    def set_max_min(self, maxs, mins):
        self.maxs = maxs
        self.mins = mins