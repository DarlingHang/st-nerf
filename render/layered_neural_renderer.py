from config import cfg
import imageio
import os
import numpy as np
import torch
from data import make_ray_data_loader_render, get_iteration_path
from modeling import build_layered_model
from utils import layered_batchify_ray, add_two_dim_dict
from .render_functions import *
from robopy import *
import time

from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp 
from scipy.interpolate import splprep, splev
import pdb

class LayeredNeuralRenderer:

    def __init__(self,cfg,scale=None,shift=None,rotation = None,s_shift = None,s_scale=None,s_alpha=None):
        self.alpha = None
        self.cfg = cfg
        self.scale = scale
        self.shift = shift
        self.rotation = rotation
        self.s_shift = s_shift
        self.s_scale = s_scale
        self.s_alpha = s_alpha


        if s_shift != None:
            self.shift = self.s_shift[0]

        if s_scale != None:
            self.scale = self.s_scale[0]

        if s_alpha != None:
            self.alpha = self.s_alpha[0]


        # The dictionary save all rendered images and videos
        self.dataset_dir = self.cfg.OUTPUT_DIR
        self.output_dir = os.path.join(self.cfg.OUTPUT_DIR,'rendered')

        self.dataset, self.model = self.load_dataset_model()

        # {0,1} dictionary, 1 means display, 0 means hide, key is [layer_id]
        self.display_layers = {}

        # (0,1,2,...,LAYER_NUM)
        for layer_id in range(cfg.DATASETS.LAYER_NUM+1):
            self.display_layers[layer_id] = 1

        # Intrinsic for all rendered image, update when firstly load dataset
        self.gt_poses = self.dataset.poses
        self.gt_Ks = self.dataset.Ks

        # self.near = 0.1
        self.far = 20.0

        # Each layer will have a min-max frame range
        self.min_frame = [1+cfg.DATASETS.FRAME_OFFSET for i in range(cfg.DATASETS.LAYER_NUM+1)]
        self.max_frame = [cfg.DATASETS.FRAME_NUM+cfg.DATASETS.FRAME_OFFSET for i in range(cfg.DATASETS.LAYER_NUM+1)]

        self.images = []
        self.depths = []

        # Total image number rendered and saved in renderer
        self.image_num = 0
        # Total frame number and layer number, use it carefully, because it may not be all loaded into model
        self.frame_num = cfg.DATASETS.FRAME_NUM
        self.layer_num = cfg.DATASETS.LAYER_NUM
        self.camera_num = self.dataset.camera_num
        self.min_camera_id = 0
        self.max_camera_id = self.camera_num-1

        self.fps = 25
        self.height = cfg.INPUT.SIZE_TEST[1]
        self.width = cfg.INPUT.SIZE_TEST[0]

        #Count for save multiple videos
        self.save_count = 0

        # All rendered poses and intrinsics aligned with images
        self.poses = []
        self.Ks = []
        # Corresponding to each pose, we will have mutiple (layer_id, frame_id) pairs to identify the visible layers and frames.
        # Example [[(0,1),(1,1)],[(0,1),(1,2)],...] represent [(layer_0, frame_1), (layer_1,frame_1)] for poses[0] and so on
        self.layer_frame_pairs = []

        # Trace one layer (lookat to the center of layer), -1 means no trace layer
        self.trace_layer = -1

        # auto saving dir
        self.dir_name = ''

    def set_save_count(self, cnt):
        self.save_count = cnt


    def load_dataset_model(self):
        para_file = get_iteration_path(self.dataset_dir)
        print(para_file)

        if para_file is None:
            assert 'training model does not exist'
        
        _, dataset = make_ray_data_loader_render(cfg)
        
        model = build_layered_model(cfg, dataset.camera_num, scale = self.scale, shift=self.shift)
        
        model.set_bkgd_bbox(dataset.datasets[0][0].bbox)    # frame0 layer0 's bbox  From pointcloud
        model.set_bboxes(dataset.bboxes)        # human's bbox  From pointscloud
        model_dict = model.state_dict()
        dict_0 = torch.load(os.path.join(para_file),map_location='cuda')

        model_dict = dict_0['model']
        model_new_dict = model.state_dict()
        offset = {k: v for k, v in model_new_dict.items() if k not in model_dict}
        for k,v in offset.items():
            model_dict[k] = v
        model.load_state_dict(model_dict)

        model.cuda()

        return dataset, model

    
    def check_label(self):
        output = os.path.join(self.output_dir,'masked_images')
        if not os.path.exists(output):
            os.makedirs(output)
        for i in range(self.frame_num):
            output_f = os.path.join(output, 'frame%d' % i)
            if not os.path.exists(output_f):
                os.makedirs(output_f)
            for j in range(self.camera_num):
                image, label = self.dataset.get_image_label(j, i)
                image = image.permute(1,2,0)
                image[label[0,...]==0] = 0
                imageio.imwrite(os.path.join(output_f,'%d.jpg'% j), image)

        return




    # The function set the pose, before using it, set the right frame duration for each layer
    def set_path_lookat(self, start,end,step_num,center,up):
        
        # Generate poses
        if self.trace_layer == -1:
            poses = generate_poses_by_path(start,end,step_num,center,up)
        else:
            centers = []
            temp = center
            for idx in range(step_num):
                frame_id = int((self.max_frame-self.min_frame)/step_num*(idx+1)) + self.min_frame
                frame_dic = self.datasets[frame_id]
                for layer_id in frame_dic:
                    if layer_id == self.trace_layer:
                        temp = frame_dic[layer_id].center
                centers.append(temp)
            poses = generate_poses_by_path_center(start,end,step_num,centers,up)
        
        self.poses = self.poses + poses

        for idx in range(len(poses)+1):
            layer_frame_pair = []
            for layer_id in range(self.layer_num+1):
                if self.is_shown_layer(layer_id):
                    frame_id = int((self.max_frame[layer_id]-self.min_frame[layer_id])/len(poses)*(idx)) + self.min_frame[layer_id]
                    layer_frame_pair.append((layer_id,frame_id))
            self.layer_frame_pairs.append(layer_frame_pair)

    def set_path_gt_poses(self):
        poses = []
        for i in range(self.dataset.poses.shape[0]):
            poses.append(self.dataset.poses[i])

        self.poses = self.poses + poses
        self.Ks = self.Ks + self.gt_Ks

        for idx in range(len(poses)+1):
            layer_frame_pair = []
            for layer_id in range(self.layer_num+1):
                if self.is_shown_layer(layer_id):
                    frame_id = int((self.max_frame[layer_id]-self.min_frame[layer_id])/len(poses)*(idx)) + self.min_frame[layer_id]
                    layer_frame_pair.append((layer_id,frame_id))
            self.layer_frame_pairs.append(layer_frame_pair)


    def set_path_fixed_gt_poses(self,id,num=None):
        poses = []
        Ks = []
        if self.s_shift != None:
            s_shift_start = np.array(self.s_shift[0])
            s_shift_end = np.array(self.s_shift[1])
            s_shift_step = (s_shift_end-s_shift_start)/(num-1)
            self.s_shift_frame = []

        if self.s_scale != None:
            s_scale_start = np.array(self.s_scale[0])
            s_scale_end = np.array(self.s_scale[1])
            s_scale_step = (s_scale_end-s_scale_start)/(num-1)
            self.s_scale_frame = []
            

        for i in range(num):
            poses.append(self.dataset.poses[id])
            K = self.dataset.Ks[id]
            
            # EXPEDIENCY
            if K == None:
                K = self.dataset.Ks[id+1]
            Ks.append(K)
            if self.s_shift != None:
                self.s_shift_frame.append((s_shift_start+i*s_shift_step).tolist())

            if self.s_scale != None:
                self.s_scale_frame.append((s_scale_start+i*s_scale_step).tolist())

        self.poses = self.poses + poses
        self.Ks = self.Ks + Ks

        for idx in range(len(poses)+1):
            layer_frame_pair = []
            for layer_id in range(self.layer_num+1):
                if self.is_shown_layer(layer_id):
                    frame_id = int((self.max_frame[layer_id]-self.min_frame[layer_id])/len(poses)*(idx)) + self.min_frame[layer_id]
                    layer_frame_pair.append((layer_id,frame_id))
            self.layer_frame_pairs.append(layer_frame_pair)


    def set_smooth_path_poses(self,step_num, around=False, smooth_time = False):

        if self.s_shift != None:
            s_shift_start = np.array(self.s_shift[0])
            s_shift_end = np.array(self.s_shift[1])
            s_shift_step = (s_shift_end-s_shift_start)/(step_num-1)
            self.s_shift_frame = []

        if self.s_alpha != None:
            s_alpha_start = self.s_alpha[0]
            s_alpha_end = self.s_alpha[1]
            s_alpha_step = (s_alpha_end-s_alpha_start)/(step_num-1)
            self.s_alpha_frame = []

        poses = []
        Rs = self.gt_poses[self.min_camera_id:self.max_camera_id+1,:3,:3].cpu().numpy()
        Ts = self.gt_poses[self.min_camera_id:self.max_camera_id+1,:3,3].cpu().numpy()
        #print(Ts)
        
        key_frames = [i for i in range(self.min_camera_id,self.max_camera_id+1)]
        # Only use the first and the last
        if not around:
            temp = [Rs[0],Rs[-1]]
            Rs = np.array(temp)
            # key_frames = [self.min_camera_id,self.max_camera_id]

        # interp_frames = [(i * (self.max_camera_id-self.min_camera_id) / (step_num-1) + self.min_camera_id) for i in range(step_num)]

        # Rs = R.from_matrix(Rs)
        # slerp = Slerp(key_frames, Rs)
        # interp_Rs = slerp(interp_frames).as_matrix()

        # x = Ts[:,0]
        # y = Ts[:,1]
        # z = Ts[:,2]

        # tck, u0 = splprep([x,y,z])
        # u_new = [i / (step_num-1)  for i in range(step_num)]
        # new_points = splev(u_new,tck)

        # new_points = np.stack(new_points, axis=1)

        K0 = self.gt_Ks[self.min_camera_id]
        K1 = self.gt_Ks[self.max_camera_id]

        if self.s_scale != None:
            s_scale_start = np.array(self.s_scale[0])
            s_scale_end = np.array(self.s_scale[1])
            s_scale_step = (s_scale_end-s_scale_start)/(step_num-1)
            self.s_scale_frame = []
        for i in range(step_num):
            pose = np.zeros((4,4))
            pose[:3,:3] = Rs[0]
            pose[:3,3] = Ts[0]
            pose[3,3] = 1
            poses.append(pose)

            # K = (K1 - K0) * i / (step_num - 1) + K0
            K = K0
        
            self.Ks.append(K)
            if self.s_scale != None:
                self.s_scale_frame.append((s_scale_start+i*s_scale_step).tolist())

            if self.s_shift != None:
                self.s_shift_frame.append((s_shift_start+i*s_shift_step).tolist())

            if self.s_alpha != None:
                self.s_alpha_frame.append((s_alpha_start+i*s_alpha_step))

        self.poses = self.poses + poses

        # Generate corresponding layer id and frame id for poses
        for idx in range(len(poses)+1):
            layer_frame_pair = []
            for layer_id in range(self.layer_num+1):
                if self.is_shown_layer(layer_id):

                    if not smooth_time:
                        frame_id = int((self.max_frame[layer_id]-self.min_frame[layer_id])/len(poses)*(idx)) + self.min_frame[layer_id]
                    else:
                        frame_id = (self.max_frame[layer_id]-self.min_frame[layer_id])/len(poses)*(idx) + self.min_frame[layer_id]
                    layer_frame_pair.append((layer_id,frame_id))
            self.layer_frame_pairs.append(layer_frame_pair)

    def load_path_poses(self,poses):
        self.poses = poses
        step_num = len(poses)
        K0 = self.gt_Ks[self.min_camera_id]
        K1 = self.gt_Ks[self.max_camera_id-1]
        for i in range(step_num):
            K = (K1 - K0) * i / (step_num - 1) + K0
            self.Ks.append(K)

        for idx in range(len(poses)+1):
            layer_frame_pair = []
            for layer_id in range(self.layer_num+1):
                if self.is_shown_layer(layer_id):
                    frame_id = int((self.max_frame[layer_id]-self.min_frame[layer_id])/len(poses)*(idx)) + self.min_frame[layer_id]
                    layer_frame_pair.append((layer_id,frame_id))
            self.layer_frame_pairs.append(layer_frame_pair)


    def load_cams_from_path(self, path):

        campose = np.load(os.path.join(path, 'RT_c2w.npy'))
        Ts = np.zeros((campose.shape[0],4,4))
        Ts[:,:3,:] = campose.reshape(-1, 3, 4)
        Ts[:,3,3] = 1.

        #scale
        Ts[:,:3,3] = self.cfg.DATASETS.SCALE * Ts[:,:3,3]

        Ks = np.load(os.path.join(path, 'K.npy'))
        Ks = Ks.reshape(-1, 3, 3)

        self.poses = Ts
        self.Ks = torch.from_numpy(Ks.astype(np.float32))

        for idx in range(len(self.poses)+1):
            layer_frame_pair = []
            for layer_id in range(self.layer_num+1):
                if self.is_shown_layer(layer_id):
                    frame_id = int((self.max_frame[layer_id]-self.min_frame[layer_id])/len(self.poses)*(idx)) + self.min_frame[layer_id]
                    layer_frame_pair.append((layer_id,frame_id))
            self.layer_frame_pairs.append(layer_frame_pair)


    def render_pose(self, pose, K, layer_frame_pair, density_threshold=0,bkgd_density_threshold=0):
        print(K)
        print(pose)
        #print(K)
        H = self.dataset.height
        W = self.dataset.width

        rays, labels, bbox, near_far = self.dataset.get_rays_by_pose_and_K(pose, K, layer_frame_pair)


        rays = rays.cuda()
        bbox = bbox.cuda()
        labels = labels.cuda()
        near_far = near_far.cuda()

        with torch.no_grad():
            stage2, stage1, stage2_layer, stage1_layer, _ = layered_batchify_ray(self.model, rays, labels, bbox, near_far=near_far, density_threshold=density_threshold,bkgd_density_threshold=bkgd_density_threshold)

            color = stage2[0].reshape(H,W,3)
            depth = stage2[1].reshape(H,W,1)
            depth[depth < 0] = 0
            depth = depth / self.far
            color_layer = [i[0].reshape(H,W,3) for i in stage2_layer]
            depth_layer = []
            for temp in stage2_layer:
                depth_1 = temp[1].reshape(H,W,1)
                depth_1[depth < 0] = 0
                depth_1 = depth_1 / self.far
                depth_layer.append(depth_1)
      
            return color,depth,color_layer,depth_layer


    


    


    def render_path(self,inverse_y_axis=False,density_threshold=0,bkgd_density_threshold=0, auto_save=True):

        if self.dir_name == '':
            save_dir = os.path.join(self.output_dir,'video_%d' % self.save_count,'mixed')
        else:
            save_dir = os.path.join(self.output_dir,self.dir_name,'cam%d' % self.save_count,'mixed')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            os.mkdir(os.path.join(save_dir,'color'))
            os.mkdir(os.path.join(save_dir,'depth'))

        file_poses = open(os.path.join(save_dir,'poses'),mode='w')
        for pose in self.poses:
            file_poses.write(str(pose)+"\n")
        file_poses.close()
        
        file_Ks = open(os.path.join(save_dir,'Ks'),mode='w')
        for K in self.Ks:
            file_Ks.write(str(K)+"\n")	
        file_Ks.close()


        self.images = []
        self.depths = []
        self.images_layer = [[] for i in range(self.layer_num+1)]
        self.depths_layer = [[] for i in range(self.layer_num+1)]

        self.image_num = 0
        
        for idx in range(len(self.poses)):
            print('Rendering image %d' % idx)
            K = self.Ks[idx]
            pose = self.poses[idx]
            layer_frame_pair = self.layer_frame_pairs[idx]
            if self.s_shift != None:
                self.model.shift = self.s_shift_frame[idx]
            if self.s_scale != None:
                self.model.scale = self.s_scale_frame[idx]
            if self.s_alpha != None:
                self.model.alpha = self.s_alpha_frame[idx]

            color,depth,color_layer,depth_layer = self.render_pose(pose, K, layer_frame_pair, density_threshold,bkgd_density_threshold)


            if inverse_y_axis:
                color = torch.flip(color,[0])
                depth = torch.flip(depth,[0])
                color_layer = [torch.flip(i,[0]) for i in color_layer]
                depth_layer = [torch.flip(i,[0]) for i in depth_layer]

            color = color.cpu()
            depth = depth.cpu()
            color_layer = [i.cpu() for i in color_layer]
            depth_layer = [i.cpu() for i in depth_layer]

            if auto_save:
                #print(rgb.shape)
                imageio.imwrite(os.path.join(save_dir,'color','%d.jpg'% self.image_num), color)
                imageio.imwrite(os.path.join(save_dir,'depth','%d.png'% self.image_num), depth)
                self.images.append(color)
                self.depths.append(depth)
                '''for layer_id in range(self.layer_num+1):
                    if self.is_shown_layer(layer_id):
                        if self.dir_name == '':
                            save_dir = os.path.join(self.output_dir,'video_%d' % self.save_count,str(layer_id))
                        else:
                            save_dir = os.path.join(self.output_dir,self.dir_name,'video_%d' % self.save_count,str(layer_id))
                        if not os.path.exists(save_dir):
                            os.makedirs(save_dir)
                            os.mkdir(os.path.join(save_dir,'color'))
                            os.mkdir(os.path.join(save_dir,'depth'))

                        imageio.imwrite(os.path.join(save_dir,'color','%d.jpg'% self.image_num), color_layer[layer_id])
                        imageio.imwrite(os.path.join(save_dir,'depth','%d.png'% self.image_num), depth_layer[layer_id])
                        self.images_layer[layer_id].append(color)
                        self.depths_layer[layer_id].append(depth)
                '''
                self.image_num += 1
        



    

    def retime_by_key_frames(self, layer_id, key_frames_layer, key_frames):
        
        assert (len(key_frames_layer) == len(key_frames))

        for i in range(len(self.layer_frame_pairs)):
            for j in range(len(self.layer_frame_pairs[i])):
                layer, frame = self.layer_frame_pairs[i][j]
                #Retiming the corresponding layer
                if layer == layer_id:
                    idx_start = -1
                    idx_end = -1
                    weight = 0
                    for idx in range(len(key_frames)):
                        if frame <= key_frames[idx]:
                            idx_end = idx
                            idx_start = idx_end-1
                            end = key_frames[idx]
                            start = 0
                            if idx == 0:
                                start = self.min_frame[layer]
                            else:
                                start = key_frames[idx-1]
                            weight = (frame-start) / (end-start)
                            # print('frame %d, start %d, end %d' % (frame,start,end))
                            # print('idx_end %d, idx_start %d' % (idx_end, idx_start))
                            break

                    new_end = 0
                    new_start = 0
                    # print('123')
                    # print('idx_end %d, idx_start %d' % (idx_end, idx_start))
                    if idx_start == -1 and idx_end == 0:
                        weight = (frame-self.min_frame[layer]) / (key_frames[0] - self.min_frame[layer])
                        new_start = self.min_frame[layer]
                        new_end = key_frames_layer[0]
                    elif idx_start >= -1 and idx_end != -1:
                        new_start = key_frames_layer[idx_start]
                        new_end = key_frames_layer[idx_start+1]
                    elif idx_start == -1 and idx_end == -1:
                        weight = (frame-key_frames[-1]) / (self.max_frame[layer] - key_frames[-1])
                        new_start = key_frames_layer[-1]
                        new_end = self.max_frame[layer]
                    else:
                        print('Undefined branch', 'start idx is %d, end idx is %d' % (idx_start,idx_end))
                        exit(-1)
 
                    new_frame = round(weight * (new_end - new_start) + new_start)
                    # print('new end is %d, new start is %d' % (new_end,new_start))
                    # print('layer %d: old frame is %d, new is %d, weight %f' % (layer,frame,new_frame,weight))
                    self.layer_frame_pairs[i][j] = (layer, new_frame)
        
        # exit(0)
                    


    def render_path_walking(self,inverse_y_axis=False,density_threshold=0,bkgd_density_threshold=0, auto_save=True):

        self.images = []
        self.depths = []
        self.images_layer = [[] for i in range(self.layer_num+1)]
        self.depths_layer = [[] for i in range(self.layer_num+1)]

        self.image_num = 0

        for idx in range(len(self.poses)):
            print('Rendering image %d' % idx)
            K = self.Ks[idx]
            pose = self.poses[idx]
            layer_frame_pair = self.layer_frame_pairs[idx]

            color,depth,color_layer,depth_layer = self.render_pose(pose, K, layer_frame_pair, density_threshold,bkgd_density_threshold)

            if inverse_y_axis:
                color = torch.flip(color,[0])
                depth = torch.flip(depth,[0])
                color_layer = [torch.flip(i,[0]) for i in color_layer]
                depth_layer = [torch.flip(i,[0]) for i in depth_layer]

            color = color.cpu()
            depth = depth.cpu()
            color_layer = [i.cpu() for i in color_layer]
            depth_layer = [i.cpu() for i in depth_layer]

            if auto_save:
                save_dir = os.path.join(self.output_dir,'mixed')
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                    os.mkdir(os.path.join(save_dir,'color'))
                    os.mkdir(os.path.join(save_dir,'depth'))

                #print(rgb.shape)
                imageio.imwrite(os.path.join(save_dir,'color','%d.jpg'% self.image_num), color)
                imageio.imwrite(os.path.join(save_dir,'depth','%d.png'% self.image_num), depth)
                self.images.append(color)
                self.depths.append(depth)
                for layer_id in range(self.layer_num+1):
                    save_dir = os.path.join(self.output_dir,str(layer_id))
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                        os.mkdir(os.path.join(save_dir,'color'))
                        os.mkdir(os.path.join(save_dir,'depth'))

                    imageio.imwrite(os.path.join(save_dir,'color','%d.jpg'% self.image_num), color_layer[layer_id])
                    imageio.imwrite(os.path.join(save_dir,'depth','%d.png'% self.image_num), depth_layer[layer_id])
                    self.images_layer[layer_id].append(color)
                    self.depths_layer[layer_id].append(depth)

                color_hide = color_layer[0].clone()
                index = depth_layer[2]<depth_layer[0]
                index = torch.cat([index,index,index],dim=2)
                index = torch.logical_and(index,color_layer[2]!=0)
                color_hide[index] = color_layer[2][index]
                save_dir = os.path.join(self.output_dir,"02")
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                    os.mkdir(os.path.join(save_dir,'color'))
                imageio.imwrite(os.path.join(save_dir,'color','%d.jpg'% self.image_num), color_hide)


                
            

            self.image_num += 1


    def save_poses(self, path):
        np.save(path, self.poses)

    # Save video for the whole rendered data
    def save_video(self):
        
        if len(self.images) != 0:
            if self.dir_name == '':
                video_dir = os.path.join(self.output_dir,'video')
                if not os.path.exists(video_dir):
                    os.mkdir(video_dir)
            else:
                video_dir = os.path.join(self.output_dir,self.dir_name,'video')
                if not os.path.exists(video_dir):
                    os.mkdir(video_dir)

            imageio.mimwrite(video_dir + '/color_%d.mp4' % self.save_count, self.images, fps = self.fps, quality = 8)
            imageio.mimwrite(video_dir + '/depth_%d.mp4' % self.save_count, self.depths, fps = self.fps, quality = 8)
            self.save_count += 1
        else:
            print('Warning: Cannot generate video for all rendered images, data is empty.')

    # Helper functions
    def set_save_dir(self, dir_name):
        self.dir_name = dir_name

    def set_fps(self, fps):
        self.fps = fps

    def get_center_frame_layer(self, frame_id, layer_id):
        return self.datasets[frame_id][layer_id].center

    # Layer display functions
    def hide_layer(self, layer_id):
        self.model.hide_layer(layer_id)
        self.display_layers[layer_id] = 0

    def show_layer(self, layer_id):
        self.model.show_layer(layer_id)
        self.display_layers[layer_id] = 1

    def is_shown_layer(self,layer_id):
        return self.display_layers[layer_id] == 1

    # TODO
    # Save rendered dataset to save_dir
    def save_dataset(self, save_dir):
        save_path = os.path.join(self.dataset_dir, save_dir)

        for image_id in self.poses:
            pass

    def set_frame_duration(self, min_frame, max_frame, layer_id = -1):
        if layer_id == -1:
            for i in range(self.layer_num+1):
                self.min_frame[i] = min_frame
                self.max_frame[i] = max_frame
        else:
            self.min_frame[layer_id] = min_frame
            self.max_frame[layer_id] = max_frame

    def set_pose_duration(self, min_camera_id, max_camera_id):
        self.min_camera_id = min_camera_id
        self.max_camera_id = max_camera_id

    def invert_poses(self):
        self.poses.reverse()
        self.Ks.reverse()

    def save_path(self):
        pass
    
    def load_path(self):
        pass

    def load_rendered_images(self):
        pass
    
    # Load all pathes and iamges
    def load_dataset(self, path):
        self.load_path()
        self.load_rendered_images()
        pass

    # High level rendering features
    def set_trace_layer(self, layer_id):
        self.trace_layer = layer_id

    def render_path_frame_layer(self, frame_id, layer_id, poses, inverse_y_axis=False,density_threshold=0, bkgd_density_threshold=0,auto_save=True):

        model = self.models[frame_id][layer_id]
        dataset = self.datasets[frame_id][layer_id]

        save_dir = ''
        if auto_save:
            save_dir = os.path.join(self.output_dir,str(layer_id))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                os.mkdir(os.path.join(save_dir,'color'))
                os.mkdir(os.path.join(save_dir,'depth'))
                

        rgbs, depths = render_path_frame_layer(model, dataset, poses, inverse_y_axis,density_threshold,bkgd_density_threshold, save_dir)

        for image_id in range(len(poses)):
            add_two_dim_dict(self.images,self.image_num + image_id, layer_id, rgbs[image_id])
            add_two_dim_dict(self.depths,self.image_num + image_id, layer_id, depths[image_id])
        
        return

    # layer_id for center, frame_id for bbox, scale for zoom in (bigger than 1)
    def zoom_in(self, layer_id, frame_id, scale):

        center = torch.Tensor(self.dataset.datasets[layer_id][frame_id].center)
        print(center)
        for idx in range(self.gt_poses.size(0)):
            self.gt_poses[idx,:3,3] = center + 1/scale * (self.gt_poses[idx,:3,3] - center)
        
        return
        
    def set_near(self, near):
        self.model.near = near