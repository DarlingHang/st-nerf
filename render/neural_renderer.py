from config import cfg
from utils import add_two_dim_dict, add_three_dim_dict
import imageio
import os
import numpy as np
import torch

from .render_functions import *
from .bkgd_renderer import PrRender

class NeuralRenderer:

    def __init__(self, cfg, load_all = False, frame_id = 0, layer_id = 0):
        # Two dimension dictionary, key is [frame_id, layer_id]
        self.datasets = {}
        # Two dimension dictionary, key is [frame_id, layer_id]
        self.models = {}
        # {0,1} dictionary, 1 means display, 0 means hide, key is [layer_id]
        self.display_layers = {}

        self.cfg = cfg


        # Intrinsic for all rendered image, update when firstly load dataset
        self.K = None
        self.near = 0.1
        self.far = 50.0

        self.min_frame = 0
        self.max_frame = 0
        
        if not load_all:
            
            if frame_id != 0 and layer_id != 0:
                self.add_frame_layer(frame_id, layer_id)
                self.min_frame = frame_id
                self.max_frame = frame_id
            else:
                print("Successfully initialize an empty nerual renderer, please feed data into it before render~")

        else:
            for frame_id in range(cfg.DATASETS.FRAME_NUM):
                for layer_id in range(cfg.DATASETS.LAYER_NUM):
                    self.add_frame_layer(frame_id+1, layer_id+1)
            self.min_frame = 1
            self.max_frame = cfg.DATASETS.FRAME_NUM

        # Two dimension dictionary (image_id, layer_id)
        self.images = {}
        self.depths = {}

        # Total image number rendered and saved in renderer
        self.image_num = 0
        # Total frame number and layer number, use it carefully, because it may not be all loaded into model
        self.frame_num = cfg.DATASETS.FRAME_NUM
        self.layer_num = cfg.DATASETS.LAYER_NUM

        self.fps = 30
        self.height = cfg.INPUT.SIZE_TEST[1]
        self.width = cfg.INPUT.SIZE_TEST[0]

        #Count for save multiple videos
        self.save_count = 0

        # All rendered poses aligned with images
        self.poses = []

        # Trace one layer (lookat to the center of layer), -1 means no trace layer
        self.trace_layer = -1

        # The dictionary save all rendered images and videos
        self.dataset_dir = self.cfg.OUTPUT_DIR
        self.output_dir = os.path.join(self.cfg.OUTPUT_DIR,'rendered')

        # Setting for background renderer
        self.bkgd_model_path = os.path.join(self.cfg.DATASETS.TRAIN, 'background', 'textured.obj')
        if not os.path.exists(self.bkgd_model_path):
            self.bkgd_renderer = None
            self.display_layers[0] = 0
            print('Warning: There is no background files in ', self.bkgd_model_path, ', so background will not be rendered.')
        else:
            # Load mesh
            self.bkgd_renderer = PrRender((self.width, self.height))
            self.bkgd_renderer.load_mesh(self.bkgd_model_path)
            self.display_layers[0] = 1
            print('Successfully load background')

    # Load [frame,layer] model and dataset into renderer
    def add_frame_layer(self, frame_id, layer_id):

        dataset, model = load_dataset_model_frame_layer(self.cfg, frame_id, layer_id)

        if dataset != None and model != None:
            add_two_dim_dict(self.datasets,frame_id,layer_id,dataset)
            add_two_dim_dict(self.models,frame_id,layer_id,model)
            # Defaultly show layer
            self.show_layer(layer_id)
            # Initialize camera intrinsic
            if self.K is None:
                self.K = dataset.K
            print('Successfully load dataset and model of frame %d, layer %d, Done.' % (frame_id, layer_id))
        else:
            print('Warning: cannot find model of frame %d, layer %d. Skip.' % (frame_id, layer_id))

    def render_path_frame_layer(self, frame_id, layer_id, poses, inverse_y_axis=False,density_threshold=0, auto_save=True):

        model = self.models[frame_id][layer_id]
        dataset = self.datasets[frame_id][layer_id]

        save_dir = ''
        if auto_save:
            save_dir = os.path.join(self.output_dir,str(layer_id))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                os.mkdir(os.path.join(save_dir,'color'))
                os.mkdir(os.path.join(save_dir,'depth'))
                

        rgbs, depths = render_path_frame_layer(model, dataset, poses, inverse_y_axis,density_threshold, save_dir)

        for image_id in range(len(poses)):
            add_two_dim_dict(self.images,self.image_num + image_id, layer_id, rgbs[image_id])
            add_two_dim_dict(self.depths,self.image_num + image_id, layer_id, depths[image_id])
        
        return

    def render_path_frame(self, frame_id, start,end,step_num,center,up,inverse_y_axis=False,density_threshold=0, auto_save=True):

        print('Rendering path for frame %d, totally %d images' %(frame_id, step_num))
        frame_dic = self.models[frame_id]

        #Generate poses for path
        poses = generate_poses_by_path(start,end,step_num,center,up)

        # Render each loaded layer
        for layer_id in frame_dic:
            if not self.is_shown_layer(layer_id):
                print('Layer %d is hidden. Skip.' % layer_id)
                continue
            print('Rendering images for frame %d, layer %d / %d.' % (frame_id, layer_id, self.layer_num))
            self.render_path_frame_layer(frame_id, layer_id, poses, inverse_y_axis,density_threshold, auto_save)            
        
        # Render background for each frame
        if self.bkgd_renderer is not None:
            self.render_path_bkgd(poses, auto_save)
        # Update total image number
        self.image_num += step_num

    def render_path(self,start,end,step_num,center,up,inverse_y_axis=False,density_threshold=0, auto_save=True):

        #Generate poses for path
        poses = []
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

        for idx in range(step_num):
            
            pose = poses[idx]

            frame_id = int((self.max_frame-self.min_frame)/step_num*(idx+1)) + self.min_frame
            print('Rendering path for frame %d, totally %d images, now %d' %(frame_id, step_num, idx+1))

            frame_dic = self.models[frame_id]
            # Render each loaded layer
            for layer_id in frame_dic:
                if not self.is_shown_layer(layer_id):
                    print('Layer %d is hidden. Skip.' % layer_id)
                    continue
                # print('Rendering images for frame %d, layer %d / %d.' % (frame_id, layer_id, self.layer_num))
                self.render_path_frame_layer(frame_id, layer_id, [pose], inverse_y_axis,density_threshold, auto_save)           
        
            # Render background for each frame
            if self.bkgd_renderer is not None and self.display_layers[0] == 1:
                self.render_path_bkgd([pose], auto_save)

            self.image_num += 1 
        
    def render_path_bkgd(self, poses, auto_save):
        for image_id in range(len(poses)):
            pose = poses[image_id]
            pinhole = (self.K[0][0], self.K[1][1], self.K[0][2], self.K[1][2])

            bkgd = self.bkgd_renderer.render(pinhole, pose, self.near, self.far)
            # Normalize
            bkgd = torch.Tensor(bkgd.copy()) / 255.0
            print('bkgd image size is: ', bkgd.shape)
            # 0 is background layer
            add_two_dim_dict(self.images,self.image_num + image_id, 0, bkgd)
            if auto_save:
                save_dir = os.path.join(self.output_dir,'background')
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                    os.mkdir(os.path.join(save_dir,'color'))
                    # os.mkdir(os.path.join(save_dir,'depth'))
                imageio.imwrite(os.path.join(save_dir,'color','%d.jpg' % (self.image_num + image_id)), bkgd)




    # Mix all rendered visible layers into one image
    def mix_image(self,image_id):
        rgb = torch.zeros((self.height,self.width,3))
        depth = self.far * torch.ones((self.height,self.width,1))

        for layer_id in self.images[image_id]:
            if self.display_layers[layer_id] == 0:
                continue
            rgb_temp = self.images[image_id][layer_id]
            idx = None
            # It's not background
            if layer_id != 0:
                depth_temp = self.depths[image_id][layer_id]
                idx = depth_temp < depth
            else:
                idx = (depth == self.far)
            
            idx = idx.reshape(self.height,self.width)

            if layer_id != 0:
                depth[idx,:] = depth_temp[idx,:]
            rgb[idx,:] = rgb_temp[idx,:]
    
        return rgb, depth

    # Save video for one layer
    def save_video_layer(self, layer_id):
        rgbs = []
        depths = []
        for image_id in range(self.image_num):
            if layer_id in self.images[image_id]:
                rgbs.append(self.images[image_id][layer_id])
                depths.append(self.depths[image_id][layer_id])
        
        if len(rgbs) != 0:
            video_dir = os.path.join(self.output_dir,str(layer),'video')
            if not os.path.exists(video_dir):
                os.mkdir(video_dir)

            imageio.mimwrite(video_dir + '/color-%d.mp4' % self.fps, rgbs, fps = self.fps, quality = 8)
            imageio.mimwrite(video_dir + '/depth-%d.mp4' % self.fps, depths, fps = self.fps, quality = 8)
        else:
            print('Warning: Cannot generate video for layer %d, data is empty.' % layer)

    # Save video for the whole rendered data
    def save_video(self):
        rgbs = []
        depths = []
        for image_id in range(self.image_num):
            print('Mixing images, %d / %d' %(image_id+1, self.image_num))
            rgb, depth = self.mix_image(image_id)
            rgbs.append(rgb)
            depths.append(depth)
        
        if len(rgbs) != 0:
            video_dir = os.path.join(self.output_dir,'video')
            if not os.path.exists(video_dir):
                os.mkdir(video_dir)

            imageio.mimwrite(video_dir + '/color_%d.mp4' % self.save_count, rgbs, fps = self.fps, quality = 8)
            imageio.mimwrite(video_dir + '/depth_%d.mp4' % self.save_count, depths, fps = self.fps, quality = 8)
            self.save_count += 1
        else:
            print('Warning: Cannot generate video for all rendered images, data is empty.')

    # Helper functions
    def set_fps(self, fps):
        self.fps = fps

    def get_center_frame_layer(self, frame_id, layer_id):
        return self.datasets[frame_id][layer_id].center


    # Layer display functions
    def hide_layer(self, layer_id):
        self.display_layers[layer_id] = 0

    def show_layer(self, layer_id):
        self.display_layers[layer_id] = 1

    def is_shown_layer(self,layer_id):
        return self.display_layers[layer_id] == 1

    # TODO
    # Save rendered dataset to save_dir
    def save_dataset(self, save_dir):
        save_path = os.path.join(self.dataset_dir, save_dir)

        for image_id in self.poses:
            pass

    def set_frame_duration(self, min_frame, max_frame):
        self.min_frame = min_frame
        self.max_frame = max_frame

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