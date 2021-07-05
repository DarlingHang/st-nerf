import torch
import numpy as np
from math import sin, cos, pi
import os
from .utils import campose_to_extrinsic, read_intrinsics
from PIL import Image
import torchvision
import torch.distributions as tdist

from .frame_dataset import FrameDataset, FrameLayerDataset
from utils import ray_sampling, ray_sampling_label_bbox, lookat, getSphericalPosition, generate_rays, ray_sampling_label_label

class Ray_Dataset(torch.utils.data.Dataset):

    def __init__(self, cfg, transforms_bkgd, transforms_layer):
        
        super(Ray_Dataset, self).__init__()

        frame_num=cfg.DATASETS.FRAME_NUM
        layer_num=cfg.DATASETS.LAYER_NUM

        frame_offset=cfg.DATASETS.FRAME_OFFSET
        bkgd_sample_rate = cfg.DATASETS.BKGD_SAMPLE_RATE  

        # [[bkgd_frame1,bkgd_frame2,...,],[layer1_frame_1,layer1_frame2,...,],[layer2_frame1,layer2_frame2,...,],...,]
        self.datasets = []
        self.bboxes = torch.zeros(frame_num+frame_offset, layer_num, 8, 3)
        for layer_id in range(layer_num+1):
            datasets_layer = []
            for frame_id in range(1+frame_offset,frame_offset+frame_num+1):
                if layer_id == 0:
                    sample_rate=bkgd_sample_rate
                    use_label_map=True
                    transform=transforms_bkgd
                else:
                    sample_rate=1
                    for i in range(len(cfg.DATASETS.FIXED_LAYER)):
                        if cfg.DATASETS.FIXED_LAYER[i] == layer_id:
                            sample_rate = 0
                    use_label_map=cfg.DATASETS.USE_LABEL
                    transform=transforms_layer
                dataset_frame_layer = Ray_Frame_Layer_Dataset(cfg, transform, frame_id, layer_id, use_label_map, sample_rate)
                if layer_id != 0:
                    self.bboxes[frame_id-1, layer_id-1] = dataset_frame_layer.layer_bbox
                datasets_layer.append(dataset_frame_layer)
            self.datasets.append(datasets_layer)
        
        self.frame_num = frame_num
        self.layer_num = layer_num

        self.bkgd_sample_rate = bkgd_sample_rate

        self.ray_length = np.zeros(layer_num+1)

        for l in range(len(self.datasets)):
            layer_datasets = self.datasets[l]
            for layer_frame_dataset in layer_datasets:
                self.ray_length[l] += len(layer_frame_dataset)
        
        for l in range(len(self.datasets)):
            print('Layer %d has %d rays' % (l, int(self.ray_length[l])))
        self.length = int(sum(self.ray_length))

        print('The whole ray number is %d' % self.length)
        self.camera_num = self.datasets[0][0].camera_num

    
    def __len__(self):
        
        return self.length

    def __getitem__(self, index):
        # if index < self.bkgd_length:
        #     index = int(index / self.bkgd_sample_rate)
        # else:
        #     index = (index-self.bkgd_length) + self.original_bkgd_length
        temp = 0
        for layer_datasets in self.datasets:
            for layer_frame_dataset in layer_datasets:
                if temp + len(layer_frame_dataset) > index:
                    return layer_frame_dataset[index-temp]
                else:
                    temp += len(layer_frame_dataset)

class Ray_Dataset_View(torch.utils.data.Dataset):

    def __init__(self, cfg, transform):
        
        super(Ray_Dataset_View, self).__init__()

        # Save input
        self.dataset_path = cfg.DATASETS.TRAIN
        self.frame_num = cfg.DATASETS.FRAME_NUM
        self.layer_num = cfg.DATASETS.LAYER_NUM
        self.frame_offset = cfg.DATASETS.FRAME_OFFSET

        self.pose_refinement = cfg.MODEL.POSE_REFINEMENT
        self.use_deform_view = cfg.MODEL.USE_DEFORM_VIEW
        self.use_deform_time = cfg.MODEL.USE_DEFORM_TIME
        self.use_space_time = cfg.MODEL.USE_SPACE_TIME
        remove_outliers = cfg.MODEL.REMOVE_OUTLIERS

        self.transform = transform

        self.layer_frame_datasets = []
        for layer_id in range(self.layer_num+1):
            datasets_layer = []
            for frame_id in range(1+self.frame_offset,self.frame_offset+self.frame_num+1):
                dataset_frame_layer = FrameLayerDataset(cfg, transform, frame_id, layer_id)
                datasets_layer.append(dataset_frame_layer)
            self.layer_frame_datasets.append(datasets_layer)
        self.camera_num = self.layer_frame_datasets[0][0].cam_num

    def __len__(self):
        return 1
    
    def get_fixed_image(self, index_view, index_frame):
        
        print(index_view)
        print(index_frame)

        bboxes = []
        K = None
        T = None
        label = None
        image = None
        for i in range(self.layer_num+1):
            image_tmp, label_tmp, K_tmp, T_tmp, _, bbox, near_far = self.layer_frame_datasets[i][index_frame].get_data(index_view)
            if K is None:
                K = K_tmp
            if T is None:
                T = T_tmp
            if label is None:
                label = label_tmp
            if image is None:
                image = image_tmp
            bboxes.append(bbox)
        
        rays, labels, rgbs, ray_mask, layered_bboxes = ray_sampling_label_bbox(image,label,K,T,bboxes=bboxes)
        # rays,rgbs = ray_sampling(K.unsqueeze(0), T.unsqueeze(0), (image.size(1),image.size(2)), images = image.unsqueeze(0) )
        if self.pose_refinement:
            rays_o, rays_d = rays[:, :3], rays[:, 3:6]
            ids=torch.ones((rays_o.size(0),1))*index
            rays=torch.cat([rays_o,ids,rays_d,ids],dim = 1)

        if self.use_deform_view:
            camera_ids=torch.ones((rays.size(0),1)) * index
            rays=torch.cat([rays, camera_ids],dim=-1)

        if self.use_deform_time or self.use_space_time:
            frame_ids = torch.Tensor([index_frame+self.frame_offset+1]).reshape(1,1).repeat(rays.shape[0],1)
            rays=torch.cat([rays, frame_ids],dim=-1)

        return rays, rgbs, labels, image, label, ray_mask, layered_bboxes, near_far.repeat(rays.size(0),1)

    def __getitem__(self, index):
        
        index_frame = np.random.randint(0,self.frame_num)
        index_view = np.random.randint(0,self.camera_num)
        _, _, _, _, _, _, _, mask = self.layer_frame_datasets[0][index_frame].get_data(index_view)
        while (mask == 0):
            
            index_view = np.random.randint(0,self.camera_num)
            _, _, _, _, _, _, _, mask = self.layer_frame_datasets[0][index_frame].get_data(index_view)
        
        print(index_view)
        print(index_frame)

        bboxes = []
        K = None
        T = None
        label = None
        image = None
        for i in range(self.layer_num+1):
            image_tmp, label_tmp, K_tmp, T_tmp, _, bbox, near_far, _ = self.layer_frame_datasets[i][index_frame].get_data(index_view)
            if K is None:
                K = K_tmp
            if T is None:
                T = T_tmp
            if label is None:
                label = label_tmp
            if image is None:
                image = image_tmp
            bboxes.append(bbox)
        
        rays, labels, rgbs, ray_mask, layered_bboxes = ray_sampling_label_bbox(image,label,K,T,bboxes=bboxes)
        # rays,rgbs = ray_sampling(K.unsqueeze(0), T.unsqueeze(0), (image.size(1),image.size(2)), images = image.unsqueeze(0) )
        if self.pose_refinement:
            rays_o, rays_d = rays[:, :3], rays[:, 3:6]
            ids=torch.ones((rays_o.size(0),1))*index
            rays=torch.cat([rays_o,ids,rays_d,ids],dim = 1)

        if self.use_deform_view:
            camera_ids=torch.ones((rays.size(0),1)) * index
            rays=torch.cat([rays, camera_ids],dim=-1)

        if self.use_deform_time or self.use_space_time:
            frame_ids = torch.Tensor([index_frame+self.frame_offset+1]).reshape(1,1).repeat(rays.shape[0],1)
            rays=torch.cat([rays, frame_ids],dim=-1)

        return rays, rgbs, labels, image, label, ray_mask, layered_bboxes, near_far.repeat(rays.size(0),1)
        
class Ray_Dataset_Render(torch.utils.data.Dataset):

    def __init__(self, cfg, transform):
        
        super(Ray_Dataset_Render, self).__init__()

        # Save input
        self.use_deform_time = cfg.MODEL.USE_DEFORM_TIME
        self.use_space_time = cfg.MODEL.USE_SPACE_TIME
        
        frame_offset = cfg.DATASETS.FRAME_OFFSET
        layer_num = cfg.DATASETS.LAYER_NUM
        frame_num = cfg.DATASETS.FRAME_NUM

        self.layer_num = layer_num

        self.datasets = []

        self.bboxes = torch.zeros(frame_num+frame_offset, layer_num, 8, 3)


        for layer_id in range(layer_num+1):
            datasets_layer = []
            for frame_id in range(1+frame_offset,frame_offset+frame_num+1):
                dataset_frame_layer = FrameLayerDataset(cfg, transform, frame_id, layer_id)
                datasets_layer.append(dataset_frame_layer)
                if layer_id != 0:
                    self.bboxes[frame_id-1, layer_id-1] = dataset_frame_layer.bbox
            self.datasets.append(datasets_layer)
        
        self.camera_num = self.datasets[0][0].cam_num
        self.poses = self.datasets[0][0].Ts
        
        # Default layer size is original size 
        self.Ks = self.datasets[0][0].Ks
        col, row = self.datasets[0][0].get_original_size()
        self.Ks[:,0,0] = self.Ks[:,0,0] * cfg.INPUT.SIZE_TEST[0] / col
        self.Ks[:,1,1] = self.Ks[:,1,1] * cfg.INPUT.SIZE_TEST[0] / col
        self.Ks[:,0,2] = self.Ks[:,0,2] * cfg.INPUT.SIZE_TEST[0] / col
        self.Ks[:,1,2] = self.Ks[:,1,2] * cfg.INPUT.SIZE_TEST[0] / col

        # Use original image size, intrinsic and bbox
        image, _, self.K, _, _, _, _, _ = self.datasets[0][0].get_data(0)

        # for i in range(len(self.datasets[0][0])):
        #     _, _, K, _, _, _, _, _ = self.datasets[0][0].get_data(i)
        #     self.Ks.append(K)

        
        _, self.height, self.width = image.shape

        self.near_far = torch.Tensor([cfg.DATASETS.FIXED_NEAR,cfg.DATASETS.FIXED_FAR]).reshape(1,2)

    def get_image_label(self, camera_id, frame_id):
        image, label, _, _, _, _, _, _ = self.datasets[frame_id][0].get_data(camera_id)
        return image, label

    def get_rays_by_pose_and_K(self, T, K, layer_frame_pair):

        T = torch.Tensor(T)
        rays, _ = generate_rays(K, T, None, self.height, self.width)

        #TODO: now bbox and near far is no use
        near_fars = self.near_far.repeat(rays.size(0),1)
        bboxes = torch.zeros(rays.size(0),8,3)
        labels = torch.zeros(rays.size(0))

        # bboxes = []
        # for layer_id, frame_id in layer_frame_pair:
        #     bboxes.append(self.bboxes[frame_id-1,layer_id])
        
        # rays,rgbs = ray_sampling(K.unsqueeze(0), T.unsqueeze(0), (image.size(1),image.size(2)), images = image.unsqueeze(0) )

        if self.use_deform_time or self.use_space_time:
            frame_ids = torch.zeros(rays.size(0),self.layer_num+1)
            for layer_id, frame_id in layer_frame_pair:
                frame_ids[:,layer_id] = frame_id

            rays=torch.cat([rays, frame_ids],dim=-1)

        return rays, labels, bboxes, near_fars
    #Use the first K of the dataset by default
    def get_rays_by_pose(self, T, layer_frame_pair):

        T = torch.Tensor(T)
        rays, _ = generate_rays(self.K, T, None, self.height, self.width)

        #TODO: now bbox and near far is no use
        near_fars = self.near_far.repeat(rays.size(0),1)
        bboxes = torch.zeros(rays.size(0),8,3)
        labels = torch.zeros(rays.size(0))

        # bboxes = []
        # for layer_id, frame_id in layer_frame_pair:
        #     bboxes.append(self.bboxes[frame_id-1,layer_id])
        
        # rays,rgbs = ray_sampling(K.unsqueeze(0), T.unsqueeze(0), (image.size(1),image.size(2)), images = image.unsqueeze(0) )

        if self.use_deform_time:
            frame_ids = torch.zeros(rays.size(0),self.layer_num+1)
            for layer_id, frame_id in layer_frame_pair:
                frame_ids[:,layer_id] = frame_id

            rays=torch.cat([rays, frame_ids],dim=-1)

        return rays, labels, bboxes, near_fars

    def get_rays_by_lookat(self,eye,center,up, layer_frame_pair):

        T = torch.Tensor(lookat(eye,center,up))
        return self.get_rays_by_pose(T, layer_frame_pair)

    def get_rays_by_spherical(self, theta, phi, radius,offsets, up, layer_frame_pair):
        up = np.array(up)
        offsets = np.array(offsets)

        pos = getSphericalPosition(radius,theta,phi)
        pos += self.center
        pos += offsets
        T = torch.Tensor(lookat(pos,self.center,up))

        return self.get_rays_by_pose(T, layer_frame_pair)
    
    def get_pose_by_lookat(self, eye,center,up):
        return torch.Tensor(lookat(eye,center,up))

    def get_pose_by_spherical(self, theta, phi, radius, offsets, up):
        up = np.array(up)
        offsets = np.array(offsets)

        pos = getSphericalPosition(radius,theta,phi)
        pos += self.center
        pos += offsets
        T = torch.Tensor(lookat(pos,self.center,up))
        return T

class Ray_Frame_Layer_Dataset(torch.utils.data.Dataset):

    def __init__(self, cfg, transform, frame_id, layer_id, use_label_map, sample_rate):

        super(Ray_Frame_Layer_Dataset, self).__init__()


        # Save input
        self.dataset_path = cfg.DATASETS.TRAIN
        self.tmp_rays = cfg.DATASETS.TMP_RAYS
        self.camera_stepsize = cfg.DATASETS.CAMERA_STEPSIZE
        
        self.pose_refinement = cfg.MODEL.POSE_REFINEMENT
        self.use_deform_view = cfg.MODEL.USE_DEFORM_VIEW
        self.use_deform_time = cfg.MODEL.USE_DEFORM_TIME
        self.use_space_time = cfg.MODEL.USE_SPACE_TIME

        self.transform = transform
        self.frame_id = frame_id
        self.layer_id = layer_id

        # Generate Frame Dataset
        self.frame_dataset = FrameLayerDataset(cfg, transform, frame_id, layer_id)
        self.camera_num = self.frame_dataset.cam_num
        # Save layered rays, rgbs, labels, bboxs, near_fars
        self.layer_rays = []
        self.layer_rgbs = []
        self.layer_labels = []
        if self.frame_dataset.bbox != None:
            self.layer_bbox = self.frame_dataset.bbox
        else:
            self.layer_bbox = torch.zeros(8,3)
        self.near_fars = []

        # Check if we already generate rays
        tmp_ray_path = os.path.join(self.dataset_path,self.tmp_rays,'frame'+str(frame_id))
        if not os.path.exists(tmp_ray_path):
            print('There is no rays generated before, generating rays...')
            os.makedirs(tmp_ray_path)

        # tranverse every camera
        tmp_layer_ray_path = os.path.join(tmp_ray_path,'layer'+str(layer_id))
        if sample_rate == 0.0:
            print('Skiping layer %d, frame %d rays for zero sample rate...' % (layer_id, frame_id))
            self.layer_rays = torch.tensor([])
            self.layer_rgbs = torch.tensor([])
            self.layer_labels = torch.tensor([])
            self.near_fars = torch.tensor([])
        elif not os.path.exists(tmp_layer_ray_path) or cfg.clean_ray:
            rays_tmp = []
            rgbs_tmp = []
            labels_tmp = []
            near_fars_tmp = []
            print('There is no rays generated for layer %d, frame %d before, generating rays...' % (layer_id, frame_id))
            for i in range(0,self.frame_dataset.cam_num,self.camera_stepsize):
                print('Generating Layer %d, Camera %d rays...'% (layer_id,i))
                
                image, label, K, T, ROI, bbox, near_far, mask = self.frame_dataset.get_data(i)

                if not mask:
                    print('Skiping Camera %d by mask'% (i))
                    continue

                if not use_label_map:
                    rays, labels, rgbs, _ = ray_sampling_label_bbox(image,label,K,T,bbox)
                else:
                    rays, labels, rgbs, _ = ray_sampling_label_label(image,label,K,T,layer_id)

                if self.pose_refinement:
                    rays_o, rays_d = rays[:, :3], rays[:, 3:6]
                    ids=torch.ones((rays_o.size(0),1))*i
                    rays=torch.cat([rays_o,ids,rays_d,ids],dim = 1)

                if self.use_deform_view:
                    camera_ids=torch.ones((rays.size(0),1))*i
                    rays=torch.cat([rays, camera_ids],dim=-1)

                if self.use_deform_time or self.use_space_time:
                    frame_ids = torch.Tensor([frame_id]).reshape(1,1).repeat(rays.shape[0],1)
                    rays=torch.cat([rays, frame_ids],dim=-1)

                near_fars_tmp.append(near_far.repeat(rays.size(0),1))
                rays_tmp.append(rays)
                rgbs_tmp.append(rgbs)
                labels_tmp.append(labels)
        
            self.layer_rays = torch.cat(rays_tmp,0)
            self.layer_rgbs = torch.cat(rgbs_tmp,0)
            self.layer_labels = torch.cat(labels_tmp,0)
            self.near_fars = torch.cat(near_fars_tmp,0)
            if sample_rate != 1:
                rand_idx = torch.randperm(self.layer_rays.size(0))
                self.layer_rays = self.layer_rays[rand_idx]
                self.layer_rgbs = self.layer_rgbs[rand_idx]
                self.layer_labels = self.layer_labels[rand_idx]
                self.near_fars = self.near_fars[rand_idx]
                end = int(self.layer_rays.size(0) * sample_rate)
                self.layer_rays = self.layer_rays[:end,:].clone().detach()
                self.layer_rgbs = self.layer_rgbs[:end,:].clone().detach()
                self.layer_labels = self.layer_labels[:end,:].clone().detach()
                self.near_fars = self.near_fars[:end,:].clone().detach()
            if not os.path.exists(tmp_layer_ray_path):
                os.mkdir(tmp_layer_ray_path)
            torch.save(self.layer_rays, os.path.join(tmp_layer_ray_path,'rays.pt'))
            torch.save(self.layer_rgbs, os.path.join(tmp_layer_ray_path,'rgbs.pt'))
            torch.save(self.layer_labels, os.path.join(tmp_layer_ray_path,'labels.pt'))
            torch.save(self.near_fars, os.path.join(tmp_layer_ray_path,'near_fars.pt'))
        else:
            print('There are rays generated for layer %d, frame %d before, loading rays...' % (layer_id, frame_id))
            self.layer_rays = torch.load(os.path.join(tmp_layer_ray_path,'rays.pt'),map_location='cpu')
            self.layer_rgbs = torch.load(os.path.join(tmp_layer_ray_path,'rgbs.pt'),map_location='cpu')
            self.layer_labels = torch.load(os.path.join(tmp_layer_ray_path,'labels.pt'),map_location='cpu')
            self.near_fars = torch.load(os.path.join(tmp_layer_ray_path,'near_fars.pt'),map_location='cpu')
        
        # Fix to the layer id
        self.layer_bbox_labels =  self.layer_id * torch.ones_like(self.layer_labels)
        print('Generating %d rays' % self.layer_rays.shape[0])
    def __len__(self):
        return self.layer_rays.shape[0]

    def __getitem__(self, index):
        return self.layer_rays[index,:], self.layer_rgbs[index,:], self.layer_labels[index,:], self.layer_bbox_labels[index,:], self.layer_bbox[0], self.near_fars[index,:]