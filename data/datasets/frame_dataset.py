import torch
import numpy as np
import os
from .utils import campose_to_extrinsic, read_intrinsics, read_mask
from PIL import Image
import torchvision
import torch.distributions as tdist
import open3d as o3d

class FrameDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_path, transform, frame_id, layer_num, file_offset):

        super(FrameDataset, self).__init__()

        # 1. Set the dataset path for the next loading
        self.file_offset = file_offset
        self.frame_id = frame_id # The frame number
        self.layer_num = layer_num # The number of layers
        self.image_path = os.path.join(dataset_path,str(self.frame_id),'images')
        self.label_path = os.path.join(dataset_path,str(self.frame_id),'labels')
        self.pointcloud_path = os.path.join(dataset_path,str(self.frame_id),'pointclouds')
        self.pose_path = os.path.join(dataset_path, 'pose')
        self.transform = transform
        # 2. Loading Intrinsics & Camera poses
        camposes = np.loadtxt(os.path.join(self.pose_path,'RT_c2w.txt'))
        # Ts are camera poses
        self.Ts = torch.Tensor(campose_to_extrinsic(camposes))
        self.Ks = torch.Tensor(read_intrinsics(os.path.join(self.pose_path,'K.txt')))
        # 3. Load pointclouds for different layers
        self.pointclouds = [] # Finally (layer_num,)
        self.bboxs = [] # Finally (layer_num,)

        

        for i in range(layer_num):
            # Start from 1.ply to layer_num.ply
            pointcloud_name = os.path.join(self.pointcloud_path, '%d.ply' % (i+1))

            if not os.path.exists(pointcloud_name):
                pointcloud_name = os.path.join(self.pointcloud_path1, '%d.ply' % (i+1))

            if not os.path.exists(pointcloud_name):
                print('Cannot find corresponding pointcloud in path: ', pointcloud_name)
            pointcloud = o3d.io.read_point_cloud(pointcloud_name)
            xyz = np.asarray(pointcloud.points)
            
            xyz = torch.Tensor(xyz)
            self.pointclouds.append(xyz)

            max_xyz = torch.max(xyz, dim=0)[0]
            min_xyz = torch.min(xyz, dim=0)[0]

            # Default scalar is 0.3
            tmp = (max_xyz - min_xyz) * 0.0

            max_xyz = max_xyz + tmp
            min_xyz = min_xyz - tmp

            minx, miny, minz = min_xyz[0],min_xyz[1],min_xyz[2]
            maxx, maxy, maxz = max_xyz[0],max_xyz[1],max_xyz[2]
            bbox = torch.Tensor([[minx,miny,minz],[maxx,miny,minz],[maxx,maxy,minz],[minx,maxy,minz],
                             [minx,miny,maxz],[maxx,miny,maxz],[maxx,maxy,maxz],[minx,maxy,maxz]])
            
            bbox = bbox.reshape(1,8,3)

            self.bboxs.append(torch.Tensor(bbox))

        print('Frame %d dataset loaded, there are totally %d layers' %(frame_id,layer_num))

    def __len__(self):
        return self.cam_num * self.layer_num

    def get_data(self, camera_id, layer_id):
        # Find K,T, bbox
        T = self.Ts[camera_id]
        K = self.Ks[camera_id]
        bbox = self.bboxs[layer_id-1]
        # Load image
        image_path = os.path.join(self.image_path, '%03d.png' % (camera_id + self.file_offset))
        image = Image.open(image_path)
        # Load label
        label_path = os.path.join(self.label_path, '%03d.npy' % (camera_id + self.file_offset))
        if not os.path.exists(label_path):
            label_path = os.path.join(self.label_path, '%03d_label.npy' % (camera_id + self.file_offset))
        label = np.load(label_path)
        
        # Transform image label K T to right scale
        image, label, K, T, ROI = self.transform(image, Ks=K, Ts=T, label=label)

        return image, label, K, T, ROI, bbox


class FrameLayerDataset(torch.utils.data.Dataset):

    def __init__(self, cfg, transform, frame_id, layer_id):

        super(FrameLayerDataset, self).__init__()

        # 1. Set the dataset path for the next loading
        dataset_path = cfg.DATASETS.TRAIN
        fixed_near, fixed_far = cfg.DATASETS.FIXED_NEAR, cfg.DATASETS.FIXED_FAR
        scale=cfg.DATASETS.SCALE
        camera_stepsize=cfg.DATASETS.CAMERA_STEPSIZE 
        self.file_offset = cfg.DATASETS.FILE_OFFSET
        
        self.frame_id = frame_id # The frame id
        self.layer_id = layer_id # The layer id
        self.image_path = os.path.join(dataset_path,'frame'+str(self.frame_id),'images')
        self.label_path = os.path.join(dataset_path,'frame'+str(self.frame_id),'labels')
        #TODO: Need to fix when background is deformable
        self.pointcloud_path1 = "None"
                
        if layer_id != 0:
            self.pointcloud_path = os.path.join(dataset_path,'frame'+str(self.frame_id),'pointclouds')
            self.pointcloud_path1 = os.path.join(dataset_path,'background')
        else:
            self.pointcloud_path = os.path.join(dataset_path,'background')
        self.pose_path = os.path.join(dataset_path, 'pose')
        self.transform = transform
        # 2. Loading Intrinsics & Camera poses
        camposes = np.loadtxt(os.path.join(self.pose_path,'RT_c2w.txt'))

        # Ts are camera poses
        self.Ts = torch.Tensor(campose_to_extrinsic(camposes))
        self.Ts[:,0:3,3] = self.Ts[:,0:3,3] * scale
        print('scale is ', scale)

        self.Ks = torch.Tensor(read_intrinsics(os.path.join(self.pose_path,'K.txt')))

        self.cfg = cfg
        if cfg.DATASETS.CAMERA_NUM == 0:
            self.cam_num = self.Ts.shape[0]
        else:
            self.cam_num = cfg.DATASETS.CAMERA_NUM

        self.mask = np.ones(self.Ts.shape[0])
        self.mask_path = cfg.DATASETS.VIEW_MASK
        if self.mask_path != None:
            if os.path.exists(self.mask_path):
                self.mask = read_mask(self.mask_path)

        pointcloud_name = os.path.join(self.pointcloud_path, '%d.ply' % (layer_id))

        self.pointcloud = None
        if not os.path.exists(pointcloud_name):
            pointcloud_name = os.path.join(self.pointcloud_path1, '%d.ply' % (layer_id))

        bbox_name = 'bbox_tmp'
        if not os.path.exists(pointcloud_name):
            print('Warning: Cannot find corresponding pointcloud in path: ', pointcloud_name)
            self.bbox = None
            self.center = torch.Tensor([0,0,0])
            tmp_bbox_path = os.path.join(dataset_path,bbox_name,'frame'+str(frame_id),'layer'+str(layer_id))
            if os.path.exists(os.path.join(tmp_bbox_path,'center.pt')):
                print('There are bbox generated for layer %d, frame %d before, loading bbox...' % (layer_id, frame_id))
                # pointcloud = o3d.io.read_point_cloud(pointcloud_name)
                # xyz = np.asarray(pointcloud.points)
                
                # xyz = torch.Tensor(xyz)
                # self.pointcloud = xyz
                self.center = torch.load(os.path.join(tmp_bbox_path,'center.pt'))
                self.bbox = torch.load(os.path.join(tmp_bbox_path,'bbox.pt'))
        else:
            tmp_bbox_path = os.path.join(dataset_path,bbox_name,'frame'+str(frame_id),'layer'+str(layer_id))
            if not os.path.exists(os.path.join(tmp_bbox_path,'center.pt')):
                print('There is no bbox generated before, generating bbox...')
                if not os.path.exists(tmp_bbox_path):
                    os.makedirs(tmp_bbox_path)
                pointcloud = o3d.io.read_point_cloud(pointcloud_name)
                xyz = np.asarray(pointcloud.points)
                
                xyz = torch.Tensor(xyz)
                self.pointcloud = xyz * scale

                max_xyz = torch.max(self.pointcloud, dim=0)[0]
                min_xyz = torch.min(self.pointcloud, dim=0)[0]

                # Default scalar is 0.3
                tmp = (max_xyz - min_xyz) * 0.0

                max_xyz = max_xyz + tmp
                min_xyz = min_xyz - tmp

                minx, miny, minz = min_xyz[0],min_xyz[1],min_xyz[2]
                maxx, maxy, maxz = max_xyz[0],max_xyz[1],max_xyz[2]
                bbox = torch.Tensor([[minx,miny,minz],[maxx,miny,minz],[maxx,maxy,minz],[minx,maxy,minz],
                                    [minx,miny,maxz],[maxx,miny,maxz],[maxx,maxy,maxz],[minx,maxy,maxz]])
                
                bbox = bbox.reshape(1,8,3)

                self.center = np.array([(min_xyz[0]+max_xyz[0])/2, (min_xyz[1]+max_xyz[1])/2, (min_xyz[2]+max_xyz[2])/2])
                self.bbox = torch.Tensor(bbox)
                if not os.path.exists(os.path.join(tmp_bbox_path,'center.pt')):
                    torch.save(self.center, os.path.join(tmp_bbox_path,'center.pt'))
                if not  os.path.exists(os.path.join(tmp_bbox_path,'bbox.pt')):
                    torch.save(self.bbox, os.path.join(tmp_bbox_path,'bbox.pt'))
            else:
                print('There are bbox generated for layer %d, frame %d before, loading bbox...' % (layer_id, frame_id))
                # pointcloud = o3d.io.read_point_cloud(pointcloud_name)
                # xyz = np.asarray(pointcloud.points)
                
                # xyz = torch.Tensor(xyz)
                # self.pointcloud = xyz
                self.center = torch.load(os.path.join(tmp_bbox_path,'center.pt'))
                self.bbox = torch.load(os.path.join(tmp_bbox_path,'bbox.pt'))


        if fixed_near == -1.0 and fixed_far == -1.0:
            near_far_name = 'near_far_tmp'
            tmp_near_far_path = os.path.join(dataset_path,near_far_name,'frame'+str(frame_id),'layer'+str(layer_id))
            if not os.path.exists(os.path.join(tmp_near_far_path,'near.pt')):
                if not os.path.exists(os.path.join(tmp_near_far_path)):
                    os.makedirs(tmp_near_far_path)
                inv_Ts = torch.inverse(self.Ts).unsqueeze(1)  #(M,1,4,4)
                
                if self.pointcloud is None:
                    pointcloud = o3d.io.read_point_cloud(pointcloud_name)
                    xyz = np.asarray(pointcloud.points)
                    
                    xyz = torch.Tensor(xyz)
                    self.pointcloud = xyz * scale
                vs = self.pointcloud.clone().unsqueeze(-1)   #(N,3,1)
                vs = torch.cat([vs,torch.ones(vs.size(0),1,vs.size(2)) ],dim=1) #(N,4,1)

                pts = torch.matmul(inv_Ts,vs) #(M,N,4,1)

                pts_max = torch.max(pts, dim=1)[0].squeeze() #(M,4)
                pts_min = torch.min(pts, dim=1)[0].squeeze() #(M,4)

                pts_max = pts_max[:,2]   #(M)
                pts_min = pts_min[:,2]   #(M)

                self.near = pts_min
                # self.near[self.near<(pts_max*0.1)] = pts_max[self.near<(pts_max*0.1)]*0.1
            
                self.far = pts_max
                torch.save(self.near,os.path.join(tmp_near_far_path,'near.pt'))
                torch.save(self.far,os.path.join(tmp_near_far_path,'far.pt'))
            else:
                self.near = torch.load(os.path.join(tmp_near_far_path,'near.pt'))
                self.far = torch.load(os.path.join(tmp_near_far_path,'far.pt'))
        else:
            self.near = torch.ones(self.Ts.shape[0]) * fixed_near
            self.far = torch.ones(self.Ts.shape[0]) * fixed_far

        print('Layer %d, Frame %d dataset loaded' %(layer_id,frame_id))

    def __len__(self):
        return self.cam_num

    def get_data(self, camera_id):
        # when camera num is not equal to zero, means we want a complete offset from camera parameters to images, else, only images
        if self.cfg.DATASETS.CAMERA_NUM != 0:
            camera_id = camera_id + self.file_offset
        if self.mask[camera_id] == 0:
            return None, None, None, None, None, None, None, 0
        # Find K,T, bbox

        T = self.Ts[camera_id]
        K = self.Ks[camera_id]
        bbox = self.bbox
        # Load image
        image_path = os.path.join(self.image_path, '%03d.png' % (camera_id))
        if not os.path.exists(image_path):
            image_path = os.path.join(self.image_path, '%d.png' % (camera_id))
        if not os.path.exists(image_path):
            image = None
        else:
            image = Image.open(image_path)
        # Load label
        label = None
        label_path = os.path.join(self.label_path, '%03d.npy' % (camera_id))
        if not os.path.exists(label_path):
            label_path = os.path.join(self.label_path, '%03d_label.npy' % (camera_id))
        if not os.path.exists(label_path):
            label_path = os.path.join(self.label_path, '%d.npy' % (camera_id))
        if not os.path.exists(label_path):
            if image == None:
                label = None
            else:
                width, height = image.size
                label = np.ones((height, width)) * self.layer_id
                print('Warning: There is no label map for this dataset, and we trying to train layer %d, for frame %d, so generate a full label map with it' % (self.layer_id, self.frame_id))
        else:
            label = np.load(label_path)
        
        # Transform image label K T to right scale
        image, label, K, T, ROI = self.transform(image, Ks=K, Ts=T, label=label)

        return image, label, K, T, ROI, bbox, torch.tensor([self.near[camera_id],self.far[camera_id]]).unsqueeze(0), self.mask[camera_id]

    def get_original_size(self):

        image_path = os.path.join(self.image_path, '%03d.png' % (0))
        if not os.path.exists(image_path):
            image_path = os.path.join(self.image_path, '%d.png' % (0))
        if not os.path.exists(image_path):
            image = None
        else:
            image = Image.open(image_path)

        return image.size


