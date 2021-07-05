
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
import time

from utils import Trigonometric_kernel




class SpaceNet(nn.Module):


    def __init__(self, c_pos=3, include_input = True, use_dir = True, use_time = False, deep_rgb = False):
        super(SpaceNet, self).__init__()


        self.tri_kernel_pos = Trigonometric_kernel(L=10,include_input = include_input)
        if use_dir:
            self.tri_kernel_dir = Trigonometric_kernel(L=4, include_input = include_input)
        if use_time:
            self.tri_kernel_time = Trigonometric_kernel(L=10, input_dim=1, include_input = include_input)

        self.c_pos = c_pos

        self.pos_dim = self.tri_kernel_pos.calc_dim(c_pos)
        if use_dir:
            self.dir_dim = self.tri_kernel_dir.calc_dim(3)
        else:
            self.dir_dim = 0

        if use_time:
            self.time_dim = self.tri_kernel_time.calc_dim(1)
        else:
            self.time_dim = 0

        self.use_dir = use_dir
        self.use_time = use_time
        backbone_dim = 256
        head_dim = 128


        self.stage1 = nn.Sequential(
                    nn.Linear(self.pos_dim, backbone_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(backbone_dim,backbone_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(backbone_dim,backbone_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(backbone_dim,backbone_dim),
                    nn.ReLU(inplace=True),
                )

        self.stage2 = nn.Sequential(
                    nn.Linear(backbone_dim+self.pos_dim, backbone_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(backbone_dim,backbone_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(backbone_dim,backbone_dim),
                    nn.ReLU(inplace=True),
                )

        self.density_net = nn.Sequential(
                    nn.Linear(backbone_dim, 1)
                )
        if deep_rgb:
            print("deep")
            self.rgb_net = nn.Sequential(
                        nn.ReLU(inplace=True),
                        nn.Linear(backbone_dim+self.dir_dim+self.time_dim, head_dim),
                        nn.ReLU(inplace=True),
                        nn.Linear(head_dim, head_dim),
                        nn.ReLU(inplace=True),
                        nn.Linear(head_dim, head_dim),
                        nn.ReLU(inplace=True),
                        nn.Linear(head_dim,3)
                    )
        else:
            self.rgb_net = nn.Sequential(
                        nn.ReLU(inplace=True),
                        nn.Linear(backbone_dim+self.dir_dim+self.time_dim, head_dim),
                        nn.ReLU(inplace=True),
                        nn.Linear(head_dim,3)
                    )


    '''
    INPUT
    pos: 3D positions (N,L,c_pos) or (N,c_pos)
    rays: corresponding rays  (N,6)
    times: corresponding time (N,1)

    OUTPUT

    rgb: color (N,L,3) or (N,3)
    density: (N,L,1) or (N,1)

    '''
    def forward(self, pos, rays, times=None, maxs=None, mins=None):

        #beg = time.time()
        rgbs = None
        if rays is not None and self.use_dir:

            dirs = rays[...,3:6]
            
        bins_mode = False
        if len(pos.size())>2:
            bins_mode = True
            L = pos.size(1)
            pos = pos.reshape((-1,self.c_pos))     #(N,c_pos)
            if rays is not None and self.use_dir:
                dirs = dirs.unsqueeze(1).repeat(1,L,1)
                dirs = dirs.reshape((-1,self.c_pos))   #(N,3)
            if rays is not None and self.use_time:
                times = times.unsqueeze(1).repeat(1,L,1)
                times = times.reshape((-1,1))   #(N,1)

            
           

        if maxs is not None:
            pos = ((pos - mins)/(maxs-mins) - 0.5) * 2

        pos = self.tri_kernel_pos(pos)
        if rays is not None and self.use_dir:
            dirs = self.tri_kernel_dir(dirs)
        if self.use_time:
            times = self.tri_kernel_time(times)
        #torch.cuda.synchronize()
        #print('transform :',time.time()-beg)

        #beg = time.time()
        x = self.stage1(pos)
        x = self.stage2(torch.cat([x,pos],dim =1))

        density = self.density_net(x)
        
        x1 = 0
        if rays is not None and self.use_dir:
            x1 = torch.cat([x,dirs],dim =1)
        else:
            x1 = x.clone()

        rgbs = None
        if self.use_time:
            x2 = torch.cat([x1,times],dim =1)
            rgbs = self.rgb_net(x2)
        else:
            rgbs = self.rgb_net(x1)
        #torch.cuda.synchronize()
        #print('fc:',time.time()-beg)

        if bins_mode:
            density = density.reshape((-1,L,1))
            rgbs = rgbs.reshape((-1,L,3))

        return rgbs, density



         


        


        

        


