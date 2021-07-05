import torch
import torch.nn as nn

from utils import Trigonometric_kernel
class MotionNet(nn.Module):
    # (x,y,z,t)
    def __init__(self, c_input=5, include_input = True, input_time = False):
        """ Init layered sampling
        """
        super(MotionNet, self).__init__()
        self.c_input = c_input
        self.input_time = input_time
        #Positional Encoding
        self.tri_kernel_pos = Trigonometric_kernel(L=10,input_dim = c_input, include_input = include_input)

        self.pos_dim = self.tri_kernel_pos.calc_dim(c_input)
        backbone_dim = 128
        head_dim = 128

        self.motion_net = nn.Sequential(
            nn.Linear(self.pos_dim, head_dim),
            nn.ReLU(inplace=False),
            nn.Linear(head_dim,backbone_dim),
            nn.ReLU(inplace=True),
            nn.Linear(backbone_dim,backbone_dim),
            nn.ReLU(inplace=True),
            nn.Linear(backbone_dim,backbone_dim),
            nn.ReLU(inplace=True),
            nn.Linear(backbone_dim ,head_dim),
            nn.ReLU(inplace=True),
            nn.Linear(head_dim,3)
        )

    def forward(self, input_0):
        """ Generate sample points
        Input:
        pos: [N,3] points in real world coordinates

        Output:
        flow: [N,3] Scene Flow in real world coordinates
        """

        bins_mode = False
        if len(input_0.size()) > 2:
            bins_mode = True
            L = input_0.size(1)
            input_0 = input_0.reshape((-1, self.c_input))  # (N,input)

        if self.input_time:
            xyz = input_0[:,:-1]
            time = input_0[:,-1:]
            lower = torch.floor(time)
            if not torch.all(torch.eq(lower, time)):
                upper = lower + 1
                weight = time - lower
                i_lower = torch.cat([xyz,lower],-1)
                i_upper = torch.cat([xyz,upper],-1)
                i_lower = self.tri_kernel_pos(i_lower)
                i_upper = self.tri_kernel_pos(i_upper)
                input_0 = (1-weight) * i_lower + weight * i_upper
            else:
                input_0 = self.tri_kernel_pos(input_0)
        else:        
            input_0 = self.tri_kernel_pos(input_0)
        
        flow = self.motion_net(input_0)

        if bins_mode:
            flow = flow.reshape(-1, L, 3)

        return flow
