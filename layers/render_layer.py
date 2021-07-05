import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


def gen_weight(sigma, delta, act_fn=F.relu):
    """Generate transmittance from predicted density
    """
    alpha = 1.-torch.exp(-act_fn(sigma.squeeze(-1))*delta)
    weight = 1.-alpha + 1e-10
    #weight = alpha * torch.cumprod(weight, dim=-1) / weight # exclusive cum_prod

    weight = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1),device = alpha.device), weight], -1), -1)[:, :-1]

    return weight

class VolumeRenderer(nn.Module):
    def __init__(self, use_mask= False, boarder_weight = 1e10):
        super(VolumeRenderer, self).__init__()
        self.boarder_weight = boarder_weight
        self.use_mask = use_mask

    def forward(self, depth, rgb, sigma, noise=0):
        """
        N - num rays; L - num samples; 
        :param depth: torch.tensor, depth for each sample along the ray. [N, L, 1]
        :param rgb: torch.tensor, raw rgb output from the network. [N, L, 3]
        :param sigma: torch.tensor, raw density (without activation). [N, L, 1]
        
        :return:
            color: torch.tensor [N, 3]
            depth: torch.tensor [N, 1]
        """

        delta = (depth[:, 1:] - depth[:, :-1]).squeeze()  # [N, L-1]
        #pad = torch.Tensor([1e10],device=delta.device).expand_as(delta[...,:1])
        pad = self.boarder_weight*torch.ones(delta[...,:1].size(),device = delta.device)
        delta = torch.cat([delta, pad], dim=-1)   # [N, L]

        if noise > 0.:
            sigma += (torch.randn(size=sigma.size(),device = delta.device) * noise)

        weights = gen_weight(sigma, delta).unsqueeze(-1)    #[N, L, 1]

        color = torch.sum(torch.sigmoid(rgb) * weights, dim=1) #[N, 3]
        depth = torch.sum(weights * depth, dim=1)   # [N, 1]
        acc_map = torch.sum(weights, dim = 1) #
        # #TODO: This scaling will make the program crash. because the summing nan value when acc_map is near to 0.
        # if acc_map.max() > 0.0001:
        #     acc_map = acc_map / acc_map.max()

        if self.use_mask:
            #TODO: Here may have a bug about multiply color at the last
            color = color + (1.-acc_map[...,None]) * color
        
        return color, depth, acc_map, weights
    
    
if __name__ == "__main__":
    N_rays = 1024
    N_samples = 64

    depth = torch.randn(N_rays, N_samples, 1)
    raw  = torch.randn(N_rays, N_samples, 3)
    sigma = torch.randn(N_rays, N_samples, 1)

    renderer = VolumeRenderer()

    color, dpt, weights = renderer(depth, raw, sigma)
    print('Predicted [CPU]: ', color.shape, dpt.shape, weights.shape)

    if torch.cuda.is_available():
        depth = depth.cuda()
        raw = raw.cuda()
        sigma = sigma.cuda()
        renderer = renderer.cuda()

        color, dpt, weights = renderer(depth, raw, sigma)
        print('Predicted [GPU]: ', color.shape, dpt.shape, weights.shape)

    print('Test load data')
    tf_depth = np.load('layers/test_output/depth_map.npy')
    tf_color = np.load('layers/test_output/rgb_map.npy')
    tf_weights = np.load('layers/test_output/weights.npy')
    print('TF output = ', tf_depth.shape, tf_color.shape, tf_weights.shape)

    raws = torch.from_numpy(np.load('layers/test_output/raws.npy'))
    ray_d = torch.from_numpy(np.load('layers/test_output/ray_d.npy'))
    z_val = torch.from_numpy(np.load('layers/test_output/z_vals.npy'))

    print('TF input = ', raws.shape, ray_d.shape, z_val.shape)

    in_depth = z_val
    print('in_depth = ', in_depth.shape)
    in_raw = raws[:, :, :3]
    print('in_raw = ', in_raw.shape)
    in_sigma = raws[:, :, 3:]
    print('in_sigma = ', in_sigma.shape)

    color, dpt, weights = renderer(in_depth.unsqueeze(-1).cuda(), in_raw.cuda(), in_sigma.cuda())
    print('Predicted-TF [GPU]: ', color.shape, dpt.shape, weights.shape)

    print('ERROR [GPU]: ', 
            np.mean(tf_color - color.detach().cpu().numpy()),
            np.mean(tf_depth - dpt.squeeze(-1).detach().cpu().numpy()),
            np.mean(tf_weights - weights.squeeze(-1).detach().cpu().numpy()))