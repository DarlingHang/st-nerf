import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import numpy as np

def corrupt_cameras(cam_poses, offset=(-0.1, 0.1), rotation=(-5, 5)):
    rand_t = np.random.rand(cam_poses.shape[0], 3)
    perturb_t = (1 - rand_t) * offset[0] + rand_t * offset[1]
    tr = cam_poses[:, :3, 3] + perturb_t
    tr = tr[..., None] # [N, 3, 1]
    
    rand_r = np.random.rand(cam_poses.shape[0], 3)
    rand_r = (1 - rand_r) * rotation[0] + rand_r * rotation[1]
    rand_r = np.deg2rad(rand_r)
    
    # Pre-compute rotation matrices
    Rx = np.stack((
        np.ones_like(rand_r[:, 0]), np.zeros_like(rand_r[:, 0]), np.zeros_like(rand_r[:, 0]),
        np.zeros_like(rand_r[:, 0]), np.cos(rand_r[:, 0]), -np.sin(rand_r[:, 0]),
        np.zeros_like(rand_r[:, 0]), np.sin(rand_r[:, 0]), np.cos(rand_r[:, 0])
    ), axis=1).reshape(-1, 3, 3)
  
    Ry = np.stack((
        np.cos(rand_r[:, 1]), np.zeros_like(rand_r[:, 1]), np.sin(rand_r[:, 1]),
        np.zeros_like(rand_r[:, 1]), np.ones_like(rand_r[:, 1]), np.zeros_like(rand_r[:, 1]),
        -np.sin(rand_r[:, 1]), np.zeros_like(rand_r[:, 1]), np.cos(rand_r[:, 1])
    ), axis=1).reshape(-1, 3, 3)

    Rz = np.stack((
        np.cos(rand_r[:, 2]), -np.sin(rand_r[:, 2]), np.zeros_like(rand_r[:, 2]),
        np.sin(rand_r[:, 2]), np.cos(rand_r[:, 2]), np.zeros_like(rand_r[:, 2]),
        np.zeros_like(rand_r[:, 2]), np.zeros_like(rand_r[:, 2]), np.ones_like(rand_r[:, 2])
    ), axis=1).reshape(-1, 3, 3)
    
    # Apply rotation sequentially
    rot = cam_poses[:, :3, :3] # [N, 3, 3]
    for perturb_r in [Rz, Ry, Rx]:
        rot = np.matmul(perturb_r, rot)
    
    return np.concatenate([rot, tr], axis=-1)

# Camera Transformation Layer
class CameraTransformer(nn.Module):

    def __init__(self, num_cams, trainable=False):
        """ Init layered sampling
        num_cams: number of training cameras
        trainable: Whether planes can be trained by optimizer
        """
        super(CameraTransformer, self).__init__()
        
        self.trainable = trainable

        identity_quat = torch.Tensor([0, 0, 0, 1]).repeat((num_cams, 1))
        identity_off = torch.Tensor([0, 0, 0]).repeat((num_cams, 1))
        if self.trainable:
            self.rvec = nn.Parameter(torch.Tensor(identity_quat)) # [N_cameras, 4]
            self.tvec = nn.Parameter(torch.Tensor(identity_off)) # [N_cameras, 3]
        else:
            self.register_buffer('rvec', torch.Tensor(identity_quat)) # [N_cameras, 4]
            self.register_buffer('tvec', torch.Tensor(identity_off)) # [N_cameras, 3]

        print("Create %d %s camera transformer" % (num_cams, 'trainable' if self.rvec.requires_grad else 'non-trainable'))

    def rot_mats(self):
        theta = torch.sqrt(1e-5 + torch.sum(self.rvec ** 2, dim=1))
        rvec = self.rvec / theta[:, None]
        return torch.stack((
            1. - 2. * rvec[:, 1] ** 2 - 2. * rvec[:, 2] ** 2,
            2. * (rvec[:, 0] * rvec[:, 1] - rvec[:, 2] * rvec[:, 3]),
            2. * (rvec[:, 0] * rvec[:, 2] + rvec[:, 1] * rvec[:, 3]),

            2. * (rvec[:, 0] * rvec[:, 1] + rvec[:, 2] * rvec[:, 3]),
            1. - 2. * rvec[:, 0] ** 2 - 2. * rvec[:, 2] ** 2,
            2. * (rvec[:, 1] * rvec[:, 2] - rvec[:, 0] * rvec[:, 3]),

            2. * (rvec[:, 0] * rvec[:, 2] - rvec[:, 1] * rvec[:, 3]),
            2. * (rvec[:, 0] * rvec[:, 3] + rvec[:, 1] * rvec[:, 2]),
            1. - 2. * rvec[:, 0] ** 2 - 2. * rvec[:, 1] ** 2
        ), dim=1).view(-1, 3, 3)

    def forward(self, rays_o, rays_d, **render_kwargs):
        """ Generate sample points
        Args:
        rays_o: [N_rays, 3+1] origin points of rays with camera id
        rays_d: [N_rays, 3+1] directions of rays with camera id

        render_kwargs: other render parameters

        Return:
        rays_o: [N_rays, 3] Transformed origin points
        rays_d: [N_rays, 3] Transformed directions of rays
        """
        assert rays_o.shape[-1] == 4
        assert (rays_o[:, 3] == rays_d[:, 3]).all()
        indx = rays_o[:, 3].type(torch.LongTensor)
        
        # Rotate ray directions w.r.t. rvec
        c2w = self.rot_mats()[indx]
        rays_d = torch.sum(rays_d[..., None, :3] * c2w[:, :3, :3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
        
        # Translate camera w.r.t. tvec
        rays_o = rays_o[..., :3] + self.tvec[indx]

        return rays_o, rays_d