import torch
import numpy as np
'''
INPUT: 

z_vals:  (N,L)
weights: (N,L)

OUPUT:

samples_z: (N,L)


'''
torch.autograd.set_detect_anomaly(True)


def sample_pdf(z_vals, weights, N_samples, det=False, pytest=False):
    # Get pdf
    bins = .5 * (z_vals[...,1:] + z_vals[...,:-1])
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1], device = z_vals.device), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples, device = z_vals.device)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples], device = z_vals.device)

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous()

    inds = torch.searchsorted(cdf, u, right = True)
    below = torch.max(torch.zeros_like(inds-1, device = inds.device), inds-1)
    above = torch.min(cdf.shape[-1]-1 * torch.ones_like(inds, device = inds.device), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom, device=denom.device), denom)
    t = (u-cdf_g[...,0])/denom
    samples_z = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples_z
