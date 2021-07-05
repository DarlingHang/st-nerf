import torch

def vis_density(model,bbox, L= 32):

    maxs = torch.max(bbox, dim=0).values
    mins = torch.min(bbox, dim=0).values


    x = torch.linspace(mins[0],maxs[0],steps=L).cuda()
    y = torch.linspace(mins[1],maxs[1],steps=L).cuda()
    z = torch.linspace(mins[2],maxs[2],steps=L).cuda()
    grid_x ,grid_y,grid_z = torch.meshgrid(x, y,z)
    xyz = torch.stack([grid_x ,grid_y,grid_z], dim = -1)  #(L,L,L,3)

    xyz = xyz.reshape((-1,3)) #(L*L*L,3)


    xyzs = xyz.split(5000, dim=0)

    sigmas = []
    for i in xyzs:
        with torch.no_grad():
            _,density = model.spacenet_fine(i, None, model.maxs, model.mins) #(L*L*L,1)
            density = torch.nn.functional.relu(density)
            sigmas.append(density.detach().cpu())

    sigmas = torch.cat(sigmas, dim=0)

    return sigmas