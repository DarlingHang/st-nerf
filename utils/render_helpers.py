import numpy as np
import torch


def lookat(eye,center,up):
    z = eye - center
    z /= np.sqrt(z.dot(z))

    y = up
    x = np.cross(y,z)
    y = np.cross(z,x)

    x /= np.sqrt(x.dot(x))
    y /= np.sqrt(y.dot(y))

    T = np.identity(4)
    T[0,:3] = x
    T[1,:3] = y
    T[2,:3] = z
    T[0,3] = -x.dot(eye)
    T[1,3] = -y.dot(eye)
    T[2,3] = -z.dot(eye)
    T[3,:] = np.array([0,0,0,1])

    # What we need is camera pose
    T = np.linalg.inv(T) 
    T[:3,1] = -T[:3,1]
    T[:3,2] = -T[:3,2]

    return T

# degree is True means using degree measure, else means using radian system
def getSphericalPosition(r,theta,phi,degree=True):
    if degree:
        theta = theta / 180 * pi
        phi = phi / 180 * pi
    x = r * cos(theta) * sin(phi)
    z = r * cos(theta) * cos(phi)
    y = r * sin(theta)
    return np.array([x,y,z])

def generate_rays(K, T, bbox, h, w):

    if bbox is not None:
        bbox = bbox.reshape(8,3)
        bbox = torch.transpose(bbox,0,1) #(3,8)
        bbox = torch.cat([bbox,torch.ones(1,bbox.shape[1])],0)
        inv_T = torch.inverse(T)

        pts = torch.mm(inv_T,bbox)

        pts = pts[:3,:]
        pixels = torch.mm(K,pts)
        pixels = pixels / pixels[2,:]
        pixels = pixels[:2,:]
        temp = torch.zeros_like(pixels)
        temp[1,:] = pixels[0,:]
        temp[0,:] = pixels[1,:]
        pixels = temp

        min_pixel = torch.min(pixels, dim=1)[0]
        max_pixel = torch.max(pixels, dim=1)[0]

        min_pixel[min_pixel < 0.0] = 0
        if min_pixel[0] >= h-1:
            min_pixel[0] = h-1
        if min_pixel[1] >= w-1:
            min_pixel[1] = w-1
        
        max_pixel[max_pixel < 0.0] = 0
        if max_pixel[0] >= h-1:
            max_pixel[0] = h-1
        if max_pixel[1] >= w-1:
            max_pixel[1] = w-1
        
        minh = int(min_pixel[0])
        minw = int(min_pixel[1])
        maxh = int(max_pixel[0])+1
        maxw = int(max_pixel[1])+1
    else:
        minh = 0
        minw = 0
        maxh = h
        maxw = w

    # print(max_pixel,min_pixel)
    # print(minh,maxh,minw,maxw)

    if minh == maxh or minw == maxw:
        print('Warning: there is a pointcloud cannot find right bbox')

    K = K.unsqueeze(0)
    T = T.unsqueeze(0)
    M = 1

    x = torch.linspace(0,h-1,steps=h,device = K.device )
    y = torch.linspace(0,w-1,steps=w,device = K.device )

    grid_x, grid_y = torch.meshgrid(x,y)
    coordinates = torch.stack([grid_y, grid_x]).unsqueeze(0).repeat(M,1,1,1)   #(M,2,H,W)
    coordinates = torch.cat([coordinates,torch.ones(coordinates.size(0),1,coordinates.size(2), 
                             coordinates.size(3),device = K.device) ],dim=1).permute(0,2,3,1).unsqueeze(-1)


    inv_K = torch.inverse(K)

    dirs = torch.matmul(inv_K,coordinates) #(M,H,W,3,1)
    dirs = dirs/torch.norm(dirs,dim=3,keepdim = True)
    dirs = torch.cat([dirs,torch.zeros(dirs.size(0),coordinates.size(1), 
                             coordinates.size(2),1,1,device = K.device) ],dim=3) #(M,H,W,4,1)


    dirs = torch.matmul(T,dirs) #(M,H,W,4,1)
    dirs = dirs[:,:,:,0:3,0]  #(M,H,W,3)

    pos = T[:,0:3,3] #(M,3)
    pos = pos.unsqueeze(1).unsqueeze(1).repeat(1,h,w,1)

    rays = torch.cat([pos,dirs],dim = 3)  #(M,H,W,6)

    rays = rays[:,minh:maxh,minw:maxw,:] #(M,H',W',6)

    rays = rays.reshape((-1,rays.size(3)))

    ray_mask = torch.zeros(h,w,1)
    ray_mask[minh:maxh,minw:maxw,:] = 1.0

    return rays, ray_mask