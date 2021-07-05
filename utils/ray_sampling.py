
import torch


'''
Sample rays from views (and images) with/without masks

--------------------------
INPUT Tensors
Ks: intrinsics of cameras (M,3,3)
Ts: extrinsic of cameras (M,4,4)
image_size: the size of image [H,W]
images: (M,C,H,W)
mask_threshold: a float threshold to mask rays
masks:(M,H,W)
-------------------
OUPUT:
list of rays:  (N,6)  dirs(3) + pos(3)
RGB:  (N,C)
'''

def ray_sampling(Ks, Ts, image_size, masks=None, mask_threshold = 0.5, images=None, outlier_map=None):
    h = image_size[0]
    w = image_size[1]
    M = Ks.size(0)


    x = torch.linspace(0,h-1,steps=h,device = Ks.device )
    y = torch.linspace(0,w-1,steps=w,device = Ks.device )

    grid_x, grid_y = torch.meshgrid(x,y)
    coordinates = torch.stack([grid_y, grid_x]).unsqueeze(0).repeat(M,1,1,1)   #(M,2,H,W)
    coordinates = torch.cat([coordinates,torch.ones(coordinates.size(0),1,coordinates.size(2), 
                             coordinates.size(3),device = Ks.device) ],dim=1).permute(0,2,3,1).unsqueeze(-1)


    inv_Ks = torch.inverse(Ks)

    dirs = torch.matmul(inv_Ks,coordinates) #(M,H,W,3,1)
    dirs = dirs/torch.norm(dirs,dim=3,keepdim = True)
    dirs = torch.cat([dirs,torch.zeros(dirs.size(0),coordinates.size(1), 
                             coordinates.size(2),1,1,device = Ks.device) ],dim=3) #(M,H,W,4,1)


    dirs = torch.matmul(Ts,dirs) #(M,H,W,4,1)
    dirs = dirs[:,:,:,0:3,0]  #(M,H,W,3)

    pos = Ts[:,0:3,3] #(M,3)
    pos = pos.unsqueeze(1).unsqueeze(1).repeat(1,h,w,1)

    if outlier_map is not None:
        ids = outlier_map.reshape([M,h,w,1])
        rays = torch.cat([pos,dirs,ids],dim = 3)  #(M,H,W,7)
    else:
        rays = torch.cat([pos,dirs],dim = 3)  #(M,H,W,6)

    if images is not None:
        rgbs = images.permute(0,2,3,1) #(M,H,W,C)
    else:
        rgbs = None

    if masks is not None:
        rays = rays[masks>mask_threshold,:]
        if rgbs is not None:
            rgbs = rgbs[masks>mask_threshold,:]

    else:
        rays = rays.reshape((-1,rays.size(3)))
        if rgbs is not None:
            rgbs = rgbs.reshape((-1, rgbs.size(3)))

    return rays,rgbs
    
# Sample rays and labels with K,T and bbox
def ray_sampling_label_bbox(image,label,K,T,bbox=None, bboxes=None):

    _,H,W = image.shape

    if bbox != None:
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

        # print(pixels)
        # print(min_pixel)
        # print(max_pixel)

        min_pixel[min_pixel < 0.0] = 0
        if min_pixel[0] >= H-1:
            min_pixel[0] = H-1
        if min_pixel[1] >= W-1:
            min_pixel[1] = W-1
        
        max_pixel[max_pixel < 0.0] = 0
        if max_pixel[0] >= H-1:
            max_pixel[0] = H-1
        if max_pixel[1] >= W-1:
            max_pixel[1] = W-1
        
        minh = int(min_pixel[0])
        minw = int(min_pixel[1])
        maxh = int(max_pixel[0])+1
        maxw = int(max_pixel[1])+1
    else:
        minh = 0
        minw = 0
        maxh = H
        maxw = W

    # print(max_pixel,min_pixel)
    # print(minh,maxh,minw,maxw)

    if minh == maxh or minw == maxw:
        print('Warning: there is a pointcloud cannot find right bbox')
    
    # minh = 0
    # minw = 0
    # maxh = H
    # maxw = W
    # image_cutted = image[:,minh:maxh,minw:maxw]
    # label_cutted = label[:,minh:maxh,minw:maxw]

    K = K.unsqueeze(0)
    T = T.unsqueeze(0)
    M = 1


    x = torch.linspace(0,H-1,steps=H,device = K.device )
    y = torch.linspace(0,W-1,steps=W,device = K.device )

    grid_x, grid_y = torch.meshgrid(x,y)
    coordinates = torch.stack([grid_y, grid_x]).unsqueeze(0).repeat(M,1,1,1)   #(M,2,H,W)
    coordinates = torch.cat([coordinates,torch.ones(coordinates.size(0),1,coordinates.size(2), 
                             coordinates.size(3),device = K.device) ],dim=1).permute(0,2,3,1).unsqueeze(-1)


    inv_Ks = torch.inverse(K)

    dirs = torch.matmul(inv_Ks,coordinates) #(M,H,W,3,1)
    dirs = dirs/torch.norm(dirs,dim=3,keepdim = True)
    dirs = torch.cat([dirs,torch.zeros(dirs.size(0),coordinates.size(1), 
                             coordinates.size(2),1,1,device = K.device) ],dim=3) #(M,H,W,4,1)


    dirs = torch.matmul(T,dirs) #(M,H,W,4,1)
    dirs = dirs[:,:,:,0:3,0]  #(M,H,W,3)

    pos = T[:,0:3,3] #(M,3)
    pos = pos.unsqueeze(1).unsqueeze(1).repeat(1,H,W,1)
    rays = torch.cat([pos,dirs],dim = 3)

    rays = rays[:,minh:maxh,minw:maxw,:] #(H',W',6)
    rays = rays.reshape((-1,rays.size(3)))

    ray_mask = torch.zeros_like(label)
    ray_mask[:,minh:maxh,minw:maxw] = 1.0
    ray_mask = ray_mask.permute(1,2,0)

    label = label[:,minh:maxh,minw:maxw].permute(1,2,0) #(H',W',1)
    image = image[:,minh:maxh,minw:maxw].permute(1,2,0) #(H',W',3)


    rays = rays.reshape(-1,6)
    label = label.reshape(-1,1) #(N,1)
    image = image.reshape(-1,3)

    if bboxes is not None:
        layered_bboxes = torch.zeros(rays.size(0),8,3)
        for i in range(len(bboxes)):
            idx = (label == i).squeeze() #(N,)
            layered_bboxes[idx] = bboxes[i]

    if bboxes is None:
        return rays, label, image, ray_mask
    else:
        return rays, label, image, ray_mask,layered_bboxes

def ray_sampling_label_label(image,label,K,T,label0):

    _,H,W = image.shape

    K = K.unsqueeze(0)
    T = T.unsqueeze(0)
    M = 1


    x = torch.linspace(0,H-1,steps=H,device = K.device )
    y = torch.linspace(0,W-1,steps=W,device = K.device )

    grid_x, grid_y = torch.meshgrid(x,y)
    coordinates = torch.stack([grid_y, grid_x]).unsqueeze(0).repeat(M,1,1,1)   #(M,2,H,W)
    coordinates = torch.cat([coordinates,torch.ones(coordinates.size(0),1,coordinates.size(2), 
                                coordinates.size(3),device = K.device) ],dim=1).permute(0,2,3,1).unsqueeze(-1)


    inv_Ks = torch.inverse(K)

    dirs = torch.matmul(inv_Ks,coordinates) #(M,H,W,3,1)
    dirs = dirs/torch.norm(dirs,dim=3,keepdim = True)
    dirs = torch.cat([dirs,torch.zeros(dirs.size(0),coordinates.size(1), 
                                coordinates.size(2),1,1,device = K.device) ],dim=3) #(M,H,W,4,1)


    dirs = torch.matmul(T,dirs) #(M,H,W,4,1)
    dirs = dirs[:,:,:,0:3,0]  #(M,H,W,3)

    pos = T[:,0:3,3] #(M,3)
    pos = pos.unsqueeze(1).unsqueeze(1).repeat(1,H,W,1)
    rays = torch.cat([pos,dirs],dim = 3)


    ray_mask = torch.zeros_like(label)
    idx = (label == label0)
    ray_mask[idx] = 1.0
    ray_mask = ray_mask.permute(1,2,0)

    rays = rays[idx,:] #(N,6)

    label = label[idx] #(N)
    label = label.reshape(-1,1)
    image = image[:,idx.squeeze()].permute(1,0) #(N,3)


    return rays, label, image, ray_mask








