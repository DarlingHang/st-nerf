import numpy as np
import glob
import os


def campose_to_extrinsic(camposes):
    if camposes.shape[1]!=12:
        raise Exception(" wrong campose data structure!")
    
    res = np.zeros((camposes.shape[0],4,4))
    
    res[:,0,:] = camposes[:,0:4]
    res[:,1,:] = camposes[:,4:8]
    res[:,2,:] = camposes[:,8:12]
    res[:,3,3] = 1.0
    
    return res


def read_intrinsics(fn_instrinsic):
    fo = open(fn_instrinsic)
    data= fo.readlines()
    i = 0
    Ks = []
    while i<len(data):
        tmp = data[i].split()
        a = [float(i) for i in tmp[0:3]]
        a = np.array(a)
        b = [float(i) for i in tmp[3:6]]
        b = np.array(b)
        c = [float(i) for i in tmp[6:9]]
        c = np.array(c)
        res = np.vstack([a,b,c])
        Ks.append(res)

        i = i+1
    Ks = np.stack(Ks)
    fo.close()

    return Ks

def get_iteration_path(root_dir, fix_iter = -1):
    if fix_iter != -1:
        return os.path.join(root_dir,'frame','layered_rfnr_checkpoint_%d.pt' % fix_iter)

    if not os.path.exists(root_dir):
        return None
    file_names = glob.glob(os.path.join(root_dir,'layered_rfnr_checkpoint_*.pt'))
    max_iter = -1
    for file_name in file_names:
        temp = file_name.split('/')[-1].split('_')
        if len(temp) != 4:
            continue
        num_name = temp[-1]
        temp = int(num_name.split('.')[0])
        if temp > max_iter:
            max_iter = temp
    if not os.path.exists(os.path.join(root_dir,'layered_rfnr_checkpoint_%d.pt' % max_iter)):
        return None
    return os.path.join(root_dir,'layered_rfnr_checkpoint_%d.pt' % max_iter)

def get_iteration_path_and_iter(root_dir, fix_iter = -1):
    if fix_iter != -1:
        return os.path.join(root_dir,'frame','layered_rfnr_checkpoint_%d.pt' % fix_iter)

    if not os.path.exists(root_dir):
        return None
    file_names = glob.glob(os.path.join(root_dir,'layered_rfnr_checkpoint_*.pt'))
    max_iter = -1
    for file_name in file_names:
        num_name = file_name.split('_')[-1]
        temp = int(num_name.split('.')[0])
        if temp > max_iter:
            max_iter = temp
    if not os.path.exists(os.path.join(root_dir,'layered_rfnr_checkpoint_%d.pt' % max_iter)):
        return None
    return os.path.join(root_dir,'layered_rfnr_checkpoint_%d.pt' % max_iter), max_iter


def read_mask(path):
    fo = open(path)
    data= fo.readlines()
    mask = []
    for i in range(len(data)):
        tmp = int(data[i])
        mask.append(tmp)
    mask = np.array(mask)
    fo.close()

    return mask