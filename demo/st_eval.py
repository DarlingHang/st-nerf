import numpy as np
import os
import PIL.Image
from termcolor import colored
import json
from skimage.metrics import structural_similarity

def save_json(file, data):
    if not os.path.exists(os.path.dirname(file)):
        os.makedirs(os.path.dirname(file))
    with open(file, 'w') as f:
        json.dump(data, f, indent=4)

def mse_metric(image_pred, image_gt):
    value=np.mean((image_pred-image_gt)**2)
    return value

def mae_metric(image_pred, image_gt):
    value=np.abs(image_pred-image_gt)
    return  np.mean(value)

def psnr_metric(image_pred, image_gt):
    mse = np.mean((image_pred - image_gt)**2)
    psnr = -10 * np.log(mse) / np.log(10)
    return psnr

def ssim_metric(image_pred, image_gt, reduction='mean'):
    ssim = structural_similarity(image_pred, image_gt, multichannel=True)
    return ssim # in [-1, 1]

def eval():
    result_dir = '/home/shenwenhao/st-nerf/outputs/'
    pred_path = '/home/shenwenhao/st-nerf/outputs/walking/rendered/origin'
    gt_path = '/home/shenwenhao/walking'
    result_path = os.path.join(result_dir, 'walking_new_metrices.json')
    os.system('mkdir -p {}'.format(os.path.dirname(result_path)))
    print(colored('The evaluation results are saved at {}'.format(result_dir), 'yellow'))
    
    bpsnr = []
    bssim = []
    for i in range(0,75):
        apsnr = []
        assim = []
        for j in range(16):
            img_pred = np.array(PIL.Image.open(os.path.join(pred_path, 'cam%d'%(j), 'mixed/color','{}.jpg'.format(i))).convert('RGB'))/255.0
            img_gt = np.array(PIL.Image.open(os.path.join(gt_path, 'frame{}'.format(i+1), 'images/%d.png'%j)).convert('RGB'))/255.0
            # amse.append(mse_metric(img_pred, img_gt))
            # amae.append(mae_metric(img_pred, img_gt))
            apsnr.append(psnr_metric(img_pred, img_gt))
            assim.append(ssim_metric(img_pred, img_gt))
        bpsnr.append(np.mean(apsnr))
        bssim.append(np.mean(assim))
    metrics = {'psnr': bpsnr, 'ssim': bssim}
    save_json(result_path, metrics)
    print('Exp: walking demo')
    print('psnr: {}'.format(np.mean(bpsnr)))
    print('ssim: {}'.format(np.mean(bssim)))
    
    result_path = os.path.join(result_dir, 'taekwondo_metrices.json')
    bpsnr = []
    bssim = []
    pred_path = '/home/shenwenhao/st-nerf/outputs/taekwondo/rendered/origin'
    gt_path = '/home/shenwenhao/taekwondo'
    for i in range(0, 101):
        #amse = []
        #amae = []
        apsnr = []
        assim = []
        for j in range(15):
            img_pred = np.array(PIL.Image.open(os.path.join(pred_path, 'cam%d'%(j), 'mixed/color','{}.jpg'.format(i))).convert('RGB'))/255.0
            img_gt = np.array(PIL.Image.open(os.path.join(gt_path, 'frame{}'.format(i+1), 'images/%d.png'%j)).convert('RGB'))/255.0
            # amse.append(mse_metric(img_pred, img_gt))
            # amae.append(mae_metric(img_pred, img_gt))
            apsnr.append(psnr_metric(img_pred, img_gt))
            assim.append(ssim_metric(img_pred, img_gt))
        bpsnr.append(np.mean(apsnr))
        bssim.append(np.mean(assim))
    metrics = {'psnr': bpsnr, 'ssim': bssim}
    save_json(result_path, metrics)
    # print(img_pred.shape, img_gt.shape)
    print('Exp: taekwondo demo')
    print('psnr: {}'.format(np.mean(apsnr)))
    print('ssim: {}'.format(np.mean(assim)))


if __name__ == '__main__':
    eval()