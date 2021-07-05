# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import logging
import imageio
import torch

from utils import layered_batchify_ray, vis_density, metrics
from utils.metrics import *
import numpy as np
import os
import time

def evaluator(val_dataset, model, loss_fn, swriter, epoch):
    
    model.eval()
    rays, rgbs, labels, image, label, mask, bbox, near_far = val_dataset[0]

    rays = rays.cuda()
    rgbs = rgbs.cuda()
    bbox = bbox.cuda()
    labels = labels.cuda()
    color_gt = image.cuda()
    mask = mask.cuda()
    near_far = near_far.cuda()

    # uv_list = (mask).squeeze().nonzero()
    # u_list = uv_list[:,0]
    # v_list = uv_list[:,1]

    with torch.no_grad():
        # TODO: Use mask to gain less query of space
        stage2, stage1, stage2_layer, stage1_layer, _ = layered_batchify_ray(model, rays, labels, bbox, near_far=near_far)
        for i in range(len(stage2_layer)):
            color_1 = stage2_layer[i][0]
            depth_1 = stage2_layer[i][1]
            acc_map_1 = stage2_layer[i][2]
            #print(color_1.shape)
            #print(depth_1.shape)
            #print(acc_map_1.shape)

            color_0 = stage1_layer[i][0]
            depth_0 = stage1_layer[i][1]
            acc_map_00 = stage1_layer[i][2]


            color_img = color_1.reshape( (color_gt.size(1), color_gt.size(2), 3) ).permute(2,0,1)
            depth_img = depth_1.reshape( (color_gt.size(1), color_gt.size(2), 1) ).permute(2,0,1)
            depth_img = (depth_img-depth_img.min())/(depth_img.max()-depth_img.min())
            acc_map = acc_map_1.reshape( (color_gt.size(1), color_gt.size(2), 1) ).permute(2,0,1)


            color_img_0 = color_0.reshape( (color_gt.size(1), color_gt.size(2), 3) ).permute(2,0,1)
            depth_img_0 = depth_0.reshape( (color_gt.size(1), color_gt.size(2), 1) ).permute(2,0,1)
            depth_img_0 = (depth_img_0-depth_img_0.min())/(depth_img_0.max()-depth_img_0.min())
            acc_map_0 = acc_map_00.reshape((color_gt.size(1), color_gt.size(2), 1) ).permute(2,0,1)


            depth_img = (depth_img-depth_img.min())/(depth_img.max()-depth_img.min())
            depth_img_0 = (depth_img_0-depth_img_0.min())/(depth_img_0.max()-depth_img_0.min())

            color_img = color_img*((mask).permute(2,0,1).repeat(3,1,1))
            color_gt = color_gt*((mask).permute(2,0,1).repeat(3,1,1))

            if i == 0:
                swriter.add_image('stage2_bkgd/rendered', color_img, epoch)
                swriter.add_image('stage2_bkgd/depth', depth_img, epoch)
                swriter.add_image('stage2_bkgd/alpha', acc_map, epoch)

                swriter.add_image('stage1_bkgd/rendered', color_img_0, epoch)
                swriter.add_image('stage1_bkgd/depth', depth_img_0, epoch)
                swriter.add_image('stage1_bkgd/alpha', acc_map_0, epoch)
                
            else:
                swriter.add_image('stage2_layer' +str(i)+ '/rendered', color_img, epoch)
                swriter.add_image('stage2_layer' +str(i)+ '/depth', depth_img, epoch)
                swriter.add_image('stage2_layer' +str(i)+ '/alpha', acc_map, epoch)

                swriter.add_image('stage1_layer' +str(i)+ '/rendered', color_img_0, epoch)
                swriter.add_image('stage1_layer' +str(i)+ '/depth', depth_img_0, epoch)
                swriter.add_image('stage1_layer' +str(i)+ '/alpha', acc_map_0, epoch)


        color_1 = stage2[0]
        depth_1 = stage2[1]
        acc_map_1 = stage2[2]
        #print(color_1.shape)
        #print(depth_1.shape)
        #print(acc_map_1.shape)

        color_0 = stage1[0]
        depth_0 = stage1[1]
        acc_map_00 = stage1[2]


        color_img = color_1.reshape( (color_gt.size(1), color_gt.size(2), 3) ).permute(2,0,1)
        depth_img = depth_1.reshape( (color_gt.size(1), color_gt.size(2), 1) ).permute(2,0,1)
        depth_img = (depth_img-depth_img.min())/(depth_img.max()-depth_img.min())
        acc_map = acc_map_1.reshape( (color_gt.size(1), color_gt.size(2), 1) ).permute(2,0,1)


        color_img_0 = color_0.reshape( (color_gt.size(1), color_gt.size(2), 3) ).permute(2,0,1)
        depth_img_0 = depth_0.reshape( (color_gt.size(1), color_gt.size(2), 1) ).permute(2,0,1)
        depth_img_0 = (depth_img_0-depth_img_0.min())/(depth_img_0.max()-depth_img_0.min())
        acc_map_0 = acc_map_00.reshape((color_gt.size(1), color_gt.size(2), 1) ).permute(2,0,1)


        depth_img = (depth_img-depth_img.min())/(depth_img.max()-depth_img.min())
        depth_img_0 = (depth_img_0-depth_img_0.min())/(depth_img_0.max()-depth_img_0.min())

        color_img = color_img*((mask).permute(2,0,1).repeat(3,1,1))
        color_gt = color_gt*((mask).permute(2,0,1).repeat(3,1,1))


        swriter.add_image('GT/Label', label * 50, epoch)
        swriter.add_image('GT/Image', color_gt, epoch)

        swriter.add_image('stage2/rendered', color_img, epoch)
        swriter.add_image('stage2/depth', depth_img, epoch)
        swriter.add_image('stage2/alpha', acc_map, epoch)

        swriter.add_image('stage1/rendered', color_img_0, epoch)
        swriter.add_image('stage1/depth', depth_img_0, epoch)
        swriter.add_image('stage1/alpha', acc_map_0, epoch)


        return loss_fn(color_img, color_gt).item()


def do_train(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        loss_fn,
        swriter,
        resume_epoch = 0,
        psnr_thres = 100
):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    output_dir = cfg.OUTPUT_DIR
    max_epochs = cfg.SOLVER.MAX_EPOCHS
    train_by_pointcloud = cfg.MODEL.TRAIN_BY_POINTCLOUD
    use_label = cfg.DATASETS.USE_LABEL
    coarse_stage = cfg.SOLVER.COARSE_STAGE
    remove_outliers = cfg.MODEL.REMOVE_OUTLIERS


    logger = logging.getLogger("LayeredRFRender.%s.train" % cfg.OUTPUT_DIR.split('/')[-1])
    logger.info("Start training")
    #global step
    global_step = 0
    
    torch.autograd.set_detect_anomaly(True)


    for epoch in range(1+resume_epoch,max_epochs):
        print('Training Epoch %d...' % epoch)
        model.cuda()

        #psnr monitor 
        psnr_monitor = []

        #epoch time recordingbatchify_ray
        epoch_start = time.time()
        for batch_idx, batch in enumerate(train_loader):

            #iteration time recording
            iters_start = time.time()
            global_step = (epoch -1) * len(train_loader) + batch_idx

            model.train()
            optimizer.zero_grad()

            rays, rgbs, labels, bbox_labels, bboxes, near_far = batch
            bbox_labels = bbox_labels.cuda()
            labels = labels.cuda()
            rays = rays.cuda()
            rgbs = rgbs.cuda()
            bboxes = bboxes.cuda()
            near_far = near_far.cuda()

            loss = 0
           
            if epoch<coarse_stage:
                stage2, stage1,stage2_layer,stage1_layer,ray_mask = model( rays, bbox_labels, bboxes,True,near_far=near_far)
            else:
                stage2, stage1,stage2_layer,stage1_layer,ray_mask = model( rays, bbox_labels, bboxes,False,near_far=near_far)

            # inliers = (labels != 0).repeat(1,3)
            # outliers = (labels == 0).repeat(1,3)
            #inliers = torch.logical_and(labels < label+0.5, labels > label-0.5)
            # print(torch.sum(outliers))
            # print(torch.sum(inliers))
            # out_threshold = 0.5

            predict_rgb_0 = stage1[0]
            predict_rgb_1 = stage2[0]

            # predict_rgb_0 = stage1_layer[0][labels.repeat(1,3) != 0]
            # predict_rgb_1 = stage2_layer[0][labels.repeat(1,3) != 0]
            # rgbs = rgbs[labels.repeat(1,3) != 0]

            # print('ray number is %d' % torch.sum(labels != 0))


            # print('layer ray number is %d, bbox layer ray number is %d, outlier number is %d, total number is %d' % (torch.sum(labels != 0), torch.sum(bbox_labels != 0), outliers_1.shape[0], rays.size(0)))


            loss1 = loss_fn(predict_rgb_0, rgbs)
            loss2 = loss_fn(predict_rgb_1, rgbs)
            if epoch < 3 and remove_outliers:
                outliers_1 = []
                outliers_2 = []
                inliers_1 =[]
                inliers_2 = []
                for i in range(len(stage1_layer)):
                    
                    if i != 0: #i!=3 for spiderman basket
                        outliers_1.append(stage1_layer[i][2][labels == 0])
                        outliers_2.append(stage2_layer[i][2][labels == 0])
                    # else:
                    #     outliers_1.append(stage1_layer[i][2][labels == 0])
                    #     outliers_2.append(stage2_layer[i][2][labels == 0])
                    inliers_1.append(stage1_layer[i][2][labels == i])
                    inliers_2.append(stage2_layer[i][2][labels == i])

                if outliers_1 != []:
                    outliers_1 = torch.cat(outliers_1,0)
                    outliers_2 = torch.cat(outliers_2,0)
                inliers_1 = torch.cat(inliers_1,0)
                inliers_2 = torch.cat(inliers_2,0)
                # print('total ray number is ', stage2[1].shape, ', the inliers number is ',predict_rgb_1.shape)
                # loss1 = loss_fn(predict_rgb_0, rgbs)
                # loss2 = loss_fn(predict_rgb_1, rgbs)

                #TODO: 100000 should be adapted
                scalar_max = 100000
                scalar = scalar_max
                #penalty 100 will make mask be smaller, 20 will be better, try 10
                penalty = 1
                if outliers_1 != []:
                    loss_mask_0 = torch.sum(torch.abs(outliers_1)) * penalty + torch.sum(torch.abs(1-inliers_1))
                    loss_mask_1 = torch.sum(torch.abs(outliers_2)) * penalty + torch.sum(torch.abs(1-inliers_2))
                else:
                    loss_mask_0 = torch.sum(torch.abs(1-inliers_1))
                    loss_mask_1 = torch.sum(torch.abs(1-inliers_2))

                # while loss_mask_1 / scalar < rays.shape[0]/(scalar_max * 2) and loss_mask_1 > 1:
                #     scalar /= 2
                #     if scalar <= 1:
                #         scalar = 1.0
                #         break

                # num_ray_mask = torch.sum(ray_mask.view(1,-1)).item()
                # print('This batch has %d rays in bbox' % num_ray_mask)

                if loss_mask_0 > rays.shape[0] * 0.0005 and remove_outliers:
                    loss_mask_0 = loss_mask_0 / scalar
                else:
                    loss_mask_0 = torch.Tensor([0]).cuda()

                if loss_mask_1 > rays.shape[0] * 0.0005 and remove_outliers:
                    loss_mask_1 = loss_mask_1 / scalar
                else:
                    loss_mask_1 = torch.Tensor([0]).cuda()
            else:
                loss_mask_0 = torch.Tensor([0]).cuda()
                loss_mask_1 = torch.Tensor([0]).cuda()


            if epoch<coarse_stage:
                loss = loss1 + loss_mask_0
            else:
                loss = loss1 + loss2 + loss_mask_0 + loss_mask_1
            
            # loss = loss1 + loss2

            loss.backward()

            optimizer.step()
            scheduler.step()

            psnr_0 = psnr(predict_rgb_0, rgbs)
            psnr_ = psnr(predict_rgb_1, rgbs)
            psnr_monitor.append(psnr_.cpu().detach().numpy())


            if batch_idx % 50 ==0:
                swriter.add_scalar('Loss/train_loss',loss.item(), global_step)
                swriter.add_scalar('TrainPsnr', psnr_, global_step)
                swriter.add_scalar('Loss/mask_loss',(loss_mask_0+loss_mask_1).item(), global_step)
                swriter.add_scalar('Loss/rgb_loss',(loss1+loss2).item(), global_step)

            if batch_idx % log_period == 0:
                for param_group in optimizer.param_groups:
                    lr = param_group['lr']
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3e}  Psnr coarse: {:.2f} Psnr fine: {:.2f} Lr: {:.2e} Speed: {:.1f}[rays/s]"
                            .format(epoch, batch_idx, len(train_loader), loss.item(), psnr_0, psnr_ ,lr,
                                    log_period * float(cfg.SOLVER.BUNCH) / (time.time() - iters_start)))
            #validation
            if global_step % 1000 == 0:
                val_vis(val_loader, model, loss_fn, swriter, logger, epoch)

            #model saving
            if global_step % checkpoint_period == 0:
                ModelCheckpoint(model, optimizer, scheduler, output_dir, epoch)
                ModelCheckpoint(model, optimizer, scheduler, output_dir, epoch, global_step)
                    
        #EPOCH COMPLETED
        ModelCheckpoint(model, optimizer, scheduler, output_dir, epoch)

        val_vis(val_loader, model ,loss_fn, swriter, logger, epoch)

        logger.info('Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[rays/s]'
                    .format(epoch, time.time() - epoch_start,
                            len(train_loader) * float(cfg.SOLVER.BUNCH) / (time.time() - epoch_start)))

        psnr_monitor = np.mean(psnr_monitor)
        
        if psnr_monitor > psnr_thres:
            logger.info("The Mean Psnr of Epoch: {:.3f}, greater than threshold: {:.3f}, Training Stopped".format(psnr_monitor, psnr_thres))
            break
        else:
            logger.info("The Mean Psnr of Epoch: {:.3f}, less than threshold: {:.3f}, Continue to Training".format(psnr_monitor, psnr_thres))

def val_vis(val_loader,model ,loss_fn, swriter, logger, epoch):
    
   
    avg_loss = evaluator(val_loader, model, loss_fn, swriter,epoch)
    logger.info("Validation Results - Epoch: {} Avg Loss: {:.3f}"
                .format(epoch,  avg_loss)
                )
    swriter.add_scalar('Loss/val_loss',avg_loss, epoch)

def ModelCheckpoint(model, optimizer, scheduler, output_dir, epoch, global_step = 0):
    # model,optimizer,scheduler saving 
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if global_step == 0:
        torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict()}, 
                    os.path.join(output_dir,'layered_rfnr_checkpoint_%d.pt' % epoch))
    else:
        torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict()}, 
                    os.path.join(output_dir,'layered_rfnr_checkpoint_%d_%d.pt' % (epoch,global_step)))
    # torch.save(model.state_dict(), os.path.join(output_dir, 'spacenet_epoch_%d.pth'%epoch))
    # torch.save(optimizer.state_dict(), os.path.join(output_dir, 'optimizer_epoch_%d.pth'%epoch))
    # torch.save(scheduler.state_dict(), os.path.join(output_dir, 'scheduler_epoch_%d.pth'%epoch))


def do_evaluate( model,val_dataset):
    mae_list = []
    psnr_list = []
    ssim_list = []

    
    model.eval()
    with torch.no_grad():
        for i in range(2):
            for j in range(50):
                rays, rgbs, labels, image, label, mask, bbox, near_far = val_dataset.get_fixed_image(i,j+1)

                rays = rays.cuda()
                rgbs = rgbs.cuda()
                bbox = bbox.cuda()
                labels = labels.cuda()
                color_gt = image.cuda()
                mask = mask.cuda()
                near_far = near_far.cuda()

                # uv_list = (mask).squeeze().nonzero()
                # u_list = uv_list[:,0]
                # v_list = uv_list[:,1]


                # TODO: Use mask to gain less query of space
                stage2, _, _, _, _ = layered_batchify_ray(model, rays, labels, bbox, near_far=near_far)

                color_1 = stage2[0]
                depth_1 = stage2[1]
                acc_map_1 = stage2[2]
                #print(color_1.shape)
                #print(depth_1.shape)
                #print(acc_map_1.shape)


                color_img = color_1.reshape( (color_gt.size(1), color_gt.size(2), 3) ).permute(2,0,1)
                
                mae = metrics.mae(color_img,color_gt)
                psnr = metrics.psnr(color_img,color_gt)
                ssim = metrics.ssim(color_img,color_gt)
                print(color_img.shape)
                print(color_gt.shape)

                #imageio.imwrite("/new_disk/zhangjk/NeuralVolumeRender-dynamic/evaluation/walking/"+str(j+1)+".png", color_img.transpose(0,2).transpose(0,1).cpu())
                

                print("mae:",mae)
                print("psnr:",psnr)
                print("ssim:",ssim)
                mae_list.append(mae)
                psnr_list.append(psnr)
                ssim_list.append(ssim)
        mae_list = np.array(mae_list)
        psnr_list = np.array(psnr_list)
        ssim_list = np.array(ssim_list)
        np.savetxt('/new_disk/zhangjk/NeuralVolumeRender-dynamic/evaluation/complete/mae.out',mae_list) 
        np.savetxt('/new_disk/zhangjk/NeuralVolumeRender-dynamic/evaluation/complete/psnr.out',psnr_list) 
        np.savetxt('/new_disk/zhangjk/NeuralVolumeRender-dynamic/evaluation/complete/ssim.out',ssim_list) 
        avg_mae = np.mean(np.array(mae_list)) 
        avg_psnr = np.mean(np.array(psnr_list))
        avg_ssim = np.mean(np.array(ssim_list))
        print("avg_mae:",avg_mae)
        print("avg_psnr:",avg_psnr)
        print("avg_ssim:",avg_ssim)
        #print(color_1.shape)
        #print(color_gt.shape)
        #print(metrics.psnr(color_img, color_gt))
        




        

