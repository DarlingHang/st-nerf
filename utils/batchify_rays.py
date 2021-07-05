import torch


def batchify_ray(model, rays, bboxes, chuncks = 1024*7, near_far=None, near_far_points = [], density_threshold=0,bkgd_density_threshold=0):
    N = rays.size(0)
    if N <chuncks:
        return model(rays, bboxes,near_far =near_far, near_far_points = near_far_points)

    else:
        rays = rays.split(chuncks, dim=0)
        if bboxes is not None:
            bboxes = bboxes.split(chuncks, dim=0)
        else:
            bboxes = [None]*len(rays)
        if near_far is not None:
            near_far = near_far.split(chuncks, dim=0)
        else:
            near_far = [None]*len(rays)

        colors = [[],[]]
        depths = [[],[]]
        acc_maps = [[],[]]

        ray_masks = []

        for i in range(len(rays)):
            stage2, stage1, ray_mask = model( rays[i], bboxes[i], near_far = near_far[i], near_far_points = near_far_points, density_threshold=density_threshold,bkgd_density_threshold=bkgd_density_threshold)
            colors[0].append(stage1[0])
            depths[0].append(stage1[1])
            acc_maps[0].append(stage1[2])

            colors[1].append(stage2[0])
            depths[1].append(stage2[1])
            acc_maps[1].append(stage2[2])
            if ray_mask is not None:
                ray_masks.append(ray_mask)

        colors[0] = torch.cat(colors[0], dim=0)
        depths[0] = torch.cat(depths[0], dim=0)
        acc_maps[0] = torch.cat(acc_maps[0], dim=0)

        colors[1] = torch.cat(colors[1], dim=0)
        depths[1] = torch.cat(depths[1], dim=0)
        acc_maps[1] = torch.cat(acc_maps[1], dim=0)
        if len(ray_masks)>0:
            ray_masks = torch.cat(ray_masks, dim=0)

        return (colors[1], depths[1], acc_maps[1]), (colors[0], depths[0], acc_maps[0]), ray_masks


def layered_batchify_ray(model, rays, labels, bboxes, chuncks = 512*7, near_far=None, near_far_points = [], density_threshold=0,bkgd_density_threshold=0):
    N = rays.size(0)
    if N <chuncks:
        return model(rays, labels, bboxes,near_far =near_far, near_far_points = near_far_points)

    else:
        rays = rays.split(chuncks, dim=0)
        if bboxes is not None:
            bboxes = bboxes.split(chuncks, dim=0)
        else:
            bboxes = [None]*len(rays)
        if near_far is not None:
            near_far = near_far.split(chuncks, dim=0)
        else:
            near_far = [None]*len(rays)

        if labels is not None:
            labels = labels.split(chuncks, dim=0)
        else:
            labels = [None]*len(rays)

        colors = [[],[]]
        depths = [[],[]]
        acc_maps = [[],[]]

        ray_masks = []

        for i in range(len(rays)):
            stage2, stage1, stage2_layer, stage1_layer, ray_mask = model( rays[i], labels[i], bboxes[i], near_far = near_far[i], near_far_points = near_far_points, density_threshold=density_threshold,bkgd_density_threshold=bkgd_density_threshold)
            
            if i == 0:
                for _ in range(len(stage2_layer)):
                    colors.append([])
                    colors.append([])
                    depths.append([])
                    depths.append([])
                    acc_maps.append([])
                    acc_maps.append([])
                    ray_masks.append([])
            
            colors[0].append(stage1[0])
            depths[0].append(stage1[1])
            acc_maps[0].append(stage1[2])

            colors[1].append(stage2[0])
            depths[1].append(stage2[1])
            acc_maps[1].append(stage2[2])

            for i in range(len(stage2_layer)):
                colors[2+i*2].append(stage1_layer[i][0])
                depths[2+i*2].append(stage1_layer[i][1])
                acc_maps[2+i*2].append(stage1_layer[i][2])

                colors[3+i*2].append(stage2_layer[i][0])
                depths[3+i*2].append(stage2_layer[i][1])
                acc_maps[3+i*2].append(stage2_layer[i][2])
            
            if ray_mask is not None:
                for i in range(len(stage2_layer)):
                    ray_masks[i].append(ray_mask[i])

        colors[0] = torch.cat(colors[0], dim=0)
        depths[0] = torch.cat(depths[0], dim=0)
        acc_maps[0] = torch.cat(acc_maps[0], dim=0)

        colors[1] = torch.cat(colors[1], dim=0)
        depths[1] = torch.cat(depths[1], dim=0)
        acc_maps[1] = torch.cat(acc_maps[1], dim=0)

        for i in range(len(stage2_layer)):
            colors[2+i*2] = torch.cat(colors[2+i*2], dim=0)
            depths[2+i*2] = torch.cat(depths[2+i*2], dim=0)
            acc_maps[2+i*2] = torch.cat(acc_maps[2+i*2], dim=0)

            colors[3+i*2] = torch.cat(colors[3+i*2], dim=0)
            depths[3+i*2] = torch.cat(depths[3+i*2], dim=0)
            acc_maps[3+i*2] = torch.cat(acc_maps[3+i*2], dim=0)

        if len(ray_masks)>0:
            for i in range(len(stage2_layer)):
                ray_masks[i] = torch.cat(ray_masks[i], dim=0)

        stage1_layer_final = []
        stage2_layer_final = []

        for i in range(len(stage2_layer)):
            stage1_layer_final.append((colors[2+i*2], depths[2+i*2], acc_maps[2+i*2]))
            stage2_layer_final.append((colors[3+i*2], depths[3+i*2], acc_maps[3+i*2]))
        return (colors[1], depths[1], acc_maps[1]), (colors[0], depths[0], acc_maps[0]),\
        stage2_layer_final, stage1_layer_final, ray_masks



def layered_batchify_ray_big(layer_big,scale,model, rays, labels, bboxes, chuncks = 512*7, near_far=None, near_far_points = [], density_threshold=0):
    N = rays.size(0)
    if N <chuncks:
        return model(rays, labels, bboxes,near_far =near_far, near_far_points = near_far_points)

    else:
        rays = rays.split(chuncks, dim=0)
        if bboxes is not None:
            bboxes = bboxes.split(chuncks, dim=0)
        else:
            bboxes = [None]*len(rays)
        if near_far is not None:
            near_far = near_far.split(chuncks, dim=0)
        else:
            near_far = [None]*len(rays)

        if labels is not None:
            labels = labels.split(chuncks, dim=0)
        else:
            labels = [None]*len(rays)

        colors = [[],[]]
        depths = [[],[]]
        acc_maps = [[],[]]

        ray_masks = []

        for i in range(len(rays)):
            stage2, stage1, stage2_layer, stage1_layer, ray_mask = model( rays[i], labels[i], bboxes[i], near_far = near_far[i], near_far_points = near_far_points, density_threshold=density_threshold,bkgd_density_threshold=bkgd_density_threshold,layer_big=layer_big,scale=scale)
            
            if i == 0:
                for _ in range(len(stage2_layer)):
                    colors.append([])
                    colors.append([])
                    depths.append([])
                    depths.append([])
                    acc_maps.append([])
                    acc_maps.append([])
                    ray_masks.append([])
            
            colors[0].append(stage1[0])
            depths[0].append(stage1[1])
            acc_maps[0].append(stage1[2])

            colors[1].append(stage2[0])
            depths[1].append(stage2[1])
            acc_maps[1].append(stage2[2])

            for i in range(len(stage2_layer)):
                colors[2+i*2].append(stage1_layer[i][0])
                depths[2+i*2].append(stage1_layer[i][1])
                acc_maps[2+i*2].append(stage1_layer[i][2])

                colors[3+i*2].append(stage2_layer[i][0])
                depths[3+i*2].append(stage2_layer[i][1])
                acc_maps[3+i*2].append(stage2_layer[i][2])
            
            if ray_mask is not None:
                for i in range(len(stage2_layer)):
                    ray_masks[i].append(ray_mask[i])

        colors[0] = torch.cat(colors[0], dim=0)
        depths[0] = torch.cat(depths[0], dim=0)
        acc_maps[0] = torch.cat(acc_maps[0], dim=0)

        colors[1] = torch.cat(colors[1], dim=0)
        depths[1] = torch.cat(depths[1], dim=0)
        acc_maps[1] = torch.cat(acc_maps[1], dim=0)

        for i in range(len(stage2_layer)):
            colors[2+i*2] = torch.cat(colors[2+i*2], dim=0)
            depths[2+i*2] = torch.cat(depths[2+i*2], dim=0)
            acc_maps[2+i*2] = torch.cat(acc_maps[2+i*2], dim=0)

            colors[3+i*2] = torch.cat(colors[3+i*2], dim=0)
            depths[3+i*2] = torch.cat(depths[3+i*2], dim=0)
            acc_maps[3+i*2] = torch.cat(acc_maps[3+i*2], dim=0)

        if len(ray_masks)>0:
            for i in range(len(stage2_layer)):
                ray_masks[i] = torch.cat(ray_masks[i], dim=0)

        stage1_layer_final = []
        stage2_layer_final = []

        for i in range(len(stage2_layer)):
            stage1_layer_final.append((colors[2+i*2], depths[2+i*2], acc_maps[2+i*2]))
            stage2_layer_final.append((colors[3+i*2], depths[3+i*2], acc_maps[3+i*2]))
        return (colors[1], depths[1], acc_maps[1]), (colors[0], depths[0], acc_maps[0]),\
        stage2_layer_final, stage1_layer_final, ray_masks