import time
import numpy as np
import saverloader
import skimage.morphology
from fire import Fire

import utils.misc
import utils.improc
from utils.basic import print_, print_stats

from pseudokittidataset import PseudoKittiDataset

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from tensorboardX import SummaryWriter
import torch.nn.functional as F

import nets.seg2dnet

import random
device = 'cuda'
random.seed(125)
np.random.seed(125)

def requires_grad(parameters, flag=True):
    for p in parameters:
        p.requires_grad = flag

def fetch_optimizer(lr, wdecay, epsilon, num_steps, params):
    """ Create the optimizer and learning rate scheduler """
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wdecay, eps=epsilon)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, num_steps+100,
        pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler

def compute_loss(pred, pos, neg, balanced=True):
    pos = pos.clone().reshape(-1)
    neg = neg.clone().reshape(-1)
    pred = pred.reshape(-1)

    label = pos*2.0 - 1.0
    a = -label * pred
    b = F.relu(a)
    loss = b + torch.log(torch.exp(-b)+torch.exp(a-b))

    mask_ = (pos+neg>0.0).float()

    if balanced:
        pos_loss = utils.basic.reduce_masked_mean(loss, pos)
        neg_loss = utils.basic.reduce_masked_mean(loss, neg)

        balanced_loss = pos_loss + neg_loss

        return balanced_loss
    else:
        loss_pos = loss[pos > 0]
        loss_neg = loss[neg > 0]
        loss = torch.cat([loss_pos, loss_neg], dim=0).mean()
        return loss
    
def run_model(B, model, d, sw, use_augs=True, debug=True):
    total_loss = torch.tensor(0.0, requires_grad=True).to(device)
    metrics = {}
    
    rgb_cam = d['rgb_cam'].cuda() # B, C, H, W; already preprocessed
    # xyz_cam = d['xyz_cam'].cuda() # B, N, 3
    pix_T_cam = d['pix_T_cam'].cuda() # B, 4, 4
    seg_xyz_cam_list = [xyz.cuda() for xyz in d['seg_xyz_cam_list']]

    B1, C, H, W = rgb_cam.shape
    # B1, V, D = xyz_cam.shape
    assert(B1==1)

    if sw is not None and sw.save_this:
        sw.summ_rgb('0_inputs/rgb_orig', rgb_cam)

    N = len(seg_xyz_cam_list)
    pos_mask = torch.zeros_like(rgb_cam[:,0:1])
    for n1 in range(N):
        seg_xyz_cam = seg_xyz_cam_list[n1]
        seg_xy_pix = utils.geom.apply_pix_T_cam(pix_T_cam, seg_xyz_cam)
        mask = utils.improc.xy2mask(seg_xy_pix, H, W, norm=False)
        mask = mask[0,0].cpu().numpy()
        mask = skimage.morphology.convex_hull.convex_hull_image(mask)
        pos_mask += torch.from_numpy(mask.reshape(1, 1, H, W)).float().cuda()
    pos_mask = pos_mask.clamp(0,1)

    # create neg by dilating the pos
    pos_wide = utils.improc.dilate2d(pos_mask, times=48)
    pos_med = utils.improc.dilate2d(pos_mask, times=24)
    neg_mask = (pos_wide - pos_med).clamp(0,1)
        
    if sw is not None and sw.save_this:
        seg_g = torch.cat([pos_mask, neg_mask], dim=1)
        seg_vis = sw.summ_soft_seg_thr('', seg_g, colormap='tab10', only_return=True)
        grey = torch.mean(rgb_cam, dim=1, keepdim=True).repeat(1, 3, 1, 1)
        seg = utils.improc.preprocess_color(seg_vis).cuda()
        sw.summ_rgb('0_inputs/seg_on_rgb_orig', (grey + seg)/2.0)

    if use_augs:
        # create a batch, with augs
        rgb_batch = []
        pos_batch = []
        neg_batch = []
        for b in range(B):
            rgb_aug = rgb_cam.clone()
            pos_aug = pos_mask.clone()
            neg_aug = neg_mask.clone()

            rgb_aug = utils.misc.apply_color_augs(rgb_aug, amount=0.8)
            rgb_aug = utils.misc.apply_random_blur_and_noise(rgb_aug)

            if random.random() > 0.5:
                rgb_aug = torch.flip(rgb_aug, [3])
                pos_aug = torch.flip(pos_aug, [3])
                neg_aug = torch.flip(neg_aug, [3])

            done = False
            tries = 0
            while not done and tries < 10:
                # min_size = 64
                # ymin = torch.randint(low=int(-H/2), high=int(H-H/2), size=[])
                # xmin = torch.randint(low=int(-W/2), high=int(W-W/2), size=[])
                # ymax = torch.randint(low=ymin.clamp(min=0)+min_size, high=int(H+H/2), size=[])
                # size_factor = (ymax-ymin)/H
                # xmax = (xmin + size_factor*W).long()

                zoomout_factor = torch.from_numpy(np.random.uniform(0.5, 2.0, size=1).astype(np.float32)).cuda()
                xc = torch.from_numpy(np.random.uniform(int(W/4), int(W-W/4), size=1)).cuda().long()
                yc = torch.from_numpy(np.random.uniform(int(H/4), int(H-H/4), size=1)).cuda().long()
                ymin = yc - (zoomout_factor*H)/2.0 
                ymax = yc + (zoomout_factor*H)/2.0
                xmin = xc - (zoomout_factor*W)/2.0 
                xmax = xc + (zoomout_factor*W)/2.0
                box2d_crop = torch.stack([ymin, xmin, ymax, xmax]).reshape(1, 4).float().cuda()
                box2d_crop = utils.geom.normalize_box2d(box2d_crop, H, W)

                rgb_aug_ = utils.geom.crop_and_resize(rgb_aug, box2d_crop, H, W)
                pos_aug_ = utils.geom.crop_and_resize(pos_aug, box2d_crop, H, W)
                neg_aug_ = utils.geom.crop_and_resize(neg_aug, box2d_crop, H, W)
                if torch.sum(pos_aug_) > 128 and torch.sum(neg_aug_) > 128:
                    rgb_batch.append(rgb_aug_)
                    pos_batch.append(pos_aug_)
                    neg_batch.append(neg_aug_)
                    done = True
                tries += 1
            if not done:
                rgb_batch.append(rgb_aug)
                pos_batch.append(pos_aug)
                neg_batch.append(neg_aug)
        rgb = torch.cat(rgb_batch, dim=0)
        pos = torch.cat(pos_batch, dim=0).round()
        neg = torch.cat(neg_batch, dim=0).round()
    else:
        assert(B==1) # B>1 doesn't make sense if augs are disabled
        rgb = rgb_cam.clone()
        pos = pos_mask.clone()
        neg = neg_mask.clone()

    if sw is not None and sw.save_this:
        
        sw.summ_rgb('1_outputs/rgb', rgb)
        seg_g = torch.cat([pos, neg], dim=1)
        seg_vis = sw.summ_soft_seg_thr('', seg_g, colormap='tab10', only_return=True)
        grey = torch.mean(rgb, dim=1, keepdim=True).repeat(1, 3, 1, 1)
        seg = utils.improc.preprocess_color(seg_vis).cuda()
        sw.summ_rgb('1_outputs/seg_g_on_rgb', (grey + seg)/2.0)

        if debug:
            # vis each element of the batch individually, just to make sure
            for b in range(B):
                sw.summ_rgb('00_debug/rgb_%d' % b, rgb[b:b+1])
                sw.summ_soft_seg_thr('00_debug/seg_%d' % b, seg_g[b:b+1], colormap='tab10')

    seg_e = model(rgb)
    # pos_g = F.interpolate(pos, scale_factor=0.25, mode='nearest').round()
    # neg_g = F.interpolate(neg, scale_factor=0.25, mode='nearest').round()

    seg_loss = compute_loss(seg_e, pos, neg, balanced=True)
    total_loss += seg_loss
    
    if sw is not None and sw.save_this:
        sw.summ_oned('1_outputs/seg_e', torch.sigmoid(seg_e))
        # seg_e = F.interpolate(torch.sigmoid(seg_e), scale_factor=4)
        seg_e = torch.sigmoid(seg_e)
        pos_e = (seg_e > 0.8).float()
        neg_e = (seg_e < 0.2).float()

        seg_e = torch.cat([pos_e, neg_e], dim=1)
        seg_vis = sw.summ_soft_seg_thr('', seg_e, colormap='tab10', only_return=True)
        grey = torch.mean(rgb, dim=1, keepdim=True).repeat(1, 3, 1, 1)
        seg = utils.improc.preprocess_color(seg_vis).cuda()
        sw.summ_rgb('1_outputs/seg_e_on_rgb', (grey + seg)/2.0)
        
    return total_loss, metrics
    
    
def main(
        input_name,
        exp_name='debug',
        max_iters=20000,
        log_freq=500,
        save_freq=1000,
        shuffle=True,
        use_augs=True,
        B=4,
        lr=1e-3,
        init_dir='',
        load_step=False,
        load_optimizer=False,
):
    # this file implements the 2d part of the M step

    # autogen a name, based on hyps
    model_name = "%02d" % (B)
    lrn = "%.1e" % lr # e.g., 5.0e-04
    lrn = lrn[0] + lrn[3:5] + lrn[-1] # e.g., 5e-4
    model_name += "_%s" % lrn
    model_name += "_kitti2d"
    model_name += "_%s" % input_name
    model_name += "_%s" % exp_name
    
    import datetime
    model_date = datetime.datetime.now().strftime('%H:%M:%S')
    model_name = model_name + '_' + model_date
    print('model_name', model_name)
    
    ckpt_dir = 'checkpoints/%s' % model_name
    log_dir = 'logs_tcr_kitti_train_2d'
    writer_t = SummaryWriter(log_dir + '/' + model_name + '/t', max_queue=10, flush_secs=60)

    train_dataset = PseudoKittiDataset(shuffle=shuffle, input_name=input_name, load_seg=True)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=shuffle,
        num_workers=4,
        drop_last=True)
    train_iterloader = iter(train_dataloader)
    
    global_step = 0

    model = nets.seg2dnet.Seg2dNet(num_classes=1, imagenet_init=True, shallow=False).to(device).train()
    
    parameters = list(model.parameters())
    optimizer, scheduler = fetch_optimizer(lr, 0.0001, 1e-8, max_iters, parameters)

    if init_dir:
        if load_step and load_optimizer:
            global_step = saverloader.load(init_dir, model, optimizer, scheduler)
        elif load_step:
            global_step = saverloader.load(init_dir, model)
        else:
            _ = saverloader.load(init_dir, model)
            global_step = 0
    requires_grad(parameters, True)
    model.train()

    n_pool = 100
    loss_pool_t = utils.misc.SimplePool(n_pool, version='np')
    
    while global_step < max_iters:
        optimizer.zero_grad()
        # torch.cuda.empty_cache()
        
        read_start_time = time.time()
        
        global_step += 1
        total_loss = torch.tensor(0.0, requires_grad=True).to(device)

        sw_t = utils.improc.Summ_writer(
            writer=writer_t,
            global_step=global_step,
            log_freq=log_freq,
            fps=12,
            scalar_freq=int(log_freq/2),
            just_gif=True)

        try:
            sample = next(train_iterloader)
        except StopIteration:
            train_iterloader = iter(train_dataloader)
            sample = next(train_iterloader)
                
        read_time = time.time()-read_start_time
        iter_start_time = time.time()

        total_loss, metrics = run_model(B, model, sample, sw_t, use_augs=use_augs)

        sw_t.summ_scalar('total_loss', total_loss)
        loss_pool_t.update([total_loss.detach().cpu().numpy()])
        sw_t.summ_scalar('pooled/total_loss', loss_pool_t.mean())

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        if np.mod(global_step, save_freq)==0:
            saverloader.save(ckpt_dir, optimizer, model, global_step, keep_latest=1, scheduler=scheduler)

        current_lr = optimizer.param_groups[0]['lr']
        sw_t.summ_scalar('_/current_lr', current_lr)
        
        iter_time = time.time()-iter_start_time
        print('%s; step %06d/%d; rtime %.2f; itime %.2f; loss = %.5f' % (
            model_name, global_step, max_iters, read_time, iter_time,
            total_loss.item()))
        
    writer_t.close()
            

if __name__ == '__main__':
    Fire(main)
