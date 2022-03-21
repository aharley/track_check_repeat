import time
import numpy as np
import saverloader
import skimage.morphology
from fire import Fire

import utils.misc
import utils.improc
import utils.vox
import utils.geom
import utils.eval
from utils.basic import print_, print_stats

from pseudokittidataset import PseudoKittiDataset
from simplekittidataset import SimpleKittiDataset

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from tensorboardX import SummaryWriter
import torch.nn.functional as F

import nets.centernet2d

import random
device = 'cuda'
random.seed(125)
np.random.seed(125)

iou_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

scene_centroid_x = 0.0
scene_centroid_y = 1.0
scene_centroid_z = 0.0

scene_centroid = np.array([scene_centroid_x,
                           scene_centroid_y,
                           scene_centroid_z]).reshape([1, 3])
scene_centroid = torch.from_numpy(scene_centroid).float().cuda()

XMIN, XMAX = -16, 16
ZMIN, ZMAX = 2, 34
YMIN, YMAX = -1, 3

bounds = (XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX)

Z, Y, X = 256, 16, 256
Z2, Y2, X2 = Z//2, Y//2, X//2
Z4, Y4, X4 = Z//4, Y//4, X//4
Z8, Y8, X8 = Z//8, Y//8, X//8

def requires_grad(parameters, flag=True):
    for p in parameters:
        p.requires_grad = flag

def fetch_optimizer(lr, wdecay, epsilon, num_steps, params):
    """ Create the optimizer and learning rate scheduler """
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wdecay, eps=epsilon)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, num_steps+100,
        pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler

def balanced_ce_loss(pred, gt, valid):
    # pred is B x 1 x Y x X
    pos = (gt > 0.95).float()
    neg = (gt < 0.05).float()

    label = pos*2.0 - 1.0
    a = -label * pred
    b = F.relu(a)
    loss = b + torch.log(torch.exp(-b)+torch.exp(a-b))
    
    pos_loss = utils.basic.reduce_masked_mean(loss, pos*valid)
    neg_loss = utils.basic.reduce_masked_mean(loss, neg*valid)
    balanced_loss = pos_loss + neg_loss
    return balanced_loss

def run_model(model, d, sw, use_augs=False, stride=8):
    total_loss = torch.tensor(0.0, requires_grad=True).to(device)
    metrics = {}
    metrics['maps_bev'] = None
    
    rgb_cam = d['rgb_cam'].float().cuda() # B, C, H, W
    xyz_cam = d['xyz_cam'].float().cuda() # B, N, 3
    pix_T_cam = d['pix_T_cam'].float().cuda() # B, 4, 4

    lrtlist_cam = d['lrtlist_cam'].float().cuda() # B, N, 9
    scorelist = d['scorelist'].float().cuda() # B, N
    tidlist = d['tidlist'].long().cuda() # B, N
    # rylist = d['rylist'].float().cuda() # B, N

    B, C, H, W = rgb_cam.shape
    B, V, D = xyz_cam.shape
    
    rgb_cam = utils.improc.preprocess_color(rgb_cam)
    
    vox_util = utils.vox.Vox_util(
        Z, Y, X,
        scene_centroid=scene_centroid,
        bounds=bounds,
        assert_cube=False)

    # compute freespace samples along the rays
    xyz_cam_free = vox_util.convert_xyz_to_visibility_samples(xyz_cam, samps=8, dist_eps=0.05, rand=True)
    xyz_cam_bak = xyz_cam.clone()

    K = tidlist.shape[1]

    scorelist = utils.misc.rescore_lrtlist_with_inbound(lrtlist_cam, scorelist, Z, Y, X, vox_util)
    if torch.sum(scorelist) == 0:
        return total_loss, None

    if use_augs:

        if random.random() > 0.5:

            # image-centric masking occlusions
            mask_size = np.random.randint(10, 100)
            xyz_cam, _ = utils.geom.random_occlusion(
                xyz_cam,
                lrtlist_cam,
                scorelist,
                pix_T_cam, H, W,
                mask_size=mask_size,
                occ_prob=0.8,
                occlude_bkg_too=True)
            V = xyz_cam.shape[1]
            

        for b in range(B):
            # random scaling
            aug_T_cam = utils.geom.get_random_scale(1, low=0.7, high=1.3) # B x 4 x 4
            xyz_cam[b:b+1] = utils.geom.apply_4x4(aug_T_cam, xyz_cam[b:b+1])
            xyz_cam_free[b:b+1] = utils.geom.apply_4x4(aug_T_cam, xyz_cam_free[b:b+1])
            xyz_cam_bak[b:b+1] = utils.geom.apply_4x4(aug_T_cam, xyz_cam_bak[b:b+1])
            lrtlist_cam[b:b+1] = utils.geom.apply_scaling_to_lrtlist(aug_T_cam, lrtlist_cam[b:b+1])
        
        for b in range(B):

            # put the objects near zero, so that random rotation doesn't shoot them out of bounds
            _, rtlist = utils.geom.split_lrtlist(lrtlist_cam[b:b+1])
            rlist, tlist = utils.geom.split_rtlist(rtlist)
            # tlist is 1,N,3
            # scorelist is B,N
            offset = -utils.basic.reduce_masked_mean(tlist, scorelist[b:b+1].reshape(1, -1, 1).repeat(1, 1, 3), dim=1)
            off_T_cam = utils.geom.merge_rt(utils.geom.eye_3x3(1), offset)

            xyz_cam[b:b+1] = utils.geom.apply_4x4(off_T_cam, xyz_cam[b:b+1])
            xyz_cam_free[b:b+1] = utils.geom.apply_4x4(off_T_cam, xyz_cam_free[b:b+1])
            xyz_cam_bak[b:b+1] = utils.geom.apply_4x4(off_T_cam, xyz_cam_bak[b:b+1])
            lrtlist_cam[b:b+1] = utils.geom.apply_4x4_to_lrtlist(off_T_cam, lrtlist_cam[b:b+1])

            aug_T_cam = utils.geom.get_random_rt(1, rx_amount=0.0, ry_amount=30.0, rz_amount=0.0, t_amount=0.0, y_zero=True)
            xyz_cam[b:b+1] = utils.geom.apply_4x4(aug_T_cam, xyz_cam[b:b+1])
            xyz_cam_free[b:b+1] = utils.geom.apply_4x4(aug_T_cam, xyz_cam_free[b:b+1])
            xyz_cam_bak[b:b+1] = utils.geom.apply_4x4(aug_T_cam, xyz_cam_bak[b:b+1])
            lrtlist_cam[b:b+1] = utils.geom.apply_4x4_to_lrtlist(aug_T_cam, lrtlist_cam[b:b+1])

            # put the objects back
            xyz_cam[b:b+1] = utils.geom.apply_4x4(off_T_cam.inverse(), xyz_cam[b:b+1])
            xyz_cam_free[b:b+1] = utils.geom.apply_4x4(off_T_cam.inverse(), xyz_cam_free[b:b+1])
            xyz_cam_bak[b:b+1] = utils.geom.apply_4x4(off_T_cam.inverse(), xyz_cam_bak[b:b+1])
            lrtlist_cam[b:b+1] = utils.geom.apply_4x4_to_lrtlist(off_T_cam.inverse(), lrtlist_cam[b:b+1])

    occ_mem = vox_util.voxelize_xyz(xyz_cam, Z, Y, X) # B, 1, Z, Y, X
    occ_feat = occ_mem.squeeze(1).permute(0, 2, 1, 3) # B, Y, Z, X (y becomes feature channel)

    Z8, Y8, X8 = Z//stride, Y//stride, X//stride
    
    # now i want to create seg gt
    pos_mem8 = torch.zeros((B, 1, Z8, Y8, X8), dtype=torch.float32, device='cuda')
    for b in range(B):
        for k in range(K):
            score = scorelist[b,k]
            inbound = utils.geom.get_pts_inbound_lrt(xyz_cam_bak[b:b+1], lrtlist_cam[b:b+1, k]) # 1 x N
            inb_pts_cnt = torch.sum(inbound)
            if inb_pts_cnt > 0 and score > 0:
                occ = vox_util.voxelize_xyz(xyz_cam_bak[b:b+1,inbound[0]], Z8, Y8, X8) # B, 1, Z, Y, X
                pos_mem8[b,0] += occ[0,0]
    pos_mem8 = pos_mem8.clamp(0,1)
    occ_mem8 = vox_util.voxelize_xyz(xyz_cam_bak, Z8, Y8, X8)
    free_mem8 = vox_util.voxelize_xyz(xyz_cam_free, Z8, Y8, X8)
    free_mem8 = (free_mem8-occ_mem8).clamp(0,1)
    pos_wide_mem8 = utils.improc.dilate3d(pos_mem8, times=8)
    pos_med_mem8 = utils.improc.dilate3d(pos_mem8, times=4)
    neg_mem8 = (pos_wide_mem8 - pos_med_mem8).clamp(0,1) * occ_mem8
    # also use freespace within the _med region as neg
    neg_mem8 = (neg_mem8 + free_mem8*pos_med_mem8).clamp(0,1)
    # note that in bev vis this looks very tight,
    # but it is only tight in freespace voxels

    if sw is not None and sw.save_this:

        pos_bev = torch.max(pos_mem8, dim=3)[0]
        neg_bev = torch.max(neg_mem8, dim=3)[0]
        seg_bev = torch.cat([pos_bev, neg_bev], dim=1)
        seg_bev = F.interpolate(seg_bev, scale_factor=stride)
        # vis each element of the batch individually, just to make sure
        for b in range(B):
            seg_vis = sw.summ_soft_seg_thr('', seg_bev[b:b+1], colormap='tab10', only_return=True)
            occ_vis = sw.summ_occ('', occ_mem[b:b+1], only_return=True)
            seg_vis = utils.improc.preprocess_color(seg_vis).cuda()
            occ_vis = utils.improc.preprocess_color(occ_vis).cuda()
            sw.summ_rgb('00_debug/seg_on_occ_%d' % b, (occ_vis + seg_vis)/2.0)

            # sw.summ_lrtlist('00_debug/lrtlist_cam_%d' % b, rgb_cam[b:b+1], lrtlist_cam[b:b+1], scorelist[b:b+1], tidlist[b:b+1], pix_T_cam[b:b+1])
            sw.summ_lrtlist_bev('00_debug/lrtlist_bev_%d' % b, occ_mem[b:b+1], lrtlist_cam[b:b+1], scorelist[b:b+1], tidlist[b:b+1], vox_util)

    # get the centers and sizes in vox coords
    lrtlist_mem = vox_util.apply_mem_T_ref_to_lrtlist(
        lrtlist_cam, Z8, Y8, X8)
    clist_cam = utils.geom.get_clist_from_lrtlist(lrtlist_cam)
    lenlist, rtlist = utils.geom.split_lrtlist(lrtlist_cam)
    sizelist = (torch.max(lenlist, dim=2)[0]).clamp(min=2)
    sizelist = sizelist.clamp(min=4)
    mask = vox_util.xyz2circles(clist_cam, sizelist/2.0, Z8, Y8, X8, already_mem=False)
    mask = mask * scorelist.reshape(B, K, 1, 1, 1)
    center_g = torch.max(mask, dim=1, keepdim=True)[0]
    center_g = torch.max(center_g, dim=3)[0] # max along Y

    valid_mask = vox_util.xyz2circles(clist_cam, sizelist*2, Z8, Y8, X8, already_mem=False)
    valid_mask = valid_mask * scorelist.reshape(B, K, 1, 1, 1)
    valid_g = torch.max(valid_mask, dim=1, keepdim=True)[0]
    valid_g = torch.max(valid_g, dim=3)[0] # max along Y
    valid_g = (valid_g > 0.5).float()

    if sw is not None and sw.save_this:
        sw.summ_oned('center2d/center_g', center_g, norm=False)
        sw.summ_oned('center2d/valid_g', valid_g, norm=False)
    
    det_loss, lrtlist_cam_e, scorelist_e, seg_e = model(occ_feat, lrtlist_cam_g=lrtlist_cam, scorelist_g=scorelist, center_g=center_g, pos_mem=pos_mem8, neg_mem=neg_mem8, valid_g=valid_g, vox_util=vox_util, sw=sw, force_export_boxlist=sw.save_this)
    total_loss += det_loss

    if sw is not None and sw.save_this:
        metrics['maps_bev'] = iou_thresholds*0
        if lrtlist_cam_e.shape[1] > 0:
            lrtlist_e, lrtlist_g, scorelist_e, scorelist_g = utils.eval.drop_invalid_lrts(
                lrtlist_cam_e[0:1], lrtlist_cam[0:1], scorelist_e[0:1], scorelist[0:1])

            if torch.sum(scorelist_g) > 0 and torch.sum(scorelist_e) > 0:

                Ne = lrtlist_e.shape[1]
                Ng = lrtlist_g.shape[1]
                ious_3d = np.zeros((Ne, Ng), dtype=np.float32)
                ious_bev = np.zeros((Ne, Ng), dtype=np.float32)
                for i in list(range(Ne)):
                    for j in list(range(Ng)):
                        iou_3d, iou_bev = utils.eval.get_iou_from_corresponded_lrtlists(lrtlist_e[:, i:i+1], lrtlist_g[:, j:j+1])
                        ious_3d[i, j] = iou_3d[0, 0]
                        ious_bev[i, j] = iou_bev[0, 0]
                ious_bev = torch.max(torch.from_numpy(ious_bev).float().cuda(), dim=1)[0]
                ious_bev = ious_bev.unsqueeze(0)

                maps_3d, maps_bev = utils.eval.get_mAP_from_lrtlist(lrtlist_e, scorelist_e, lrtlist_g, iou_thresholds)
                metrics['maps_bev'] = maps_bev

                lrtlist_full = torch.cat([lrtlist_g, lrtlist_e], dim=1)
                scorelist_full = torch.cat([scorelist_g, ious_bev], dim=1)
                tidlist_full = torch.cat([5*torch.ones_like(scorelist_g), 2*torch.ones_like(scorelist_e)], dim=1).long()
                sw.summ_lrtlist_bev('outputs/lrtlist_bev', occ_mem, lrtlist_full, scorelist_full, tidlist_full, vox_util, frame_id=maps_bev[4], include_zeros=True)
                
        seg_e_sig = F.interpolate(torch.sigmoid(seg_e), scale_factor=stride)
        pos_e = (seg_e_sig > 0.8).float()
        neg_e = (seg_e_sig < 0.2).float()

        # show the occ estimates
        pos_e = pos_e * occ_mem
        neg_e = neg_e * occ_mem
        pos_bev = torch.max(pos_e, dim=3)[0]
        neg_bev = torch.max(neg_e, dim=3)[0]
        seg_bev = torch.cat([pos_bev, neg_bev], dim=1)
        seg_vis = sw.summ_soft_seg_thr('', seg_bev, colormap='tab10', only_return=True)
        occ_vis = sw.summ_occ('', occ_mem, only_return=True)
        seg_vis = utils.improc.preprocess_color(seg_vis).cuda()
        occ_vis = utils.improc.preprocess_color(occ_vis).cuda()
        sw.summ_rgb('outputs/seg_e_on_occ', (occ_vis + seg_vis)/2.0)
                    
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
        do_val=True,
        val_freq=10,
        init_dir='',
        load_step=False,
        load_optimizer=False,
):
    # this file implements the 3d part of the M step
    
    # autogen a name, based on hyps
    model_name = "%02d" % (B)
    lrn = "%.1e" % lr # e.g., 5.0e-04
    lrn = lrn[0] + lrn[3:5] + lrn[-1] # e.g., 5e-4
    model_name += "_%s" % lrn
    model_name += "_kitti3d"
    model_name += "_%s" % input_name
    model_name += "_%s" % exp_name

    import datetime
    model_date = datetime.datetime.now().strftime('%H:%M:%S')
    model_name = model_name + '_' + model_date
    print('model_name', model_name)
    
    ckpt_dir = 'checkpoints/%s' % model_name
    log_dir = 'logs_tcr_kitti_train_3d'
    writer_t = SummaryWriter(log_dir + '/' + model_name + '/t', max_queue=10, flush_secs=60)
    if do_val:
        writer_v = SummaryWriter(log_dir + '/' + model_name + '/v', max_queue=10, flush_secs=60)

    train_dataset = PseudoKittiDataset(shuffle=shuffle, input_name=input_name)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=shuffle,
        num_workers=4,
        drop_last=True)
    train_iterloader = iter(train_dataloader)

    if do_val:
        val_dataset = SimpleKittiDataset(S=1,kitti_data_seqlen=2,shuffle=shuffle,dset='v',return_valid=True)
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=B,
            shuffle=shuffle,
            num_workers=1,
            drop_last=True)
        val_iterloader = iter(val_dataloader)
    
    global_step = 0

    stride = 4
    model = nets.centernet2d.Centernet2d(Y=Y, K=20, show_thresh=0.5, stride=stride).cuda()
    
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
    map_bev_pools_t = [utils.misc.SimplePool(n_pool, version='np') for i in list(range(len(iou_thresholds)))]

    if do_val:
        loss_pool_v = utils.misc.SimplePool(n_pool, version='np')
        map_bev_pools_v = [utils.misc.SimplePool(n_pool, version='np') for i in list(range(len(iou_thresholds)))]
    
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
            
        total_loss, metrics = run_model(model, sample, sw_t, use_augs=use_augs, stride=stride)
        
        if metrics is not None:
            sw_t.summ_scalar('total_loss', total_loss)
            loss_pool_t.update([total_loss.detach().cpu().numpy()])
            sw_t.summ_scalar('pooled/total_loss', loss_pool_t.mean())

            if metrics['maps_bev'] is not None:
                for i,m in enumerate(metrics['maps_bev']):
                    map_bev_pools_t[i].update([m])
            for i in range(len(iou_thresholds)):
                sw_t.summ_scalar('map_bev/iou_%.1f' % iou_thresholds[i], map_bev_pools_t[i].mean())
            
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        # else we returned early
            
        if do_val and (global_step) % val_freq == 0:
            # torch.cuda.empty_cache()
            # let's do a val iter
            model.eval()
            sw_v = utils.improc.Summ_writer(
                writer=writer_v,
                global_step=global_step,
                log_freq=log_freq,
                fps=12,
                scalar_freq=int(log_freq/2),
                just_gif=True)
            try:
                sample = next(val_iterloader)
            except StopIteration:
                val_iterloader = iter(val_dataloader)
                sample = next(val_iterloader)
            with torch.no_grad():
                total_loss, metrics = run_model(model, sample, sw_v, use_augs=False, stride=stride)
            if metrics is not None:
                sw_v.summ_scalar('total_loss', total_loss)
                loss_pool_v.update([total_loss.detach().cpu().numpy()])
                sw_v.summ_scalar('pooled/total_loss', loss_pool_v.mean())
                if metrics['maps_bev'] is not None:
                    for i,m in enumerate(metrics['maps_bev']):
                        map_bev_pools_v[i].update([m])
                for i in range(len(iou_thresholds)):
                    sw_v.summ_scalar('map_bev/iou_%.1f' % iou_thresholds[i], map_bev_pools_v[i].mean())
            model.train()
            
        if np.mod(global_step, save_freq)==0:
            saverloader.save(ckpt_dir, optimizer, model, global_step, keep_latest=1, scheduler=scheduler)

        current_lr = optimizer.param_groups[0]['lr']
        sw_t.summ_scalar('_/current_lr', current_lr)
        
        iter_time = time.time()-iter_start_time
        if metrics is not None:
            print('%s; step %06d/%d; rtime %.2f; itime %.2f; loss = %.5f' % (
                model_name, global_step, max_iters, read_time, iter_time,
                total_loss.item()))
        else:
            print('%s; step %06d/%d; rtime %.2f; itime %.2f' % (
                model_name, global_step, max_iters, read_time, iter_time))
            
    writer_t.close()
    if do_val:
        writer_v.close()
            

if __name__ == '__main__':
    Fire(main)

