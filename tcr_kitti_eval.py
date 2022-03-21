import time
import os
import numpy as np
import imageio
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
import nets.seg2dnet

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

vis_dir = './tcr_vis'
utils.basic.mkdir(vis_dir)

def requires_grad(parameters, flag=True):
    for p in parameters:
        p.requires_grad = flag
    
def save_vis(rgb, name):
    rgb = rgb.cpu().numpy()[0].transpose(1,2,0) # H x W x 3
    vis_fn = os.path.join(vis_dir, '%s.png' % (name))
    imageio.imwrite(vis_fn, rgb)
    print('saved %s' % vis_fn)
    
def run_model(B, model_3d, d, sw, export_vis=False, step_name='temp', export_npzs=False):
    total_loss = torch.tensor(0.0, requires_grad=True).to(device)
    metrics = {}
    metrics['maps_bev'] = [i*0 for i in iou_thresholds] # mAP=0 by default
    metrics['maps_per'] = [i*0 for i in iou_thresholds] # mAP=0 by default

    # ------------------
    # set up the data
    # ------------------
    
    rgb_cam = d['rgb_cam'].float().cuda() # B, 3, H, W
    xyz_cam = d['xyz_cam'].float().cuda() # B, V, 3
    pix_T_cam = d['pix_T_cam'].float().cuda() # B, 4, 4
    lrtlist_cam_g = d['lrtlist_cam'].float().cuda() # B, 4, 4
    scorelist_g = d['scorelist'].float().cuda() # B, 4, 4
    tidlist_g = d['tidlist'].long().cuda() # B, 4, 4

    B, C, H, W = rgb_cam.shape
    assert(B==1)
    assert(C==3)
    B, V, D = xyz_cam.shape
    assert(B==1)
    assert(D==3)

    rgb_cam = utils.improc.preprocess_color(rgb_cam)
    
    vox_util = utils.vox.Vox_util(
        Z, Y, X,
        scene_centroid=scene_centroid,
        bounds=bounds,
        assert_cube=False)

    occ_mem = vox_util.voxelize_xyz(xyz_cam, Z, Y, X)
    occ_feat = occ_mem.squeeze(1).permute(0, 2, 1, 3) # B, Y, Z, X (y becomes feature channel)
    _, lrtlist_cam_e, scorelist_e, _ = model_3d(occ_feat, vox_util=vox_util, force_export_boxlist=True)

    # since we only have annotations within the frustum,
    # discard preds outside the frustum
    clist_e = utils.geom.get_clist_from_lrtlist(lrtlist_cam_e)
    xy_e = utils.geom.apply_pix_T_cam(pix_T_cam, clist_e)
    # ok = xy_e[0,:,0] <
    inds = utils.geom.get_image_inbounds(pix_T_cam, clist_e, H, W)
    lrtlist_cam_e = lrtlist_cam_e[0:1,inds[0]]
    scorelist_e = scorelist_e[0:1,inds[0]]

    # also, discard gt outside our 3d bounds
    scorelist_g = utils.misc.rescore_lrtlist_with_inbound(lrtlist_cam_g, scorelist_g, Z, Y, X, vox_util, pad=0.0)

    if sw is not None and sw.save_this:
        sw.summ_lrtlist_bev(
            '0_bevdet/lrtlist_mem_e',
            occ_mem,
            lrtlist_cam_e[0:1],
            scorelist_e[0:1], # scores
            torch.ones_like(scorelist_e[0:1]).long(), # tids
            vox_util,
            already_mem=False)
        sw.summ_lrtlist_bev(
            '0_bevdet/lrtlist_mem_g',
            occ_mem,
            lrtlist_cam_g[0:1],
            scorelist_g[0:1], # scores
            torch.ones_like(scorelist_g[0:1]).long(), # tids
            vox_util,
            already_mem=False)

    lrtlist_e, lrtlist_g, scorelist_e, scorelist_g = utils.eval.drop_invalid_lrts(
        lrtlist_cam_e, lrtlist_cam_g, scorelist_e, scorelist_g)
    boxlist_e = utils.geom.get_boxlist2d_from_lrtlist(pix_T_cam, lrtlist_e, H, W)
    boxlist_g = utils.geom.get_boxlist2d_from_lrtlist(pix_T_cam, lrtlist_g, H, W)

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

        ious_per = np.zeros((Ne), dtype=np.float32)
        boxlist_e_np = boxlist_e.detach().cpu().numpy()
        boxlist_g_np = boxlist_g.detach().cpu().numpy()
        for i in list(range(Ne)):
            iou_2d = utils.box.boxlist_2d_iou(boxlist_e_np[:,i:i+1].repeat(Ng, axis=1), boxlist_g_np)
            ious_per[i] = np.max(iou_2d)
        ious_per = torch.from_numpy(ious_per).float().cuda().reshape(1, Ne)

        maps_3d, maps_bev = utils.eval.get_mAP_from_lrtlist(lrtlist_e, scorelist_e, lrtlist_g, iou_thresholds)
        metrics['maps_bev'] = maps_bev
        # print('maps_bev', maps_bev)

        maps_per = utils.eval.get_mAP_from_2d_boxlists(boxlist_e, scorelist_e, boxlist_g, iou_thresholds)
        metrics['maps_per'] = maps_per
        # print('maps_per', maps_per)

        tidlist_e = 2*torch.ones_like(scorelist_e).long()
        tidlist_g = 5*torch.ones_like(scorelist_g).long()
        if sw is not None and sw.save_this:
            sw.summ_lrtlist('1_boxes/lrtlist_e', rgb_cam, lrtlist_e, ious_bev, tidlist_e, pix_T_cam, frame_id=maps_bev[0], include_zeros=True)
            sw.summ_boxlist2d('1_boxes/boxlist_e', rgb_cam, boxlist_e, ious_per, tidlist_e, frame_id=maps_per[0])
            sw.summ_lrtlist('1_boxes/lrtlist_g', rgb_cam, lrtlist_g, scorelist_g, tidlist_g, pix_T_cam)
            sw.summ_boxlist2d('1_boxes/boxlist_g', rgb_cam, boxlist_g, scorelist_g, tidlist_g)
    elif torch.sum(scorelist_g==0) and torch.sum(scorelist_e==0):
        # mAP unaffected
        metrics['maps_bev'] = None
        metrics['maps_per'] = None

    if sw.save_this and export_vis:
        tidlist_e = 2*torch.ones_like(scorelist_e).long()

        if metrics['maps_bev'] is not None:
            mAP = metrics['maps_bev'][4] # map@0.5
        else:
            mAP = 0
            
        # get perspective vis
        vis_per = sw.summ_lrtlist('', rgb_cam, lrtlist_e, scorelist_e, tidlist_e, pix_T_cam, frame_id=mAP, include_zeros=True, only_return=True)

        # get bev vis
        occ_mem_high = vox_util.voxelize_xyz(xyz_cam, Z*2, Y*2, X*2)
        vis_bev = sw.summ_lrtlist_bev('',occ_mem_high,lrtlist_cam_e,scorelist_e,tidlist_e,vox_util,only_return=True)

        # pad bev to match the (wider) perspective vis
        pad_w = int(W-Z*2)//2
        vis_bev = F.pad(vis_bev, (pad_w, pad_w))

        # cat and save
        vis_both = torch.cat([vis_per, vis_bev], dim=2)
        save_vis(vis_both, step_name)
        
        
    return total_loss, metrics

    
def main(
        init_dir_3d,
        exp_name='eval',
        max_iters=665, # size of val set
        log_freq=100,
        export_npzs=True,
        export_vis=False,
        shuffle=False,
        seq_name='any',
        sort=True,
        skip_to=0,
        show_thresh=0.5,
):
    
    ## autogen a name
    model_name = "%s" % exp_name
    model_name += "_%s" % init_dir_3d.split('/')[-1]
    import datetime
    model_date = datetime.datetime.now().strftime('%H:%M:%S')
    model_name = model_name + '_' + model_date
    print('model_name', model_name)
    
    ckpt_dir = 'checkpoints/%s' % model_name
    log_dir = 'logs_tcr_kitti_eval'
    writer_t = SummaryWriter(log_dir + '/' + model_name + '/t', max_queue=10, flush_secs=60)

    B = 1
    S = 2
    train_dataset = SimpleKittiDataset(S=1,shuffle=shuffle,dset='v',kitti_data_seqlen=2,seq_name=seq_name,sort=sort)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=B,
        shuffle=shuffle,
        num_workers=4,
        drop_last=True)
    train_iterloader = iter(train_dataloader)
    
    global_step = 0

    stride = 4
    model_3d = nets.centernet2d.Centernet2d(Y=Y, K=20, show_thresh=show_thresh, stride=stride).cuda()
    parameters = list(model_3d.parameters())
    _ = saverloader.load(init_dir_3d, model_3d)
    requires_grad(parameters, False)
    model_3d.eval()

    n_pool = max_iters*2
    loss_pool_t = utils.misc.SimplePool(n_pool, version='np')
    map_bev_pools = [utils.misc.SimplePool(n_pool, version='np') for i in list(range(len(iou_thresholds)))]
    map_per_pools = [utils.misc.SimplePool(n_pool, version='np') for i in list(range(len(iou_thresholds)))]
    
    while global_step < max_iters:
        torch.cuda.empty_cache()
        
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

        if global_step >= skip_to:
            step_name = '%s_%s_%04d' % (seq_name, exp_name, global_step)
            
            _, metrics = run_model(B, model_3d, sample, sw_t, export_vis=export_vis, export_npzs=export_npzs, step_name=step_name)
            
            if metrics['maps_bev'] is not None:
                for i,m in enumerate(metrics['maps_bev']):
                    map_bev_pools[i].update([m])
                for i,m in enumerate(metrics['maps_per']):
                    map_per_pools[i].update([m])

            for i in range(len(iou_thresholds)):
                sw_t.summ_scalar('map_bev/iou_%.1f' % iou_thresholds[i], map_bev_pools[i].mean())
                sw_t.summ_scalar('map_per/iou_%.1f' % iou_thresholds[i], map_per_pools[i].mean())

        iter_time = time.time()-iter_start_time
        print('%s; step %06d/%d; rtime %.2f; itime %.2f; map@%.1f %.2f; map@%.1f %.2f; map@%.1f %.2f; map@%.1f %.2f' % (
            model_name, global_step, max_iters, read_time, iter_time,
            iou_thresholds[0], map_bev_pools[0].mean(),
            iou_thresholds[2], map_bev_pools[2].mean(),
            iou_thresholds[4], map_bev_pools[4].mean(),
            iou_thresholds[6], map_bev_pools[6].mean(),
        ))

    writer_t.close()

    print('-'*10)
    print('BEV accuracy summary:')
    for i in range(len(iou_thresholds)):
        print('map@iou=%.1f: %.2f' % (iou_thresholds[i], map_bev_pools[i].mean()))
    print('-'*10)
    print('perspective accuracy summary:')
    for i in range(len(iou_thresholds)):
        print('map@iou=%.1f: %.2f' % (iou_thresholds[i], map_per_pools[i].mean()))
    print('-'*10)

if __name__ == '__main__':
    Fire(main)
    
