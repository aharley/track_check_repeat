import time
import argparse
import numpy as np
import timeit
import imageio
import io
import os
import math
import sys
# import matplotlib
# from PIL import Image
# matplotlib.use('Agg') # suppress plot showing
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
import cv2
import saverloader
import skimage.morphology

def requires_grad(parameters, flag=True):
    for p in parameters:
        p.requires_grad = flag

import utils.eval
import utils.py
import utils.box
import utils.misc
import utils.improc
import utils.vox
import utils.grouping
import random
import glob
import color2d

from utils.basic import print_, print_stats

from fire import Fire

import kittidataset

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from tensorboardX import SummaryWriter

import torch.nn.functional as F

import nets.seg2dnet
import nets.seg3dnet
import nets.bevdet
from nets.centernet2d import Centernet2d

device = 'cuda'
patch_size = 8
random.seed(125)
np.random.seed(125)

iou_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

scene_centroid_x = 0.0
# scene_centroid_y = 1.0 #hyp.YMIN
# scene_centroid_z = 18.0
scene_centroid_y = 1.0
scene_centroid_z = 0.0

scene_centroid = np.array([scene_centroid_x,
                           scene_centroid_y,
                           scene_centroid_z]).reshape([1, 3])
scene_centroid = torch.from_numpy(scene_centroid).float().cuda()

XMIN, XMAX = -32, 32
ZMIN, ZMAX = 2, 66
YMIN, YMAX = -3, 5


XMIN, XMAX = -16, 16
ZMIN, ZMAX = 2, 34
YMIN, YMAX = -1, 3

bounds = (XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX)

sc = 8
# Z, Y, X = ZMAX*sc, YMAX*sc, XMAX*sc
Z, Y, X = 1024, 40, 1024
Z, Y, X = 512, 32, 512
Z, Y, X = 256, 16, 256

Z2, Y2, X2 = Z//2, Y//2, X//2
Z4, Y4, X4 = Z//4, Y//4, X//4
Z8, Y8, X8 = Z//8, Y//8, X//8

vis_dir = './tcr_vis'
npz_dir = './tcr_pseudo'
utils.basic.mkdir(npz_dir)
utils.basic.mkdir(vis_dir)

def requires_grad(parameters, flag=True):
    for p in parameters:
        p.requires_grad = flag

def save_vis(rgb, name):
    rgb = rgb.cpu().numpy()[0].transpose(1,2,0) # H x W x 3
    vis_fn = os.path.join(vis_dir, '%s.png' % (name))
    imageio.imwrite(vis_fn, rgb)
    print('saved %s' % vis_fn)
    
def run_model(B, model_3d, d, sw, export_vis=False, export_npzs=False):
    total_loss = torch.tensor(0.0, requires_grad=True).to(device)
    metrics = {}
    metrics['maps_bev'] = [i*0 for i in iou_thresholds] # mAP=0 by default
    metrics['maps_per'] = [i*0 for i in iou_thresholds] # mAP=0 by default

    # ------------------
    # set up the data
    # ------------------
    
    rgb_camXs = d['rgb_camXs'].float() # B, S, C, H, W
    xyz_veloXs = d['xyz_veloXs'].float() # B, N, 3
    cam_T_velos = d['cam_T_velos'].float() # B, S, 4, 4
    rect_T_cams = d['rect_T_cams'].float() # B, S, 4, 4
    pix_T_rects = d['pix_T_rects'].float() # B, S, 4, 4

    rgb_cam = rgb_camXs[:,0].cuda()
    xyz_velo = xyz_veloXs[:,0].cuda()
    pix_T_rect = pix_T_rects[:,0].cuda()
    rect_T_cam = rect_T_cams[:,0].cuda()
    cam_T_velo = cam_T_velos[:,0].cuda()

    B1, V, D = xyz_velo.shape
    assert(B1==1)

    _, _, H, W = rgb_cam.shape
    
    vox_util = utils.vox.Vox_util(
        Z, Y, X,
        scene_centroid=scene_centroid,
        bounds=bounds,
        assert_cube=False)

    xyz_velo_free = vox_util.convert_xyz_to_visibility_samples(xyz_velo, samps=16, dist_eps=0.05, rand=True)
    xyz_cam = utils.geom.apply_4x4(cam_T_velo, xyz_velo)
    xyz_rect = utils.geom.apply_4x4(rect_T_cam, xyz_cam)
    xyz_cam_free = utils.geom.apply_4x4(cam_T_velo, xyz_velo_free)

    occ_mem = vox_util.voxelize_xyz(xyz_cam, Z4, Y4, X4)
    free_mem = vox_util.voxelize_xyz(xyz_cam_free, Z4, Y4, X4)
    free_mem = (free_mem - utils.improc.dilate3d(occ_mem)).clamp(0,1)

    input_occ_mem = vox_util.voxelize_xyz(xyz_cam, Z, Y, X)
    input_free_mem = vox_util.voxelize_xyz(xyz_cam_free, Z, Y, X)
    input_occ_mem_fat = utils.improc.dilate3d(input_occ_mem).clamp(0,1)
    input_free_mem = (input_free_mem - input_occ_mem_fat).clamp(0,1)
    
    # ------------------
    # get the estimates from the cnns
    # ------------------
    occ_feat = input_occ_mem.squeeze(1).permute(0, 2, 1, 3) # B, Y, Z, X (y becomes feature channel)
    # _, lrtlist_cam_e, scorelist_e, seg_e = model_3d(occ_feat, vox_util=vox_util, force_export_boxlist=True)
    _, lrtlist_cam_e, scorelist_e, _ = model_3d(occ_feat, vox_util=vox_util, force_export_boxlist=True)

    # since we only have annotations within the frustum,
    # discard preds outside the frustum
    clist_e = utils.geom.get_clist_from_lrtlist(lrtlist_cam_e)
    xy_e = utils.geom.apply_pix_T_cam(pix_T_rect, clist_e)
    # ok = xy_e[0,:,0] <
    inds = utils.geom.get_image_inbounds(pix_T_rect, clist_e, H, W)
    lrtlist_cam_e = lrtlist_cam_e[0:1,inds[0]]
    scorelist_e = scorelist_e[0:1,inds[0]]

    boxlist_cam_g = d['boxlist_camXs'].float().cuda()[:,0] # note this is already in rectified coords
    tidlist_g = d['tidlist_s'].long().cuda()[:,0]
    scorelist_g = d['scorelist_s'].float().cuda()[:,0]
    lrtlist_cam_g = utils.geom.convert_boxlist_to_lrtlist(boxlist_cam_g)
    scorelist_g = utils.misc.rescore_lrtlist_with_inbound(lrtlist_cam_g, scorelist_g, Z, Y, X, vox_util, pad=0.0)

    # if torch.sum(scorelist_g)==0:
    #     metrics['maps_bev'] = None
    #     metrics['maps_per'] = None
    #     return total_loss, metrics
    
    if sw is not None and sw.save_this:
        sw.summ_lrtlist_bev(
            '0_bevdet/lrtlist_mem_e',
            input_occ_mem,
            lrtlist_cam_e[0:1],
            scorelist_e[0:1], # scores
            torch.ones_like(scorelist_e[0:1]).long(), # tids
            vox_util,
            already_mem=False)

        sw.summ_lrtlist_bev(
            '0_bevdet/lrtlist_mem_g',
            input_occ_mem,
            lrtlist_cam_g[0:1],
            scorelist_g[0:1], # scores
            torch.ones_like(scorelist_g[0:1]).long(), # tids
            vox_util,
            already_mem=False)
    
    # seg3d_e = torch.sigmoid(model_3d(input_occ_mem)) # B, 1, Z4, Y4, X4
    # if torch.max(seg3d_e) > 0.6: # push the max to 1.0
    #     seg3d_e = seg3d_e / (1e-4 + torch.max(seg3d_e))
    # seg3d_e = F.interpolate(seg3d_e, scale_factor=4, mode='trilinear') # B, 1, Z, Y, X

    rgb_cam = utils.improc.preprocess_color(rgb_cam)
    # seg2d_e = torch.sigmoid(model_2d(rgb_cam)) # B, 1, H4, W4
    # if torch.max(seg2d_e) > 0.6: # push the max to 1.0
    #     seg2d_e = seg2d_e / (1e-4 + torch.max(seg2d_e))
    # seg2d_e = F.interpolate(seg2d_e, scale_factor=4) # B, 1, H, W

    # # ------------------
    # # assemble the ensemble
    # # ------------------
    # seg2d_mem = vox_util.unproject_image_to_mem(seg2d_e, Z, Y, X, utils.basic.matmul2(pix_T_rect, rect_T_cam), assert_cube=False)
    # prod_mem = seg2d_mem * seg3d_e

    # # ------------------
    # # cluster regions into objects
    # # ------------------
    # N = 32 # max number of objects
    # # use two thresholds, one tight and one loose
    # # binary1_mem = ((prod_mem * input_occ_mem_fat) > 0.95).float()
    # # binary2_mem = ((prod_mem * input_occ_mem_fat) > 0.9).float()

    # binary1_mem = ((prod_mem * (1-input_free_mem)) > 0.95).float()
    # binary2_mem = ((prod_mem * (1-input_free_mem)) > 0.9).float()
    
    # boxlist1_mem, scorelist1, tidlist1, connlist1 = utils.misc.get_any_boxes_from_binary(
    #     binary1_mem.squeeze(1), N, min_voxels=64, min_side=1, count_mask=binary1_mem)
    # boxlist2_mem, scorelist2, tidlist2, connlist2 = utils.misc.get_any_boxes_from_binary(
    #     binary2_mem.squeeze(1), N, min_voxels=64, min_side=1, count_mask=binary2_mem)

    # # now the idea is:
    # # for each object in connlist1,
    # # if there is an overlapping object in connlist2, add it (to extend slightly)
    # boxlist3_mem = boxlist1_mem.clone()
    # connlist3 = connlist1.clone()
    # tidlist3 = tidlist1.clone()
    # scorelist3 = scorelist1.clone()
    # for n1 in range(N):
    #     for n2 in range(N):
    #         conn1 = connlist1[:,n1]
    #         conn2 = connlist2[:,n2]
    #         if torch.sum(conn1*conn2) > 0:
    #             connlist3[:,n1] = (connlist1[:,n1] + connlist2[:,n2]).clamp(0,1) # use the union of the two
    #             boxlist3_mem[:,n1] = boxlist2_mem[:,n2] # use the larger box

    # if torch.sum(connlist3) == 0:
    #     # return early
    #     return total_loss, metrics
    
    # # check that the 2d projection has some minimum area
    # conn_seg = []
    # seg_xyz_cam_list = []
    # xyz_mem = utils.basic.gridcloud3d(B, Z, Y, X)
    # for n1 in range(N):
    #     conn3 = connlist3[:,n1]
    #     xyz_mem_here = xyz_mem[:,conn3.reshape(-1) > 0]
    #     xyz_cam_here = vox_util.Mem2Ref(xyz_mem_here, Z, Y, X)
    #     xy_pix = utils.geom.apply_pix_T_cam(pix_T_rect, utils.geom.apply_4x4(rect_T_cam, xyz_cam_here))
    #     mask = utils.improc.xy2mask(xy_pix, H, W, norm=False)
    #     mask = mask[0,0].cpu().numpy()
    #     if np.sum(mask) > 4:
    #         seg_xyz_cam_list.append(xyz_cam_here)
    #         # print('validated conn segment %d; sum(conn3); np.sum(mask)' % n1, torch.sum(conn3).item(), np.sum(mask))
    #         # close the hull
    #         mask = skimage.morphology.convex_hull.convex_hull_image(mask)
    #         conn_seg.append(torch.from_numpy(mask.reshape(1, H, W)).float().cuda())

    # if len(conn_seg) and sw is not None and sw.save_this:
    #     conn_seg = torch.stack(conn_seg, dim=1)
    #     sw.summ_soft_seg_thr('3_objects/conn_seg', conn_seg, frame_id=torch.sum(conn_seg))
    #     conn_vis = sw.summ_soft_seg_thr('4_objects/conn_seg', conn_seg, only_return=True)
    #     grey = torch.mean(rgb_cam, dim=1, keepdim=True).repeat(1, 3, 1, 1)
    #     seg = utils.improc.preprocess_color(conn_vis).cuda()
    #     sw.summ_rgb('3_objects/conn_on_rgb', (grey + seg)/2.0)
    #     if export_vis:
    #         seg_vis = sw.summ_rgb('', (grey + seg)/2.0, only_return=True)
    #         save_vis(seg_vis, step_name)

    # # ------------------
    # # export pseudolabels for the next stage
    # # ------------------
    # if export_npzs:
    #     npz_fn = os.path.join(npz_dir, '%s.npz' % (step_name))
    #     seg_xyz_cam_list_py = [xyz[0].detach().cpu().numpy() for xyz in seg_xyz_cam_list]
    #     np.savez(npz_fn,
    #              rgb_cam=rgb_cam[0].detach().cpu().numpy(), 
    #              xyz_cam=xyz_rect[0].detach().cpu().numpy(), # use rect==cam, for simplicity
    #              pix_T_cam=pix_T_rect[0].detach().cpu().numpy(),
    #              seg_xyz_cam_list=seg_xyz_cam_list_py,
    #     )
    #     print('saved %s' % npz_fn)

    # # ------------------
    # # vis everything 
    # # ------------------
    # if sw is not None and sw.save_this:
    #     sw.summ_rgb('0_inputs/rgb', rgb_cam)
    #     sw.summ_oned('1_outputs/seg2d_e', seg2d_e, norm=True)
    #     sw.summ_oned('1_outputs/seg2d_mem', torch.mean(seg2d_mem, dim=3), norm=True)

    #     pos_e = (seg2d_e > 0.8).float()
    #     neg_e = (seg2d_e < 0.2).float()
    #     seg_vis = torch.cat([pos_e, neg_e], dim=1)
    #     seg_vis = sw.summ_soft_seg_thr('', seg_vis, colormap='tab10', only_return=True)
    #     grey = torch.mean(rgb_cam, dim=1, keepdim=True).repeat(1, 3, 1, 1)
    #     seg = utils.improc.preprocess_color(seg_vis).cuda()
    #     sw.summ_rgb('1_outputs/seg2d_e_on_rgb', (grey + seg)/2.0)

    #     pos_e = (seg3d_e > 0.90).float()
    #     neg_e = (seg3d_e < 0.05).float()

    #     # just show me the occ estimates
    #     # pos_e = pos_e * input_occ_mem
    #     # neg_e = neg_e * input_occ_mem
    #     pos_bev = torch.max(pos_e, dim=3)[0]
    #     neg_bev = torch.max(neg_e, dim=3)[0]
    #     seg_bev = torch.cat([pos_bev, neg_bev], dim=1)
    #     seg_vis = sw.summ_soft_seg_thr('', seg_bev, colormap='tab10', only_return=True)
    #     occ_vis = sw.summ_occ('', input_occ_mem, only_return=True)
    #     seg_vis = utils.improc.preprocess_color(seg_vis).cuda()
    #     occ_vis = utils.improc.preprocess_color(occ_vis).cuda()
    #     sw.summ_rgb('1_outputs/seg3d_e_on_occ', (occ_vis + seg_vis)/2.0)
    #     sw.summ_oned('1_outputs/seg3d_e', torch.mean(seg3d_e, dim=3), norm=True)


    #     sw.summ_oned('2_ensemble/prod_mem', torch.mean(prod_mem, dim=3), norm=True)
    #     # sw.summ_oned('2_ensemble/prod_mem_occ', torch.mean(prod_mem*input_occ_mem, dim=3), norm=True)
    #     # sw.summ_oned('2_ensemble/prod_mem_occ', torch.max(prod_mem*input_occ_mem_fat, dim=3)[0], norm=True)
    #     sw.summ_oned('2_ensemble/prod_mem_occ', torch.max(binary1_mem, dim=3)[0], norm=True)
    #     pos_e = (prod_mem > 0.90).float()
    #     neg_e = (prod_mem < 0.05).float()
    #     # pos_e = pos_e * input_occ_mem
    #     # neg_e = neg_e * input_occ_mem
    #     pos_bev = torch.max(pos_e, dim=3)[0]
    #     neg_bev = torch.max(neg_e, dim=3)[0]
    #     seg_bev = torch.cat([pos_bev, neg_bev], dim=1)
    #     seg_vis = sw.summ_soft_seg_thr('', seg_bev, colormap='tab10', only_return=True)
    #     occ_vis = sw.summ_occ('', input_occ_mem, only_return=True)
    #     seg_vis = utils.improc.preprocess_color(seg_vis).cuda()
    #     occ_vis = utils.improc.preprocess_color(occ_vis).cuda()
    #     sw.summ_rgb('2_ensemble/prod_mem_on_occ', (occ_vis + seg_vis)/2.0)


    #     sw.summ_occ('3_objects/connlist1', connlist1.sum(dim=1, keepdim=True), bev=True)
    #     sw.summ_occ('3_objects/connlist2', connlist2.sum(dim=1, keepdim=True), bev=True)
    #     sw.summ_occ('3_objects/connlist3', connlist3.sum(dim=1, keepdim=True), bev=True)
    #     seg_vis = sw.summ_soft_seg_thr('', torch.max(connlist3, dim=3)[0], colormap='tab10', only_return=True)
    #     seg_vis = utils.improc.preprocess_color(seg_vis).cuda()
    #     sw.summ_rgb('3_objects/connlist3_on_occ', (occ_vis + seg_vis)/2.0)

    # ------------------
    # evaluate mAP
    # ------------------
    # boxlist_rect_g = d['boxlist_camXs'].float().cuda()[:,0] # note this is already in rectified coords
    # tidlist_g = d['tidlist_s'].long().cuda()[:,0]
    # scorelist_g = d['scorelist_s'].float().cuda()[:,0]
    # lrtlist_rect_g = utils.geom.convert_boxlist_to_lrtlist(boxlist_rect_g)
    # scorelist_g = utils.misc.rescore_lrtlist_with_inbound(lrtlist_rect_g, scorelist_g, Z, Y, X, vox_util, pad=0.0)

    # lrtlist_mem = utils.geom.convert_boxlist_to_lrtlist(boxlist3_mem)
    # lrtlist_cam = vox_util.apply_ref_T_mem_to_lrtlist(lrtlist_mem, Z, Y, X)
    # lrtlist_rect_e = utils.geom.apply_4x4_to_lrtlist(rect_T_cam, lrtlist_cam)

    lrtlist_e, lrtlist_g, scorelist_e, scorelist_g = utils.eval.drop_invalid_lrts(
        lrtlist_cam_e, lrtlist_cam_g, scorelist_e, scorelist_g)
    boxlist_e = utils.geom.get_boxlist2d_from_lrtlist(pix_T_rect, lrtlist_e, H, W)
    boxlist_g = utils.geom.get_boxlist2d_from_lrtlist(pix_T_rect, lrtlist_g, H, W)

    if torch.sum(scorelist_g) > 0 and torch.sum(scorelist_e) > 0:
        Ne = lrtlist_e.shape[1]
        Ng = lrtlist_g.shape[1]
        ious_3d = np.zeros((Ne, Ng), dtype=np.float32)
        ious_bev = np.zeros((Ne, Ng), dtype=np.float32)
        for i in list(range(Ne)):
            for j in list(range(Ng)):
                iou_3d, iou_bev = utils.geom.get_iou_from_corresponded_lrtlists(lrtlist_e[:, i:i+1], lrtlist_g[:, j:j+1])
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

        if sw is not None and sw.save_this:
            sw.summ_lrtlist('1_boxes/lrtlist_e', rgb_cam, lrtlist_e, ious_bev, 2*torch.ones_like(scorelist_e).long(), pix_T_rect, frame_id=maps_bev[0], include_zeros=True)
            sw.summ_boxlist2d('1_boxes/boxlist_e', rgb_cam, boxlist_e, ious_per, 2*torch.ones_like(scorelist_e).long(), frame_id=maps_per[0])
            sw.summ_lrtlist('1_boxes/lrtlist_g', rgb_cam, lrtlist_g, scorelist_g, 5*torch.ones_like(scorelist_g).long(), pix_T_rect)
            sw.summ_boxlist2d('1_boxes/boxlist_g', rgb_cam, boxlist_g, scorelist_g, 5*torch.ones_like(scorelist_g).long())
    elif torch.sum(scorelist_g==0) and torch.sum(scorelist_e==0):
        # mAP unaffected
        metrics['maps_bev'] = None
        metrics['maps_per'] = None
        
            
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
    # train_dataset = kittidataset.KittiDataset(S=S,shuffle=shuffle,dset='a',kitti_data_seqlen=2,seq_name=seq_name,sort=sort)
    train_dataset = kittidataset.KittiDataset(S=S,shuffle=shuffle,dset='v',kitti_data_seqlen=2,seq_name=seq_name,sort=sort)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=B,
        shuffle=shuffle,
        num_workers=4,
        drop_last=True)
    train_iterloader = iter(train_dataloader)
    
    global_step = 0

    stride = 4
    model_3d = Centernet2d(Y=Y, show_thresh=0.5, K=20, stride=stride).cuda()
    # model_3d = nets.bevdet.Bevdet(Y=Y, show_thresh=0.3, K=20).cuda()
    parameters = list(model_3d.parameters())
    _ = saverloader.load(init_dir_3d, model_3d)
    requires_grad(parameters, False)
    model_3d.eval()

    # model_2d = nets.seg2dnet.Seg2dNet(num_classes=1).to(device).eval()
    # parameters = list(model_2d.parameters())
    # _ = saverloader.load(init_dir_2d, model_2d)
    # requires_grad(parameters, False)
    # model_2d.eval()

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
            # _, metrics = run_model(B, model_2d, model_3d, sample, sw_t,
            #                        export_vis=export_vis, export_npzs=export_npzs)
            _, metrics = run_model(B, model_3d, sample, sw_t,
                                   export_vis=export_vis, export_npzs=export_npzs)
            
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
    
