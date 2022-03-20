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

from simplekittidataset import SimpleKittiDataset

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from tensorboardX import SummaryWriter

import torch.nn.functional as F

import nets.raftnet
import nets.seg2dnet
# import nets.seg3dnet
# import nets.bevdet
import nets.centernet2d

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

XMIN, XMAX = -16, 16
ZMIN, ZMAX = 2, 34
YMIN, YMAX = -1, 3

bounds = (XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX)

sc = 8
# Z, Y, X = ZMAX*sc, YMAX*sc, XMAX*sc
# Z, Y, X = 512, 32, 512
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
    
def run_model(B, raft, model_2d, model_3d, d, sw, export_vis=False, export_npzs=False, step_name='temp', force_vis=False):
    total_loss = torch.tensor(0.0, requires_grad=True).to(device)
    metrics = {}
    metrics['maps_bev'] = [i*0 for i in iou_thresholds] # mAP=0 by default
    metrics['maps_per'] = [i*0 for i in iou_thresholds] # mAP=0 by default

    # ------------------
    # set up the data
    # ------------------
    
    rgb_cams = d['rgb_cams'].float().cuda() # B, S, C, H, W
    xyz_cams = d['xyz_cams'].float().cuda() # B, S, V, 3
    pix_T_cams = d['pix_T_cams'].float().cuda() # B, S, 4, 4

    B, S, C, H, W = rgb_cams.shape
    assert(B==1)
    assert(C==3)
    B, S, V, D = xyz_cams.shape
    assert(B==1)
    assert(D==3)

    __p = lambda x: utils.basic.pack_seqdim(x, B)
    __u = lambda x: utils.basic.unpack_seqdim(x, B)

    rgb_cams = utils.improc.preprocess_color(rgb_cams)
    rgb_cam0 = rgb_cams[:,0]
    rgb_cam1 = rgb_cams[:,1]

    # ------------------
    # get 2d flow, and perform cycle-consistency checks
    # ------------------
    
    flow_01, _ = raft(rgb_cam0, rgb_cam1, iters=8)
    flow_10, _ = raft(rgb_cam1, rgb_cam0, iters=8)
    flow_01 = flow_01.detach()
    flow_10 = flow_10.detach()

    # note that the valid mask here indicates the pixels in 1 that that have a corresp in 0
    valid_1 = utils.samp.get_backwarp_mask(flow_01)
    valid_0 = utils.samp.get_backwarp_mask(flow_10)
    valid_0_aligned_to_1 = utils.samp.backwarp_using_2d_flow(valid_0, flow_01)
    valid_1_aligned_to_0 = utils.samp.backwarp_using_2d_flow(valid_1, flow_10)

    # get mask of pixels that have one-to-one mapping between frames
    oto_0 = valid_0.round() * valid_1_aligned_to_0.round()
    oto_1 = valid_1.round() * valid_0_aligned_to_1.round()

    # get mask cycle-consistency mask
    flow_10_aligned_to_0 = utils.samp.backwarp_using_2d_flow(flow_10, flow_01)
    flow_01_aligned_to_1 = utils.samp.backwarp_using_2d_flow(flow_01, flow_10)

    residual_0 = torch.norm((flow_01 + flow_10_aligned_to_0), dim=1) # B x H x W
    residual_1 = torch.norm((flow_10 + flow_01_aligned_to_1), dim=1) # B x H x W

    flow_01_cat = torch.cat([flow_01, flow_10_aligned_to_0], dim=1)
    flow_10_cat = torch.cat([flow_10, flow_01_aligned_to_1], dim=1)

    flow_01_mask = (residual_0 < 0.05*torch.norm(flow_01_cat, dim=1, keepdim=True) + 0.5).float()
    flow_10_mask = (residual_1 < 0.05*torch.norm(flow_10_cat, dim=1, keepdim=True) + 0.5).float()

    rely_0 = flow_01_mask*oto_0
    rely_1 = flow_10_mask*oto_1
    rely_1_mask_aligned_to_0 = utils.samp.backwarp_using_2d_flow(rely_1, flow_01).round()
    rely_0_mask_aligned_to_1 = utils.samp.backwarp_using_2d_flow(rely_0, flow_10).round()
    rely_0 = rely_0 * rely_1_mask_aligned_to_0
    rely_1 = rely_1 * rely_0_mask_aligned_to_1
    

    # ------------------
    # get egomotion flow
    # ------------------
    
    pix_T_cam = pix_T_cams[:,0]
    xyz_cam0 = xyz_cams[:,0]
    xyz_cam1 = xyz_cams[:,1]

    # only valid stuff please
    xyz_cam0 = xyz_cam0[:,xyz_cam0[0,:,2]>1]
    xyz_cam1 = xyz_cam1[:,xyz_cam1[0,:,2]>1]
    # print_stats('xyz_cam0', xyz_cam0.shape)

    # estimate egomotion, using the flow, depth, and RANSAC
    inlier_thresh_3d = 0.25 # meters; max disagreement with the rigid motion
    inlier_thresh_2d = 2.0 # pixels
    corresp_thresh_3d = 0.1 # meters; max displacement from an existing point, to call it a new corresp
    cam1_T_cam0, align_error, corresp_tuple = utils.grouping.get_cycle_consistent_transform(
        xyz_cam0, xyz_cam1,
        flow_01, flow_10,
        pix_T_cam, H, W,
        flow_01_valid=rely_0,
        flow_10_valid=rely_1,
        inlier_thresh=inlier_thresh_3d)
    cam0_T_cam1 = utils.geom.safe_inverse(cam1_T_cam0)
    xyz_cam0_c, xyz_cam1_c = corresp_tuple
    xyz_cam1_c_stab = utils.geom.apply_4x4(cam0_T_cam1, xyz_cam1_c)
    # measure error in the corresponded pointclouds
    align_error_c = torch.norm(xyz_cam0_c-xyz_cam1_c_stab, dim=2)
    xyz_cam0_c_ = xyz_cam0_c.squeeze(0)
    xyz_cam1_c_ = xyz_cam1_c_stab.squeeze(0)
    align_error_ = align_error_c.squeeze(0)
    # i = inlier; o = outlier
    xyz_cam0_i = xyz_cam0_c_[align_error_ <= inlier_thresh_3d].unsqueeze(0)
    xyz_cam1_i = xyz_cam1_c_[align_error_ <= inlier_thresh_3d].unsqueeze(0)
    xyz_cam0_o = xyz_cam0_c_[align_error_ > inlier_thresh_3d].unsqueeze(0)
    xyz_cam1_o = xyz_cam1_c_[align_error_ > inlier_thresh_3d].unsqueeze(0)

    # go to mem
    vox_util = utils.vox.Vox_util(
        Z, Y, X,
        scene_centroid=scene_centroid,
        bounds=bounds,
        assert_cube=False)
    occ_mem0 = vox_util.voxelize_xyz(xyz_cam0, Z, Y, X)
    occ_mem1 = vox_util.voxelize_xyz(xyz_cam1, Z, Y, X)
    occ_mem0_i = vox_util.voxelize_xyz(xyz_cam0_i, Z, Y, X)
    occ_mem1_i = vox_util.voxelize_xyz(xyz_cam1_i, Z, Y, X)
    occ_mem0_o = vox_util.voxelize_xyz(xyz_cam0_o, Z, Y, X)
    occ_mem1_o = vox_util.voxelize_xyz(xyz_cam1_o, Z, Y, X)
    explained_by_ego = utils.improc.dilate3d(occ_mem0_i)
    occ_mem0_ext = vox_util.voxelize_xyz(torch.cat([xyz_cam0, xyz_cam0*1.01, xyz_cam0*1.02], dim=1), Z, Y, X)
    free_mem0 = vox_util.get_freespace(xyz_cam0, occ_mem0)
    occ_mem0_ext = (occ_mem0_ext - free_mem0).clamp(0,1)
    # occ_mem0_fat = utils.improc.dilate3d(occ_mem0).clamp(0,1)

    max_align_error = torch.max(align_error)
    # print_('max_align_error', max_align_error)

    if torch.max(max_align_error) > 0.1:
        # print('egomotion estimation failed: max_align_error', max_align_error)
        
        # return early, without trying to find objects
        if export_vis:
            grey = torch.mean(rgb_cams[:,0], dim=1, keepdim=True).repeat(1, 3, 1, 1)
            save_vis(utils.improc.back2color((grey+0.5)*0.5-0.5), step_name)

        return total_loss, metrics

    # stabilize things according to the estimated motion, just for vis 
    xyz_cam1_stab = utils.geom.apply_4x4(cam0_T_cam1, xyz_cam1)
    occ_mem1_stab = vox_util.voxelize_xyz(xyz_cam1_stab, Z, Y, X)

    # ------------------
    # find evidence of independent motion
    # ------------------

    # convert the pointcloud into a depth map
    depth_cam0, valid_cam0 = utils.geom.create_depth_image(pix_T_cam, xyz_cam0, H, W, offset_amount=0)
    # depth_cam1, valid_cam1 = utils.geom.create_depth_image(pix_T_cams[:,1], xyz_cam1, H, W, offset_amount=0)

    # convert egomotion+depth into a flow field, and take a 2d difference
    flow_01_ego = utils.geom.depthrt2flow(depth_cam0, cam1_T_cam0, pix_T_cam) * valid_cam0
    flow_dist = torch.norm(flow_01_ego - flow_01*valid_cam0, dim=1, keepdim=True) * rely_0 * valid_cam0
    cos = nn.CosineSimilarity(dim=1)
    flow_cos = cos(flow_01_ego, flow_01)

    xy = utils.basic.gridcloud2d(B, H, W)
    xyd = torch.cat([xy, depth_cam0.reshape(1, -1, 1)], dim=2)

    # use thresholds on angle and distance, so that both thresholds can be loose
    # we use lower thresholds here than we did in the discover mode, since we have additional signals from models
    xyd_keepA = xyd[:,flow_cos.reshape(-1) < 0.8/4.0] # do not aim in the exact same dir
    xyz_keepA = utils.geom.xyd2pointcloud(xyd_keepA, pix_T_cam)
    occ_keepA = vox_util.voxelize_xyz(xyz_keepA, Z, Y, X) * (1-explained_by_ego) # discard static stuff
    
    xyd_keepB = xyd[:,(flow_dist.reshape(-1) > 8/4.0)]
    xyz_keepB = utils.geom.xyd2pointcloud(xyd_keepB, pix_T_cam)
    occ_keepB = vox_util.voxelize_xyz(xyz_keepB, Z, Y, X) * (1-explained_by_ego) # discard static stuff

    # ------------------
    # run the models
    # ------------------
    occ_feat = occ_mem0.squeeze(1).permute(0, 2, 1, 3) # B, Y, Z, X (y becomes feature channel)
    _, lrtlist_e, scorelist_e, seg3d_e = model_3d(occ_feat, vox_util=vox_util, force_export_boxlist=True)
    seg3d_e = F.interpolate(torch.sigmoid(seg3d_e), scale_factor=4)
    # print_stats('seg3d_e', seg3d_e)

    seg2d_e = torch.sigmoid(model_2d(rgb_cam0)) # B, 1, H4, W4
    # print_stats('seg2d_e', seg2d_e)
    seg2d_e = F.interpolate(seg2d_e, scale_factor=4) # B, 1, H, W
    seg2d_mem = vox_util.unproject_image_to_mem(
        seg2d_e, Z, Y, X, pix_T_cam, assert_cube=False)
    # print_stats('seg2d_mem', seg2d_mem)
    prod_mem = seg2d_mem * seg3d_e
    # bin_mem = (prod_mem>0.95).float() * (1-utils.improc.dilate3d(free_mem0))
    bin_mem = (prod_mem>0.9).float()
    bin_mem = utils.improc.erode3d(bin_mem)

    # # bin_mem_clean = utils.improc.erode3d(utils.improc.dilate3d(occ_mem0_ext, times=2))
    # bin_mem_clean = utils.improc.erode3d(utils.improc.dilate3d(occ_mem0, times=2))
    # bin_mem_clean = bin_mem * bin_mem_clean

    bin_mem_clean = bin_mem * utils.improc.erode3d(utils.improc.dilate3d(occ_mem0_ext, times=3)) * (1-free_mem0)

    # ------------------
    # cluster regions into objects
    # ------------------

    N = 16 # max number of regions

    # find one set of objects using motion
    boxlist1_mem, scorelist1, tidlist1, connlist1 = utils.misc.get_any_boxes_from_binary(
        (occ_keepB*occ_keepA).squeeze(1), N, min_voxels=128, min_side=2, count_mask=occ_keepB*occ_keepA)
    connlist1 = connlist1 # * occ_keepB * occ_keepA

    # find a second set using the models
    boxlist2_mem, scorelist2, tidlist2, connlist2 = utils.misc.get_any_boxes_from_binary(
        bin_mem_clean.squeeze(1), N, min_voxels=128, min_side=2, count_mask=bin_mem_clean*occ_mem0)

    # print('connlist1', torch.sum(connlist1))
    # print('connlist2', torch.sum(connlist2))
    
    # now the idea is:
    # for each object in connlist1,
    # if there is an overlapping object in connlist2,
    # then use the union
    # also, check that the 2d projection has some minimum area.
    boxlist_mem = []
    connlist_3d = []
    connlist_2d = []
    seg_xyz_cam_list = []
    tidlist = []
    xyz_mem = utils.basic.gridcloud3d(B, Z, Y, X)
    min_area = 4
    for n1 in range(N):
        for n2 in range(N):
            conn1 = connlist1[:,n1]
            conn2 = connlist2[:,n2]

            if torch.sum(conn1*conn2) > 0:
                conn_union = (conn1+conn2).clamp(0,1) # use the union of the two
                xyz_mem_here = xyz_mem[:,conn_union.reshape(-1) > 0]
                xyz_cam_here = vox_util.Mem2Ref(xyz_mem_here, Z, Y, X)
                xy_pix = utils.geom.apply_pix_T_cam(pix_T_cam, xyz_cam_here)
                mask = utils.improc.xy2mask(xy_pix, H, W, norm=False)
                mask = mask[0,0].cpu().numpy()
                if np.sum(mask) > min_area:
                    seg_xyz_cam_list.append(xyz_cam_here)
                    mask = skimage.morphology.convex_hull.convex_hull_image(mask)
                    connlist_2d.append(torch.from_numpy(mask.reshape(1, H, W)).float().cuda())
                    connlist_3d.append(conn_union)
                    boxlist_mem.append(boxlist2_mem[:,n2]) # use the larger box
                    tidlist.append(tidlist2[:,n2])
    
    if len(connlist_3d)==0 and not force_vis:
        # we did not find anything
        return total_loss, metrics

    if len(connlist_3d)==0 and force_vis:
        # make some empty tensors, so that the vis step doesn't complain
        connlist_3d = torch.zeros_like(occ_mem0)
        connlist_2d = torch.zeros_like(rgb_cam0[:,0:1])
        boxlist_mem = torch.zeros((B, N, 9), dtype=torch.float32, device='cuda')
        scorelist = torch.zeros((B, N), dtype=torch.float32, device='cuda')
        tidlist = torch.zeros((B, N), dtype=torch.int32, device='cuda')
    else:
        # stack the object lists into tensors
        connlist_3d = torch.stack(connlist_3d, dim=1)
        connlist_2d = torch.stack(connlist_2d, dim=1)
        boxlist_mem = torch.stack(boxlist_mem, dim=1)
        tidlist = torch.stack(tidlist, dim=1)
        scorelist = torch.ones_like(boxlist_mem[:,:,0])

    N = connlist_3d.shape[1]

    lrtlist_mem = utils.geom.convert_boxlist_to_lrtlist(boxlist_mem)
    lrtlist_cam = vox_util.apply_ref_T_mem_to_lrtlist(lrtlist_mem, Z, Y, X)

    if force_vis or torch.sum(connlist_3d) > 0:
        if sw is not None and sw.save_this:
            sw.summ_rgbs('0_inputs/rgb_cams', rgb_cams.unbind(1))
            sw.summ_rgb('0_inputs/rgb_0', rgb_cam0)
            sw.summ_rgb('0_inputs/rgb_1', rgb_cam1)

            sw.summ_flow('1_flow/flow_01', flow_01, clip=200)
            sw.summ_flow('1_flow/flow_10', flow_10, clip=200)
            sw.summ_oned('1_flow/rely_0', rely_0, norm=False)
            sw.summ_oned('1_flow/rely_1', rely_1, norm=False)

            sw.summ_occs('2_ego/occ_mem01_orig', [occ_mem0, occ_mem1])
            sw.summ_occs('2_ego/occ_mem01_stab', [occ_mem0, occ_mem1_stab], frame_ids=[max_align_error.item()]*2)
            sw.summ_oned('2_ego/depth_cam0', depth_cam0, max_val=30)
            
            sw.summ_occ('3_diff/occ_keepA', occ_keepA, bev=True)
            sw.summ_occ('3_diff/occ_keepB', occ_keepB, bev=True)

            sw.summ_oned('4_models/0_seg3d_e', torch.mean(seg3d_e, dim=3), norm=True)
            sw.summ_oned('4_models/0_seg2d_mem', torch.mean(seg2d_mem, dim=3), norm=True)
            sw.summ_oned('4_models/1_prod_mem', torch.mean(prod_mem, dim=3), norm=True)
            sw.summ_occ('4_models/2_bin_mem', bin_mem)
            sw.summ_occ('4_models/3_bin_mem_clean', bin_mem_clean)
            sw.summ_lrtlist_bev(
                '4_models/lrtlist_mem',
                occ_mem0,
                lrtlist_e,
                scorelist_e,
                torch.ones_like(lrtlist_e[:,:,0]).long(),
                vox_util,
                already_mem=False)
            
            sw.summ_soft_seg_thr('5_objects/0_connlist1', torch.max(connlist1, dim=3)[0])
            sw.summ_soft_seg_thr('5_objects/1_connlist_2d', connlist_2d)
            sw.summ_soft_seg_thr('5_objects/1_connlist_3d', torch.max(connlist_3d, dim=3)[0])
            sw.summ_lrtlist_bev(
                '5_objects/lrtlist_mem',
                occ_mem0,
                lrtlist_cam,
                scorelist,
                tidlist,
                vox_util,
                already_mem=False)
            sw.summ_lrtlist(
                '5_objects/lrtlist_cam',
                rgb_cam0,
                lrtlist_cam,
                scorelist,
                tidlist,
                pix_T_cam)

    # ------------------
    # export pseudolabels for the next stage
    # ------------------
    if torch.sum(connlist_3d):
        if export_npzs:
            npz_fn = os.path.join(npz_dir, '%s.npz' % (step_name))
            seg_xyz_cam_list_py = [xyz[0].detach().cpu().numpy() for xyz in seg_xyz_cam_list]
            np.savez(npz_fn,
                     rgb_cam=rgb_cam0[0].detach().cpu().numpy(), 
                     xyz_cam=xyz_cam0[0].detach().cpu().numpy(),
                     pix_T_cam=pix_T_cam[0].detach().cpu().numpy(),
                     seg_xyz_cam_list=seg_xyz_cam_list_py,
                     xyz_cam_i=xyz_cam0_i[0].detach().cpu().numpy(), # egomotion inliers
                     lrtlist_cam=lrtlist_cam[0].detach().cpu().numpy(),
            )
            print('saved %s' % npz_fn)
            
    return total_loss, metrics

    
def main(
        init_dir_2d,
        init_dir_3d,
        output_name='temp',
        exp_name='ke00',
        max_iters=4000,
        log_freq=50,
        export_npzs=False,
        export_vis=False,
        force_vis=False,
        shuffle=False,
        seq_name='any',
        dset='t',
        sort=True,
        skip_to=0,
):
    # this file implements the 2d-3d ensemble, for E steps past the first 
    
    ## autogen a name
    model_name = "%s" % exp_name
    if export_npzs:
        model_name += "_export_%s" % output_name
    import datetime
    model_date = datetime.datetime.now().strftime('%H:%M:%S')
    model_name = model_name + '_' + model_date
    print('model_name', model_name)
    
    ckpt_dir = 'checkpoints/%s' % model_name
    log_dir = 'logs_tcr_kitti_ensemble2'
    writer_t = SummaryWriter(log_dir + '/' + model_name + '/t', max_queue=10, flush_secs=60)

    B = 1
    train_dataset = SimpleKittiDataset(S=2,shuffle=shuffle,dset=dset,kitti_data_seqlen=2,seq_name=seq_name,sort=sort)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=B,
        shuffle=shuffle,
        num_workers=4,
        drop_last=True)
    train_iterloader = iter(train_dataloader)
    
    global_step = 0

    raft = nets.raftnet.RaftNet(ckpt_name='../RAFT/models/raft-sintel.pth').cuda()
    requires_grad(raft.parameters(), False)
    raft.eval()

    stride = 4
    model_3d = nets.centernet2d.Centernet2d(Y=Y, K=20, show_thresh=0.5, stride=stride).cuda()
    parameters = list(model_3d.parameters())
    _ = saverloader.load(init_dir_3d, model_3d)
    requires_grad(parameters, False)
    model_3d.eval()

    model_2d = nets.seg2dnet.Seg2dNet(num_classes=1).to(device).eval()
    parameters = list(model_2d.parameters())
    _ = saverloader.load(init_dir_2d, model_2d)
    requires_grad(parameters, False)
    model_2d.eval()

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
            step_name = '%s_%s_%04d' % (seq_name, output_name, global_step)
            _, metrics = run_model(B, raft, model_2d, model_3d, sample, sw_t,
                                   export_vis=export_vis, export_npzs=export_npzs, step_name=step_name,
                                   force_vis=force_vis)
            
            if metrics['maps_bev'] is not None:
                for i,m in enumerate(metrics['maps_bev']):
                    map_bev_pools[i].update([m])
                for i,m in enumerate(metrics['maps_per']):
                    map_per_pools[i].update([m])

            for i in range(len(iou_thresholds)):
                sw_t.summ_scalar('map_bev/iou_%.1f' % iou_thresholds[i], map_bev_pools[i].mean())
                sw_t.summ_scalar('map_per/iou_%.1f' % iou_thresholds[i], map_per_pools[i].mean())

        iter_time = time.time()-iter_start_time
        print('%s; step %06d/%d; rtime %.2f; itime %.2f; map@%.1f %.2f; map@%.1f %.2f; map@%.1f %.2f' % (
            model_name, global_step, max_iters, read_time, iter_time,
            iou_thresholds[0], map_bev_pools[0].mean(),
            iou_thresholds[2], map_bev_pools[2].mean(),
            iou_thresholds[4], map_bev_pools[4].mean()))

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
