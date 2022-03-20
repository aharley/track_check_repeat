import time
import argparse
import numpy as np
import timeit
import imageio
# import tensorflow as tf
# import scipy.misc
import io
import os
import math
# import matplotlib
# from PIL import Image
# matplotlib.use('Agg') # suppress plot showing
# import sys
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
import cv2
import saverloader
import skimage.morphology

from fire import Fire
import nets.raftnet

def requires_grad(parameters, flag=True):
    for p in parameters:
        p.requires_grad = flag

# import utils.eval
# import utils.py
# import utils.box
# import utils.misc
# import utils.improc
# import utils.vox
# import utils.grouping
# import random
# import glob

from utils.basic import print_, print_stats

from simplekittidataset import SimpleKittiDataset

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from tensorboardX import SummaryWriter

import torch.nn.functional as F

device = 'cuda'
patch_size = 8
random.seed(125)
np.random.seed(125)

iou_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

scene_centroid_x = 0.0
scene_centroid_y = 1.0
scene_centroid_z = 20.0

scene_centroid = np.array([scene_centroid_x,
                           scene_centroid_y,
                           scene_centroid_z]).reshape([1, 3])
scene_centroid = torch.from_numpy(scene_centroid).float().cuda()

XMIN, XMAX = -16, 16
ZMIN, ZMAX = -16, 16
YMIN, YMAX = -1, 1
bounds = (XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX)

Z, Y, X = 256, 16, 256
Z2, Y2, X2 = Z//2, Y//2, X//2
Z4, Y4, X4 = Z//4, Y//4, X//4
Z8, Y8, X8 = Z//8, Y//8, X//8

vis_dir = './tcr_vis'
npz_dir = './tcr_pseudo'
utils.basic.mkdir(npz_dir)
utils.basic.mkdir(vis_dir)

def save_vis(rgb, name):
    rgb = rgb.cpu().numpy()[0].transpose(1,2,0) # H x W x 3
    vis_fn = os.path.join(vis_dir, '%s.png' % (name))
    imageio.imwrite(vis_fn, rgb)
    print('saved %s' % vis_fn)

def run_model(raft, d, sw, export_vis=False, export_npzs=False, step_name='temp', vis_early=False):
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

    rgb_camXs = rgb_camXs.cuda()
    xyz_veloXs = xyz_veloXs.cuda()
    pix_T_rects = pix_T_rects.cuda()
    rect_T_cams = rect_T_cams.cuda()
    cam_T_velos = cam_T_velos.cuda()

    B, S, C, H, W = rgb_camXs.shape
    B, S, V, D = xyz_veloXs.shape
    assert(B==1)
    assert(D==3)

    __p = lambda x: utils.basic.pack_seqdim(x, B)
    __u = lambda x: utils.basic.unpack_seqdim(x, B)

    xyz_cams = __u(utils.geom.apply_4x4(__p(cam_T_velos), __p(xyz_veloXs)))
    xyz_rects = __u(utils.geom.apply_4x4(__p(rect_T_cams), __p(xyz_cams)))

    rgb_camXs = utils.improc.preprocess_color(rgb_camXs)
    rgb_cam = rgb_camXs[:,0]
    if vis_early and sw is not None and sw.save_this:
        sw.summ_rgbs('0_inputs/rgb_camXs', rgb_camXs.unbind(1))
        sw.summ_rgb('0_inputs/rgb_0', rgb_camXs[:,0])
        sw.summ_rgb('0_inputs/rgb_1', rgb_camXs[:,1])

    # ------------------
    # get 2d flow, and perform cycle-consistency checks
    # ------------------
    
    flow_01, _ = raft(rgb_camXs[:,0], rgb_camXs[:,1], iters=8)
    flow_10, _ = raft(rgb_camXs[:,1], rgb_camXs[:,0], iters=8)
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
    
    if vis_early and sw is not None and sw.save_this:
        sw.summ_flow('1_flow/flow_01', flow_01, clip=200)
        sw.summ_flow('1_flow/flow_10', flow_10, clip=200)
        sw.summ_oned('1_flow/rely_0', rely_0, norm=False)
        sw.summ_oned('1_flow/rely_1', rely_1, norm=False)

    # ------------------
    # get egomotion flow
    # ------------------
    
    # slightly abuse the cam/rect notation, and use rect==cam
    pix_T_cam = pix_T_rects[:,0]
    xyz_cam0 = xyz_rects[:,0]
    xyz_cam1 = xyz_rects[:,1]

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
        assert_cube=True)
    occ_mem0 = vox_util.voxelize_xyz(xyz_cam0, Z, Y, X)
    occ_mem1 = vox_util.voxelize_xyz(xyz_cam1, Z, Y, X)
    occ_mem0_i = vox_util.voxelize_xyz(xyz_cam0_i, Z, Y, X)
    occ_mem1_i = vox_util.voxelize_xyz(xyz_cam1_i, Z, Y, X)
    occ_mem0_o = vox_util.voxelize_xyz(xyz_cam0_o, Z, Y, X)
    occ_mem1_o = vox_util.voxelize_xyz(xyz_cam1_o, Z, Y, X)
    explained_by_ego = utils.improc.dilate3d(occ_mem0_i)

    max_align_error = torch.max(align_error)
    # print_('max_align_error', max_align_error)

    if torch.max(max_align_error) > 0.1:
        # print('egomotion estimation failed: max_align_error', max_align_error)
        
        # return early, without trying to find objects
        if export_vis:
            grey = torch.mean(rgb_camXs[:,0], dim=1, keepdim=True).repeat(1, 3, 1, 1)
            save_vis(utils.improc.back2color((grey+0.5)*0.5-0.5), step_name)

        return total_loss, metrics

    # stabilize things according to the estimated motion, just for vis 
    xyz_cam1_stab = utils.geom.apply_4x4(cam0_T_cam1, xyz_cam1)
    occ_mem1_stab = vox_util.voxelize_xyz(xyz_cam1_stab, Z, Y, X)
    if vis_early and sw is not None and sw.save_this:
        sw.summ_occs('2_ego/occ_mem01_orig', [occ_mem0, occ_mem1])
        sw.summ_occs('2_ego/occ_mem01_stab', [occ_mem0, occ_mem1_stab], frame_ids=[max_align_error.item()]*2)

    # ------------------
    # find evidence of independent motion
    # ------------------

    # convert the pointcloud into a depth map
    depth_cam0, valid_cam0 = utils.geom.create_depth_image(pix_T_rects[:,0], xyz_cam0, H, W, offset_amount=0)
    # depth_cam1, valid_cam1 = utils.geom.create_depth_image(pix_T_rects[:,1], xyz_cam1, H, W, offset_amount=0)
    if vis_early and sw is not None and sw.save_this:
        sw.summ_oned('2_ego/depth_cam0', depth_cam0, max_val=30)
        # sw.summ_oned('2_ego/depth_cam1', depth_cam1, max_val=30)

    # convert egomotion+depth into a flow field, and take a 2d difference
    flow_01_ego = utils.geom.depthrt2flow(depth_cam0, cam1_T_cam0, pix_T_cam) * valid_cam0
    flow_dist = torch.norm(flow_01_ego - flow_01*valid_cam0, dim=1, keepdim=True) * rely_0 * valid_cam0
    cos = nn.CosineSimilarity(dim=1)
    flow_cos = cos(flow_01_ego, flow_01)

    xy = utils.basic.gridcloud2d(B, H, W)
    xyd = torch.cat([xy, depth_cam0.reshape(1, -1, 1)], dim=2)

    # use thresholds on angle and distance, so that both thresholds can be loose
    xyd_keepA = xyd[:,flow_cos.reshape(-1) < 0.8] # do not aim in the exact same dir
    xyz_keepA = utils.geom.xyd2pointcloud(xyd_keepA, pix_T_cam)
    occ_keepA = vox_util.voxelize_xyz(xyz_keepA, Z, Y, X) * (1-explained_by_ego) # discard static stuff
    
    xyd_keepB = xyd[:,(flow_dist.reshape(-1) > 8)]
    xyz_keepB = utils.geom.xyd2pointcloud(xyd_keepB, pix_T_cam)
    occ_keepB = vox_util.voxelize_xyz(xyz_keepB, Z, Y, X) * (1-explained_by_ego) # discard static stuff

    # ------------------
    # cluster regions into objects
    # ------------------
    
    # next is connected components, and center-surround
    N = 32 # max number of regions
    vis_mem0 = vox_util.convert_xyz_to_visibility(xyz_cam0, Z, Y, X, target_T_given=None)
    vis_mem1_stab = vox_util.convert_xyz_to_visibility(xyz_cam1, Z, Y, X, target_T_given=cam0_T_cam1)
    vis_both = vis_mem0 * vis_mem1_stab
    
    boxlist_mem, scorelist, tidlist, connlist = utils.misc.get_any_boxes_from_binary(
        occ_keepB.squeeze(1), N, min_voxels=128, min_side=1, count_mask=occ_keepB*occ_keepA)
    connlist = connlist * occ_keepB * occ_keepA
    # print_('scorelist', scorelist)

    if torch.sum(connlist) > 0:
        if vis_early and sw is not None and sw.save_this:
            sw.summ_soft_seg_thr('4_objects/connlist', torch.max(connlist, dim=3)[0])

    # check that the 2d projection has some minimum area,
    # and check center-surround

    # lrtlist_mem = utils.geom.convert_boxlist_to_lrtlist(boxlist_mem)
    # lrtlist_cam = vox_util.apply_ref_T_mem_to_lrtlist(lrtlist_mem, Z, Y, X)

    # masklist_1 = vox_util.assemble_padded_obj_masklist(
    #     lrtlist_cam, scorelist, Z, Y, X, coeff=1.0)
    # masklist_2 = vox_util.assemble_padded_obj_masklist(
    #     lrtlist_cam, scorelist, Z, Y, X, coeff=1.0)
    # masklist_3 = vox_util.assemble_padded_obj_masklist(
    #     lrtlist_cam, scorelist, Z, Y, X, coeff=1.0, additive_coeff=1.0)
    # # these are B x N x 1 x Z2 x Y2 x X2

    # # the idea of a center-surround feature is:
    # # there should be stuff in the center but not in the surround
    # # so, i need the density of the center
    # # and the density of the surround
    # # then, the score is center minus surround
    # center_mask = (masklist_1).squeeze(2)
    # surround_mask = (masklist_3-masklist_2).squeeze(2)
    # # these are B x N x Z x Y x X

    # # # occ_ = utils.improc.dilate3d(occ_keepA, times=2).clamp(0,1) # B, 1, Z, Y, X
    # # occ_ = (occ_keepA + occ_keepB + occ_mem0*0.1).clamp(0,1)
    # # occ_ = (occ_keepA + occ_keepB + occ_mem0*0.1).clamp(0,1)
    # # occ_ = (occ_keepA + occ_keepB).clamp(0,1)
    # occ_ = (occ_keepA + occ_keepB).clamp(0,1)
    # # occ_ = occ_mem0.clamp(0,1)
    # # occ_ = utils.improc.dilate3d(occ_keepA, times=2).clamp(0,1) # B, 1, Z, Y, X
    # occ_ = occ_.repeat(1, N, 1, 1, 1)
    # # center_ = utils.basic.reduce_masked_mean(occ_, center_mask*occ_mem0, dim=[2,3,4]) # B, N
    # # surround_ = utils.basic.reduce_masked_mean(occ_, surround_mask*occ_mem0, dim=[2,3,4]) # B, N

    # # center_ = utils.basic.reduce_masked_mean(occ_, center_mask*occ_mem0, dim=[2,3,4]) # B, N
    # surround_ = torch.sum(occ_*surround_mask*occ_mem0, dim=[2,3,4]) # B, N
    
    # print_('surround_', surround_)
    
    # # scorelist = center_ - surround_
    # # # scorelist is B x N, in the range [0,1], because both are in [0,1]
    # scorelist = 1 - surround_
    
    # # scorelist = (center_ - surround_)/center_
    # # scorelist is B x N, in the range [0,1]
    # print_('surr scorelist', scorelist)

    # # cs_thr = 0.0
    # cs_thr = -10000.0
    
    conn_seg = []
    seg_xyz_cam_list = []
    xyz_mem = utils.basic.gridcloud3d(B, Z, Y, X)
    new_boxlist_mem = []
    # new_scorelist = []
    
    if torch.sum(connlist) > 0:
        print_stats('scorelist', scorelist)

    for n1 in range(N):
        conn1 = connlist[:,n1]
        xyz_mem_here = xyz_mem[:,conn1.reshape(-1) > 0]
        xyz_cam_here = vox_util.Mem2Ref(xyz_mem_here, Z, Y, X)
        xy_pix = utils.geom.apply_pix_T_cam(pix_T_cam, xyz_cam_here)
        mask = utils.improc.xy2mask(xy_pix, H, W, norm=False)
        mask = mask[0,0].cpu().numpy()
        # if np.sum(mask) > 4 and scorelist[0,n1] > cs_thr:
        if np.sum(mask) > 4 and scorelist[0,n1]:
            seg_xyz_cam_list.append(xyz_cam_here)
            new_boxlist_mem.append(boxlist_mem[:,n1])
            # new_scorelist.append(scorelist[:,n1])
            # print('validated conn segment %d; sum(conn); np.sum(mask)' % n1, torch.sum(conn1).item(), np.sum(mask))
            # close the hull
            mask = skimage.morphology.convex_hull.convex_hull_image(mask)
            conn_seg.append(torch.from_numpy(mask.reshape(1, H, W)).float().cuda())

    if len(conn_seg):
        print('found %d objects' % len(conn_seg))
        conn_seg = torch.stack(conn_seg, dim=1)
        boxlist_mem = torch.stack(new_boxlist_mem, dim=1)
        # scorelist = torch.stack(new_scorelist, dim=1)
        # scorelist = scorelist[:,scorelist[0]>cs_thr]
        print('boxlist_mem', boxlist_mem.shape)
        # print('scorelist', scorelist.shape)
    else:
        conn_seg = torch.zeros_like(occ_mem0)

    # if torch.sum(connlist) > 0:
    #     # ------------------
    #     # measure center-surround, to discard half-objects
    #     # ------------------

    #     lrtlist_mem = utils.geom.convert_boxlist_to_lrtlist(boxlist_mem)
    #     lrtlist_cam = vox_util.apply_ref_T_mem_to_lrtlist(lrtlist_mem, Z, Y, X)

    #     masklist_1 = vox_util.assemble_padded_obj_masklist(
    #         lrtlist_cam, scorelist, Z, Y, X, coeff=1.0)
    #     masklist_2 = vox_util.assemble_padded_obj_masklist(
    #         lrtlist_cam, scorelist, Z, Y, X, coeff=1.0)
    #     masklist_3 = vox_util.assemble_padded_obj_masklist(
    #         lrtlist_cam, scorelist, Z, Y, X, coeff=1.2, additive_coeff=0.5)
    #     # these are B x N x 1 x Z2 x Y2 x X2

    #     # the idea of a center-surround feature is:
    #     # there should be stuff in the center but not in the surround
    #     # so, i need the density of the center
    #     # and the density of the surround
    #     # then, the score is center minus surround
    #     center_mask = (masklist_1).squeeze(2)
    #     surround_mask = (masklist_3-masklist_2).squeeze(2)
    #     # these are B x N x Z x Y x X

    #     occ_ = utils.improc.dilate3d(occ_mem0, times=2).clamp(0,1) # B, 1, Z, Y, X
    #     occ_ = occ_.repeat(1, N, 1, 1, 1)
    #     center_ = utils.basic.reduce_masked_mean(occ_, center_mask, dim=[2,3,4]) # B, N
    #     print_('center_', center_)
    #     surround_ = utils.basic.reduce_masked_mean(occ_, surround_mask, dim=[2,3,4]) # B, N
    #     scorelist = (center_ - surround_)/center_
    #     # scorelist is B x N, in the range [0,1]
    #     print_('surr scorelist', scorelist)

    #     cs_thr = 0.4
    #     inds = scorelist[0] > cs_thr
    #     boxlist_mem = boxlist_mem[:,inds]
    #     new_seg_xyz_cam_list = []
    #     for n1 in range(N):
    #         if scorelist[0,n1] > cs_thr:
    #             conn1 = connlist[:,n1]
    #             xyz_mem_here = xyz_mem[:,conn1.reshape(-1) > 0]
    #             xyz_cam_here = vox_util.Mem2Ref(xyz_mem_here, Z, Y, X)
    #             xy_pix = utils.geom.apply_pix_T_cam(pix_T_cam, xyz_cam_here)
    #             mask = utils.improc.xy2mask(xy_pix, H, W, norm=False)
    #             mask = mask[0,0].cpu().numpy()
    #             if np.sum(mask) > 4:
    #                 seg_xyz_cam_list.append(xyz_cam_here)
    #                 # print('validated conn segment %d; sum(conn); np.sum(mask)' % n1, torch.sum(conn1).item(), np.sum(mask))
    #                 # close the hull
    #                 mask = skimage.morphology.convex_hull.convex_hull_image(mask)
    #                 conn_seg.append(torch.from_numpy(mask.reshape(1, H, W)).float().cuda())

    # ------------------
    # evaluate mAP
    # ------------------
    boxlist_rect_g = d['boxlist_camXs'].float().cuda()[:,0] # note this is already in rectified coords
    tidlist_g = d['tidlist_s'].long().cuda()[:,0]
    scorelist_g = d['scorelist_s'].float().cuda()[:,0]
    lrtlist_rect_g = utils.geom.convert_boxlist_to_lrtlist(boxlist_rect_g)
    scorelist_g = utils.misc.rescore_lrtlist_with_inbound(lrtlist_rect_g, scorelist_g, Z, Y, X, vox_util, pad=0.0)

    lrtlist_mem = utils.geom.convert_boxlist_to_lrtlist(boxlist_mem)
    lrtlist_cam = vox_util.apply_ref_T_mem_to_lrtlist(lrtlist_mem, Z, Y, X)
    lrtlist_rect_e = utils.geom.apply_4x4_to_lrtlist(rect_T_cams[:,0], lrtlist_cam)
    rylist = boxlist_mem[:,:,7]

    lrtlist_e, lrtlist_g, scorelist_e, scorelist_g = utils.eval.drop_invalid_lrts(
        lrtlist_rect_e, lrtlist_rect_g, scorelist, scorelist_g)
    boxlist_e = utils.geom.get_boxlist2d_from_lrtlist(pix_T_rects[:,0], lrtlist_e, H, W)
    boxlist_g = utils.geom.get_boxlist2d_from_lrtlist(pix_T_rects[:,0], lrtlist_g, H, W)

    rylist = rylist[:,:boxlist_e.shape[1]]

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

        if vis_early and sw is not None and sw.save_this:
            sw.summ_lrtlist('5_boxes/lrtlist_e', rgb_cam, lrtlist_e, ious_bev, 2*torch.ones_like(scorelist_e).long(), pix_T_rects[:,0], frame_id=maps_bev[0])
            sw.summ_boxlist2d('5_boxes/boxlist_e', rgb_cam, boxlist_e, ious_per, 2*torch.ones_like(scorelist_e).long(), frame_id=maps_per[0])
            sw.summ_lrtlist('5_boxes/lrtlist_g', rgb_cam, lrtlist_g, scorelist_g, 2*torch.ones_like(scorelist_g).long(), pix_T_rects[:,0])
            sw.summ_boxlist2d('5_boxes/boxlist_g', rgb_cam, boxlist_g, scorelist_g, 2*torch.ones_like(scorelist_g).long())
            
    elif torch.sum(scorelist_g==0) and torch.sum(scorelist_e==0):
        # mAP unaffected
        metrics['maps_bev'] = None
        metrics['maps_per'] = None
        # ious_bev = torch.ones_like(lrtlist_e[:,:,0])
        # ious_per = torch.ones_like(lrtlist_e[:,:,0])
        ious_bev = None
        ious_per = None
        # maps_bev = None
    else:
        ious_bev = None
        ious_per = None
        # ious_bev = torch.ones_like(lrtlist_e[:,:,0])
        # ious_per = torch.ones_like(lrtlist_e[:,:,0])

    # ------------------
    # export pseudolabels for the next stage
    # ------------------
    if torch.sum(conn_seg):
        if export_npzs:
            npz_fn = os.path.join(npz_dir, '%s.npz' % (step_name))
            seg_xyz_cam_list_py = [xyz[0].detach().cpu().numpy() for xyz in seg_xyz_cam_list]
            np.savez(npz_fn,
                     rgb_cam=rgb_camXs[0,0].detach().cpu().numpy(), 
                     xyz_cam=xyz_cam0[0].detach().cpu().numpy(),
                     pix_T_cam=pix_T_cam[0].detach().cpu().numpy(),
                     seg_xyz_cam_list=seg_xyz_cam_list_py,
                     xyz_cam_i=xyz_cam0_i[0].detach().cpu().numpy(), # egomotion inliers
                     lrtlist_cam=lrtlist_e[0].detach().cpu().numpy(), # valid rectified 3d boxes
                     # scorelist=scorelist[0].detach().cpu().numpy(), 
                     # rylist=rylist[0].detach().cpu().numpy(), 
            )
            print('saved %s' % npz_fn)
    
    # ------------------
    # vis everything 
    # ------------------
    if (not vis_early) and sw is not None and sw.save_this:
        if torch.sum(conn_seg):

            sw.summ_rgbs('0_inputs/rgb_camXs', rgb_camXs.unbind(1))
            sw.summ_rgb('0_inputs/rgb_0', rgb_camXs[:,0])
            sw.summ_rgb('0_inputs/rgb_1', rgb_camXs[:,1])
            
            sw.summ_flow('1_flow/flow_01', flow_01, clip=200)
            sw.summ_flow('1_flow/flow_10', flow_10, clip=200)
            sw.summ_oned('1_flow/rely_0', rely_0, norm=False)
            sw.summ_oned('1_flow/rely_1', rely_1, norm=False)

            sw.summ_occs('2_ego/occ_mem01_orig', [occ_mem0, occ_mem1])
            sw.summ_occs('2_ego/occ_mem01_stab', [occ_mem0, occ_mem1_stab], frame_ids=[max_align_error.item()]*2)

            sw.summ_flow('3_diff/flow_01_ego', flow_01_ego, clip=20)
            sw.summ_flow('3_diff/flow_01_raw_masked', flow_01, clip=20)
            # sw.summ_flow('3_diff/flow_01_raw_masked', flow_01*valid_cam0, clip=20)
            sw.summ_oned('3_diff/flow_dist', flow_dist)
            sw.summ_oned('3_diff/flow_dist_thr1', (flow_dist > 1).float())
            sw.summ_oned('3_diff/flow_dist_thr2', (flow_dist > 2).float())
            sw.summ_oned('3_diff/flow_dist_thr4', (flow_dist > 4).float())
            sw.summ_oned('3_diff/flow_dist_thr8', (flow_dist > 8).float())
            sw.summ_occ('3_diff/occ_keepA', occ_keepA, bev=True)
            # sw.summ_occ('3_diff/occ_keepA_fat', occ_keepA_fat, bev=True)
            sw.summ_occ('3_diff/occ_keepB', occ_keepB, bev=True)
            # sw.summ_occ('3_diff/occ_keep2', occ_keep2, bev=True)

            connlist_bev = torch.max(connlist, dim=3)[0]
            sw.summ_soft_seg_thr('4_objects/connlist_merged_bev', connlist_bev, frame_id=torch.sum(connlist))
            
            conn_vis = sw.summ_soft_seg_thr('', conn_seg, only_return=True)
            grey = torch.mean(rgb_camXs[:,0], dim=1, keepdim=True).repeat(1, 3, 1, 1)
            seg = utils.improc.preprocess_color(conn_vis).cuda()
            sw.summ_rgb('4_objects/conn_on_rgb', (grey+seg)/2.0, frame_id=torch.sum(conn_seg))

            if ious_bev is not None:
                sw.summ_lrtlist('5_boxes/lrtlist_e', rgb_cam, lrtlist_e, ious_bev, 2*torch.ones_like(scorelist_e).long(), pix_T_rects[:,0], frame_id=maps_bev[0])
                sw.summ_boxlist2d('5_boxes/boxlist_e', rgb_cam, boxlist_e, ious_per, 2*torch.ones_like(scorelist_e).long(), frame_id=maps_per[0])
                sw.summ_lrtlist('5_boxes/lrtlist_g', rgb_cam, lrtlist_g, scorelist_g, 2*torch.ones_like(scorelist_g).long(), pix_T_rects[:,0])
                sw.summ_boxlist2d('5_boxes/boxlist_g', rgb_cam, boxlist_g, scorelist_g, 2*torch.ones_like(scorelist_g).long())
            
            if export_vis:
                seg_vis = sw.summ_rgb('', (grey+seg)/2.0, only_return=True)
                save_vis(seg_vis, step_name)
                
        else: # no object found
            if export_vis:
                grey = torch.mean(rgb_camXs[:,0], dim=1, keepdim=True).repeat(1, 3, 1, 1)
                save_vis(utils.improc.back2color((grey+0.5)*0.5-0.5), step_name)
                
    return total_loss, metrics

    
def main(
        output_name,
        exp_name='kd00',
        max_iters=4000,
        log_freq=1, # note we only log if we discovered something, so log1 is OK
        export_npzs=True,
        export_vis=False,
        shuffle=False,
        seq_name='any',
        sort=True,
        skip_to=0,
        dset='t', # let's collect pseudolabels from the "training" subset
):
    # this file implements the first E step

    ## autogen a name
    model_name = "%s" % output_name
    model_name += "_%s" % exp_name
    import datetime
    model_date = datetime.datetime.now().strftime('%H:%M:%S')
    model_name = model_name + '_' + model_date
    print('model_name', model_name)
    
    ckpt_dir = 'checkpoints/%s' % model_name
    log_dir = 'logs_tcr_kitti_discover'
    writer_t = SummaryWriter(log_dir + '/' + model_name + '/t', max_queue=10, flush_secs=60)

    B = 1
    S = 2
    train_dataset = SimpleKittiDataset(S=2,shuffle=shuffle,dset=dset,kitti_data_seqlen=2,seq_name=seq_name,sort=sort)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=B,
        shuffle=shuffle,
        num_workers=4,
        drop_last=True)
    train_iterloader = iter(train_dataloader)
    
    global_step = 0

    vox_util = utils.vox.Vox_util(
        Z, Y, X,
        scene_centroid=scene_centroid,
        bounds=bounds,
        assert_cube=True)
    
    stride = 8
    H, W = 128, 416
    
    raft = nets.raftnet.RaftNet(ckpt_name='../RAFT/models/raft-sintel.pth').cuda()
    requires_grad(raft.parameters(), False)
    raft.eval()

    n_pool = max_iters*2
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
            total_loss, metrics = run_model(raft, sample, sw_t, export_vis=export_vis, export_npzs=export_npzs, step_name=step_name)

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

        del total_loss
        
    writer_t.close()
            

if __name__ == '__main__':
    Fire(main)
