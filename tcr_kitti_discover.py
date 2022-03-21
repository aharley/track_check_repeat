import time
import numpy as np
import imageio
import os
import saverloader
import skimage.morphology

from fire import Fire

import nets.raftnet

import utils.basic
import utils.vox
import utils.improc
import utils.misc
import utils.samp
import utils.improc
from utils.basic import print_, print_stats

from simplekittidataset import SimpleKittiDataset

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from tensorboardX import SummaryWriter

import torch.nn.functional as F

import random
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

def requires_grad(parameters, flag=True):
    for p in parameters:
        p.requires_grad = flag

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
    cam1_T_cam0, align_error, corresp_tuple = utils.misc.get_cycle_consistent_transform(
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

    max_align_error = torch.max(align_error)
    # print_('max_align_error', max_align_error)

    if torch.max(max_align_error) > 0.1:
        # print('egomotion estimation failed: max_align_error', max_align_error)
        
        # return early, without trying to find objects
        if export_vis:
            grey = torch.mean(rgb_cam0, dim=1, keepdim=True).repeat(1, 3, 1, 1)
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
    
    N = 16 # max number of regions
    
    boxlist_mem, scorelist, tidlist, connlist = utils.misc.get_any_boxes_from_binary(
        occ_keepB.squeeze(1), N, min_voxels=128, min_side=1, count_mask=occ_keepB*occ_keepA)
    connlist = connlist * occ_keepB * occ_keepA

    # now the idea is:
    # for each object in connlist,
    # check that the 2d projection has some minimum area.
    min_area = 4
    conn_seg = []
    seg_xyz_cam_list = []
    xyz_mem = utils.basic.gridcloud3d(B, Z, Y, X)
    new_boxlist_mem = []
    for n1 in range(N):
        conn1 = connlist[:,n1]
        xyz_mem_here = xyz_mem[:,conn1.reshape(-1) > 0]
        xyz_cam_here = vox_util.Mem2Ref(xyz_mem_here, Z, Y, X)
        xy_pix = utils.geom.apply_pix_T_cam(pix_T_cam, xyz_cam_here)
        mask = utils.improc.xy2mask(xy_pix, H, W, norm=False)
        mask = mask[0,0].cpu().numpy()
        if np.sum(mask) > min_area and scorelist[0,n1]:
            seg_xyz_cam_list.append(xyz_cam_here)
            new_boxlist_mem.append(boxlist_mem[:,n1])
            # close the hull
            mask = skimage.morphology.convex_hull.convex_hull_image(mask)
            conn_seg.append(torch.from_numpy(mask.reshape(1, H, W)).float().cuda())

    if len(conn_seg):
        print('found %d objects' % len(conn_seg))
        conn_seg = torch.stack(conn_seg, dim=1)
        boxlist_mem = torch.stack(new_boxlist_mem, dim=1)
        print('boxlist_mem', boxlist_mem.shape)
    else:
        conn_seg = torch.zeros_like(occ_mem0)

    lrtlist_mem = utils.geom.convert_boxlist_to_lrtlist(boxlist_mem)
    lrtlist_cam = vox_util.apply_ref_T_mem_to_lrtlist(lrtlist_mem, Z, Y, X)
    scorelist = torch.ones_like(lrtlist_cam[:,:,0])
    tidlist = torch.ones_like(lrtlist_cam[:,:,0]).long()

    # ------------------
    # export pseudolabels for the next stage
    # ------------------
    if torch.sum(conn_seg):
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
    
    # ------------------
    # vis everything 
    # ------------------
    if (not vis_early) and sw is not None and sw.save_this:
        if torch.sum(conn_seg):

            sw.summ_rgbs('0_inputs/rgb_cams', rgb_cams.unbind(1))
            sw.summ_rgb('0_inputs/rgb_0', rgb_cam0)
            sw.summ_rgb('0_inputs/rgb_1', rgb_cam1)
            
            sw.summ_flow('1_flow/flow_01', flow_01, clip=200)
            sw.summ_flow('1_flow/flow_10', flow_10, clip=200)
            sw.summ_oned('1_flow/rely_0', rely_0, norm=False)
            sw.summ_oned('1_flow/rely_1', rely_1, norm=False)

            sw.summ_occs('2_ego/occ_mem01_orig', [occ_mem0, occ_mem1])
            sw.summ_occs('2_ego/occ_mem01_stab', [occ_mem0, occ_mem1_stab], frame_ids=[max_align_error.item()]*2)

            sw.summ_flow('3_diff/flow_01_ego', flow_01_ego, clip=20)
            sw.summ_flow('3_diff/flow_01_raw_masked', flow_01, clip=20)
            sw.summ_oned('3_diff/flow_dist', flow_dist)
            sw.summ_oned('3_diff/flow_dist_thr1', (flow_dist > 1).float())
            sw.summ_oned('3_diff/flow_dist_thr2', (flow_dist > 2).float())
            sw.summ_oned('3_diff/flow_dist_thr4', (flow_dist > 4).float())
            sw.summ_oned('3_diff/flow_dist_thr8', (flow_dist > 8).float())
            sw.summ_occ('3_diff/occ_keepA', occ_keepA, bev=True)
            sw.summ_occ('3_diff/occ_keepB', occ_keepB, bev=True)

            connlist_bev = torch.max(connlist, dim=3)[0]
            sw.summ_soft_seg_thr('4_objects/connlist_merged_bev', connlist_bev, frame_id=torch.sum(connlist))
            
            conn_vis = sw.summ_soft_seg_thr('', conn_seg, only_return=True)
            grey = torch.mean(rgb_cam0, dim=1, keepdim=True).repeat(1, 3, 1, 1)
            seg = utils.improc.preprocess_color(conn_vis).cuda()
            sw.summ_rgb('4_objects/conn_on_rgb', (grey+seg)/2.0, frame_id=torch.sum(conn_seg))

            sw.summ_lrtlist('5_boxes/lrtlist_cam', rgb_cam0, lrtlist_cam, scorelist, tidlist, pix_T_cam)
            sw.summ_lrtlist_bev(
                '5_boxes/lrtlist_mem',
                occ_mem0,
                lrtlist_cam,
                scorelist,
                tidlist,
                vox_util,
                already_mem=False)
            
            if export_vis:
                seg_vis = sw.summ_rgb('', (grey+seg)/2.0, only_return=True)
                save_vis(seg_vis, step_name)
                
        else: # no object found
            if export_vis:
                grey = torch.mean(rgb_cam0, dim=1, keepdim=True).repeat(1, 3, 1, 1)
                save_vis(utils.improc.back2color((grey+0.5)*0.5-0.5), step_name)
                
    return total_loss, metrics

    
def main(
        output_name,
        exp_name='debug',
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
    model_name = "discover"
    model_name += "_%s" % output_name
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

        iter_time = time.time()-iter_start_time
        print('%s; step %06d/%d; rtime %.2f; itime %.2f' % (
            model_name, global_step, max_iters, read_time, iter_time))

        del total_loss
        
    writer_t.close()
            

if __name__ == '__main__':
    Fire(main)
