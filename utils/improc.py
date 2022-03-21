import torch
import numpy as np
import torch.nn.functional as F
import utils.basic
import cv2
import matplotlib
from itertools import combinations

# color conversion libs, for flow vis
from skimage.color import (
    rgb2lab, rgb2yuv, rgb2ycbcr, lab2rgb, yuv2rgb, ycbcr2rgb,
    rgb2hsv, hsv2rgb, rgb2xyz, xyz2rgb, rgb2hed, hed2rgb)

def _convert(input_, type_):
    return {
        'float': input_.float(),
        'double': input_.double(),
    }.get(type_, input_)

def _generic_transform_sk_4d(transform, in_type='', out_type=''):
    def apply_transform(input_):
        to_squeeze = (input_.dim() == 3)
        device = input_.device
        input_ = input_.cpu()
        input_ = _convert(input_, in_type)

        if to_squeeze:
            input_ = input_.unsqueeze(0)

        input_ = input_.permute(0, 2, 3, 1).numpy()
        transformed = transform(input_)
        output = torch.from_numpy(transformed).float().permute(0, 3, 1, 2)
        if to_squeeze:
            output = output.squeeze(0)
        output = _convert(output, out_type)
        return output.to(device)
    return apply_transform

def _generic_transform_sk_3d(transform, in_type='', out_type=''):
    def apply_transform_individual(input_):
        device = input_.device
        input_ = input_.cpu()
        input_ = _convert(input_, in_type)

        input_ = input_.permute(1, 2, 0).detach().numpy()
        transformed = transform(input_)
        output = torch.from_numpy(transformed).float().permute(2, 0, 1)
        output = _convert(output, out_type)
        return output.to(device)

    def apply_transform(input_):
        to_stack = []
        for image in input_:
            to_stack.append(apply_transform_individual(image))
        return torch.stack(to_stack)
    return apply_transform

# # --- Cie*LAB ---
rgb_to_lab = _generic_transform_sk_4d(rgb2lab)
lab_to_rgb = _generic_transform_sk_3d(lab2rgb, in_type='double', out_type='float')
# # --- YUV ---
# rgb_to_yuv = _generic_transform_sk_4d(rgb2yuv)
# yuv_to_rgb = _generic_transform_sk_4d(yuv2rgb)
# # --- YCbCr ---
# rgb_to_ycbcr = _generic_transform_sk_4d(rgb2ycbcr)
# ycbcr_to_rgb = _generic_transform_sk_4d(ycbcr2rgb, in_type='double', out_type='float')
# # --- HSV ---
# rgb_to_hsv = _generic_transform_sk_3d(rgb2hsv)
hsv_to_rgb = _generic_transform_sk_3d(hsv2rgb)

class Summ_writer(object):
    def __init__(self, writer, global_step, log_freq=10, fps=8, scalar_freq=100, just_gif=False):
        self.writer = writer
        self.global_step = global_step
        self.log_freq = log_freq
        self.fps = fps
        self.just_gif = just_gif
        self.maxwidth = 10000
        self.save_this = (self.global_step % self.log_freq == 0)
        self.scalar_freq = max(scalar_freq,1)
        

    def summ_gif(self, name, tensor, blacken_zeros=False):
        # tensor should be in B x S x C x H x W
        
        assert tensor.dtype in {torch.uint8,torch.float32}
        shape = list(tensor.shape)

        if tensor.dtype == torch.float32:
            tensor = back2color(tensor, blacken_zeros=blacken_zeros)

        video_to_write = tensor[0:1]

        S = video_to_write.shape[1]
        if S==1:
            # video_to_write is 1 x 1 x C x H x W
            self.writer.add_image(name, video_to_write[0,0], global_step=self.global_step)
        else:
            self.writer.add_video(name, video_to_write, fps=self.fps, global_step=self.global_step)
            
        return video_to_write

    def summ_rgbs(self, name, ims, frame_ids=None, blacken_zeros=False, only_return=False):
        if self.save_this:

            ims = gif_and_tile(ims, just_gif=self.just_gif)
            vis = ims

            assert vis.dtype in {torch.uint8,torch.float32}

            if vis.dtype == torch.float32:
                vis = back2color(vis, blacken_zeros)           

            B, S, C, H, W = list(vis.shape)

            if frame_ids is not None:
                assert(len(frame_ids)==S)
                for s in range(S):
                    vis[:,s] = draw_frame_id_on_vis(vis[:,s], frame_ids[s])

            if int(W) > self.maxwidth:
                vis = vis[:,:,:,:self.maxwidth]

            if only_return:
                return vis
            else:
                return self.summ_gif(name, vis, blacken_zeros)

    def summ_rgb(self, name, ims, blacken_zeros=False, frame_id=None, only_return=False, halfres=False):
        if self.save_this:
            assert ims.dtype in {torch.uint8,torch.float32}

            if ims.dtype == torch.float32:
                ims = back2color(ims, blacken_zeros)

            #ims is B x C x H x W
            vis = ims[0:1] # just the first one
            B, C, H, W = list(vis.shape)

            if halfres:
                vis = F.interpolate(vis, scale_factor=0.5)

            if frame_id is not None:
                vis = draw_frame_id_on_vis(vis, frame_id)

            if int(W) > self.maxwidth:
                vis = vis[:,:,:,:self.maxwidth]

            if only_return:
                return vis
            else:
                return self.summ_gif(name, vis.unsqueeze(1), blacken_zeros)

    def summ_occs(self, name, occs, bev=False, fro=False, reduce_axes=[3], frame_ids=None):
        if self.save_this:
            B, C, D, H, W = list(occs[0].shape)
            if bev:
                reduce_axes = [3]
            elif fro:
                reduce_axes = [2]
            for reduce_axis in reduce_axes:
                heights = [convert_occ_to_height(occ, reduce_axis=reduce_axis) for occ in occs]
                self.summ_oneds(name=('%s_ax%d' % (name, reduce_axis)), ims=heights, norm=False, frame_ids=frame_ids)
            
    def summ_occ(self, name, occ, reduce_axes=[3], bev=False, fro=False, pro=False, frame_id=None, only_return=False):
        if self.save_this:
            B, C, D, H, W = list(occ.shape)
            if bev:
                reduce_axes = [3]
            elif fro:
                reduce_axes = [2]
            elif pro:
                reduce_axes = [4]
            for reduce_axis in reduce_axes:
                height = convert_occ_to_height(occ, reduce_axis=reduce_axis)
                # if only_return:
                #     return height
                if reduce_axis == reduce_axes[-1]:
                    return self.summ_oned(name=('%s_ax%d' % (name, reduce_axis)), im=height, norm=False, frame_id=frame_id, only_return=only_return)
                else:
                    self.summ_oned(name=('%s_ax%d' % (name, reduce_axis)), im=height, norm=False, frame_id=frame_id, only_return=only_return)
                    
    def summ_oneds(self, name, ims, frame_ids=None, bev=False, fro=False, logvis=False, reduce_max=False, max_val=0.0, norm=True, only_return=False):
        if self.save_this:
            if bev: 
                B, C, H, _, W = list(ims[0].shape)
                if reduce_max:
                    ims = [torch.max(im, dim=3)[0] for im in ims]
                else:
                    ims = [torch.mean(im, dim=3) for im in ims]
            elif fro: 
                B, C, _, H, W = list(ims[0].shape)
                if reduce_max:
                    ims = [torch.max(im, dim=2)[0] for im in ims]
                else:
                    ims = [torch.mean(im, dim=2) for im in ims]


            if len(ims) != 1: # sequence
                im = gif_and_tile(ims, just_gif=self.just_gif)
            else:
                im = torch.stack(ims, dim=1) # single frame

            B, S, C, H, W = list(im.shape)
            
            if logvis and max_val:
                max_val = np.log(max_val)
                im = torch.log(torch.clamp(im, 0)+1.0)
                im = torch.clamp(im, 0, max_val)
                im = im/max_val
                norm = False
            elif max_val:
                im = torch.clamp(im, 0, max_val)
                im = im/max_val
                norm = False
                
            if norm:
                # normalize before oned2inferno,
                # so that the ranges are similar within B across S
                im = utils.basic.normalize(im)

            im = im.view(B*S, C, H, W)
            vis = oned2inferno(im, norm=norm)
            vis = vis.view(B, S, 3, H, W)

            if frame_ids is not None:
                assert(len(frame_ids)==S)
                for s in range(S):
                    vis[:,s] = draw_frame_id_on_vis(vis[:,s], frame_ids[s])

            if W > self.maxwidth:
                vis = vis[...,:self.maxwidth]

            if only_return:
                return vis
            else:
                self.summ_gif(name, vis)
                
    def summ_oned(self, name, im, bev=False, fro=False, logvis=False, max_val=0, max_along_y=False, norm=True, frame_id=None, only_return=False):
        if self.save_this:

            if bev: 
                B, C, H, _, W = list(im.shape)
                if max_along_y:
                    im = torch.max(im, dim=3)[0]
                else:
                    im = torch.mean(im, dim=3)
            elif fro:
                B, C, _, H, W = list(im.shape)
                if max_along_y:
                    im = torch.max(im, dim=2)[0]
                else:
                    im = torch.mean(im, dim=2)
            else:
                B, C, H, W = list(im.shape)
                
            im = im[0:1] # just the first one
            assert(C==1)
            
            if logvis and max_val:
                max_val = np.log(max_val)
                im = torch.log(im)
                im = torch.clamp(im, 0, max_val)
                im = im/max_val
                norm = False
            elif max_val:
                im = torch.clamp(im, 0, max_val)/max_val
                norm = False

            vis = oned2inferno(im, norm=norm)
            # vis = vis.view(B, 3, H, W)
            if W > self.maxwidth:
                vis = vis[...,:self.maxwidth]
            # self.writer.add_images(name, vis, global_step=self.global_step, dataformats='NCHW')
            return self.summ_rgb(name, vis, blacken_zeros=False, frame_id=frame_id, only_return=only_return)
        
    def summ_scalar(self, name, value):
        if (not (isinstance(value, int) or isinstance(value, float) or isinstance(value, np.float32))) and ('torch' in value.type()):
            value = value.detach().cpu().numpy()
        if not np.isnan(value):
            if (self.log_freq == 1):
                self.writer.add_scalar(name, value, global_step=self.global_step)
            elif self.save_this or np.mod(self.global_step, self.scalar_freq)==0:
                self.writer.add_scalar(name, value, global_step=self.global_step)

    def summ_flow(self, name, im, clip=0.0, only_return=False, frame_id=None):
        # flow is B x C x D x W
        if self.save_this:
            return self.summ_rgb(name, flow2color(im, clip=clip), only_return=only_return, frame_id=frame_id)
        else:
            return None

    def summ_soft_seg(self, name, seg, bev=False, max_along_y=False, only_return=False, frame_id=None, colormap='tab20', label_colors=None):
        if not self.save_this:
            return

        if bev:
            B,N,D,H,W = seg.shape
            if max_along_y:
                seg = torch.max(seg, dim=3)[0]
            else:
                seg = torch.mean(seg, dim=3)
        B,N,H,W = seg.shape
            
        # the values along N should sum to 1
        
        if N > 10:
            colormap = 'tab20'

        if label_colors is None:
            custom_label_colors = False
            label_colors = matplotlib.cm.get_cmap(colormap).colors
            label_colors = [[int(i*255) for i in l] for l in label_colors]
        else:
            custom_label_colors = True
            
        color_map = torch.zeros([B, 3, N, H, W], dtype=torch.float32).cuda()
        seg_ = seg.unsqueeze(1)
        # this is B x 1 x N x H x W

        # print('label_colors', label_colors, len(label_colors))

        for label in range(0,N):
            if (not custom_label_colors) and (N > 20):
                label_ = label % 20
            else:
                label_ = label
            # print('label_colors[%d]' % (label_), label_colors[label_])
            color_map[:,0,label_] = label_colors[label_][0]
            color_map[:,1,label_] = label_colors[label_][1]
            color_map[:,2,label_] = label_colors[label_][2]

        out = torch.sum(color_map * seg_, dim=2)
        out = out.type(torch.ByteTensor)
        return self.summ_rgb(name, out, only_return=only_return, frame_id=frame_id)
    
    def summ_soft_seg_thr(self, name, seg_e_sig, thr=0.5, only_return=False, colormap='tab20', frame_id=None, label_colors=None):
        B, N, H, W = list(seg_e_sig.shape)
        assert(thr > 0.0)
        assert(thr < 1.0)
        seg_e_hard = (seg_e_sig > thr).float()
        single_class = (torch.sum(seg_e_hard, dim=1, keepdim=True)==1).float()
        seg_e_hard = seg_e_hard * single_class
        seg_e_sig = (seg_e_sig - thr).clamp(0, (1-thr))/(1-thr)
        return self.summ_soft_seg(name, seg_e_hard * seg_e_sig, only_return=only_return, colormap=colormap, frame_id=frame_id, label_colors=label_colors)
        
    def summ_lrtlist(self, name, rgbR, lrtlist, scorelist, tidlist, pix_T_cam, only_return=False, frame_id=None, include_zeros=False, halfres=False, show_ids=False):
        # rgb is B x H x W x C
        # lrtlist is B x N x 19
        # scorelist is B x N
        # tidlist is B x N
        # pix_T_cam is B x 4 x 4

        if self.save_this:

            B, C, H, W = list(rgbR.shape)
            B, N, D = list(lrtlist.shape)

            xyzlist_cam = utils.geom.get_xyzlist_from_lrtlist(lrtlist)
            # this is B x N x 8 x 3

            clist_cam = utils.geom.get_clist_from_lrtlist(lrtlist)
            arrowlist_cam = utils.geom.get_arrowheadlist_from_lrtlist(lrtlist)

            boxes_vis = self.draw_corners_on_image(rgbR,
                                                   xyzlist_cam,
                                                   clist_cam, 
                                                   scorelist,
                                                   tidlist,
                                                   pix_T_cam,
                                                   arrowlist_cam=arrowlist_cam,
                                                   frame_id=frame_id,
                                                   include_zeros=include_zeros,
                                                   show_ids=show_ids)
            return self.summ_rgb(name, boxes_vis, only_return=only_return, halfres=halfres)
    
    
    def summ_lrtlist_bev(self, name, occ_memR, lrtlist, scorelist, tidlist, vox_util, lrt=None, already_mem=False, only_return=False, frame_id=None, include_zeros=False, show_ids=False):
        if self.save_this:
            # rgb is B x C x Z x Y x X
            # lrtlist is B x N x 19
            # scorelist is B x N
            # tidlist is B x N

            # print('occ_memR', occ_memR.shape)
            # print('lrtlist', lrtlist.shape)
            # print('scorelist', scorelist.shape)
            # print('tidlist', tidlist.shape)
            # if lrt is not None:
            #     print('lrt', lrt.shape)

            B, _, Z, Y, X = list(occ_memR.shape)
            B, N, D = list(lrtlist.shape)

            corners_cam = utils.geom.get_xyzlist_from_lrtlist(lrtlist)
            centers_cam = utils.geom.get_clist_from_lrtlist(lrtlist)
            arrowlist_cam = utils.geom.get_arrowheadlist_from_lrtlist(lrtlist)
            
            if lrt is None:
                if not already_mem:
                    corners_mem = vox_util.Ref2Mem(corners_cam.reshape(B, N*8, 3), Z, Y, X, assert_cube=False).reshape(B, N, 8, 3)
                    # this is B x N x 8 x 3
                    centers_mem = vox_util.Ref2Mem(centers_cam, Z, Y, X, assert_cube=False).reshape(B, N, 1, 3)
                    # this is B x N x 1 x 3
                    arrowlist_mem = vox_util.Ref2Mem(arrowlist_cam, Z, Y, X, assert_cube=False).reshape(B, N, 1, 3)
                else:
                    corners_mem = corners_cam.clone().reshape(B, N, 8, 3)
                    centers_mem = centers_cam.clone().reshape(B, N, 1, 3)
                    arrowlist_mem = arrowlist_cam.clone().reshape(B, N, 1, 3)
                    
            else:
                # use the lrt to know where to voxelize
                corners_mem = vox_util.Ref2Zoom(corners_cam.reshape(B, N*8, 3), lrt, Z, Y, X).reshape(B, N, 8, 3)
                centers_mem = vox_util.Ref2Zoom(centers_cam, lrt, Z, Y, X).reshape(B, N, 1, 3)
                arrowlist_mem = vox_util.Ref2Zoom(arrowlist_cam, lrt, Z, Y, X).reshape(B, N, 1, 3)

            # rgb = utils.basic.reduce_masked_mean(unp_memR, occ_memR.repeat(1, C, 1, 1, 1), dim=3)
            rgb_vis = self.summ_occ('', occ_memR, only_return=True)
            # utils.py.print_stats('rgb_vis', rgb_vis.cpu().numpy())
            # print('rgb', rgb.shape)
            # rgb_vis = back2color(rgb)
            # this is in [0, 255]

            # print('rgb_vis', rgb_vis.shape)

            if False:
                # alt method
                box_mem = torch.cat([centers_mem, corners_mem], dim=2).reshape(B, N*9, 3)
                box_vox = vox_util.voxelize_xyz(box_mem, Z, Y, X, already_mem=True)
                box_vis = self.summ_occ('', box_vox, reduce_axes=[3], only_return=True)

                box_vis = convert_occ_to_height(box_vox, reduce_axis=3)
                box_vis = utils.basic.normalize(box_vis)
                box_vis = oned2inferno(box_vis, norm=False)
                # this is in [0, 255]

                # replace black with occ vis
                box_vis[box_vis==0] = (rgb_vis[box_vis==0].float()*0.5).byte() # darken the bkg a bit
                box_vis = preprocess_color(box_vis)
                return self.summ_rgb(('%s' % (name)), box_vis, only_return=only_return)#, only_return=only_return)

            # take the xz part
            centers_mem = torch.stack([centers_mem[:,:,:,0], centers_mem[:,:,:,2]], dim=3)
            corners_mem = torch.stack([corners_mem[:,:,:,0], corners_mem[:,:,:,2]], dim=3)
            arrowlist_mem = torch.stack([arrowlist_mem[:,:,:,0], arrowlist_mem[:,:,:,2]], dim=3)

            if frame_id is not None:
                rgb_vis = draw_frame_id_on_vis(rgb_vis, frame_id)

            out = self.draw_boxes_on_image_py(rgb_vis[0].detach().cpu().numpy(),
                                              corners_mem[0].detach().cpu().numpy(),
                                              centers_mem[0].detach().cpu().numpy(),
                                              scorelist[0].detach().cpu().numpy(),
                                              tidlist[0].detach().cpu().numpy(),
                                              arrowlist_pix=arrowlist_mem[0].detach().cpu().numpy(),
                                              frame_id=frame_id,
                                              include_zeros=include_zeros,
                                              show_ids=show_ids)
            # utils.py.print_stats('py out', out)
            out = torch.from_numpy(out).type(torch.ByteTensor).permute(2, 0, 1)
            out = torch.unsqueeze(out, dim=0)
            out = preprocess_color(out)
            return self.summ_rgb(name, out, only_return=only_return)
            # box_vis = torch.reshape(out, [1, 3, Z, X]).byte()
            # out = torch.reshape(out, [1, C, X, Z])
            # out = out.permute(0, 1, 3, 2)

            # box_vis = preprocess_color(out)
            # utils.py.print_stats('box_vis', box_vis.cpu().numpy())

            # if not only_return:
            #     self.summ_rgb(name, box_vis)
            # return box_vis
            #     self.summ_rgb(name, box_vis)

    def draw_corners_on_image(self, rgb, corners_cam, centers_cam, scores, tids, pix_T_cam, frame_id=None, arrowlist_cam=None, include_zeros=False, show_ids=False):
        # first we need to get rid of invalid gt boxes
        # gt_boxes = trim_gt_boxes(gt_boxes)
        B, C, H, W = list(rgb.shape)
        assert(C==3)
        B2, N, D, E = list(corners_cam.shape)
        assert(B2==B)
        assert(D==8) # 8 corners
        assert(E==3) # 3D

        rgb = back2color(rgb)

        corners_cam_ = torch.reshape(corners_cam, [B, N*8, 3])
        centers_cam_ = torch.reshape(centers_cam, [B, N*1, 3])
        
        corners_pix_ = utils.geom.apply_pix_T_cam(pix_T_cam, corners_cam_)
        centers_pix_ = utils.geom.apply_pix_T_cam(pix_T_cam, centers_cam_)
        
        corners_pix = torch.reshape(corners_pix_, [B, N, 8, 2])
        centers_pix = torch.reshape(centers_pix_, [B, N, 1, 2])
        
        if arrowlist_cam is not None:
            arrowlist_cam_ = torch.reshape(arrowlist_cam, [B, N*1, 3])
            arrowlist_pix_ = utils.geom.apply_pix_T_cam(pix_T_cam, arrowlist_cam_)
            arrowlist_pix = torch.reshape(arrowlist_pix_, [B, N, 1, 2])
            
        if frame_id is not None:
            rgb = draw_frame_id_on_vis(rgb, frame_id)

        out = self.draw_boxes_on_image_py(rgb[0].detach().cpu().numpy(),
                                          corners_pix[0].detach().cpu().numpy(),
                                          centers_pix[0].detach().cpu().numpy(),
                                          scores[0].detach().cpu().numpy(),
                                          tids[0].detach().cpu().numpy(),
                                          frame_id=frame_id,
                                          arrowlist_pix=arrowlist_pix[0].detach().cpu().numpy(),
                                          include_zeros=include_zeros,
                                          show_ids=show_ids)
        out = torch.from_numpy(out).type(torch.ByteTensor).permute(2, 0, 1)
        out = torch.unsqueeze(out, dim=0)
        out = preprocess_color(out)
        out = torch.reshape(out, [1, C, H, W])
        return out
    
    def draw_boxes_on_image_py(self, rgb, corners_pix, centers_pix, scores, tids, boxes=None, thickness=1, frame_id=None, arrowlist_pix=None, include_zeros=False, show_ids=False):
        # all inputs are numpy tensors
        # rgb is H x W x 3
        # corners_pix is N x 8 x 2, in xy order
        # centers_pix is N x 1 x 2, in xy order
        # scores is N
        # tids is N
        # boxes is N x 9 < this is only here to print some rotation info

        # cv2.cvtColor seems to cause an Illegal instruction error on compute-0-38; no idea why

        rgb = np.transpose(rgb, [1, 2, 0]) # put channels last
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) 

        H, W, C = rgb.shape
        assert(C==3)
        N, D, E = corners_pix.shape
        assert(D==8)
        assert(E==2)

        N2 = scores.shape
        N3 = tids.shape
        # assert(N==N2)
        # assert(N==N3)

        if boxes is not None:
            rx = boxes[:,6].clone()
            ry = boxes[:,7].clone()
            rz = boxes[:,8].clone()
        else:
            rx = 0
            ry = 0
            rz = 0

        color_map = matplotlib.cm.get_cmap('tab20')
        color_map = color_map.colors

        corners_pix = corners_pix.astype(np.int32)
        centers_pix = centers_pix.astype(np.int32)

        # else:
        #     print('frame_id is none')
            
        # draw
        for ind, corners in enumerate(corners_pix):
            # corners is 8 x 2

            if include_zeros or (not np.isclose(scores[ind], 0.0)):

                # print('ind', ind)
                # print('score = %.2f' % scores[ind])
                color_id = tids[ind] % 20
                # print('color_id', color_id)
                # print('color_map', color_map)
                color = color_map[color_id]
                color = np.array(color)*255.0
                color = color[::-1]
                # color = (0,191,255)
                # color = (255,191,0)
                # print 'tid = %d; score = %.3f' % (tids[ind], scores[ind])

                # # draw center
                # cv2.circle(rgb,(centers_pix[ind,0,0],centers_pix[ind,0,1]),1,color,-1)

                if False:
                    if arrowlist_pix is not None:
                        cv2.arrowedLine(rgb,(centers_pix[ind,0,0],centers_pix[ind,0,1]),(arrowlist_pix[ind,0,0],arrowlist_pix[ind,0,1]),color,
                                        thickness=1,line_type=cv2.LINE_AA,tipLength=0.25)


                # if scores[ind] < 1.0 and scores[ind] > 0.0:
                # if False:
                if scores[ind] < 1.0:
                    # print('for example, putting this one at', np.min(corners[:,0]), np.min(corners[:,1]))
                    cv2.putText(rgb,
                                '%.2f' % (scores[ind]), 
                                # '%.2f match' % (scores[ind]), 
                                # '%.2f IOU' % (scores[ind]), 
                                # '%d (%.2f)' % (tids[ind], scores[ind]), 
                                (np.min(corners[:,0]), np.min(corners[:,1])),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, # font scale (float)
                                color,
                                1) # font thickness (int)

                if show_ids: # write tid
                    cv2.putText(rgb,
                                '%d' % (tids[ind]),
                                # '%.2f match' % (scores[ind]), 
                                # '%.2f IOU' % (scores[ind]), 
                                # '%d (%.2f)' % (tids[ind], scores[ind]), 
                                (np.max(corners[:,0]), np.max(corners[:,1])),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, # font scale (float)
                                color,
                                1) # font thickness (int)
                        

                # for c in corners:
                #     # rgb[pt1[0], pt1[1], :] = 255
                #     # rgb[pt2[0], pt2[1], :] = 255
                #     # rgb[np.clip(int(c[0]), 0, W), int(c[1]), :] = 255
                #     c0 = np.clip(int(c[0]), 0,  W-1)
                #     c1 = np.clip(int(c[1]), 0,  H-1)
                #     rgb[c1, c0, :] = 255
                    
                    
                # we want to distinguish between in-plane edges and out-of-plane ones
                # so let's recall how the corners are ordered:

                # (new clockwise ordering)
                xs = np.array([1/2., 1/2., -1/2., -1/2., 1/2., 1/2., -1/2., -1/2.])
                ys = np.array([1/2., 1/2., 1/2., 1/2., -1/2., -1/2., -1/2., -1/2.])
                zs = np.array([1/2., -1/2., -1/2., 1/2., 1/2., -1/2., -1/2., 1/2.])

                # for ii in list(range(0,2)):
                #     cv2.circle(rgb,(corners_pix[ind,ii,0],corners_pix[ind,ii,1]),1,color,-1)
                # for ii in list(range(2,4)):
                #     cv2.circle(rgb,(corners_pix[ind,ii,0],corners_pix[ind,ii,1]),1,color,-1)

                xs = np.reshape(xs, [8, 1])
                ys = np.reshape(ys, [8, 1])
                zs = np.reshape(zs, [8, 1])
                offsets = np.concatenate([xs, ys, zs], axis=1)

                corner_inds = list(range(8))
                combos = list(combinations(corner_inds, 2))

                for combo in combos:
                    pt1 = offsets[combo[0]]
                    pt2 = offsets[combo[1]]
                    # draw this if it is an in-plane edge
                    eqs = pt1==pt2
                    if np.sum(eqs)==2:
                        i, j = combo
                        pt1 = (corners[i, 0], corners[i, 1])
                        pt2 = (corners[j, 0], corners[j, 1])
                        retval, pt1, pt2 = cv2.clipLine((0, 0, W, H), pt1, pt2)
                        if retval:
                            cv2.line(rgb, pt1, pt2, color, thickness, cv2.LINE_AA)

                        # rgb[pt1[0], pt1[1], :] = 255
                        # rgb[pt2[0], pt2[1], :] = 255
        rgb = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_BGR2RGB)
        # utils.basic.print_stats_py('rgb_uint8', rgb)
        # imageio.imwrite('boxes_rgb.png', rgb)
        return rgb
        
def preprocess_color(x):
    if isinstance(x, np.ndarray):
        return x.astype(np.float32) * 1./255 - 0.5
    else:
        return x.float() * 1./255 - 0.5
    
def dilate2d(im, times=1, device='cuda'):
    weights2d = torch.ones(1, 1, 3, 3, device=device)
    for time in range(times):
        im = F.conv2d(im, weights2d, padding=1).clamp(0, 1)
    return im

def dilate3d(im, times=1, device='cuda'):
    weights3d = torch.ones(1, 1, 3, 3, 3, device=device)
    for time in range(times):
        im = F.conv3d(im, weights3d, padding=1).clamp(0, 1)
    return im

def erode2d(im, times=1, device='cuda'):
    weights2d = torch.ones(1, 1, 3, 3, device=device)
    for time in range(times):
        im = 1.0 - F.conv2d(1.0 - im, weights2d, padding=1).clamp(0, 1)
    return im

def erode3d(im, times=1, device='cuda'):
    weights3d = torch.ones(1, 1, 3, 3, 3, device=device)
    for time in range(times):
        im = 1.0 - F.conv3d(1.0 - im, weights3d, padding=1).clamp(0, 1)
    return im

def colorize(d):
    # this is actually just grayscale right now
    
    # # d is C x H x W or H x W
    # if d.ndim==3:
    #     d = d.squeeze(dim=0)
    # else:
    #     assert(d.ndim==2)

    if d.ndim==2:
        d = d.unsqueeze(dim=0)
    else:
        assert(d.ndim==3)
    # copy to the three chans
    d = d.repeat(3, 1, 1)
    return d

def oned2inferno(d, norm=True):
    # convert a 1chan input to a 3chan image output

    # if it's just B x H x W, add a C dim
    if d.ndim==3:
        d = d.unsqueeze(dim=1)
    # d should be B x C x H x W, where C=1
    B, C, H, W = list(d.shape)
    assert(C==1)

    if norm:
        d = utils.basic.normalize(d)
        
    rgb = torch.zeros(B, 3, H, W)
    for b in list(range(B)):
        rgb[b] = colorize(d[b])

    rgb = (255.0*rgb).type(torch.ByteTensor)

    # rgb = tf.cast(255.0*rgb, tf.uint8)
    # rgb = tf.reshape(rgb, [-1, hyp.H, hyp.W, 3])
    # rgb = tf.expand_dims(rgb, axis=0)
    return rgb

def xy2mask_single(xy, H, W):
    # xy is N x 2
    x, y = torch.unbind(xy, axis=1)
    x = x.long()
    y = y.long()

    x = torch.clamp(x, 0, W-1)
    y = torch.clamp(y, 0, H-1)
    
    inds = utils.basic.sub2ind(H, W, y, x)

    valid = (inds > 0).byte() & (inds < H*W).byte()
    inds = inds[torch.where(valid)]

    mask = torch.zeros(H*W, dtype=torch.float32, device=torch.device('cuda'))
    mask[inds] = 1.0
    mask = torch.reshape(mask, [1,H,W])
    return mask

def xy2mask(xy, H, W, norm=False):
    # xy is B x N x 2, in either pixel coords or normalized coordinates (depending on norm)
    # returns a mask shaped B x 1 x H x W, with a 1 at each specified xy
    B = list(xy.shape)[0]
    if norm:
        # convert to pixel coords
        x, y = torch.unbind(xy, axis=2)
        x = x*float(W)
        y = y*float(H)
        xy = torch.stack(xy, axis=2)
        
    mask = torch.zeros([B, 1, H, W], dtype=torch.float32, device=torch.device('cuda'))
    for b in list(range(B)):
        mask[b] = xy2mask_single(xy[b], H, W)
    return mask

def gif_and_tile(ims, just_gif=False):
    S = len(ims) 
    # each im is B x H x W x C
    # i want a gif in the left, and the tiled frames on the right
    # for the gif tool, this means making a B x S x H x W tensor
    # where the leftmost part is sequential and the rest is tiled
    gif = torch.stack(ims, dim=1)
    if just_gif:
        return gif
    til = torch.cat(ims, dim=2)
    til = til.unsqueeze(dim=1).repeat(1, S, 1, 1, 1)
    im = torch.cat([gif, til], dim=3)
    return im

def back2color(i, blacken_zeros=False):
    if blacken_zeros:
        const = torch.tensor([-0.5])
        i = torch.where(i==0.0, const.cuda() if i.is_cuda else const, i)
        return back2color(i)
    else:
        return ((i+0.5)*255).type(torch.ByteTensor)


def flow2color(flow, clip=50.0):
    # flow is B x C x H x W

    B, C, H, W = list(flow.size())

    flow = flow.clone().detach()

    abs_image = torch.abs(flow)
    flow_mean = abs_image.mean(dim=[1,2,3])
    flow_std = abs_image.std(dim=[1,2,3])

    if clip:
        flow = torch.clamp(flow, -clip, clip)/clip
    else:
        # Apply some kind of normalization. Divide by the perceived maximum (mean + std*2)
        flow_max = flow_mean + flow_std*2 + 1e-10
        for b in range(B):
            flow[b] = flow[b].clamp(-flow_max[b].item(), flow_max[b].item()) / flow_max[b].clamp(min=1)

    radius = torch.sqrt(torch.sum(flow**2, dim=1, keepdim=True)) #B x 1 x H x W
    radius_clipped = torch.clamp(radius, 0.0, 1.0)

    angle = torch.atan2(flow[:, 1:], flow[:, 0:1]) / np.pi #B x 1 x H x W

    hue = torch.clamp((angle + 1.0) / 2.0, 0.0, 1.0)
    saturation = torch.ones_like(hue) * 0.75
    value = radius_clipped
    hsv = torch.cat([hue, saturation, value], dim=1) #B x 3 x H x W

    #flow = tf.image.hsv_to_rgb(hsv)
    flow = hsv_to_rgb(hsv)
    flow = (flow*255.0).type(torch.ByteTensor)
    return flow

def convert_occ_to_height(occ, reduce_axis=3):
    B, C, D, H, W = list(occ.shape)
    assert(C==1)
    # note that height increases DOWNWARD in the tensor
    # (like pixel/camera coordinates)
    
    G = list(occ.shape)[reduce_axis]
    values = torch.linspace(float(G), 1.0, steps=G, dtype=torch.float32, device=occ.device)
    if reduce_axis==2:
        # fro view
        values = values.view(1, 1, G, 1, 1)
    elif reduce_axis==3:
        # top view
        values = values.view(1, 1, 1, G, 1)
    elif reduce_axis==4:
        # lateral view
        values = values.view(1, 1, 1, 1, G)
    else:
        assert(False) # you have to reduce one of the spatial dims (2-4)
    values = torch.max(occ*values, dim=reduce_axis)[0]/float(G)
    # values = values.view([B, C, D, W])
    return values

def draw_frame_id_on_vis(vis, frame_id, scale=0.5, left=5, top=20):

    rgb = vis.detach().cpu().numpy()[0]
    rgb = np.transpose(rgb, [1, 2, 0]) # put channels last
    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) 
    color = (255, 255, 255)
    # print('putting frame id', frame_id)

    frame_str = utils.basic.strnum(frame_id)
    
    cv2.putText(
        rgb,
        frame_str,
        (left, top), # from left, from top
        cv2.FONT_HERSHEY_SIMPLEX,
        scale, # font scale (float)
        color, 
        1) # font thickness (int)
    rgb = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_BGR2RGB)
    vis = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    return vis
