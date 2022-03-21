import torch
import torch.nn as nn
import torch.nn.functional as F

import utils.improc
import utils.misc
import utils.basic
import utils.geom
import utils.samp
import numpy as np
from utils.basic import print_

def compute_seg_loss(pred, pos, neg, balanced=True):
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
    
def balanced_ce_loss(pred, gt, valid=None):
    # pred is B x 1 x Y x X
    pos = (gt > 0.95).float()
    neg = (gt < 0.05).float()

    if valid is None:
        valid = torch.ones_like(pos)

    label = pos*2.0 - 1.0
    a = -label * pred
    b = F.relu(a)
    loss = b + torch.log(torch.exp(-b)+torch.exp(a-b))
    
    pos_loss = utils.basic.reduce_masked_mean(loss, pos*valid)
    neg_loss = utils.basic.reduce_masked_mean(loss, neg*valid)
    balanced_loss = pos_loss + neg_loss
    return balanced_loss

class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(ResidualBlock, self).__init__()
  
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.BatchNorm2d(planes)
        
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not stride == 1:
                self.norm3 = nn.Sequential()

        if stride == 1:
            self.downsample = None
        
        else:    
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)


    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)
    
class BasicEncoder(nn.Module):
    def __init__(self, input_dim=3, stride=8, output_dim=128, norm_fn='batch', dropout=0.0):
        super(BasicEncoder, self).__init__()
        self.norm_fn = norm_fn
        self.stride = stride

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)
            
        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(64)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(input_dim, 64, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 64
        self.layer1 = self._make_layer(64,  stride=1)
        self.layer2 = self._make_layer(96, stride=2)
        self.layer3 = self._make_layer(128, stride=2)
        self.layer4 = self._make_layer(128, stride=2)

        self.conv2 = nn.Conv2d(128+128+96, output_dim, kernel_size=1)
        # # output convolution
        # if self.stride==4:
        #     # self.conv2 = nn.Conv2d(128+96, output_dim, kernel_size=1)
        # else:
        #     self.conv2 = nn.Conv2d(128, output_dim, kernel_size=1)

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        # layer3 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        # layers = (layer1, layer2, layer3)
        layers = (layer1, layer2)
        
        self.in_planes = dim
        return nn.Sequential(*layers)


    def forward(self, x):

        _, _, H, W = x.shape
        
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        y = self.layer3(x)
        z = self.layer4(y)

        # print('x', x.shape)
        # print('y', y.shape)
        # print('z', z.shape)

        x = F.interpolate(x, (H//self.stride, W//self.stride), mode='bilinear', align_corners=True)
        y = F.interpolate(y, (H//self.stride, W//self.stride), mode='bilinear', align_corners=True)
        z = F.interpolate(z, (H//self.stride, W//self.stride), mode='bilinear', align_corners=True)
        x = self.conv2(torch.cat([x,y,z], dim=1))

        if self.training and self.dropout is not None:
            x = self.dropout(x)

        return x
    
def _topk(objectness, K=10):
    B, C, Z, X = list(objectness.shape)
    assert(C==1)
      
    scorelist, indlist = torch.topk(objectness.view(B, C, -1), K)

    # indlist_z = indlist // (X)
    indlist_z = torch.div(indlist, X, rounding_mode='trunc')
    indlist_x = indlist % (X)

    scorelist = scorelist.reshape(B, K)
    indlist_z = indlist_z.reshape(B, K)
    indlist_x = indlist_x.reshape(B, K)

    xzlist = torch.stack([indlist_x, indlist_z], dim=2).float()
    return scorelist, xzlist

def _nms(heat, kernel=11):
    pad = (kernel - 1) // 2
    hmax = F.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float() #* (hmax > 0.9).float()
    return heat * keep

class Centernet2d(nn.Module):
    def _sigmoid(self, x):
        y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
        return y
    
    def __init__(self, Y=32, K=10, show_thresh=0.5, stride=8):
        super(Centernet2d, self).__init__()

        print('Centernet2d...')

        self.stride = stride
        Y_ = Y//self.stride
        self.Y = Y
        self.Y_ = Y_
        self.K = K
        self.thresh = show_thresh # only show/return boxes with this conf


        self.num_rot_bins = 16
        self.heading_unaware = True
        
        if self.heading_unaware:
            # for rotation, i have decided to be heading-unaware
            # so, 0 and 180 are treated as equivalent
            bin_angles = np.linspace(0, np.pi, self.num_rot_bins, endpoint=False)
            bin_complements = bin_angles + np.pi
            all_bins = np.concatenate([bin_angles, bin_complements], axis=0)
            all_inds = np.concatenate([np.arange(self.num_rot_bins), np.arange(self.num_rot_bins)], axis=0)
        else:
            all_bins = np.linspace(0, np.pi*2, self.num_rot_bins, endpoint=False)
            all_inds = np.arange(self.num_rot_bins)
            bin_angles = all_bins
        
        
        self.bin_angles = torch.from_numpy(bin_angles).float().cuda()
        self.all_bins = torch.from_numpy(all_bins).float().cuda()
        self.all_inds = torch.from_numpy(all_inds).long().cuda()
        
        obj_channels = 1
        size_channels = 3
        offset_channels = 4
        rot_channels = self.num_rot_bins # just ry
        seg_channels = Y_
        
        self.output_channels = obj_channels + size_channels + offset_channels + rot_channels + seg_channels

        self.net = BasicEncoder(input_dim=Y, stride=stride, output_dim=self.output_channels, norm_fn='instance', dropout=0)

        self.mse = torch.nn.MSELoss(reduction='none')
        self.smoothl1 = torch.nn.SmoothL1Loss(reduction='none')
        
    def balanced_mse_loss(self, pred, gt, valid=None):
        # pos_inds = gt.eq(1).float()
        # neg_inds = gt.lt(1).float()
        pos_mask = gt.gt(0.5).float()
        neg_mask = gt.lt(0.5).float()

        # utils.basic.print_stats('pos_mask', pos_mask)
        # utils.basic.print_stats('neg_mask', neg_mask)
        # utils.basic.print_stats('pred', pred)
        # utils.basic.print_stats('gt', gt)
        if valid is None:
            valid = torch.ones_like(pos_mask)

        mse_loss = self.mse(pred, gt)
        pos_loss = utils.basic.reduce_masked_mean(mse_loss, pos_mask*valid)
        neg_loss = utils.basic.reduce_masked_mean(mse_loss, neg_mask*valid)

        loss = (pos_loss + neg_loss)*0.5
        return loss

    def compute_rot_loss(self, rot_prob, rot_g, valid):
        # rot_prob is B x N x self.rot_bins
        # rot_g is B x N, with angles in radians
        # valid is B x N

        B, N = list(rot_g.shape)
        rot_prob = rot_prob.reshape(B*N, self.num_rot_bins)
        valid = valid.reshape(B*N)
        rot_g = rot_g.reshape(B*N, 1)

        # i need to assign rot_g into bins
        dist = utils.geom.angular_l1_dist(rot_g, self.all_bins.reshape(1, -1))
        # this is B*N x num_rot_bins
        min_inds = torch.argmin(dist, dim=1)
        # this is B*N and long
        # for safety, let's not parallelize the gather here
        labels = torch.zeros(B*N).long().cuda()
        for b in list(range(B*N)):
            labels[b] = self.all_inds[min_inds[b]]
        # print('labels', labels.detach().cpu().numpy())
        # print('rot_prob', rot_prob.shape)
        loss_vec = F.cross_entropy(rot_prob, labels, reduction='none')
        
        # rather than take a straight mean, we will balance across classes
        losses = []
        for cls in list(range(self.num_rot_bins)):
            mask = (labels==cls).float()
            cls_loss = utils.basic.reduce_masked_mean(loss_vec, mask*valid)
            if torch.sum(mask) >= 1:
                # print('adding loss for rot bin %d' % cls)
                losses.append(cls_loss)
        total_loss = torch.mean(torch.stack(losses))
        return total_loss
    
    
    def forward(self, occ_feat, lrtlist_cam_g=None, scorelist_g=None, center_g=None, valid_g=None, pos_mem=None, neg_mem=None, vox_util=None, sw=None, force_export_boxlist=False, K=None):
        total_loss = torch.tensor(0.0).cuda()

        occ_mem = occ_feat.permute(0, 2, 1, 3).unsqueeze(1)
        
        B, Y, Z, X = list(occ_feat.shape)
        out_feat = self.net(occ_feat)
        # print('out_feat', out_feat.shape)

        Z8 = Z//self.stride
        X8 = X//self.stride
        Y8 = Y//self.stride

        # pred = out_feat.reshape(B, self.Y8, self.output_channels, Z8, X8).permute(0, 2, 3, 1, 4) # B, C, Z, Y, X
        # print('pred', pred.shape)

        pred = out_feat.reshape(B, self.output_channels, Z8, X8) # B, C, Z, X

        # dy, dx = utils.basic.gradient2d(pred, absolute=True)
        # smooth_loss = torch.mean(dy+dx)
        # total_loss += smooth_loss*0.01

        center_e = pred[:,0:1]
        center_e_sig = torch.sigmoid(center_e)

        size_e = F.softplus(pred[:,1:4]) + 0.01
        
        offset_e = pred[:, 4:7]
        ycoord_e = pred[:, 7:8]
        ry_e = pred[:, 8:8+self.num_rot_bins]

        seg_e = pred[:, 8+self.num_rot_bins:].reshape(B, 1, Y8, Z8, X8).permute(0, 1, 3, 2, 4)
        # print('center_e', center_e.shape)
        # print('center_g', center_g.shape)
            
        if lrtlist_cam_g is not None:
            # assume other _g tensors are present
            
            B2, N, D = list(lrtlist_cam_g.shape)
            assert(B==B2)
            assert(D==19)

            seg_loss = compute_seg_loss(seg_e, pos_mem, neg_mem, balanced=True)
            total_loss = utils.misc.add_loss('center2d/seg_loss', total_loss, seg_loss, 1.0, sw)
            
            lrtlist_mem_g = vox_util.apply_mem_T_ref_to_lrtlist(lrtlist_cam_g, Z8, Y8, X8, assert_cube=False)

            prob_loss = self.balanced_mse_loss(center_e_sig, center_g, valid_g)
            total_loss = utils.misc.add_loss('center2d/prob_loss', total_loss, prob_loss, 1.0, sw)

            clist_g = utils.geom.get_clist_from_lrtlist(lrtlist_mem_g)
            # clist_g is B x N x 3
            sizelist_g, rtlist_cam_g = utils.geom.split_lrtlist(lrtlist_cam_g) # note these are from cam, unlike centers
            # sizelist_g, rtlist_cam_g = utils.geom.split_lrtlist(lrtlist_mem_g)
            
            # print_('rylist_g', rylist_g)
            rlist_, tlist_ = utils.geom.split_rt(rtlist_cam_g.reshape(B*N, 4, 4))
            # compute ry using trigonometry
            x_vec = torch.zeros((B*N, 3), dtype=torch.float32, device=clist_g.device)
            x_vec[:, 2] = 1 # 0,0,1 
            x_rot = torch.matmul(rlist_, x_vec.unsqueeze(2)).squeeze(2)
            rylist_g = torch.atan2(x_rot[:,0],x_rot[:,2]).reshape(B, N)

            sizelist_e = utils.samp.bilinear_sample2d(size_e, clist_g[:,:,0], clist_g[:,:,2]).permute(0, 2, 1)
            sizelist_diff = torch.sum(self.smoothl1(sizelist_e, sizelist_g), dim=2)
            # this is B x N
            size_loss = utils.basic.reduce_masked_mean(sizelist_diff, scorelist_g)
            total_loss = utils.misc.add_loss('center2d/size_loss', total_loss, size_loss, 1.0, sw)

            ycoordlist_e = utils.samp.bilinear_sample2d(ycoord_e, clist_g[:,:,0], clist_g[:,:,2]).permute(0, 2, 1)
            ycoordlist_g = clist_g[:,:,1:2]
            ycoordlist_diff = torch.sum(self.smoothl1(ycoordlist_e, ycoordlist_g), dim=2)
            ycoord_loss = utils.basic.reduce_masked_mean(ycoordlist_diff, scorelist_g)
            total_loss = utils.misc.add_loss('center2d/ycoord_loss', total_loss, ycoord_loss, 1.0, sw)

            offsetlist_e = utils.samp.bilinear_sample2d(offset_e, clist_g[:,:,0], clist_g[:,:,2]).permute(0, 2, 1)
            offsetlist_g = clist_g - torch.round(clist_g) # get the decimal part
            offsetlist_diff = torch.sum(self.smoothl1(offsetlist_e, offsetlist_g), dim=2)
            offset_loss = utils.basic.reduce_masked_mean(offsetlist_diff, scorelist_g)
            total_loss = utils.misc.add_loss('center2d/offset_loss', total_loss, offset_loss, 1.0, sw)

            rylist_e = utils.samp.bilinear_sample2d(ry_e, clist_g[:,:,0], clist_g[:,:,2]).permute(0, 2, 1)
            ry_loss = self.compute_rot_loss(rylist_e, rylist_g, scorelist_g)
            total_loss = utils.misc.add_loss('center2d/ry_loss', total_loss, ry_loss, 1.0, sw)


        # now, let's convert the estimates into discrete boxes
        # this means: extract topk peaks from the centerness map,
        # and at those locations, extract the rotation and size estimates
        center_e_clean = center_e_sig.clone()
        center_e_clean = _nms(center_e_clean, kernel=15)
        if sw is not None:
            sw.summ_oned('center2d/center_e_clean', center_e_clean, norm=False)

        scorelist_e, xzlist_mem_e = _topk(center_e_clean, K=self.K)

        sizelist_e = utils.samp.bilinear_sample2d(size_e, xzlist_mem_e[:,:,0], xzlist_mem_e[:,:,1]).permute(0, 2, 1)
        offsetlist_e = utils.samp.bilinear_sample2d(offset_e, xzlist_mem_e[:,:,0], xzlist_mem_e[:,:,1]).permute(0, 2, 1)
        ycoordlist_e = utils.samp.bilinear_sample2d(ycoord_e, xzlist_mem_e[:,:,0], xzlist_mem_e[:,:,1]).permute(0, 2, 1)
        rylist_e = utils.samp.bilinear_sample2d(ry_e, xzlist_mem_e[:,:,0], xzlist_mem_e[:,:,1]).permute(0, 2, 1)

        # note that the predicted ycoord is in mem coords
        xyzlist_mem_e = torch.stack([xzlist_mem_e[:,:,0],
                                     ycoordlist_e[:,:,0],
                                     xzlist_mem_e[:,:,1]], dim=2)
        xyzlist_cam_e = vox_util.Mem2Ref(xyzlist_mem_e, Z8, Y8, X8, assert_cube=False)
        # # fancy new idea:
        # # at these peaks, apply another loss, using the nearest gt
        # # e.g., we would like offsets away from the object to point to the object
        # if (lrtlist_mem_g is not None):

        #     extra_size_loss = 0.0
        #     extra_offset_loss = 0.0
        #     extra_rot_loss = 0.0

        #     normalizer = 0.0
        #     for b in list(range(B)):
        #         for k in list(range(self.K)):
        #             xyz_e = xyzlist_mem_e[b:b+1, k]
        #             size_e = sizelist_e[b:b+1, k]
        #             offset_e = offsetlist_e[b:b+1, k]
        #             # these are 1 x 3
        #             # rx_e = rxlist_e[b:b+1, k]
        #             ry_e = rylist_e[b:b+1, k]
        #             # these are 1 x num_rot_bins
        #             # rz = rzlist_e[b:b+1, k]
        #             # these are 1 x 1
        #             xyz_g = clist_g[b:b+1]
        #             score_g = scorelist_g[b:b+1]
        #             xyz_g[score_g < 1.0] = 100000 # discard for mindist
        #             # this is 1 x N x 3
        #             dist = utils.basic.sql2_on_axis(xyz_g - xyz_e.unsqueeze(1), 2)
        #             # this is 1 x N
        #             ind = torch.argmin(dist, dim=1).squeeze()
        #             # print('ind', ind.detach().cpu().numpy(), ind.shape)
        #             xyz_g = clist_g[b:b+1,ind]
        #             size_g = sizelist_g[b:b+1,ind]
        #             score_g = scorelist_g[b:b+1,ind]
        #             mindist = dist[:,ind]

        #             # only proceed if the nn is valid, and not too far away
        #             if score_g.squeeze() == 1.0 and mindist.squeeze() < 8.0:
        #                 # offset_g = offsetlist_g[b:b+1,ind]
        #                 # for offset, we actually need to recompute
        #                 offset_g = xyz_g - xyz_e
        #                 # rx_g = rxlist_g[b:b+1,ind]
        #                 ry_g = rylist_g[b:b+1,ind]

        #                 # all the tensors of interest are 1x3, or 1xnum_bins for rots

        #                 # extra_rot_loss += 0.5 * self.compute_rot_loss(rx_e.unsqueeze(1), rx_g.unsqueeze(1), torch.ones_like(rx_g.unsqueeze(1)))
        #                 # extra_rot_loss += 0.5 * self.compute_rot_loss(ry_e.unsqueeze(1), ry_g.unsqueeze(1), torch.ones_like(ry_g.unsqueeze(1)))
        #                 extra_size_loss += torch.mean(torch.sum(self.smoothl1(size_e, size_g), dim=1))
        #                 extra_offset_loss += torch.mean(torch.sum(self.smoothl1(offset_e, offset_g), dim=1))
        #                 extra_rot_loss += self.compute_rot_loss(ry_e.unsqueeze(1), ry_g.unsqueeze(1), torch.ones_like(ry_g.unsqueeze(1)))

        #                 normalizer += 1
        #             else:
        #                 # print('discarding; mindist:', mindist.squeeze().detach().cpu().numpy())
        #                 pass

        #     if normalizer > 0:
        #         total_loss = utils.misc.add_loss('center2d/extra_size_loss', total_loss, extra_size_loss/normalizer, 0.1, sw)
        #         total_loss = utils.misc.add_loss('center2d/extra_offset_loss', total_loss, extra_offset_loss/normalizer, 0.1, sw)
        #         total_loss = utils.misc.add_loss('center2d/extra_rot_loss', total_loss, extra_rot_loss/normalizer, 0.1, sw)
                
        if (sw is not None and sw.save_this) or force_export_boxlist:

            # xyzlist_cam_e = vox_util.Mem2Ref(xyzlist_mem_e + offsetlist_e, Z, Y, X, assert_cube=False)

            boxlist = scorelist_e.new_zeros((B, self.K, 9))
            scorelist = scorelist_e.new_zeros((B, self.K))
            for b in list(range(B)):
                boxlist_b = []
                scorelist_b = []
                for k in list(range(self.K)):
                    score = scorelist_e[b:b+1, k]
                    # print('score', score.shape)
                    # print('score', score.squeeze().shape)
                    # let's call it a real object
                    if score.squeeze() > self.thresh:
                        # xyz = xyzlist_mem_e[b:b+1, k]
                        xyz = xyzlist_cam_e[b:b+1, k] # 1,3
                        size = sizelist_e[b:b+1, k] # 1,3
                        
                        ry = rylist_e[b:b+1, k] # 1,num_rot_bins
                        # i need to convert this into an actual rot
                        ry = ry.squeeze()
                        ry_ind = torch.argmax(ry)
                        ry = self.bin_angles[ry_ind].reshape(1)
                        rz = torch.zeros_like(ry)
                        rx = torch.zeros_like(ry)
                        rot = torch.stack([rx, ry, rz], dim=1) # 1, 3

                        box = torch.cat([xyz, size, rot], dim=1)
                        boxlist_b.append(box)
                        scorelist_b.append(score)
                if len(boxlist_b) > 0:
                    boxlist_b = torch.stack(boxlist_b, dim=1) # 1 x ? x 3
                    scorelist_b = torch.stack(scorelist_b, dim=1) # 1 x ? x 1
                    boxlist_b = torch.cat((boxlist_b, torch.zeros([1, self.K, 9]).cuda()), dim=1)
                    scorelist_b = torch.cat((scorelist_b, torch.zeros([1, self.K]).cuda()), dim=1)
                    boxlist_b = boxlist_b[:, :self.K]
                    scorelist_b = scorelist_b[:, :self.K]
                else:
                    boxlist_b = torch.zeros([1, self.K, 9]).cuda()
                    scorelist_b = torch.zeros([1, self.K]).cuda()
                boxlist[b:b+1] = boxlist_b
                scorelist[b:b+1] = scorelist_b
            lrtlist_cam = utils.geom.convert_boxlist_to_lrtlist(boxlist)

            if sw is not None and sw.save_this:
                sw.summ_lrtlist_bev(
                    'center2d/lrtlist_mem_e',
                    occ_mem,
                    lrtlist_cam[0:1],
                    scorelist[0:1], # scores
                    torch.ones(1,self.K).long().cuda(), # tids
                    vox_util,
                    already_mem=False)
                sw.summ_lrtlist_bev(
                    'center2d/lrtlist_mem_g',
                    occ_mem,
                    lrtlist_cam_g[0:1],
                    scorelist_g[0:1], # scores
                    torch.ones_like(scorelist_g).long(), # tids
                    vox_util,
                    already_mem=False)
        else:
            lrtlist_cam = None
            scorelist = None

        if sw is not None and sw.save_this:
            sw.summ_oned('center2d/center_e_sig', center_e_sig, norm=False)

            seg_e_sig = F.interpolate(torch.sigmoid(seg_e), scale_factor=self.stride)
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
            sw.summ_rgb('center2d/seg_e_on_occ', (occ_vis + seg_vis)/2.0)
            sw.summ_oned('center2d/seg_e_sig', torch.mean(seg_e_sig, dim=3), norm=True)
            
                
        return total_loss, lrtlist_cam, scorelist, seg_e
