import torch
import utils.basic
import utils.geom
import numpy as np
from cc3d import connected_components
import cv2

def get_any_boxes_from_binary(binary, N, min_voxels=3, min_side=1, count_mask=None):
    B, Z, Y, X = list(binary.shape)
        
    # turn the binary map into labels with connected_components

    assert(B==1) # later i will extend
    binary = binary[0]
    # binary is Z x Y x X

    if count_mask is None:
        count_mask = torch.ones_like(binary)
    else:
        count_mask = count_mask.reshape(Z, Y, X)
    
    mask = binary.detach().cpu().numpy().astype(np.int32)
    count_mask = count_mask.detach().cpu().numpy()

    from cc3d import connected_components

    boxlist = np.zeros([N, 9], dtype=np.float32)
    scorelist = np.zeros([N], dtype=np.float32)
    connlist = np.zeros([N, Z, Y, X], dtype=np.float32)
    boxcount = 0

    zg, yg, xg = utils.basic.meshgrid3d_py(Z, Y, X, stack=False, norm=False)
    box3d_list = []

    labels = connected_components(mask)
    segids = [ x for x in np.unique(labels) if x != 0 ]
    for si, segid in enumerate(segids):
        extracted_vox = (labels == segid)

        z = zg[extracted_vox==1]
        y = yg[extracted_vox==1]
        x = xg[extracted_vox==1]

        zmin = np.min(z)
        zmax = np.max(z)
        ymin = np.min(y)
        ymax = np.max(y)
        xmin = np.min(x)
        xmax = np.max(x)

        if (zmax-zmin > min_side and
            ymax-ymin > min_side and
            xmax-xmin > min_side and
            np.sum(extracted_vox*count_mask) > min_voxels):

            # find the oriented box in birdview
            im = np.sum(extracted_vox, axis=1) # reduce on the Y dim
            im = im.astype(np.uint8)

            # somehow the versions change 
            # _, contours, hier = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours, hier = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2:]
            if contours:
                cnt = contours[0]
                rect = cv2.minAreaRect(cnt)

                ymin = np.min(y)
                ymax = np.max(y)

                hei = ymax-ymin
                yc = (ymax+ymin)/2.0

                (xc,zc),(wid,dep),theta = rect
                theta = -theta

                box = cv2.boxPoints(rect)
                if dep < wid:
                    # dep goes along the long side of an oriented car
                    theta += 90.0
                    wid, dep = dep, wid
                theta = utils.geom.deg2rad(theta)
                
                if boxcount < N:#  and (yc > ymin_) and (yc < ymax_):
                    # bx, by = np.split(box, axis=1)
                    # boxpoints[boxcount,:] = box

                    box3d = [xc, yc, zc, wid, hei, dep, 0, theta, 0]
                    box3d = np.array(box3d).astype(np.float32)

                    already_have = False
                    for box3d_ in box3d_list:
                        if np.all(box3d_==box3d):
                            already_have = True

                    if not already_have:
                        box = np.int0(box)

                        boxlist[boxcount,:] = box3d
                        scorelist[boxcount] = 1.0

                        conn_ = np.zeros([Z, Y, X], np.float32)
                        conn_[extracted_vox] = 1.0
                        connlist[boxcount] = conn_

                        boxcount += 1
                        box3d_list.append(box3d)
                    else:
                        # print('skipping a box that already exists')
                        pass
                # endif boxcount
            # endif contours
        # endif sides
    # endloop over segments
        
    boxlist = torch.from_numpy(boxlist).float().to('cuda').unsqueeze(0)
    scorelist = torch.from_numpy(scorelist).float().to('cuda').unsqueeze(0)
    connlist = torch.from_numpy(connlist).float().to('cuda').unsqueeze(0)
    tidlist = torch.linspace(1.0, N, N).long().to('cuda')
    tidlist = tidlist.unsqueeze(0)
    return boxlist, scorelist, tidlist, connlist
            

class SimplePool():
    def __init__(self, pool_size, version='pt'):
        self.pool_size = pool_size
        self.version = version
        # random.seed(125)
        if self.pool_size > 0:
            self.num = 0
            self.items = []
        if not (version=='pt' or version=='np'):
            print('version = %s; please choose pt or np')
            assert(False) # please choose pt or np
            
    def __len__(self):
        return len(self.items)
    
    def mean(self, min_size='none'):
        if min_size=='half':
            pool_size_thresh = self.pool_size/2
        else:
            pool_size_thresh = 0
            
        if self.version=='np':
            if len(self.items) >= pool_size_thresh:
                return np.sum(self.items)/len(self.items)
            else:
                return np.nan
        if self.version=='pt':
            if len(self.items) >= pool_size_thresh:
                return torch.sum(self.items)/len(self.items)
            else:
                return torch.from_numpy(np.nan)
    
    def sample(self):
        idx = np.random.randint(len(self.items))
        return self.items[idx]
    
    def fetch(self, num=None):
        if self.version=='pt':
            item_array = torch.stack(self.items)
        elif self.version=='np':
            item_array = np.stack(self.items)
        if num is not None:
            # there better be some items
            assert(len(self.items) >= num)
                
            # if there are not that many elements just return however many there are
            if len(self.items) < num:
                return item_array
            else:
                idxs = np.random.randint(len(self.items), size=num)
                return item_array[idxs]
        else:
            return item_array
            
    def is_full(self):
        full = self.num==self.pool_size
        # print 'num = %d; full = %s' % (self.num, full)
        return full
    
    def empty(self):
        self.items = []
        self.num = 0
            
    def update(self, items):
        for item in items:
            if self.num < self.pool_size:
                # the pool is not full, so let's add this in
                self.num = self.num + 1
            else:
                # the pool is full
                # pop from the front
                self.items.pop(0)
            # add to the back
            self.items.append(item)
        return self.items


def get_cycle_consistent_transform_helper(
        xyz_cam0, xyz_cam1,
        flow_01,
        pix_T_cam, H, W,
        flow_valid=None,
        inlier_thresh=0.25):
    # this just does one direction

    xyz_cam1_i, valid_i = utils.geom.get_point_correspondence_from_flow(
        xyz_cam0, xyz_cam1, flow_01, pix_T_cam, H, W, flow_valid=flow_valid)
    xyz_cam0_i = xyz_cam0[valid_i>0]
    xyz_cam1_i = xyz_cam1_i[valid_i>0]
    
    cam1_T_cam0_i = get_rigid_transform(
        xyz_cam0_i, xyz_cam1_i,
        inlier_thresh=inlier_thresh,
        ransac_steps=512,
        recompute_with_inliers=True)
    # xyz_cam0_o = xyz_cam0[valid_i==0]
    # xyz_cam1_o = xyz_cam1[valid_i==0]

    corresp_tuple = (xyz_cam0_i.unsqueeze(0), xyz_cam1_i.unsqueeze(0))
    return cam1_T_cam0_i.unsqueeze(0), corresp_tuple
# (xyz_cam0_o.unsqueeze(0), xyz_cam1_o.unsqueeze(0)))

def get_cycle_consistent_transform(
        xyz_cam0, xyz_cam1,
        flow_01, flow_10,
        pix_T_cam, H, W,
        flow_01_valid=None,
        flow_10_valid=None,
        inlier_thresh=0.25):

    # forward direction
    cam1_T_cam0_fw, corresp_tuple = get_cycle_consistent_transform_helper(
        xyz_cam0, xyz_cam1,
        flow_01,
        pix_T_cam, H, W,
        flow_valid=flow_01_valid,
        inlier_thresh=inlier_thresh)
    cam0_T_cam1_bw, _ = get_cycle_consistent_transform_helper(
        xyz_cam1, xyz_cam0,
        flow_10,
        pix_T_cam, H, W,
        flow_valid=flow_10_valid,
        inlier_thresh=inlier_thresh)

    # now we want to see if these are inverses of each other
    
    # first gather the valids
    xyz_cam0 = xyz_cam0.reshape(-1, 3)
    # valid_cam0 = torch.norm(xyz_cam0, dim=1) > 1e-4
    # xyz_cam0 = xyz_cam0[xyz_cam0[:,2] > 0]
    # xyz_cam0 = xyz_cam0.unsqueeze(0)
    # # print('xyz_cam0', xyz_cam0.shape)
    # xyz_cam1 = xyz_cam1.reshape(-1, 3)
    # xyz_cam1 = xyz_cam1[xyz_cam1[:,2] > 0]
    # xyz_cam1 = xyz_cam1.unsqueeze(0)
    # print('xyz_cam1', xyz_cam1.shape)
    xyz_cam0 = xyz_cam0[torch.norm(xyz_cam0, dim=1) > 1e-4]
    xyz_cam0 = xyz_cam0.unsqueeze(0)

    cam0_T_cam0 = utils.basic.matmul2(cam0_T_cam1_bw, cam1_T_cam0_fw)
    xyz_cam0_prime = utils.geom.apply_4x4(cam0_T_cam0, xyz_cam0)

    dist = torch.norm(xyz_cam0-xyz_cam0_prime, dim=2)

    return cam1_T_cam0_fw, dist, corresp_tuple#, noncorresp_tuple
    

def get_rigid_transform(xyz0, xyz1, inlier_thresh=0.04, ransac_steps=256, recompute_with_inliers=False):
    xyz0 = xyz0.detach().cpu().numpy()
    xyz1 = xyz1.detach().cpu().numpy()
    # xyz0 and xyz1 are each N x 3
    assert len(xyz0) == len(xyz1)

    # utils.py.print_stats('xyz0', xyz0)
    # utils.py.print_stats('xyz1', xyz1)
    
    N = xyz0.shape[0] # total points
    nPts = 8
    # assert(N > nPts)
    if N < nPts:
        print('grt: too few points; returning translation')
        R = np.eye(3, dtype=np.float32)
        t = np.mean(xyz1-xyz0, axis=0)
        print('t', t)
        rt = utils.py.merge_rt(R, t)
        return torch.from_numpy(rt).cuda()

    # print('N = %d' % N)
    # print('doing ransac')
    rts = []
    errs = []
    inliers = []
    for step in list(range(ransac_steps)):
        # assert(N > nPts) 
        perm = np.random.permutation(N)
        cam1_T_cam0, _ = rigid_transform_3d_py_helper(xyz0[perm[:nPts]], xyz1[perm[:nPts]])
        # i got some errors in matmul when the arrays were too big,
        # so let's just use 1k points for the error 
        perm = np.random.permutation(N)
        xyz1_prime = utils.geom.apply_4x4_py(cam1_T_cam0, xyz0[perm[:min([1000,N])]])
        xyz1_actual = xyz1[perm[:min([1000,N])]]
        # N x 3
        
        # print('xyz1_prime', xyz1_prime.shape)
        # print('xyz1_actual', xyz1_prime.shape)
        # err = np.mean(np.sum(np.abs(xyz1_prime-xyz1_actual), axis=1))
        err = np.linalg.norm(xyz1_prime-xyz1_actual, axis=1)
        # utils.py.print_stats('err', err)
        # print('err', err)
        inlier = (err < inlier_thresh).astype(np.float32)
        # print('inlier', inlier)
        inlier_count = np.sum(err < inlier_thresh)
        # print('inlier_count', inlier_count)
        # input()
        rts.append(cam1_T_cam0)
        errs.append(np.mean(err))
        inliers.append(inlier_count)
    # print('errs', errs)
    # print('inliers', inliers)
    ind0 = np.argmin(errs)
    ind1 = np.argmax(inliers)
    # print('ind0=%d, err=%.3f, inliers=%d' % (ind0, errs[ind0], inliers[ind0]))
    # print('ind1=%d, err=%.3f, inliers=%d' % (ind1, errs[ind1], inliers[ind1]))
    rt0 = rts[ind0]
    rt1 = rts[ind1]
    # print('rt0', rt0)
    # print('rt1', rt1)

    cam1_T_cam0 = rt1

    if recompute_with_inliers:
        xyz1_prime = utils.geom.apply_4x4_py(cam1_T_cam0, xyz0)
        # N x 3
        err = np.linalg.norm(xyz1_prime-xyz1, axis=1)
        inlier = (err < inlier_thresh).astype(np.float32)
        xyz0_inlier = xyz0[inlier > 0]
        xyz1_inlier = xyz1[inlier > 0]
        cam1_T_cam0, _ = rigid_transform_3d_py_helper(xyz0_inlier, xyz1_inlier)
        
    cam1_T_cam0 = torch.from_numpy(cam1_T_cam0).cuda().float()
    
    return cam1_T_cam0

def rigid_transform_3d_py_helper(xyz0, xyz1, do_scaling=False):
    assert len(xyz0) == len(xyz1)
    N = xyz0.shape[0] # total points
    if N >= 3:
        centroid_xyz0 = np.mean(xyz0, axis=0)
        centroid_xyz1 = np.mean(xyz1, axis=0)
        # print('centroid_xyz0', centroid_xyz0)
        # print('centroid_xyz1', centroid_xyz1)

        # center the points
        xyz0 = xyz0 - np.tile(centroid_xyz0, (N, 1))
        xyz1 = xyz1 - np.tile(centroid_xyz1, (N, 1))

        H = np.dot(xyz0.T, xyz1) / N

        U, S, Vt = np.linalg.svd(H)

        R = np.dot(Vt.T, U.T)

        # special reflection case
        if np.linalg.det(R) < 0:
           Vt[2,:] *= -1
           S[-1] *= -1
           R = np.dot(Vt.T, U.T) 

        # varP = np.var(xyz0, axis=0).sum()
        # c = 1/varP * np.sum(S) # scale factor
        varP = np.var(xyz0, axis=0)
        # varQ_aligned = np.var(np.dot(xyz1, R.T), axis=0)
        varQ_aligned = np.var(np.dot(xyz1, R), axis=0)

        c = np.sqrt(varQ_aligned / varP) # anisotropic

        if not do_scaling:
            # c = 1.0 # keep it 1.0
            c = np.ones(3) # keep it 1.0

        # t = c * np.dot(-R, centroid_xyz0.T) + centroid_xyz1.T
        t = -np.dot(np.dot(R, np.diag(c)), centroid_xyz0.T) + centroid_xyz1.T

        t = np.reshape(t, [3])
    else:
        # print('too few points; returning identity')
        # R = np.eye(3, dtype=np.float32)
        # t = np.zeros(3, dtype=np.float32)

        print('too few points; returning translation')
        R = np.eye(3, dtype=np.float32)
        t = np.mean(xyz1-xyz0, axis=0)
        # c = 1.0
        c = np.ones(3) # keep it 1.0
        
    rt = utils.geom.merge_rt_py(R, t)
    return rt, c
