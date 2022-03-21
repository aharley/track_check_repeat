import torch
import utils.basic
import numpy as np
import torchvision.ops as ops

def eye_3x3(B, device='cuda'):
    rt = torch.eye(3, device=torch.device(device)).view(1,3,3).repeat([B, 1, 1])
    return rt

def eye_4x4(B, device='cuda'):
    rt = torch.eye(4, device=torch.device(device)).view(1,4,4).repeat([B, 1, 1])
    return rt

def safe_inverse(a):
    B, _, _ = list(a.shape)
    inv = a.clone()
    r_transpose = a[:, :3, :3].transpose(1,2) # inverse of rotation matrix
    inv[:, :3, :3] = r_transpose
    inv[:, :3, 3:4] = -torch.matmul(r_transpose, a[:, :3, 3:4])
    return inv

def merge_rt(r, t):
    # r is B x 3 x 3
    # t is B x 3
    B, C, D = list(r.shape)
    B2, D2 = list(t.shape)
    assert(C==3)
    assert(D==3)
    assert(B==B2)
    assert(D2==3)
    t = t.view(B, 3)
    rt = eye_4x4(B, device=t.device)
    rt[:,:3,:3] = r
    rt[:,:3,3] = t
    return rt

def pixels2camera3(xyz,pix_T_cam):
    x,y,z = xyz[:,:,0],xyz[:,:,1],xyz[:,:,2]
    fx, fy, x0, y0 = split_intrinsics(pix_T_cam)
    return pixels2camera(x,y,z,fx,fy,x0,y0)

def pixels2camera2(x,y,z,pix_T_cam):
    fx, fy, x0, y0 = split_intrinsics(pix_T_cam)
    xyz = pixels2camera(x,y,z,fx,fy,x0,y0)
    return xyz

def pixels2camera(x,y,z,fx,fy,x0,y0):
    # x and y are locations in pixel coordinates, z is a depth in meters
    # they can be images or pointclouds
    # fx, fy, x0, y0 are camera intrinsics
    # returns xyz, sized B x N x 3

    B = x.shape[0]
    
    fx = torch.reshape(fx, [B,1])
    fy = torch.reshape(fy, [B,1])
    x0 = torch.reshape(x0, [B,1])
    y0 = torch.reshape(y0, [B,1])

    x = torch.reshape(x, [B,-1])
    y = torch.reshape(y, [B,-1])
    z = torch.reshape(z, [B,-1])
    
    # unproject
    x = (z/fx)*(x-x0)
    y = (z/fy)*(y-y0)
    
    xyz = torch.stack([x,y,z], dim=2)
    # B x N x 3
    return xyz

def camera2pixels(xyz, pix_T_cam):
    # xyz is shaped B x H*W x 3
    # returns xy, shaped B x H*W x 2
    
    fx, fy, x0, y0 = split_intrinsics(pix_T_cam)
    x, y, z = torch.unbind(xyz, dim=-1)
    B = list(z.shape)[0]

    fx = torch.reshape(fx, [B,1])
    fy = torch.reshape(fy, [B,1])
    x0 = torch.reshape(x0, [B,1])
    y0 = torch.reshape(y0, [B,1])
    x = torch.reshape(x, [B,-1])
    y = torch.reshape(y, [B,-1])
    z = torch.reshape(z, [B,-1])

    EPS = 1e-4
    z = torch.clamp(z, min=EPS)
    x = (x*fx)/z + x0
    y = (y*fy)/z + y0
    xy = torch.stack([x, y], dim=-1)
    return xy

def eul2rotm(rx, ry, rz):
    # inputs are shaped B
    # this func is copied from matlab
    # R = [  cy*cz   sy*sx*cz-sz*cx    sy*cx*cz+sz*sx
    #        cy*sz   sy*sx*sz+cz*cx    sy*cx*sz-cz*sx
    #        -sy            cy*sx             cy*cx]
    rx = torch.unsqueeze(rx, dim=1)
    ry = torch.unsqueeze(ry, dim=1)
    rz = torch.unsqueeze(rz, dim=1)
    # these are B x 1
    sinz = torch.sin(rz)
    siny = torch.sin(ry)
    sinx = torch.sin(rx)
    cosz = torch.cos(rz)
    cosy = torch.cos(ry)
    cosx = torch.cos(rx)
    r11 = cosy*cosz
    r12 = sinx*siny*cosz - cosx*sinz
    r13 = cosx*siny*cosz + sinx*sinz
    r21 = cosy*sinz
    r22 = sinx*siny*sinz + cosx*cosz
    r23 = cosx*siny*sinz - sinx*cosz
    r31 = -siny
    r32 = sinx*cosy
    r33 = cosx*cosy
    r1 = torch.stack([r11,r12,r13],dim=2)
    r2 = torch.stack([r21,r22,r23],dim=2)
    r3 = torch.stack([r31,r32,r33],dim=2)
    r = torch.cat([r1,r2,r3],dim=1)
    return r

def apply_4x4(RT, xyz):
    B, N, _ = list(xyz.shape)
    ones = torch.ones_like(xyz[:,:,0:1])
    xyz1 = torch.cat([xyz, ones], 2)
    xyz1_t = torch.transpose(xyz1, 1, 2)
    # this is B x 4 x N
    xyz2_t = torch.matmul(RT, xyz1_t)
    xyz2 = torch.transpose(xyz2_t, 1, 2)
    # xyz2 = xyz2 / xyz2[:,:,3:4]
    xyz2 = xyz2[:,:,:3]
    return xyz2

def merge_rt_py(r, t):
    # r is 3 x 3
    # t is 3 or maybe 3 x 1
    t = np.reshape(t, [3, 1])
    rt = np.concatenate((r,t), axis=1)
    # rt is 3 x 4
    br = np.reshape(np.array([0,0,0,1], np.float32), [1, 4])
    # br is 1 x 4
    rt = np.concatenate((rt, br), axis=0)
    # rt is 4 x 4
    return rt

def split_rt_py(rt):
    r = rt[:3,:3]
    t = rt[:3,3]
    r = np.reshape(r, [3, 3])
    t = np.reshape(t, [3, 1])
    return r, t

def apply_4x4_py(rt, xyz):
    # rt is 4 x 4
    # xyz is N x 3
    r, t = split_rt_py(rt)
    xyz = np.transpose(xyz, [1, 0])
    # xyz is xyz1 x 3 x N
    xyz = np.dot(r, xyz)
    # xyz is xyz1 x 3 x N
    xyz = np.transpose(xyz, [1, 0])
    # xyz is xyz1 x N x 3
    t = np.reshape(t, [1, 3])
    xyz = xyz + t
    return xyz

def merge_lrtlist(lenlist, rtlist):
    # lenlist is B x N x 3
    # rtlist is B x N x 4 x 4
    # merges these into a B x N x 19 tensor
    B, N, D = list(lenlist.shape)
    assert(D==3)
    B2, N2, E, F = list(rtlist.shape)
    assert(B==B2)
    assert(N==N2)
    assert(E==4 and F==4)
    rtlist = rtlist.reshape(B, N, 16)
    lrtlist = torch.cat([lenlist, rtlist], axis=2)
    return lrtlist

def convert_boxlist_to_lrtlist(boxlist):
    B, N, D = list(boxlist.shape)
    assert(D==9)
    boxlist_ = boxlist.view(B*N, D)
    rtlist_ = convert_box_to_ref_T_obj(boxlist_)
    rtlist = rtlist_.view(B, N, 4, 4)
    lenlist = boxlist[:,:,3:6].reshape(B, N, 3)
    lenlist = lenlist.clamp(min=0.01)
    lrtlist = merge_lrtlist(lenlist, rtlist)
    return lrtlist

def convert_box_to_ref_T_obj(box3d):
    # turn the box into obj_T_ref (i.e., obj_T_cam)
    B = list(box3d.shape)[0]

    # box3d is B x 9
    x, y, z, lx, ly, lz, rx, ry, rz = torch.unbind(box3d, axis=1)
    rot0 = eye_3x3(B, device=box3d.device)
    tra = torch.stack([x, y, z], axis=1)
    center_T_ref = merge_rt(rot0, -tra)
    # center_T_ref is B x 4 x 4
    
    t0 = torch.zeros([B, 3], device=box3d.device)
    rot = eul2rotm(rx, ry, rz)
    rot = torch.transpose(rot, 1, 2) # other dir
    obj_T_center = merge_rt(rot, t0)
    # this is B x 4 x 4

    # we want obj_T_ref
    # first we to translate to center,
    # and then rotate around the origin
    obj_T_ref = utils.basic.matmul2(obj_T_center, center_T_ref)

    # return the inverse of this, so that we can transform obj corners into cam coords
    ref_T_obj = obj_T_ref.inverse()
    return ref_T_obj

def get_point_correspondence_from_flow(xyz0, xyz1, flow_f, pix_T_cam, H, W, flow_valid=None):
    # flow_f is the forward flow, from frame0 to frame1
    # xyz0 and xyz1 are pointclouds, in cam coords
    # we want to get a new xyz1, with points that correspond to xyz0
    B, N, D = list(xyz0.shape)

    # discard depths that are beyond this distance, since they are probably fake
    max_dist = 200.0
    
    # now sample the 2d flow vectors at the xyz0 locations
    # ah wait!:
    # it's important here to only use positions in front of the camera
    xy = apply_pix_T_cam(pix_T_cam, xyz0)
    z0 = xyz0[:, :, 2] # B x N
    x0 = xy[:, :, 0] # B x N
    y0 = xy[:, :, 1] # B x N
    uv = utils.samp.bilinear_sample2d(flow_f, x0, y0) # B x 2 x N

    frustum0_valid = get_image_inbounds(pix_T_cam, xyz0, H, W)

    # next we want to use these to sample into the depth of the next frame 
    # depth0, valid0 = create_depth_image(pix_T_cam, xyz0, H, W)
    depth1, valid1 = create_depth_image(pix_T_cam, xyz1, H, W)
    # valid0 = valid0 * (depth0 < max_dist).float()
    valid1 = valid1 * (depth1 < max_dist).float()
    
    u = uv[:, 0] # B x N
    v = uv[:, 1] # B x N
    x1 = x0 + u
    y1 = y0 + v

    # round to the nearest pixel, since the depth image has holes
    # x0 = torch.clamp(torch.round(x0), 0, W-1).long()
    # y0 = torch.clamp(torch.round(y0), 0, H-1).long()
    x1 = torch.clamp(torch.round(x1), 0, W-1).long()
    y1 = torch.clamp(torch.round(y1), 0, H-1).long()
    z1 = torch.zeros(B, N, dtype=torch.float32, device=torch.device('cuda'))
    valid = torch.zeros(B, N, dtype=torch.float32, device=torch.device('cuda'))
    # since we rounded and clamped, we can index directly, instead of bilinear sampling

    for b in range(B):
        # depth0_b = depth0[b] # 1 x H x W
        # valid0_b = valid0[b]
        # valid0_b_ = valid0_b[0, y0[b], x0[b]] # N
        # z0_b_ = depth0_b[0, y1[b], x1[b]] # N
        
        depth1_b = depth1[b] # 1 x H x W
        valid1_b = valid1[b]
        valid1_b_ = valid1_b[0, y1[b], x1[b]] # N
        z1_b_ = depth1_b[0, y1[b], x1[b]] # N
        
        z1[b] = z1_b_
        # valid[b] = valid0_b_ * valid1_b_ * frustum0_valid[b]
        valid[b] = valid1_b_ * frustum0_valid[b]

        if flow_valid is not None:
            validf_b = flow_valid[b] # 1 x H x W
            validf_b_ = validf_b[0, y1[b], x1[b]] # N
            valid[b] = valid[b] * validf_b_

    fx, fy, x0, y0 = split_intrinsics(pix_T_cam)
    xyz1 = pixels2camera(x1,y1,z1,fx,fy,x0,y0)
    # xyz1 is B x N x 3
    # valid is B x N
    return xyz1, valid

def apply_pix_T_cam(pix_T_cam, xyz):

    fx, fy, x0, y0 = split_intrinsics(pix_T_cam)
    
    # xyz is shaped B x H*W x 3
    # returns xy, shaped B x H*W x 2
    
    B, N, C = list(xyz.shape)
    assert(C==3)
    
    x, y, z = torch.unbind(xyz, axis=-1)

    fx = torch.reshape(fx, [B, 1])
    fy = torch.reshape(fy, [B, 1])
    x0 = torch.reshape(x0, [B, 1])
    y0 = torch.reshape(y0, [B, 1])

    EPS = 1e-4
    z = torch.clamp(z, min=EPS)
    x = (x*fx)/(z)+x0
    y = (y*fy)/(z)+y0
    xy = torch.stack([x, y], axis=-1)
    return xy

def scale_intrinsics(K, sx, sy):
    fx, fy, x0, y0 = split_intrinsics(K)
    fx = fx*sx
    fy = fy*sy
    x0 = x0*sx
    y0 = y0*sy
    K = pack_intrinsics(fx, fy, x0, y0)
    return K

def split_intrinsics(K):
    # K is B x 3 x 3 or B x 4 x 4
    fx = K[:,0,0]
    fy = K[:,1,1]
    x0 = K[:,0,2]
    y0 = K[:,1,2]
    return fx, fy, x0, y0

def pack_intrinsics(fx, fy, x0, y0):
    B = list(fx.shape)[0]
    K = torch.zeros(B, 4, 4, dtype=torch.float32, device=torch.device('cuda'))
    K[:,0,0] = fx
    K[:,1,1] = fy
    K[:,0,2] = x0
    K[:,1,2] = y0
    K[:,2,2] = 1.0
    K[:,3,3] = 1.0
    return K

def get_image_inbounds(pix_T_cam, xyz_cam, H, W, padding=0.0):
    # pix_T_cam is B x 4 x 4
    # xyz_cam is B x N x 3
    # padding should be 0 unless you are trying to account for some later cropping
    
    xy_pix = utils.geom.apply_pix_T_cam(pix_T_cam, xyz_cam)

    x = xy_pix[:,:,0]
    y = xy_pix[:,:,1]
    z = xyz_cam[:,:,2]

    # print('x', x.detach().cpu().numpy())
    # print('y', y.detach().cpu().numpy())
    # print('z', z.detach().cpu().numpy())

    x_valid = ((x-padding)>-0.5).bool() & ((x+padding)<float(W-0.5)).bool()
    y_valid = ((y-padding)>-0.5).bool() & ((y+padding)<float(H-0.5)).bool()
    z_valid = ((z>0.0)).bool()

    inbounds = x_valid & y_valid & z_valid
    return inbounds.bool()

def create_depth_image_single(xy, z, H, W, force_positive=True, max_val=100.0, serial=False, slices=20):
    # turn the xy coordinates into image inds
    xy = torch.round(xy).long()
    depth = torch.zeros(H*W, dtype=torch.float32, device=xy.device)
    depth[:] = max_val
    
    # lidar reports a sphere of measurements
    # only use the inds that are within the image bounds
    # also, only use forward-pointing depths (z > 0)
    valid_inds = (xy[:,0] <= W-1) & (xy[:,1] <= H-1) & (xy[:,0] >= 0) & (xy[:,1] >= 0) & (z[:] > 0)

    # gather these up
    xy = xy[valid_inds]
    z = z[valid_inds]

    inds = utils.basic.sub2ind(H, W, xy[:,1], xy[:,0]).long()
    if not serial:
        depth[inds] = z
    else:
        if False:
            for (index, replacement) in zip(inds, z):
                if depth[index] > replacement:
                    depth[index] = replacement
        # ok my other idea is:
        # sort the depths by distance
        # create N depth maps
        # merge them back-to-front

        # actually maybe you don't even need the separate maps

        sort_inds = torch.argsort(z, descending=True)
        xy = xy[sort_inds]
        z = z[sort_inds]
        N = len(sort_inds)
        def split(a, n):
            k, m = divmod(len(a), n)
            return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

        slice_inds = split(range(N), slices)
        for si in slice_inds:
            mini_z = z[si]
            mini_xy = xy[si]
            inds = utils.basic.sub2ind(H, W, mini_xy[:,1], mini_xy[:,0]).long()
            depth[inds] = mini_z
        # cool; this is rougly as fast as the parallel, and as accurate as the serial
        
        if False:
            print('inds', inds.shape)
            unique, inverse, counts = torch.unique(inds, return_inverse=True, return_counts=True)
            print('unique', unique.shape)

            perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
            inverse, perm = inverse.flip([0]), perm.flip([0])
            perm = inverse.new_empty(unique.size(0)).scatter_(0, inverse, perm)

            # new_inds = inds[inverse_inds]
            # depth[new_inds] = z[unique_inds]

            depth[unique] = z[perm]

            # now for the duplicates...

            dup = counts > 1
            dup_unique = unique[dup]
            print('dup_unique', dup_unique.shape)
            depth[dup_unique] = 0.5
        
    if force_positive:
        # valid = (depth > 0.0).float()
        depth[torch.where(depth == 0.0)] = max_val
    # else:
    #     valid = torch.ones_like(depth)

    valid = (depth > 0.0).float() * (depth < max_val).float()
    
    depth = torch.reshape(depth, [1, H, W])
    valid = torch.reshape(valid, [1, H, W])
    return depth, valid

def create_depth_image(pix_T_cam, xyz_cam, H, W, offset_amount=0, max_val=100.0, serial=False, slices=20):
    B, N, D = list(xyz_cam.shape)
    assert(D==3)
    B2, E, F = list(pix_T_cam.shape)
    assert(B==B2)
    assert(E==4)
    assert(F==4)
    xy = apply_pix_T_cam(pix_T_cam, xyz_cam)
    z = xyz_cam[:,:,2]

    depth = torch.zeros(B, 1, H, W, dtype=torch.float32, device=xyz_cam.device)
    valid = torch.zeros(B, 1, H, W, dtype=torch.float32, device=xyz_cam.device)
    for b in list(range(B)):
        xy_b, z_b = xy[b], z[b]
        ind = z_b > 0
        xy_b = xy_b[ind]
        z_b = z_b[ind]
        depth_b, valid_b = create_depth_image_single(xy_b, z_b, H, W, max_val=max_val, serial=serial, slices=slices)
        if offset_amount:
            depth_b = depth_b.reshape(-1)
            valid_b = valid_b.reshape(-1)
            
            for off_x in range(offset_amount):
                for off_y in range(offset_amount):
                    for sign in [-1,1]:
                        offset = np.array([sign*off_x,sign*off_y]).astype(np.float32)
                        offset = torch.from_numpy(offset).reshape(1, 2).to(xyz_cam.device)
                        # offsets.append(offset)
                        depth_, valid_ = create_depth_image_single(xy_b + offset, z_b, H, W, max_val=max_val)
                        depth_ = depth_.reshape(-1)
                        valid_ = valid_.reshape(-1)
                        # at invalid locations, use this new value
                        depth_b[valid_b==0] = depth_[valid_b==0]
                        valid_b[valid_b==0] = valid_[valid_b==0]
                    
            depth_b = depth_b.reshape(1, H, W)
            valid_b = valid_b.reshape(1, H, W)
        depth[b] = depth_b
        valid[b] = valid_b
    return depth, valid

def depthrt2flow(depth_cam0, cam1_T_cam0, pix_T_cam):
    B, C, H, W = list(depth_cam0.shape)
    assert(C==1)

    # get the two pointclouds
    xyz_cam0 = depth2pointcloud(depth_cam0, pix_T_cam)
    xyz_cam1 = apply_4x4(cam1_T_cam0, xyz_cam0)

    # project, and get 2d flow
    flow = pointcloud2flow(xyz_cam1, pix_T_cam, H, W)
    return flow

def depth2pointcloud(z, pix_T_cam):
    B, C, H, W = list(z.shape)
    device = z.device
    y, x = utils.basic.meshgrid2d(B, H, W, device=device)
    z = torch.reshape(z, [B, H, W])
    fx, fy, x0, y0 = split_intrinsics(pix_T_cam)
    xyz = pixels2camera(x, y, z, fx, fy, x0, y0)
    return xyz

def pointcloud2flow(xyz1, pix_T_cam, H, W):
    # project xyz1 down, so that we get the 2d location of all of these pixels,
    # then subtract these 2d locations from the original ones to get optical flow
    
    B, N, C = list(xyz1.shape)
    assert(N==H*W)
    assert(C==3)
    
    # we assume xyz1 is the unprojection of the regular grid
    grid_y0, grid_x0 = utils.basic.meshgrid2d(B, H, W)

    xy1 = camera2pixels(xyz1, pix_T_cam)
    x1, y1 = torch.unbind(xy1, dim=2)
    x1 = x1.reshape(B, H, W)
    y1 = y1.reshape(B, H, W)

    flow_x = x1 - grid_x0
    flow_y = y1 - grid_y0
    flow = torch.stack([flow_x, flow_y], axis=1)
    # flow is B x 2 x H x W
    return flow

def get_boxlist2d_from_lrtlist(pix_T_cam, lrtlist_cam, H, W, pad=0, clamp=False):
    B, N, D = list(lrtlist_cam.shape)
    assert(D==19)
    corners_cam = get_xyzlist_from_lrtlist(lrtlist_cam)
    # this is B x N x 8 x 3
    corners_cam_ = torch.reshape(corners_cam, [B, N*8, 3])
    corners_pix_ = apply_pix_T_cam(pix_T_cam, corners_cam_)
    corners_pix = torch.reshape(corners_pix_, [B, N, 8, 2])

    xmin = torch.min(corners_pix[:,:,:,0], dim=2)[0]
    xmax = torch.max(corners_pix[:,:,:,0], dim=2)[0]
    ymin = torch.min(corners_pix[:,:,:,1], dim=2)[0]
    ymax = torch.max(corners_pix[:,:,:,1], dim=2)[0]
    # these are B x N

    if pad > 0:
        xmin = xmin - pad
        ymin = ymin - pad
        xmax = xmax + pad
        ymax = ymax + pad

    boxlist2d = torch.stack([ymin, xmin, ymax, xmax], dim=2)
    boxlist2d = normalize_boxlist2d(boxlist2d, H, W)

    if clamp:
        boxlist2d = boxlist2d.clamp(0,1)
    return boxlist2d

def xyd2pointcloud(xyd, pix_T_cam):
    # xyd is like a pointcloud but in pixel coordinates;
    # this means xy comes from a meshgrid with bounds H, W, 
    # and d comes from a depth map
    B, N, C = list(xyd.shape)
    assert(C==3)
    fx, fy, x0, y0 = split_intrinsics(pix_T_cam)
    xyz = pixels2camera(xyd[:,:,0], xyd[:,:,1], xyd[:,:,2], fx, fy, x0, y0)
    return xyz

def rad2deg(rad):
    return rad*180.0/np.pi

def deg2rad(deg):
    return deg/180.0*np.pi

def wrap2pi(rad_angle):
    # puts the angle into the range [-pi, pi]
    return torch.atan2(torch.sin(rad_angle), torch.cos(rad_angle))

def split_rt_single(rt):
    r = rt[:3, :3]
    t = rt[:3, 3].view(3)
    return r, t

def split_rt(rt):
    r = rt[:, :3, :3]
    t = rt[:, :3, 3].view(-1, 3)
    return r, t

def split_lrt(lrt):
    # splits a B x 19 tensor
    # into B x 3 (lens)
    # and B x 4 x 4 (rts)
    B, D = list(lrt.shape)
    assert(D==19)
    lrt = lrt.unsqueeze(1)
    l, rt = split_lrtlist(lrt)
    l = l.squeeze(1)
    rt = rt.squeeze(1)
    return l, rt

def split_lrtlist(lrtlist):
    # splits a B x N x 19 tensor
    # into B x N x 3 (lens)
    # and B x N x 4 x 4 (rts)
    B, N, D = list(lrtlist.shape)
    assert(D==19)
    lenlist = lrtlist[:,:,:3].reshape(B, N, 3)
    ref_T_objs_list = lrtlist[:,:,3:].reshape(B, N, 4, 4)
    return lenlist, ref_T_objs_list

def get_arrowheadlist_from_lrtlist(lrtlist):
    B, N, D = list(lrtlist.shape)
    assert(D==19)

    lenlist, rtlist = split_lrtlist(lrtlist)
    # lenlist is B x N x 3
    # rtlist is B x N x 4 x 4

    xyzlist_obj = torch.zeros((B, N, 1, 3), device=lrtlist.device)
    # xyzlist_obj is B x N x 8 x 3

    # unit vector in Z direction
    arrow_head_init = torch.Tensor([[0,0,1]]).to(lrtlist.device).repeat(B*N,1,1)
    # arrow_head_init = torch.Tensor([[1,0,0]]).cuda().repeat(B*N,1,1) 
    lenlist_ = lenlist.reshape(B*N,1,3)

    arrow_head_ = arrow_head_init * lenlist_

    rtlist_ = rtlist.reshape(B*N, 4, 4)
    xyzlist_obj_ = xyzlist_obj.reshape(B*N, 1, 3) + arrow_head_

    xyzlist_cam_ = apply_4x4(rtlist_, xyzlist_obj_)
    xyzlist_cam = xyzlist_cam_.reshape(B, N, 3)
    return xyzlist_cam

def get_clist_from_lrtlist(lrtlist):
    B, N, D = list(lrtlist.shape)
    assert(D==19)

    lenlist, rtlist = split_lrtlist(lrtlist)
    # lenlist is B x N x 3
    # rtlist is B x N x 4 x 4

    xyzlist_obj = torch.zeros((B, N, 1, 3), device=lrtlist.device)
    # xyzlist_obj is B x N x 8 x 3

    rtlist_ = rtlist.reshape(B*N, 4, 4)
    xyzlist_obj_ = xyzlist_obj.reshape(B*N, 1, 3)
    xyzlist_cam_ = apply_4x4(rtlist_, xyzlist_obj_)
    xyzlist_cam = xyzlist_cam_.reshape(B, N, 3)
    return xyzlist_cam

def get_xyzlist_from_lenlist(lenlist):
    B, N, D = list(lenlist.shape)
    assert(D==3)
    lx, ly, lz = torch.unbind(lenlist, axis=2)

    xs = torch.stack([lx/2., lx/2., -lx/2., -lx/2., lx/2., lx/2., -lx/2., -lx/2.], axis=2)
    ys = torch.stack([ly/2., ly/2., ly/2., ly/2., -ly/2., -ly/2., -ly/2., -ly/2.], axis=2)
    zs = torch.stack([lz/2., -lz/2., -lz/2., lz/2., lz/2., -lz/2., -lz/2., lz/2.], axis=2)
    
    # these are B x N x 8
    xyzlist = torch.stack([xs, ys, zs], axis=3)
    # this is B x N x 8 x 3
    return xyzlist

def get_xyzlist_from_lrtlist(lrtlist, include_clist=False):
    B, N, D = list(lrtlist.shape)
    assert(D==19)

    lenlist, rtlist = split_lrtlist(lrtlist)
    # lenlist is B x N x 3
    # rtlist is B x N x 4 x 4

    xyzlist_obj = get_xyzlist_from_lenlist(lenlist)
    # xyzlist_obj is B x N x 8 x 3

    rtlist_ = rtlist.reshape(B*N, 4, 4)
    xyzlist_obj_ = xyzlist_obj.reshape(B*N, 8, 3)
    xyzlist_cam_ = apply_4x4(rtlist_, xyzlist_obj_)
    xyzlist_cam = xyzlist_cam_.reshape(B, N, 8, 3)
    
    if include_clist:
        clist_cam = get_clist_from_lrtlist(lrtlist).unsqueeze(2)
        xyzlist_cam = torch.cat([xyzlist_cam, clist_cam], dim=2)
    return xyzlist_cam

def normalize_boxlist2d(boxlist2d, H, W):
    boxlist2d = boxlist2d.clone()
    ymin, xmin, ymax, xmax = torch.unbind(boxlist2d, dim=2)
    ymin = ymin / float(H)
    ymax = ymax / float(H)
    xmin = xmin / float(W)
    xmax = xmax / float(W)
    boxlist2d = torch.stack([ymin, xmin, ymax, xmax], dim=2)
    return boxlist2d

def unnormalize_boxlist2d(boxlist2d, H, W):
    boxlist2d = boxlist2d.clone()
    ymin, xmin, ymax, xmax = torch.unbind(boxlist2d, dim=2)
    ymin = ymin * float(H)
    ymax = ymax * float(H)
    xmin = xmin * float(W)
    xmax = xmax * float(W)
    boxlist2d = torch.stack([ymin, xmin, ymax, xmax], dim=2)
    return boxlist2d

def unnormalize_box2d(box2d, H, W):
    return unnormalize_boxlist2d(box2d.unsqueeze(1), H, W).squeeze(1)

def normalize_box2d(box2d, H, W):
    return normalize_boxlist2d(box2d.unsqueeze(1), H, W).squeeze(1)

def crop_and_resize(im, box2d, PH, PW, box2d_is_normalized=True):
    B, C, H, W = im.shape
    B2, D = box2d.shape
    assert(B==B2)
    assert(D==4)
    # PH, PW is the size to resize to

    # output is B x C x PH x PW

    # pt wants xy xy, unnormalized
    if box2d_is_normalized:
        box2d_unnorm = unnormalize_boxlist2d(box2d.unsqueeze(1), H, W).squeeze(1)
    else:
        box2d_unnorm = box2d
        
    ymin, xmin, ymax, xmax = box2d_unnorm.unbind(1)
    # box2d_pt = torch.stack([box2d_unnorm[:,1], box2d_unnorm[:,0], box2d_unnorm[:,3], box2d_unnorm[:,2]], dim=1)
    box2d_pt = torch.stack([xmin, ymin, xmax, ymax], dim=1)
    # we want a B-len list of K x 4 arrays
    box2d_list = list(box2d_pt.unsqueeze(1).unbind(0))
    rgb_crop = ops.roi_align(im, box2d_list, output_size=(PH, PW))

    return rgb_crop
    
