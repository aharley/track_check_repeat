import torch
import utils.basic
import numpy as np
import torchvision.ops as ops
import math

def eye_3x3(B, device='cuda'):
    rt = torch.eye(3, device=torch.device(device)).view(1,3,3).repeat([B, 1, 1])
    return rt

def eye_4x4(B, device='cuda'):
    rt = torch.eye(4, device=torch.device(device)).view(1,4,4).repeat([B, 1, 1])
    return rt

def angular_l1_norm(e, g, dim=1, keepdim=False):
    # inputs are shaped B x N
    # returns a tensor sized B x N, with the dist in every slot
    
    # if our angles are in [0, 360] we can follow this stack overflow answer:
    # https://gamedev.stackexchange.com/questions/4467/comparing-angles-and-working-out-the-difference
    # wrap2pi brings the angles to [-180, 180]; adding pi puts them in [0, 360]
    e = wrap2pi(e)+np.pi
    g = wrap2pi(g)+np.pi
    # now our angles are in [0, 360]
    l = torch.abs(np.pi - torch.abs(torch.abs(e-g) - np.pi))
    return torch.sum(l, dim=dim, keepdim=keepdim)

def angular_l1_dist(e, g):
    # inputs are shaped B x N
    # returns a tensor sized B x N, with the dist in every slot
    
    # if our angles are in [0, 360] we can follow this stack overflow answer:
    # https://gamedev.stackexchange.com/questions/4467/comparing-angles-and-working-out-the-difference
    # wrap2pi brings the angles to [-180, 180]; adding pi puts them in [0, 360]
    e = wrap2pi(e)+np.pi
    g = wrap2pi(g)+np.pi
    # now our angles are in [0, 360]
    l = torch.abs(np.pi - torch.abs(torch.abs(e-g) - np.pi))
    return l

def safe_inverse(a):
    B, _, _ = list(a.shape)
    inv = a.clone()
    r_transpose = a[:, :3, :3].transpose(1,2) # inverse of rotation matrix
    inv[:, :3, :3] = r_transpose
    inv[:, :3, 3:4] = -torch.matmul(r_transpose, a[:, :3, 3:4])
    return inv

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

def split_rtlist(rtlist):
    B, N, D, E = list(rtlist.shape)
    assert(D==4)
    assert(E==4)

    __p = lambda x: utils.basic.pack_seqdim(x, B)
    __u = lambda x: utils.basic.unpack_seqdim(x, B)
    rlist_, tlist_ = split_rt(__p(rtlist))
    rlist, tlist = __u(rlist_), __u(tlist_)
    return rlist, tlist

def merge_rtlist(rlist, tlist):
    B, N, D, E = list(rlist.shape)
    assert(D==3)
    assert(E==3)
    B, N, F = list(tlist.shape)
    assert(F==3)

    __p = lambda x: utils.basic.pack_seqdim(x, B)
    __u = lambda x: utils.basic.unpack_seqdim(x, B)
    rlist_, tlist_ = __p(rlist), __p(tlist)
    rtlist_ = merge_rt(rlist_, tlist_)
    rtlist = __u(rtlist_)
    return rtlist


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
    
def get_random_rt(B,
                  rx_amount=5.0,
                  ry_amount=5.0,
                  rz_amount=5.0,
                  t_amount=1.0,
                  sometimes_zero=False,
                  return_pieces=False,
                  y_zero=False):
    # t_amount is in meters
    # r_amount is in degrees
    
    rx_amount = np.pi/180.0*rx_amount
    ry_amount = np.pi/180.0*ry_amount
    rz_amount = np.pi/180.0*rz_amount

    ## translation
    tx = np.random.uniform(-t_amount, t_amount, size=B).astype(np.float32)
    ty = np.random.uniform(-t_amount/2.0, t_amount/2.0, size=B).astype(np.float32)
    tz = np.random.uniform(-t_amount, t_amount, size=B).astype(np.float32)

    if y_zero:
        ty = ty * 0
    
    ## rotation
    rx = np.random.uniform(-rx_amount, rx_amount, size=B).astype(np.float32)
    ry = np.random.uniform(-ry_amount, ry_amount, size=B).astype(np.float32)
    rz = np.random.uniform(-rz_amount, rz_amount, size=B).astype(np.float32)

    if sometimes_zero:
        rand = np.random.uniform(0.0, 1.0, size=B).astype(np.float32)
        prob_of_zero = 0.5
        rx = np.where(np.greater(rand, prob_of_zero), rx, np.zeros_like(rx))
        ry = np.where(np.greater(rand, prob_of_zero), ry, np.zeros_like(ry))
        rz = np.where(np.greater(rand, prob_of_zero), rz, np.zeros_like(rz))
        tx = np.where(np.greater(rand, prob_of_zero), tx, np.zeros_like(tx))
        ty = np.where(np.greater(rand, prob_of_zero), ty, np.zeros_like(ty))
        tz = np.where(np.greater(rand, prob_of_zero), tz, np.zeros_like(tz))
        
    t = np.stack([tx, ty, tz], axis=1)
    t = torch.from_numpy(t)
    rx = torch.from_numpy(rx)
    ry = torch.from_numpy(ry)
    rz = torch.from_numpy(rz)
    r = eul2rotm(rx, ry, rz)
    rt = merge_rt(r, t).cuda()

    if return_pieces:
        return t.cuda(), rx.cuda(), ry.cuda(), rz.cuda()
    else:
        return rt
    
def get_random_scale(B, low=0.5, high=1.5):
    # return a scale matrix
    scale = torch.rand(B, 1, 1, device=torch.device('cuda')) * (high  - low) + low
    scale_matrix = scale * eye_4x4(B)
    scale_matrix[:, 3, 3] = 1.0 # fix the last element
    return scale_matrix

def get_pts_inbound_lrt(xyz, lrt, mult_pad=1.0, add_pad=0.0):
    B, N, D = list(xyz.shape)
    B1, C = lrt.shape
    assert(B == B1)
    assert(C == 19)
    assert(D == 3)

    lens, cam_T_obj = split_lrt(lrt)
    obj_T_cam = safe_inverse(cam_T_obj)

    xyz_obj = apply_4x4(obj_T_cam, xyz) # B x N x 3
    x = xyz_obj[:, :, 0] # B x N
    y = xyz_obj[:, :, 1]
    z = xyz_obj[:, :, 2]
    lx = lens[:, 0:1] * mult_pad + add_pad # B
    ly = lens[:, 1:2] * mult_pad + add_pad # B
    lz = lens[:, 2:3] * mult_pad + add_pad # B

    x_valid = (x >= -lx/2.0).bool() & (x <= lx/2.0).bool()
    y_valid = (y >= -ly/2.0).bool() & (y <= ly/2.0).bool()
    z_valid = (z >= -lz/2.0).bool() & (z <= lz/2.0).bool()
    inbounds = x_valid.bool() & y_valid.bool() & z_valid.bool() # B x N

    return inbounds

def random_occlusion(xyz, lrtlist, scorelist, pix_T_cam, H, W, mask_size=20, occ_prob=0.5, occlude_bkg_too=False):
    # with occ_prob, we create a random mask. else no operation
    num_try = 10
    max_dist = 200.0
    # lrtlist is B x 19
    B, N, D = list(xyz.shape)
    B, N_obj, C = lrtlist.shape
    assert(C == 19)
    depth, valid = create_depth_image(pix_T_cam, xyz, H, W) # B x 1 x H x W

    clist_cam = get_clist_from_lrtlist(lrtlist) # B x N_obj x 3
    clist_pix = camera2pixels(clist_cam, pix_T_cam) # B x N_obj x 2
    clist_pix = torch.round(clist_pix).long()
    # we create a mask around the center of the box
    xyz_new_s = torch.zeros(B, H*W, 3, device=torch.device('cuda'))

    # print(N_obj)

    mask = torch.ones_like(depth)

    for b in range(B):
        for n in range(N_obj):
            if np.random.uniform() < occ_prob and scorelist[b, n]:
                inbound = get_pts_inbound_lrt(xyz[b:b+1], lrtlist[b:b+1, n]) # 1 x N
                inb_pts_cnt = torch.sum(inbound)

                # print('inb_ori:', inb_pts_cnt)

                for _ in range(num_try):
                    rand_offset = torch.randint(-mask_size//2, mask_size//2, size=(1, 2), device=torch.device('cuda'))
                    mask_center = clist_pix[b, n:n+1] + rand_offset # 1 x 2
                    mask_lower_bound = mask_center - mask_size // 2
                    mask_upper_bound = mask_center + mask_size // 2
                    mask_lower_bound_x = mask_lower_bound[:, 0]
                    mask_lower_bound_y = mask_lower_bound[:, 1]
                    mask_upper_bound_x = mask_upper_bound[:, 0]
                    mask_upper_bound_y = mask_upper_bound[:, 1]

                    mask_lower_bound_x = torch.clamp(mask_lower_bound_x, 0, W-1) # each shape 1
                    mask_upper_bound_x = torch.clamp(mask_upper_bound_x, 0, W-1)
                    mask_lower_bound_y = torch.clamp(mask_lower_bound_y, 0, H-1)
                    mask_upper_bound_y = torch.clamp(mask_upper_bound_y, 0, H-1)

                    # do the masking
                    depth_b = depth[b:b+1].clone() # 1 x 1 x H x W
                    mask_b = torch.ones_like(depth_b)
                    mask_b[:, :, mask_lower_bound_y:mask_upper_bound_y, mask_lower_bound_x:mask_upper_bound_x] = 0
                    depth_b[:, :, mask_lower_bound_y:mask_upper_bound_y, mask_lower_bound_x:mask_upper_bound_x] = max_dist
                    # set to a large value, i.e. mask out these area

                    if occlude_bkg_too:
                        bkg_mask_size = mask_size * 2
                        mask_center_x = torch.randint(bkg_mask_size//2, W - bkg_mask_size//2, size=(1,), device=torch.device('cuda'))
                        mask_center_y = torch.randint(bkg_mask_size//2, H - bkg_mask_size//2, size=(1,), device=torch.device('cuda'))
                        mask_center = torch.stack([mask_center_x, mask_center_y], dim=1)
                        mask_lower_bound = mask_center - bkg_mask_size // 2
                        mask_upper_bound = mask_center + bkg_mask_size // 2
                        mask_lower_bound_x = mask_lower_bound[:, 0]
                        mask_lower_bound_y = mask_lower_bound[:, 1]
                        mask_upper_bound_x = mask_upper_bound[:, 0]
                        mask_upper_bound_y = mask_upper_bound[:, 1]

                        mask_lower_bound_x = torch.clamp(mask_lower_bound_x, 0, W-1) # each shape 1
                        mask_upper_bound_x = torch.clamp(mask_upper_bound_x, 0, W-1)
                        mask_lower_bound_y = torch.clamp(mask_lower_bound_y, 0, H-1)
                        mask_upper_bound_y = torch.clamp(mask_upper_bound_y, 0, H-1)

                        # do the additional masking
                        mask_b[:, :, mask_lower_bound_y:mask_upper_bound_y, mask_lower_bound_x:mask_upper_bound_x] = 0
                        depth_b[:, :, mask_lower_bound_y:mask_upper_bound_y, mask_lower_bound_x:mask_upper_bound_x] = max_dist
                        # set to a large value, i.e. mask out these area

                    xyz_new = depth2pointcloud(depth_b, pix_T_cam[b:b+1]) # 1 x N x 3
                    inbound_new = get_pts_inbound_lrt(xyz_new, lrtlist[b:b+1, n]) # 1 x N
                    inb_pts_cnt_new = torch.sum(inbound_new)

                    # print(inb_pts_cnt_new)

                    if (inb_pts_cnt_new < inb_pts_cnt and
                        inb_pts_cnt_new > (inb_pts_cnt / 8.0) and
                        inb_pts_cnt_new >= 3): # if we occlude part but not all of the obj, they we are good
                        depth[b:b+1] = depth_b
                        mask[b:b+1] = mask_b
                        # all good
                        break
                

        # convert back to pointcloud
        xyz_new = depth2pointcloud(depth[b:b+1], pix_T_cam[b:b+1]) # 1 x N x 3
        xyz_new_s[b:b+1] = xyz_new

    return xyz_new_s, mask

def apply_scaling_to_lrt(Y_T_X, lrt_X):
    return apply_scaling_to_lrtlist(Y_T_X, lrt_X.unsqueeze(1)).squeeze(1)

def apply_scaling_to_lrtlist(Y_T_X, lrtlist_X): 
    B, N, D = list(lrtlist_X.shape)
    assert(D==19)
    B2, E, F = list(Y_T_X.shape)
    assert(B2==B)
    assert(E==4 and F==4)

    # Y_T_X is a scaling matrix, i.e. all off-diagnol terms are 0
    lenlist_X, rtlist_X = split_lrtlist(lrtlist_X)
    # rtlist_X is B x N x 4 x 4

    # lenlist is B x N x 3
    rtlist_X_ = rtlist_X.reshape(B*N, 4, 4)
    rlist_X_, tlist_X_ = split_rt(rtlist_X_) # B*N x 3 x 3 and B*N x 3

    lenlist_Y_ = apply_4x4(Y_T_X, lenlist_X).reshape(B*N, 3)
    tlist_Y_ = apply_4x4(Y_T_X, tlist_X_.reshape(B, N, 3)).reshape(B*N, 3)
    rlist_Y_ = rlist_X_ 

    rtlist_Y = merge_rt(rlist_Y_, tlist_Y_).reshape(B, N, 4, 4)
    lenlist_Y = lenlist_Y_.reshape(B, N, 3)
    lrtlist_Y = merge_lrtlist(lenlist_Y, rtlist_Y)

    return lrtlist_Y

def apply_4x4_to_lrtlist(Y_T_X, lrtlist_X):
    B, N, D = list(lrtlist_X.shape)
    assert(D==19)
    B2, E, F = list(Y_T_X.shape)
    assert(B2==B)
    assert(E==4 and F==4)
    
    lenlist, rtlist_X = split_lrtlist(lrtlist_X)
    # rtlist_X is B x N x 4 x 4

    Y_T_Xs = Y_T_X.unsqueeze(1).repeat(1, N, 1, 1)
    Y_T_Xs_ = Y_T_Xs.view(B*N, 4, 4)
    rtlist_X_ = rtlist_X.reshape(B*N, 4, 4)
    rtlist_Y_ = utils.basic.matmul2(Y_T_Xs_, rtlist_X_)
    rtlist_Y = rtlist_Y_.reshape(B, N, 4, 4)
    lrtlist_Y = merge_lrtlist(lenlist, rtlist_Y)
    return lrtlist_Y

def apply_4x4_to_lrt(Y_T_X, lrt_X):
    B, D = list(lrt_X.shape)
    assert(D==19)
    B2, E, F = list(Y_T_X.shape)
    assert(B2==B)
    assert(E==4 and F==4)
    return apply_4x4_to_lrtlist(Y_T_X, lrt_X.unsqueeze(1)).squeeze(1)

def apply_4x4s_to_lrts(Ys_T_Xs, lrt_Xs):
    B, S, D = list(lrt_Xs.shape)
    assert(D==19)
    B2, S2, E, F = list(Ys_T_Xs.shape)
    assert(B2==B)
    assert(S2==S)
    assert(E==4 and F==4)
    
    lenlist, rtlist_X = split_lrtlist(lrt_Xs)
    # rtlist_X is B x N x 4 x 4

    Ys_T_Xs_ = Ys_T_Xs.view(B*S, 4, 4)
    rtlist_X_ = rtlist_X.view(B*S, 4, 4)
    rtlist_Y_ = utils.basic.matmul2(Ys_T_Xs_, rtlist_X_)
    rtlist_Y = rtlist_Y_.view(B, S, 4, 4)
    lrtlist_Y = merge_lrtlist(lenlist, rtlist_Y)
    return lrtlist_Y

def rotm2eul_py(R):
    # R is 3x3
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    if sy > 1e-6: # singular
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else:
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    return x, y, z
