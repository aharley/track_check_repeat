import torch
import utils.geom
import utils.basic
import utils.samp
import numpy as np


class Vox_util(object):
    def __init__(self, Z, Y, X, scene_centroid, bounds, pad=None, assert_cube=False):
        # on every step, we create this object
        
        self.XMIN, self.XMAX, self.YMIN, self.YMAX, self.ZMIN, self.ZMAX = bounds
        # self.XMIN = self.XMIN.cpu().item()
        # self.XMAX = self.XMAX.cpu().item()
        # self.YMIN = self.YMIN.cpu().item()
        # self.YMAX = self.YMAX.cpu().item()
        # self.ZMIN = self.ZMIN.cpu().item()
        # self.ZMAX = self.ZMAX.cpu().item()
            
        # print('bounds for this iter:',
        #       'X = %.2f to %.2f' % (self.XMIN, self.XMAX), 
        #       'Y = %.2f to %.2f' % (self.YMIN, self.YMAX), 
        #       'Z = %.2f to %.2f' % (self.ZMIN, self.ZMAX), 
        # )
        # scene_centroid is B x 3
        B, D = list(scene_centroid.shape)
        # this specifies the location of the world box

        self.Z, self.Y, self.X = Z, Y, X

        scene_centroid = scene_centroid.detach().cpu().numpy()
        x_centroid, y_centroid, z_centroid = scene_centroid[0]
        self.XMIN += x_centroid
        self.XMAX += x_centroid
        self.YMIN += y_centroid
        self.YMAX += y_centroid
        self.ZMIN += z_centroid
        self.ZMAX += z_centroid
        # print('bounds for this iter:',
        #       'X = %.2f to %.2f' % (self.XMIN, self.XMAX), 
        #       'Y = %.2f to %.2f' % (self.YMIN, self.YMAX), 
        #       'Z = %.2f to %.2f' % (self.ZMIN, self.ZMAX), 
        # )

        self.default_vox_size_X = (self.XMAX-self.XMIN)/float(X)
        self.default_vox_size_Y = (self.YMAX-self.YMIN)/float(Y)
        self.default_vox_size_Z = (self.ZMAX-self.ZMIN)/float(Z)
        # print('self.default_vox_size_X', self.default_vox_size_X)
        # print('self.default_vox_size_Y', self.default_vox_size_Y)
        # print('self.default_vox_size_Z', self.default_vox_size_Z)

        if pad:
            Z_pad, Y_pad, X_pad = pad
            self.ZMIN -= self.default_vox_size_Z * Z_pad
            self.ZMAX += self.default_vox_size_Z * Z_pad
            self.YMIN -= self.default_vox_size_Y * Y_pad
            self.YMAX += self.default_vox_size_Y * Y_pad
            self.XMIN -= self.default_vox_size_X * X_pad
            self.XMAX += self.default_vox_size_X * X_pad

        if assert_cube:
            # we assume cube voxels
            if (not np.isclose(self.default_vox_size_X, self.default_vox_size_Y)) or (not np.isclose(self.default_vox_size_X, self.default_vox_size_Z)):
                print('Z, Y, X', Z, Y, X)
                print('bounds for this iter:',
                      'X = %.2f to %.2f' % (self.XMIN, self.XMAX), 
                      'Y = %.2f to %.2f' % (self.YMIN, self.YMAX), 
                      'Z = %.2f to %.2f' % (self.ZMIN, self.ZMAX), 
                )
                print('self.default_vox_size_X', self.default_vox_size_X)
                print('self.default_vox_size_Y', self.default_vox_size_Y)
                print('self.default_vox_size_Z', self.default_vox_size_Z)
            assert(np.isclose(self.default_vox_size_X, self.default_vox_size_Y))
            assert(np.isclose(self.default_vox_size_X, self.default_vox_size_Z))


    def Ref2Mem(self, xyz, Z, Y, X, assert_cube=False):
        # xyz is B x N x 3, in ref coordinates
        # transforms ref coordinates into mem coordinates
        B, N, C = list(xyz.shape)
        device = xyz.device
        assert(C==3)
        mem_T_ref = self.get_mem_T_ref(B, Z, Y, X, assert_cube=assert_cube, device=device)
        xyz = utils.geom.apply_4x4(mem_T_ref, xyz)
        return xyz
    
    def Mem2Ref(self, xyz_mem, Z, Y, X, assert_cube=False):
        # xyz is B x N x 3, in mem coordinates
        # transforms mem coordinates into ref coordinates
        B, N, C = list(xyz_mem.shape)
        ref_T_mem = self.get_ref_T_mem(B, Z, Y, X, assert_cube=assert_cube)
        xyz_ref = utils.geom.apply_4x4(ref_T_mem, xyz_mem)
        return xyz_ref

    def get_ref_T_mem(self, B, Z, Y, X, assert_cube=False, device='cuda'):
        mem_T_ref = self.get_mem_T_ref(B, Z, Y, X, assert_cube=assert_cube, device=device)
        # note safe_inverse is inapplicable here,
        # since the transform is nonrigid
        ref_T_mem = mem_T_ref.inverse()
        return ref_T_mem

    def get_mem_T_ref(self, B, Z, Y, X, assert_cube=False, device='cuda'):
        # sometimes we want the mat itself
        # note this is not a rigid transform

        # note we need to (re-)compute the vox sizes, for this new resolution
        vox_size_X = (self.XMAX-self.XMIN)/float(X)
        vox_size_Y = (self.YMAX-self.YMIN)/float(Y)
        vox_size_Z = (self.ZMAX-self.ZMIN)/float(Z)

        # for safety, let's check that this is cube
        if assert_cube:
            if (not np.isclose(vox_size_X, vox_size_Y)) or (not np.isclose(vox_size_X, vox_size_Z)):
                print('Z, Y, X', Z, Y, X)
                print('bounds for this iter:',
                      'X = %.2f to %.2f' % (self.XMIN, self.XMAX), 
                      'Y = %.2f to %.2f' % (self.YMIN, self.YMAX), 
                      'Z = %.2f to %.2f' % (self.ZMIN, self.ZMAX), 
                )
                print('vox_size_X', vox_size_X)
                print('vox_size_Y', vox_size_Y)
                print('vox_size_Z', vox_size_Z)
            assert(np.isclose(vox_size_X, vox_size_Y))
            assert(np.isclose(vox_size_X, vox_size_Z))

        # translation
        # (this makes the left edge of the leftmost voxel correspond to XMIN)
        center_T_ref = utils.geom.eye_4x4(B, device=device)
        center_T_ref[:,0,3] = -self.XMIN-vox_size_X/2.0
        center_T_ref[:,1,3] = -self.YMIN-vox_size_Y/2.0
        center_T_ref[:,2,3] = -self.ZMIN-vox_size_Z/2.0

        # scaling
        # (this makes the right edge of the rightmost voxel correspond to XMAX)
        mem_T_center = utils.geom.eye_4x4(B, device=device)
        mem_T_center[:,0,0] = 1./vox_size_X
        mem_T_center[:,1,1] = 1./vox_size_Y
        mem_T_center[:,2,2] = 1./vox_size_Z
        mem_T_ref = utils.basic.matmul2(mem_T_center, center_T_ref)

        return mem_T_ref

    def get_inbounds(self, xyz, Z, Y, X, already_mem=False, padding=0.0, assert_cube=False):
        # xyz is B x N x 3
        # padding should be 0 unless you are trying to account for some later cropping
        if not already_mem:
            xyz = self.Ref2Mem(xyz, Z, Y, X, assert_cube=assert_cube)

        x = xyz[:,:,0]
        y = xyz[:,:,1]
        z = xyz[:,:,2]

        x_valid = ((x-padding)>-0.5).byte() & ((x+padding)<float(X-0.5)).byte()
        y_valid = ((y-padding)>-0.5).byte() & ((y+padding)<float(Y-0.5)).byte()
        z_valid = ((z-padding)>-0.5).byte() & ((z+padding)<float(Z-0.5)).byte()
        nonzero = (~(z==0.0)).byte()

        inbounds = x_valid & y_valid & z_valid & nonzero
        return inbounds.bool()

    def get_inbounds_single(self, xyz, Z, Y, X, already_mem=False):
        # xyz is N x 3
        xyz = xyz.unsqueeze(0)
        inbounds = self.get_inbounds(xyz, Z, Y, X, already_mem=already_mem)
        inbounds = inbounds.squeeze(0)
        return inbounds

    def voxelize_xyz(self, xyz_ref, Z, Y, X, already_mem=False, assert_cube=False, clean_eps=0):
        B, N, D = list(xyz_ref.shape)
        assert(D==3)
        if already_mem:
            xyz_mem = xyz_ref
        else:
            xyz_mem = self.Ref2Mem(xyz_ref, Z, Y, X, assert_cube=assert_cube)
        vox = self.get_occupancy(xyz_mem, Z, Y, X, clean_eps=clean_eps)
        return vox


    def get_occupancy(self, xyz, Z, Y, X, clean_eps=0):
        # xyz is B x N x 3 and in mem coords
        # we want to fill a voxel tensor with 1's at these inds
        B, N, C = list(xyz.shape)
        assert(C==3)

        # these papers say simple 1/0 occupancy is ok:
        #  http://openaccess.thecvf.com/content_cvpr_2018/papers/Yang_PIXOR_Real-Time_3d_CVPR_2018_paper.pdf
        #  http://openaccess.thecvf.com/content_cvpr_2018/papers/Luo_Fast_and_Furious_CVPR_2018_paper.pdf
        # cont fusion says they do 8-neighbor interp
        # voxelnet does occupancy but with a bit of randomness in terms of the reflectance value i think

        inbounds = self.get_inbounds(xyz, Z, Y, X, already_mem=True)
        x, y, z = xyz[:,:,0], xyz[:,:,1], xyz[:,:,2]
        mask = torch.zeros_like(x)
        mask[inbounds] = 1.0

        if clean_eps > 0:
            # only take points that are already near centers
            xyz_round = torch.round(xyz) # B, N, 3
            dist = torch.norm(xyz_round - xyz, dim=2)
            mask[dist > clean_eps] = 0
        
        # set the invalid guys to zero
        # we then need to zero out 0,0,0
        # (this method seems a bit clumsy)
        x = x*mask
        y = y*mask
        z = z*mask

        x = torch.round(x)
        y = torch.round(y)
        z = torch.round(z)
        x = torch.clamp(x, 0, X-1).int()
        y = torch.clamp(y, 0, Y-1).int()
        z = torch.clamp(z, 0, Z-1).int()

        x = x.view(B*N)
        y = y.view(B*N)
        z = z.view(B*N)

        dim3 = X
        dim2 = X * Y
        dim1 = X * Y * Z

        base = torch.arange(0, B, dtype=torch.int32, device=xyz.device)*dim1
        base = torch.reshape(base, [B, 1]).repeat([1, N]).view(B*N)

        vox_inds = base + z * dim2 + y * dim3 + x
        voxels = torch.zeros(B*Z*Y*X, device=xyz.device).float()
        voxels[vox_inds.long()] = 1.0
        # zero out the singularity
        voxels[base.long()] = 0.0
        voxels = voxels.reshape(B, 1, Z, Y, X)
        # B x 1 x Z x Y x X
        return voxels
    
    def unproject_image_to_mem(self, rgb_camB, Z, Y, X, pixB_T_camA, assert_cube=False):
        # rgb_camB is B x C x H x W
        # pixB_T_camA is B x 4 x 4

        # rgb lives in B pixel coords
        # we want everything in A memory coords

        # this puts each C-dim pixel in the rgb_camB
        # along a ray in the voxelgrid
        B, C, H, W = list(rgb_camB.shape)

        xyz_memA = utils.basic.gridcloud3d(B, Z, Y, X, norm=False)
        # grid_z, grid_y, grid_x = utils.basic.meshgrid3d(B, Z, Y, X)
        # # these are B x Z x Y x X
        # # these represent the mem grid coordinates

        # # we need to convert these to pixel coordinates
        # x = torch.reshape(grid_x, [B, -1])
        # y = torch.reshape(grid_y, [B, -1])
        # z = torch.reshape(grid_z, [B, -1])
        # # these are B x N
        # xyz_mem = torch.stack([x, y, z], dim=2)

        xyz_camA = self.Mem2Ref(xyz_memA, Z, Y, X, assert_cube=assert_cube)

        xyz_pixB = utils.geom.apply_4x4(pixB_T_camA, xyz_camA)
        normalizer = torch.unsqueeze(xyz_pixB[:,:,2], 2)
        EPS=1e-6
        xy_pixB = xyz_pixB[:,:,:2]/torch.clamp(normalizer, min=EPS)
        # this is B x N x 2
        # this is the (floating point) pixel coordinate of each voxel
        x_pixB, y_pixB = xy_pixB[:,:,0], xy_pixB[:,:,1]
        # these are B x N

        if (0):
            # handwritten version
            values = torch.zeros([B, C, Z*Y*X], dtype=torch.float32)
            for b in list(range(B)):
                values[b] = utils.samp.bilinear_sample_single(rgb_camB[b], x_pixB[b], y_pixB[b])
        else:
            # native pytorch version
            y_pixB, x_pixB = utils.basic.normalize_grid2d(y_pixB, x_pixB, H, W)
            # since we want a 3d output, we need 5d tensors
            z_pixB = torch.zeros_like(x_pixB)
            xyz_pixB = torch.stack([x_pixB, y_pixB, z_pixB], axis=2)
            rgb_camB = rgb_camB.unsqueeze(2)
            xyz_pixB = torch.reshape(xyz_pixB, [B, Z, Y, X, 3])
            values = F.grid_sample(rgb_camB, xyz_pixB)

        values = torch.reshape(values, (B, C, Z, Y, X))
        return values

    def apply_ref_T_mem_to_lrtlist(self, lrtlist_mem, Z, Y, X, assert_cube=False):
        # lrtlist is B x N x 19, in mem coordinates
        # transforms them into cam coordinates, including a scale change for the lengths
        B, N, C = list(lrtlist_mem.shape)
        assert(C==19)
        cam_T_mem = self.get_ref_T_mem(B, Z, Y, X, assert_cube=assert_cube)

        # apply_4x4 will work for the t part
        lenlist_mem, rtlist_mem = utils.geom.split_lrtlist(lrtlist_mem)
        __p = lambda x: utils.basic.pack_seqdim(x, B)
        __u = lambda x: utils.basic.unpack_seqdim(x, B)
        rlist_mem_, tlist_mem_ = utils.geom.split_rt(__p(rtlist_mem))
        # rlist_cam_ is B*N x 3 x 3
        # tlist_cam_ is B*N x 3
        tlist_cam_ = __p(utils.geom.apply_4x4(cam_T_mem, __u(tlist_mem_)))
        # rlist does not need to change, since cam is aligned with mem
        rlist_cam_ = rlist_mem_.clone()
        rtlist_cam = __u(utils.geom.merge_rt(rlist_cam_, tlist_cam_))
        # this is B x N x 4 x 4

        # next we need to scale the lengths
        lenlist_mem, _ = utils.geom.split_lrtlist(lrtlist_mem)
        # this is B x N x 3
        vox_size_X = (self.XMAX-self.XMIN)/float(X)
        vox_size_Y = (self.YMAX-self.YMIN)/float(Y)
        vox_size_Z = (self.ZMAX-self.ZMIN)/float(Z)
        xlist, ylist, zlist = lenlist_mem.chunk(3, dim=2)
        lenlist_cam = torch.cat([xlist * vox_size_X,
                                 ylist * vox_size_Y,
                                 zlist * vox_size_Z], dim=2)
        # merge up
        lrtlist_cam = utils.geom.merge_lrtlist(lenlist_cam, rtlist_cam)
        return lrtlist_cam
    
