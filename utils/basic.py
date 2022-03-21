import os
import torch
import numpy as np
EPS = 1e-6

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_lr_str(lr):
    lrn = "%.1e" % lr # e.g., 5.0e-04
    lrn = lrn[0] + lrn[3:5] + lrn[-1] # e.g., 5e-4
    return lrn
    
def strnum(x):
    s = '%g' % x
    if '.' in s:
        if x < 1.0:
            s = s[s.index('.'):]
    return s

def assert_same_shape(t1, t2):
    for (x, y) in zip(list(t1.shape), list(t2.shape)):
        assert(x==y)
        
def print_stats(name, tensor):
    shape = tensor.shape
    tensor = tensor.detach().cpu().numpy()
    print('%s (%s) min = %.2f, mean = %.2f, max = %.2f' % (
        name, tensor.dtype, np.min(tensor), np.mean(tensor), np.max(tensor)), shape)

def print_(name, tensor):
    tensor = tensor.detach().cpu().numpy()
    print(name, tensor, tensor.shape)

def matmul2(mat1, mat2):
    return torch.matmul(mat1, mat2)

def matmul3(mat1, mat2, mat3):
    return torch.matmul(mat1, torch.matmul(mat2, mat3))

def matmul4(mat1, mat2, mat3, mat4):
    return torch.matmul(torch.matmul(mat1, torch.matmul(mat2, mat3)), mat4)

def sub2ind(height, width, y, x):
    return y*width + x

def pack_seqdim(tensor, B):
    shapelist = list(tensor.shape)
    B_, S = shapelist[:2]
    assert(B==B_)
    otherdims = shapelist[2:]
    tensor = torch.reshape(tensor, [B*S]+otherdims)
    return tensor

def unpack_seqdim(tensor, B):
    shapelist = list(tensor.shape)
    BS = shapelist[0]
    assert(BS%B==0)
    otherdims = shapelist[1:]
    S = int(BS/B)
    tensor = torch.reshape(tensor, [B,S]+otherdims)
    return tensor

def normalize_single(d):
    # d is a whatever shape torch tensor
    dmin = torch.min(d)
    dmax = torch.max(d)
    d = (d-dmin)/(EPS+(dmax-dmin))
    return d

def normalize(d):
    # d is B x whatever. normalize within each element of the batch
    out = torch.zeros(d.size())
    if d.is_cuda:
        out = out.cuda()
    B = list(d.size())[0]
    for b in list(range(B)):
        out[b] = normalize_single(d[b])
    return out

def reduce_masked_mean(x, mask, dim=None, keepdim=False):
    # x and mask are the same shape, or at least broadcastably so < actually it's safer if you disallow broadcasting
    # returns shape-1
    # axis can be a list of axes
    for (a,b) in zip(x.size(), mask.size()):
        # if not b==1: 
        assert(a==b) # some shape mismatch!
    # assert(x.size() == mask.size())
    prod = x*mask
    if dim is None:
        numer = torch.sum(prod)
        denom = EPS+torch.sum(mask)
    else:
        numer = torch.sum(prod, dim=dim, keepdim=keepdim)
        denom = EPS+torch.sum(mask, dim=dim, keepdim=keepdim)
        
    mean = numer/denom
    return mean

def gridcloud3d(B, Z, Y, X, norm=False, device='cuda'):
    # we want to sample for each location in the grid
    grid_z, grid_y, grid_x = meshgrid3d(B, Z, Y, X, norm=norm, device=device)
    x = torch.reshape(grid_x, [B, -1])
    y = torch.reshape(grid_y, [B, -1])
    z = torch.reshape(grid_z, [B, -1])
    # these are B x N
    xyz = torch.stack([x, y, z], dim=2)
    # this is B x N x 3
    return xyz

def gridcloud2d(B, Y, X, norm=False, device='cuda'):
    # we want to sample for each location in the grid
    grid_y, grid_x = meshgrid2d(B, Y, X, norm=norm, device=device)
    x = torch.reshape(grid_x, [B, -1])
    y = torch.reshape(grid_y, [B, -1])
    # these are B x N
    xy = torch.stack([x, y], dim=2)
    # this is B x N x 2
    return xy

def meshgrid2d(B, Y, X, stack=False, norm=False, device='cuda'):
    # returns a meshgrid sized B x Y x X

    grid_y = torch.linspace(0.0, Y-1, Y, device=torch.device(device))
    grid_y = torch.reshape(grid_y, [1, Y, 1])
    grid_y = grid_y.repeat(B, 1, X)

    grid_x = torch.linspace(0.0, X-1, X, device=torch.device(device))
    grid_x = torch.reshape(grid_x, [1, 1, X])
    grid_x = grid_x.repeat(B, Y, 1)

    if norm:
        grid_y, grid_x = normalize_grid2d(
            grid_y, grid_x, Y, X)

    if stack:
        # note we stack in xy order
        # (see https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.grid_sample)
        grid = torch.stack([grid_x, grid_y], dim=-1)
        return grid
    else:
        return grid_y, grid_x
    
def meshgrid3d_py(Z, Y, X, stack=False, norm=False):
    grid_z = np.linspace(0.0, Z-1, Z)
    grid_z = np.reshape(grid_z, [Z, 1, 1])
    grid_z = np.tile(grid_z, [1, Y, X])

    grid_y = np.linspace(0.0, Y-1, Y)
    grid_y = np.reshape(grid_y, [1, Y, 1])
    grid_y = np.tile(grid_y, [Z, 1, X])

    grid_x = np.linspace(0.0, X-1, X)
    grid_x = np.reshape(grid_x, [1, 1, X])
    grid_x = np.tile(grid_x, [Z, Y, 1])

    if norm:
        grid_z, grid_y, grid_x = normalize_grid3d(
            grid_z, grid_y, grid_x, Z, Y, X)

    if stack:
        # note we stack in xyz order
        # (see https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.grid_sample)
        grid = np.stack([grid_x, grid_y, grid_z], dim=-1)
        return grid
    else:
        return grid_z, grid_y, grid_x
    
def meshgrid3d(B, Z, Y, X, stack=False, norm=False, device='cuda'):
    # returns a meshgrid sized B x Z x Y x X
    
    grid_z = torch.linspace(0.0, Z-1, Z, device=device)
    grid_z = torch.reshape(grid_z, [1, Z, 1, 1])
    grid_z = grid_z.repeat(B, 1, Y, X)

    grid_y = torch.linspace(0.0, Y-1, Y, device=device)
    grid_y = torch.reshape(grid_y, [1, 1, Y, 1])
    grid_y = grid_y.repeat(B, Z, 1, X)

    grid_x = torch.linspace(0.0, X-1, X, device=device)
    grid_x = torch.reshape(grid_x, [1, 1, 1, X])
    grid_x = grid_x.repeat(B, Z, Y, 1)

    # if cuda:
    #     grid_z = grid_z.cuda()
    #     grid_y = grid_y.cuda()
    #     grid_x = grid_x.cuda()
        
    if norm:
        grid_z, grid_y, grid_x = normalize_grid3d(
            grid_z, grid_y, grid_x, Z, Y, X)

    if stack:
        # note we stack in xyz order
        # (see https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.grid_sample)
        grid = torch.stack([grid_x, grid_y, grid_z], dim=-1)
        return grid
    else:
        return grid_z, grid_y, grid_x
    
def normalize_gridcloud3d(xyz, Z, Y, X, clamp_extreme=True):
    # make things in [-1,1]
    x = xyz[...,0]
    y = xyz[...,1]
    z = xyz[...,2]
    
    z = 2.0*(z / float(Z-1)) - 1.0
    y = 2.0*(y / float(Y-1)) - 1.0
    x = 2.0*(x / float(X-1)) - 1.0

    xyz = torch.stack([x,y,z], dim=-1)
    
    if clamp_extreme:
        xyz = torch.clamp(xyz, min=-2.0, max=2.0)
    return xyz

def normalize_gridcloud2d(xy, Y, X, clamp_extreme=True):
    # make things in [-1,1]
    x = xy[...,0]
    y = xy[...,1]
    
    y = 2.0*(y / float(Y-1)) - 1.0
    x = 2.0*(x / float(X-1)) - 1.0

    xy = torch.stack([x,y], dim=-1)
    
    if clamp_extreme:
        xy = torch.clamp(xy, min=-2.0, max=2.0)
    return xy



