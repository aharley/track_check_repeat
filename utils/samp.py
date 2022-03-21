import torch
import utils.basic
import torch.nn.functional as F

def bilinear_sample2d(im, x, y, return_inbounds=False):
    # x and y are each B, N
    # output is B, C, N
    B, C, H, W = list(im.shape)
    N = list(x.shape)[1]

    x = x.float()
    y = y.float()
    H_f = torch.tensor(H, dtype=torch.float32)
    W_f = torch.tensor(W, dtype=torch.float32)
    
    # inbound_mask = (x>-0.5).float()*(y>-0.5).float()*(x<W_f+0.5).float()*(y<H_f+0.5).float()

    max_y = (H_f - 1).int()
    max_x = (W_f - 1).int()

    x0 = torch.floor(x).int()
    x1 = x0 + 1
    y0 = torch.floor(y).int()
    y1 = y0 + 1
    
    x0_clip = torch.clamp(x0, 0, max_x)
    x1_clip = torch.clamp(x1, 0, max_x)
    y0_clip = torch.clamp(y0, 0, max_y)
    y1_clip = torch.clamp(y1, 0, max_y)
    dim2 = W
    dim1 = W * H

    base = torch.arange(0, B, dtype=torch.int64).cuda()*dim1
    base = torch.reshape(base, [B, 1]).repeat([1, N])

    base_y0 = base + y0_clip * dim2
    base_y1 = base + y1_clip * dim2

    idx_y0_x0 = base_y0 + x0_clip
    idx_y0_x1 = base_y0 + x1_clip
    idx_y1_x0 = base_y1 + x0_clip
    idx_y1_x1 = base_y1 + x1_clip

    # use the indices to lookup pixels in the flat image
    # im is B x C x H x W
    # move C out to last dim
    im_flat = (im.permute(0, 2, 3, 1)).reshape(B*H*W, C)
    i_y0_x0 = im_flat[idx_y0_x0.long()]
    i_y0_x1 = im_flat[idx_y0_x1.long()]
    i_y1_x0 = im_flat[idx_y1_x0.long()]
    i_y1_x1 = im_flat[idx_y1_x1.long()]

    # Finally calculate interpolated values.
    x0_f = x0.float()
    x1_f = x1.float()
    y0_f = y0.float()
    y1_f = y1.float()

    w_y0_x0 = ((x1_f - x) * (y1_f - y)).unsqueeze(2)
    w_y0_x1 = ((x - x0_f) * (y1_f - y)).unsqueeze(2)
    w_y1_x0 = ((x1_f - x) * (y - y0_f)).unsqueeze(2)
    w_y1_x1 = ((x - x0_f) * (y - y0_f)).unsqueeze(2)

    output = w_y0_x0 * i_y0_x0 + w_y0_x1 * i_y0_x1 + \
             w_y1_x0 * i_y1_x0 + w_y1_x1 * i_y1_x1
    # output is B*N x C
    output = output.view(B, -1, C)
    output = output.permute(0, 2, 1)
    # output is B x C x N

    if return_inbounds:
        x_valid = (x > -0.5).byte() & (x < float(W_f - 0.5)).byte()
        y_valid = (y > -0.5).byte() & (y < float(H_f - 0.5)).byte()
        inbounds = (x_valid & y_valid).float()
        inbounds = inbounds.reshape(B, N) # something seems wrong here for B>1; i'm getting an error here (or downstream if i put -1)
        return output, inbounds

    return output # B, C, N

def bilinear_sample_single(im, x, y, return_mask=False):
    C, H, W = list(im.shape)
    N = list(x.shape)
    N2 = list(y.shape)
    assert(N==N2)

    x = x.float()
    y = y.float()
    h_f = torch.tensor(H, dtype=torch.float32).cuda()
    w_f = torch.tensor(W, dtype=torch.float32).cuda()

    inbound_mask = (x>-0.5).float()*(y>-0.5).float()*(x<w_f+0.5).float()*(y<h_f+0.5).float()

    x = torch.clamp(x, 0, w_f-1)
    y = torch.clamp(y, 0, h_f-1)

    x0_f = torch.floor(x)
    y0_f = torch.floor(y)
    x1_f = x0_f + 1
    y1_f = y0_f + 1
    x0 = x0_f.int()
    y0 = y0_f.int()
    x1 = torch.min(x1_f, w_f-1).int()
    y1 = torch.min(y1_f, h_f-1).int()
    dim2 = W
    dim1 = W*H
    idx_a = utils.basic.sub2ind(H, W, y0, x0)
    idx_b = utils.basic.sub2ind(H, W, y1, x0)
    idx_c = utils.basic.sub2ind(H, W, y0, x1)
    idx_d = utils.basic.sub2ind(H, W, y1, x1)

    # use the indices to lookup pixels in the flat image
    im_flat = (im.permute(1, 2, 0)).view(H*W, C)
    Ia = im_flat[idx_a.long()]
    Ib = im_flat[idx_b.long()]
    Ic = im_flat[idx_c.long()]
    Id = im_flat[idx_d.long()]
    # calculate interpolated values
    wa = ((x1_f-x) * (y1_f-y)).unsqueeze(1)
    wb = ((x1_f-x) * (y-y0_f)).unsqueeze(1)
    wc = ((x-x0_f) * (y1_f-y)).unsqueeze(1)
    wd = ((x-x0_f) * (y-y0_f)).unsqueeze(1)
    interp = wa*Ia+wb*Ib+wc*Ic+wd*Id
    
    interp = interp*inbound_mask.unsqueeze(1)
    # interp is N x C
    interp = interp.permute(1, 0)
    # interp is C x N

    if not return_mask:
        return interp
    else:
        mask = torch.zeros_like(im_flat[:,0:1])
        mask[idx_a.long()] = 1
        mask[idx_b.long()] = 1
        mask[idx_c.long()] = 1
        mask[idx_d.long()] = 1
        return interp, mask

def get_backwarp_mask(flow0):
    # flow points from 0 to 1
    # im1 is in coords1
    # returns im0 
    B, C, Y, X = list(flow0.shape)
    cloud0 = utils.basic.gridcloud2d(B, Y, X)
    cloud0_displacement = flow0.reshape(B, 2, Y*X).permute(0, 2, 1)
    resampling_coords = cloud0 + cloud0_displacement
    # resampling_coords = resampling_coords.long()
    mask = torch.zeros_like(flow0[:,0:1])
    for b in range(B):
        _, mask_ = bilinear_sample_single(mask[b].reshape(-1, Y, X), resampling_coords[b,:,0], resampling_coords[b,:,1], return_mask=True)
        mask[b] = mask_.reshape(1, Y, X)
    # out = empty.scatter_(0, resampling_coords, torch.ones_like(empty))
    # return out
    return mask

def backwarp_using_2d_flow(im1, flow0, binary_feat=False):
    # flow points from 0 to 1
    # im1 is in coords1
    # returns im0 
    B, C, Y, X = list(im1.shape)
    cloud0 = utils.basic.gridcloud2d(B, Y, X)
    cloud0_displacement = flow0.reshape(B, 2, Y*X).permute(0, 2, 1)
    resampling_coords = cloud0 + cloud0_displacement
    return resample2d(im1, resampling_coords, binary_feat=binary_feat)

def resample2d(im, xy, binary_feat=False):
    # im is some image feats
    # xy is some 2d coordinates, e.g., from gridcloud2d
    B, C, Y, X = list(im.shape)
    xy = utils.basic.normalize_gridcloud2d(xy, Y, X)
    xy = torch.reshape(xy, [B, Y, X, 2])
    im = F.grid_sample(im, xy)
    if binary_feat:
        im = im.round()
    return im


