import numpy as np
import torch
import torch.types
from math import *

# preserves perfect symmetry
# Due to restrictions imposed by torch.jit.script, only axes of type
#   int are supported. Moreover, enable_assoc is presented as an
#   argument with default value instead of a global variable.
def associative_float_sum(arr : torch.Tensor,
                          axis : int,
                          enable_assoc : bool = False):
    if not enable_assoc:
        return arr.sum(axis)
    total_size = torch.tensor(arr.shape[axis])
    total_k = int(torch.log2(total_size)) + 1
    shift_n = 60 - total_k

    M = torch.max(torch.abs(arr)) + 2 ** (-500)
    k = int(torch.log2(M) + 1000) - 999
    multip = 2 ** k
    arr = torch.round(arr * ((1 << shift_n) / multip)).to(torch.long)
    arr = arr.sum(axis).to(torch.double)
    arr /= 1 << shift_n
    return arr * float(multip)

# Supported (shape of v1, shape of v2):
#   1. ((3,) + Y, (3,) + Z + Y); Y, Z arbitrary
#   2. ((3,) + Z + Y, (3,) + Y); Y, Z arbitrary
def cross_prod(v1 : torch.Tensor, v2 : torch.Tensor):
    v1x, v1y, v1z = v1
    v2x, v2y, v2z = v2
    if tuple(v1x.shape) != ():
        ret = torch.zeros((3,) + tuple(v1x.shape)).cuda()
    elif tuple(v2x.shape) != ():
        ret = torch.zeros((3,) + tuple(v2x.shape)).cuda()
    else:
        ret = torch.zeros(3).cuda()
    ret[0] = v1y * v2z - v1z * v2y
    ret[1] = v1z * v2x - v1x * v2z
    ret[2] = v1x * v2y - v1y * v2x
    return ret

# 3d slice of 4d (multi-segment) line
# The slice plane is determined by an arbitrary point "center" on it
# and its direction vector
def get_3d_slice_of_4d_line(xs, ys, zs, ts, center, dir):
    if len(xs) == 0:
        return [[], [], [], []]
    center             = np.array(center)
    dir                = np.array(dir)
    xs : np.ndarray    = np.array(xs)
    ys : np.ndarray    = np.array(ys)
    zs : np.ndarray    = np.array(zs)
    ts : np.ndarray    = np.array(ts)
    p1x, p1y, p1z, p1t = xs[:-1], ys[:-1], zs[:-1], ts[:-1]
    p2x, p2y, p2z, p2t = xs[1:],  ys[1:],  zs[1:],  ts[1:]
    dp1                = p1x * dir[0] + p1y * dir[1] + p1z * dir[2] + p1t * dir[3]
    dp2                = p2x * dir[0] + p2y * dir[1] + p2z * dir[2] + p2t * dir[3]
    dp                 = (center * dir).sum()
    intersected        = np.nonzero((dp1 - dp) * (dp2 - dp) < 0)[0]
    if len(intersected) == 0:
        return [[], [], [], []]
    p1ix, p1iy, p1iz, p1it = p1x[intersected], p1y[intersected], p1z[intersected], p1t[intersected]
    p2ix, p2iy, p2iz, p2it = p2x[intersected], p2y[intersected], p2z[intersected], p2t[intersected]
    pdx,  pdy,  pdz,  pdt  = p2ix - p1ix, p2iy - p1iy, p2iz - p1iz, p2it - p1it
    d2                 = dp2[intersected] - dp
    d21                = dp2[intersected] - dp1[intersected]
    rat                = d2 / d21
    rx, ry, rz, rt     = p1ix + rat * pdx, p1iy + rat * pdy, p1iz + rat * pdz, p1it + rat * pdt
    return [rx, ry, rz, rt]

# indices : (N,) int64 array. The indices that each data point corresponds to
# data    : (N, shape) array
# We want to accumulate data of the same indice. The function returns a pair of arrays.
#    The first array is an array of indices, the second array is the sum of data points
#    corresponding to the index.
# For example, tally_sum([0, 1, 2, 0, 1], [3, 5, 4, 6, 7]) will return
#    ([0, 1, 2], [9, 12, 4])
# If there is any NAN in the tensor, the return value is undefined.
# If any partial sum of the array overflows, the return value is undefined.
def tally_sum(indices : torch.Tensor, data : torch.Tensor):
    assert len(indices.shape) == 1
    assert indices.shape[0] == data.shape[0]
    args = torch.argsort(indices)
    indices = indices[args]
    data = data[args]
    cum = torch.cumsum(data, dim=0)
    _, coun = torch.unique_consecutive(indices, return_counts=True)
    coun_cum = torch.cumsum(coun, dim=0)
    endpts = coun_cum - 1
    endvals = cum[endpts]
    startvals = torch.zeros(endvals.shape).cuda()
    startvals[1:] = endvals[:-1]
    startvals[0] = 0
    return indices[endpts], endvals - startvals

# Similar to `tally_sum`, but assumes that `indices` is already sorted
def tally_sum_sorted(indices : torch.Tensor, data : torch.Tensor):
    assert len(indices.shape) == 1
    assert indices.shape[0] == data.shape[0]
    cum = torch.cumsum(data, dim=0)
    _, coun = torch.unique_consecutive(indices, return_counts=True)
    coun_cum = torch.cumsum(coun, dim=0)
    endpts = coun_cum - 1
    endvals = cum[endpts]
    startvals = torch.zeros(endvals.shape).cuda()
    startvals[1:] = endvals[:-1]
    startvals[0] = 0
    return indices[endpts], endvals - startvals