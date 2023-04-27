import torch

# chids : (N,) array
def chain(chids : torch.Tensor, k):
    cnt, = chids.shape
    ret = torch.zeros((cnt - k + 1, k), dtype=torch.int64)
    for i in range(k):
        ret[:, i] = chids[i:cnt - k + 1 + i]
    return ret