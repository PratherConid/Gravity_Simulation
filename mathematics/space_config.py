import torch

def translate(positions, dx):
    dim, cnt = positions.shape
    return positions + dx.broadcast_to((cnt, dim)).transpose(0, 1)

# slen : side length
def cubic_array(dim : int, k : int, slen):
    cnt = 1
    for i in range(dim):
        cnt = cnt * k
    pos = torch.zeros((dim, cnt))
    t = 1
    for i in range(dim):
        t = t * k
        for j in range(cnt):
            pos[i][j] = slen * ((j - (j // t) * t) // (t // k) - ((k - 1) / 2))
    return pos