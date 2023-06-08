from physics.base import *


# Note that energy might not conserve if multiple
# particles collide with each other at the same time
class IdealGas(Physics):

    def __init__(self, key_r : str, key_rad : str):
        super().__init__('v')
        self.key_r = key_r
        self.key_rad = key_rad
    
    def dv(self, sim):
        dim      : int          = sim.dim
        assert sim.pr[self.key_r].shape == (sim.N,)
        relevant : torch.Tensor = torch.nonzero(sim.pr[self.key_r]).flatten()
        cnt      : int          = relevant.shape[0]
        pos      : torch.Tensor = sim.pos[:, relevant]
        vel      : torch.Tensor = sim.vel[:, relevant]
        rad      : torch.Tensor = sim.pr[self.key_rad][relevant]
        mass     : torch.Tensor = sim.pr["mass"][relevant]
        # pos_br[:][i][j] = position for i-th sphere
        pos_br   : torch.Tensor = torch.broadcast_to(pos, (cnt, dim, cnt)).permute((1, 2, 0))
        # diff[:][i][j] = position for i-th sphere   -   position for j-th sphere
        diff     : torch.Tensor = pos_br - pos_br.permute((0, 2, 1))
        dist     : torch.Tensor = associative_float_sum(diff * diff, 0) ** 0.5
        iszero   : torch.Tensor = (dist == 0).to(torch.float64)
        dist += iszero
        # vel_br[:][i][j] = velocity for i-th sphere
        vel_br   : torch.Tensor = torch.broadcast_to(vel, (cnt, dim, cnt)).permute((1, 2, 0))
        # vdiff[:][i][j] = velocity of i-th sphere   -   velocity of j-th sphere
        vdiff    : torch.Tensor = vel_br - vel_br.permute((0, 2, 1))
        # vdiff_n[:][i][j] = vdiff[:][i][j] component along (diff[:][i][j] / dist)
        vdiff_n  : torch.Tensor = associative_float_sum(diff * vdiff, 0) / dist
        rad_br   : torch.Tensor = torch.broadcast_to(rad, (cnt, cnt))
        sumrad   : torch.Tensor = rad_br + rad_br.transpose(0, 1)
        collide  : torch.Tensor = (sumrad > dist) * (vdiff_n < 0)
        # mass_br[i][j] = mass of the j-th sphere
        mass_br  : torch.Tensor = torch.broadcast_to(mass, (cnt, cnt))
        summass  : torch.Tensor = mass_br + mass_br.transpose(0, 1)
        dv_mat   : torch.Tensor = - (2 * vdiff_n * collide * mass_br / summass) * (diff / dist)
        dv       : torch.Tensor = associative_float_sum(dv_mat, -1)
        ret      : torch.Tensor = torch.zeros((dim, sim.N), dtype=dv.dtype).cuda()
        ret[:, relevant] = dv
        return ret
    

# Energy conserved version of Ideal Gas
# A and B collides iff A is the closest to B and B is the closest to A (radius substracted)
class IdealGas_E(Physics):

    def __init__(self, key_r : str, key_rad : str):
        super().__init__('v')
        self.key_r = key_r
        self.key_rad = key_rad
    
    def dv(self, sim):
        dim      : int          = sim.dim
        assert sim.pr[self.key_r].shape == (sim.N,)
        relevant : torch.Tensor = torch.nonzero(sim.pr[self.key_r]).flatten()
        cnt      : int          = relevant.shape[0]
        pos      : torch.Tensor = sim.pos[:, relevant]
        vel      : torch.Tensor = sim.vel[:, relevant]
        rad      : torch.Tensor = sim.pr[self.key_rad][relevant]
        mass     : torch.Tensor = sim.pr["mass"][relevant]
        # pos_br[:][i][j] = position for i-th sphere
        pos_br   : torch.Tensor = torch.broadcast_to(pos, (cnt, dim, cnt)).permute((1, 2, 0))
        # diff[:][i][j] = position for i-th sphere   -   position for j-th sphere
        diff     : torch.Tensor = pos_br - pos_br.permute((0, 2, 1))
        dist     : torch.Tensor = associative_float_sum(diff * diff, 0) ** 0.5
        rad_br   : torch.Tensor = torch.broadcast_to(rad, (cnt, cnt))
        sumrad   : torch.Tensor = rad_br + rad_br.transpose(0, 1)
        iszero   : torch.Tensor = (dist == 0).to(torch.float64)
        dist += iszero * (torch.max(dist) + torch.max(sumrad) + 1)
        disc_dist: torch.Tensor = dist - sumrad
        mindist: torch.Tensor = torch.min(disc_dist, 0).indices
        mintab   : torch.Tensor = torch.zeros((cnt, cnt)).cuda()
        lins     : torch.Tensor = torch.linspace(0, cnt - 1, cnt, dtype=torch.int64)
        mintab[lins, mindist] += 1
        mintab[mindist, lins] += 1
        valid    : torch.Tensor = (mintab == 2)
        # vel_br[:][i][j] = velocity for i-th sphere
        vel_br   : torch.Tensor = torch.broadcast_to(vel, (cnt, dim, cnt)).permute((1, 2, 0))
        # vdiff[:][i][j] = velocity of i-th sphere   -   velocity of j-th sphere
        vdiff    : torch.Tensor = vel_br - vel_br.permute((0, 2, 1))
        # vdiff_n[:][i][j] = vdiff[:][i][j] component along (diff[:][i][j] / dist)
        vdiff_n  : torch.Tensor = associative_float_sum(diff * vdiff, 0) / dist
        collide  : torch.Tensor = (disc_dist < 0) * (vdiff_n < 0) * valid
        # mass_br[i][j] = mass of the j-th sphere
        mass_br  : torch.Tensor = torch.broadcast_to(mass, (cnt, cnt))
        summass  : torch.Tensor = mass_br + mass_br.transpose(0, 1)
        dv_mat   : torch.Tensor = - (2 * vdiff_n * collide * mass_br / summass) * (diff / dist)
        dv       : torch.Tensor = associative_float_sum(dv_mat, -1)
        ret      : torch.Tensor = torch.zeros((dim, sim.N), dtype=dv.dtype).cuda()
        ret[:, relevant] = dv
        return ret


# Elastic collision between particle array determined by key_r_1 and key_r_2
# Important Note: The two arrays should be non-intersecting
class IdealGas_Mut(Physics):

    def __init__(self, key_r_1 : str, key_r_2 : str, key_rad_1 : str, key_rad_2):
        super().__init__('v')
        self.key_r_1 = key_r_1
        self.key_r_2 = key_r_2
        self.key_rad_1 = key_rad_1
        self.key_rad_2 = key_rad_2
    
    def dv(self, sim):
        dim      : int          = sim.dim
        assert sim.pr[self.key_r_1].shape == (sim.N,)
        assert sim.pr[self.key_r_2].shape == (sim.N,)
        relev_1  : torch.Tensor = torch.nonzero(sim.pr[self.key_r_1]).flatten()
        relev_2  : torch.Tensor = torch.nonzero(sim.pr[self.key_r_2]).flatten()
        # Non-intersecting
        assert torch.all(sim.pr[self.key_r_1] * sim.pr[self.key_r_2] == 0)
        cnt_1    : int          = relev_1.shape[0]
        cnt_2    : int          = relev_2.shape[0]
        pos_1    : torch.Tensor = sim.pos[:, relev_1]
        pos_2    : torch.Tensor = sim.pos[:, relev_2]
        vel_1    : torch.Tensor = sim.vel[:, relev_1]
        vel_2    : torch.Tensor = sim.vel[:, relev_2]
        rad_1    : torch.Tensor = sim.pr[self.key_rad_1][relev_1]
        rad_2    : torch.Tensor = sim.pr[self.key_rad_2][relev_2]
        mass_1   : torch.Tensor = sim.pr["mass"][relev_1]
        mass_2   : torch.Tensor = sim.pr["mass"][relev_2]
        # pos_br_1[:][i][j] = position for i-th sphere
        pos_br_1 : torch.Tensor = torch.broadcast_to(pos_1, (cnt_2, dim, cnt_1)).permute((1, 2, 0))
        # pos_br_2[:][i][j] = position for j-th sphere
        pos_br_2 : torch.Tensor = torch.broadcast_to(pos_2, (cnt_1, dim, cnt_2)).permute((1, 0, 2))
        # diff[:][i][j] = position for i-th sphere   -   position for j-th sphere
        diff     : torch.Tensor = pos_br_1 - pos_br_2
        dist     : torch.Tensor = associative_float_sum(diff * diff, 0) ** 0.5
        # rad_br_1[i][j] = radius of i-th sphere
        rad_br_1 : torch.Tensor = torch.broadcast_to(rad_1, (cnt_2, cnt_1)).transpose(0, 1)
        # rad_br_2[i][j] = radius of j-th sphere
        rad_br_2 : torch.Tensor = torch.broadcast_to(rad_2, (cnt_1, cnt_2))
        sumrad   : torch.Tensor = rad_br_1 + rad_br_2
        iszero   : torch.Tensor = (dist == 0).to(torch.float64)
        dist += iszero * (1 + torch.max(dist) + torch.max(sumrad))
        disc_dist = dist - sumrad
        mintab   : torch.Tensor = torch.zeros((cnt_1, cnt_2)).cuda()
        lins_1   : torch.Tensor = torch.linspace(0, cnt_1 - 1, cnt_1, dtype=torch.int64)
        mintab[lins_1, torch.min(disc_dist, 1).indices] += 1
        del lins_1
        lins_2   : torch.Tensor = torch.linspace(0, cnt_2 - 1, cnt_2, dtype=torch.int64)
        mintab[torch.min(disc_dist, 0).indices, lins_2] += 1
        del lins_2
        valid    : torch.Tensor = (mintab == 2)
        del mintab
        # vel_br_1[:][i][j] = velocity for i-th sphere
        vel_br_1 : torch.Tensor = torch.broadcast_to(vel_1, (cnt_2, dim, cnt_1)).permute((1, 2, 0))
        # vel_br_2[:][i][j] = velocity for j-th sphere
        vel_br_2 : torch.Tensor = torch.broadcast_to(vel_2, (cnt_1, dim, cnt_2)).permute((1, 0, 2))
        # vdiff[:][i][j] = velocity of i-th sphere   -   velocity of j-th sphere
        vdiff    : torch.Tensor = vel_br_1 - vel_br_2
        # vdiff_n[:][i][j] = vdiff[:][i][j] component along (diff[:][i][j] / dist)
        vdiff_n  : torch.Tensor = associative_float_sum(diff * vdiff, 0) / dist
        collide  : torch.Tensor = (disc_dist < 0) * (vdiff_n < 0)
        # mass_br[i][j] = mass of the j-th sphere
        mass_br_1: torch.Tensor = torch.broadcast_to(mass_1, (cnt_2, cnt_1)).transpose(0, 1)
        mass_br_2: torch.Tensor = torch.broadcast_to(mass_2, (cnt_1, cnt_2))
        summass  : torch.Tensor = mass_br_1 + mass_br_2
        dv_mat_1 : torch.Tensor = - (2 * vdiff_n * collide * mass_br_2 * valid / summass) * (diff / dist)
        dv_1     : torch.Tensor = associative_float_sum(dv_mat_1, -1)
        dv_mat_2 : torch.Tensor = (2 * vdiff_n * collide * mass_br_1 * valid / summass) * (diff / dist)
        dv_2     : torch.Tensor = associative_float_sum(dv_mat_2, -2)
        ret      : torch.Tensor = torch.zeros((dim, sim.N), dtype=dv_1.dtype).cuda()
        ret[:, relev_1] = dv_1
        ret[:, relev_2] = dv_2
        return ret
    

# Energy conserved version of IdealGas_Mut
# Important Note: The two arrays should be non-intersecting
class IdealGas_Mut_E(Physics):

    def __init__(self, key_r_1 : str, key_r_2 : str, key_rad_1 : str, key_rad_2):
        super().__init__('v')
        self.key_r_1 = key_r_1
        self.key_r_2 = key_r_2
        self.key_rad_1 = key_rad_1
        self.key_rad_2 = key_rad_2
    
    def dv(self, sim):
        dim      : int          = sim.dim
        assert sim.pr[self.key_r_1].shape == (sim.N,)
        assert sim.pr[self.key_r_2].shape == (sim.N,)
        return _IdealGas_Mut_E_dv_Core(sim.N, dim,
                                       sim.pos, sim.vel, sim.pr[self.key_rad_1], sim.pr[self.key_rad_2],
                                       sim.pr["mass"], sim.pr[self.key_r_1], sim.pr[self.key_r_2])

@torch.jit.script
def _IdealGas_Mut_E_dv_Core(Num: int, dim: int,
                            pos, vel, full_rad_1, full_rad_2,
                            mass, key_r_1_arr, key_r_2_arr):
    # Non-intersecting
    assert torch.all(key_r_1_arr * key_r_2_arr == 0)
    relev_1  : torch.Tensor = torch.nonzero(key_r_1_arr).flatten()
    relev_2  : torch.Tensor = torch.nonzero(key_r_2_arr).flatten()
    cnt_1    : int          = relev_1.shape[0]
    cnt_2    : int          = relev_2.shape[0]
    pos_1    : torch.Tensor = pos[:, relev_1]
    pos_2    : torch.Tensor = pos[:, relev_2]
    vel_1    : torch.Tensor = vel[:, relev_1]
    vel_2    : torch.Tensor = vel[:, relev_2]
    mass_1   : torch.Tensor = mass[relev_1]
    mass_2   : torch.Tensor = mass[relev_2]
    rad_1    : torch.Tensor = full_rad_1[relev_1]
    rad_2    : torch.Tensor = full_rad_2[relev_2]
    # pos_br_1[:][i][j] = position for i-th sphere
    pos_br_1 : torch.Tensor = torch.broadcast_to(pos_1, (cnt_2, dim, cnt_1)).permute((1, 2, 0))
    # pos_br_2[:][i][j] = position for j-th sphere
    pos_br_2 : torch.Tensor = torch.broadcast_to(pos_2, (cnt_1, dim, cnt_2)).permute((1, 0, 2))
    # diff[:][i][j] = position for i-th sphere   -   position for j-th sphere
    diff     : torch.Tensor = pos_br_1 - pos_br_2
    dist     : torch.Tensor = associative_float_sum(diff * diff, 0) ** 0.5
    iszero   : torch.Tensor = (dist == 0).to(torch.float64)
    dist += iszero
    # vel_br_1[:][i][j] = velocity for i-th sphere
    vel_br_1 : torch.Tensor = torch.broadcast_to(vel_1, (cnt_2, dim, cnt_1)).permute((1, 2, 0))
    # vel_br_2[:][i][j] = velocity for j-th sphere
    vel_br_2 : torch.Tensor = torch.broadcast_to(vel_2, (cnt_1, dim, cnt_2)).permute((1, 0, 2))
    # vdiff[:][i][j] = velocity of i-th sphere   -   velocity of j-th sphere
    vdiff    : torch.Tensor = vel_br_1 - vel_br_2
    # vdiff_n[:][i][j] = vdiff[:][i][j] component along (diff[:][i][j] / dist)
    vdiff_n  : torch.Tensor = associative_float_sum(diff * vdiff, 0) / dist
    # rad_br_1[i][j] = radius of i-th sphere
    rad_br_1 : torch.Tensor = torch.broadcast_to(rad_1, (cnt_2, cnt_1)).transpose(0, 1)
    # rad_br_2[i][j] = radius of j-th sphere
    rad_br_2 : torch.Tensor = torch.broadcast_to(rad_2, (cnt_1, cnt_2))
    sumrad   : torch.Tensor = rad_br_1 + rad_br_2
    collide  : torch.Tensor = (sumrad > dist) * (vdiff_n < 0)
    # mass_br[i][j] = mass of the j-th sphere
    mass_br_1: torch.Tensor = torch.broadcast_to(mass_1, (cnt_2, cnt_1)).transpose(0, 1)
    mass_br_2: torch.Tensor = torch.broadcast_to(mass_2, (cnt_1, cnt_2))
    summass  : torch.Tensor = mass_br_1 + mass_br_2
    dv_mat_1 : torch.Tensor = - (2 * vdiff_n * collide * mass_br_2 / summass) * (diff / dist)
    dv_1     : torch.Tensor = associative_float_sum(dv_mat_1, -1)
    dv_mat_2 : torch.Tensor = (2 * vdiff_n * collide * mass_br_1 / summass) * (diff / dist)
    dv_2     : torch.Tensor = associative_float_sum(dv_mat_2, -2)
    ret      : torch.Tensor = torch.zeros((dim, Num), dtype=dv_1.dtype).cuda()
    ret[:, relev_1] = dv_1
    ret[:, relev_2] = dv_2
    return ret


class Simple_Van_Der_Walls(Physics):

    # Simple Van Der Walls interaction
    # Potential Energy:
    #       Ep_12 = (vew_k_1 * vdw_k_2) * (1 / r^exp2 - K / r^exp1)
    #       K = exp2 / exp1 * (r_vdw_1 + r_vdw_2) ^ (exp1 - exp2)
    #       Minimal energy at `r = r_vdw_1 + r_vdw_2`
    # Force magnitude:
    #       F_12 = (vew_k_1 * vdw_k_2) * (exp2 / r^(exp2 + 1) - exp1 * K / r^(exp1 + 1))
    # Normally, exp1 < exp2
    def __init__(self, key_r, key_rad : str, key_k : str, exp1, exp2):
        super().__init__('a')
        self.key_r = key_r
        self.key_rad = key_rad
        self.key_k = key_k
        self.exp1 = exp1
        self.exp2 = exp2

    def a(self, sim):
        dim      : int          = sim.dim
        assert sim.pr[self.key_r].shape == (sim.N,)
        relevant : torch.Tensor = torch.nonzero(sim.pr[self.key_r]).flatten()
        cnt      : int          = relevant.shape[0]
        pos      : torch.Tensor = sim.pos[:, relevant]
        mass     : torch.Tensor = sim.pr["mass"][relevant]
        vdw_rad  : torch.Tensor = sim.pr[self.key_rad][relevant]
        vdw_k    : torch.Tensor = sim.pr[self.key_k][relevant]
        # pos_br[:][i][j] = position for i-th sphere
        pos_br   : torch.Tensor = torch.broadcast_to(pos, (cnt, dim, cnt)).permute((1, 2, 0))
        # diff[:][i][j] = position for i-th sphere   -   position for j-th sphere
        diff     : torch.Tensor = pos_br - pos_br.permute((0, 2, 1))
        dist     : torch.Tensor = associative_float_sum(diff * diff, 0) ** 0.5
        iszero   : torch.Tensor = (dist == 0).to(torch.float64)
        dist += iszero
        vdw_rad_br : torch.Tensor = torch.broadcast_to(vdw_rad, (cnt, cnt))
        vdw_k_br : torch.Tensor = torch.broadcast_to(vdw_k, (cnt, cnt))
        sumvrad  : torch.Tensor = vdw_rad_br + vdw_rad_br.transpose(0, 1)
        prodvk   : torch.Tensor = vdw_k_br * vdw_k_br.transpose(0, 1)
        K        : torch.Tensor = self.exp2 / self.exp1 * (sumvrad ** (self.exp1 - self.exp2))
        forceM   : torch.Tensor = prodvk * (self.exp2 * dist ** (-self.exp2 - 1) - self.exp1 * K * (dist ** (-self.exp1 - 1)))\
                                * (1 - iszero) # mask zero
        forcearr : torch.Tensor = (forceM / dist) * diff
        ret      : torch.Tensor = torch.zeros((dim, sim.N), dtype=forcearr.dtype).cuda()
        ret[:, relevant] = associative_float_sum(forcearr, -1) / mass
        return ret

    def dv(self, sim):
        return self.a(sim) * sim.dt

    def Ep(self, sim):
        dim      : int          = sim.dim
        assert sim.pr[self.key_r].shape == (sim.N,)
        relevant : torch.Tensor = torch.nonzero(sim.pr[self.key_r]).flatten()
        cnt      : int          = relevant.shape[0]
        pos      : torch.Tensor = sim.pos[relevant]
        vdw_rad  : torch.Tensor = sim.pr["vdw_rad"][relevant]
        vdw_k    : torch.Tensor = sim.pr["vdw_k"][relevant]
        # pos_br[:][i][j] = position for i-th sphere
        pos_br   : torch.Tensor = torch.broadcast_to(pos, (cnt, dim, cnt)).transpose((1, 2, 0))
        # diff[:][i][j] = position for i-th sphere   -   position for j-th sphere
        diff     : torch.Tensor = pos_br - pos_br.transpose((0, 2, 1))
        dist     : torch.Tensor = (diff * diff).sum(0) ** 0.5
        iszero   : torch.Tensor = (dist == 0).to(torch.float64)
        dist += iszero
        vdw_rad_br : torch.Tensor = torch.broadcast_to(vdw_rad, (cnt, cnt))
        vdw_k_br : torch.Tensor = torch.broadcast_to(vdw_k, (cnt, cnt))
        sumvrad  : torch.Tensor = vdw_rad_br + vdw_rad_br.transpose(0, 1)
        prodvk   : torch.Tensor = vdw_k_br * vdw_k_br.transpose(0, 1)
        K        : torch.Tensor = self.exp2 / self.exp1 * (sumvrad ** (self.exp1 - self.exp2))
        ep       : torch.Tensor = prodvk * (dist ** (-self.exp2) - K * (dist ** (-self.exp1)))
        ep       : torch.Tensor = ep * (1 - iszero) # mask zero
        return float(ep.sum()) / 2