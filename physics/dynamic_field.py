from physics.base import *


# Force following inverse square law, with particles being spheres with
#   charge uniformly distributed on it.
class SphericalInverseSquare(Physics):

    def __init__(self, cp : float, key_r, key_charge, key_rad):
        super().__init__('a')
        self.cp = cp # Coupling constant
        self.key_r = key_r
        self.key_charge = key_charge
        self.key_rad = key_rad

    def a(self, sim):
        dim      : int          = sim.dim
        relevant : torch.Tensor = torch.nonzero(sim.pr[self.key_r]).flatten()
        cnt      : int          = relevant.shape[0]
        pos      : torch.Tensor = sim.pos[:, relevant]
        mass     : torch.Tensor = sim.pr["mass"][relevant]
        charge   : torch.Tensor = sim.pr[self.key_charge][relevant]
        rad      : torch.Tensor = sim.pr[self.key_rad][relevant]
        cp       : float        = self.cp
        # pos_br[:][i][j] = position for i-th sphere
        pos_br   : torch.Tensor = torch.broadcast_to(pos, (cnt, dim, cnt)).permute((1, 2, 0))
        # diff[:][i][j] = position for i-th sphere   -   position for j-th sphere
        diff     : torch.Tensor = pos_br - pos_br.permute((0, 2, 1))
        dist     : torch.Tensor = associative_float_sum(diff * diff, 0) ** 0.5
        iszero   : torch.Tensor = (dist == 0).to(torch.float64)
        dist += iszero
        rad_br   : torch.Tensor = torch.broadcast_to(rad, (cnt, cnt))
        sumrad   : torch.Tensor = rad_br + rad_br.transpose(0, 1)
        # mass_br[i][j] = mass of i-th sphere
        mass_br  : torch.Tensor = torch.broadcast_to(mass, (cnt, cnt)).transpose(0, 1)
        # chg_j[i][j] = charge of j-th sphere
        chg_j    : torch.Tensor = torch.broadcast_to(charge, (cnt, cnt))
        # t1 = (dist < sumrad) ? dist / (sumrad ** dim) : (dist ** (1 - dim))
        lt       : torch.Tensor = (dist < sumrad).to(torch.float64)
        t1       : torch.Tensor = lt * (dist / (sumrad ** dim)) + (1 - lt) * (dist ** (1 - dim))
        accelM   : torch.Tensor = (cp * t1) * (1 - iszero) * chg_j * chg_j.transpose(0, 1) / mass_br # mask zero
        # accelarr[i][j] = acceleration of i-th sphere caused by j-th sphere
        accelarr : torch.Tensor = (accelM / dist) * (-diff)
        ret      : torch.Tensor = torch.zeros((dim, sim.N)).cuda()
        ret[:, relevant] = associative_float_sum(accelarr, -1).to(dtype=torch.float32)
        return ret

    def dv(self, sim):
        return self.a(sim) * sim.dt

    def Ep(self, sim):
        dim      : int          = sim.dim
        relevant : torch.Tensor = torch.nonzero(sim.pr[self.key_r]).flatten()
        cnt      : int          = relevant.shape[0]
        pos      : torch.Tensor = sim.pos[:, relevant]
        charge   : torch.Tensor = sim.pr[self.key_charge][relevant]
        rad      : torch.Tensor = sim.pr[self.key_rad][relevant]
        cp       : float        = self.cp
        # pos_br[:][i][j] = position for i-th sphere
        pos_br   : torch.Tensor = torch.broadcast_to(pos, (cnt, dim, cnt)).permute((1, 2, 0))
        # diff[:][i][j] = position for i-th sphere   -   position for j-th sphere
        diff     : torch.Tensor = pos_br - pos_br.permute((0, 2, 1))
        dist     : torch.Tensor = (diff * diff).sum(0) ** 0.5
        iszero   : torch.Tensor = (dist == 0).to(torch.float64)
        dist += iszero
        rad_br   : torch.Tensor = torch.broadcast_to(rad, (cnt, cnt))
        sumrad   : torch.Tensor = rad_br + rad_br.transpose(0, 1)
        # mass_br[i][j] = mass of j-th sphere
        chg_br   : torch.Tensor = torch.broadcast_to(charge, (cnt, cnt))
        chg_prd  : torch.Tensor = chg_br * (chg_br.transpose(0, 1))
        # t3 = (dist < sumrad) ? (dist ** 2) / (2 * sumrad ** dim) :
        #      (- (dist ** (2 - dim)) / (dim - 2) + (1 / 2 + 1 / (dim - 2)) * sumrad ** (2 - dim))
        t1       : torch.Tensor = dist ** 2 / (2 * sumrad ** dim)
        t2       : torch.Tensor = - (dist ** (2 - dim)) / (dim - 2) + (1 / 2 + 1 / (dim - 2)) * sumrad ** (2 - dim)
        lt       : torch.Tensor = (dist < sumrad).to(dtype=torch.float32)
        t3       : torch.Tensor = lt * t1 + (1 - lt) * t2
        ep       : torch.Tensor = cp * chg_prd * t3 * (1 - iszero)
        return float(ep.sum()) / 2