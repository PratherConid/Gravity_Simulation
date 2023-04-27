from physics.base import *


# External acceleration on a single particle
class AccelOnParticle(Physics):

    def __init__(self, index : int, a : torch.Tensor):
        super().__init__('a')
        self.index = index
        self._a = a.cuda()
    
    def a(self, sim):
        ret   = torch.zeros((sim.dim, sim.N)).cuda()
        ret[:, self.index] = self._a
        return ret
    
    def dv(self, sim):
        return self.a(sim) * sim.dt
    
    def Ep(self, sim):
        return None
    

class AccelerationField(Physics):

    # `A`, `Ep` : (dim, N) torch.Tensor -> (dim, N) torch.Tensor
    def __init__(self, a, Ep = None):
        super().__init__('a')
        self._a = a
        self._Ep = Ep

    def a(self, sim):
        return self._a(sim.pos)

    def dv(self, sim):
        return self._a(sim.pos) * sim.dt

    def Ep(self, sim):
        if self._Ep is not None:
            return self._Ep(sim.pos)
        return None
    

# A 3D bowl made from elastic material
# Center : Center of the bowl
# r      : (Approximate) Radius of the bowl
# k      : Stiffness of the bowl
def static_bowl_accel(center : torch.Tensor, r, k, poses : torch.Tensor):
    assert center.shape == (3,)
    assert poses.shape[0] == 3
    dim      : int          = 3
    cnt      : int          = poses.shape[1]
    center = center.cuda()
    poses = poses.cuda()
    dpos = poses - center.broadcast_to((cnt, dim)).transpose(0, 1)
    x, y, z = dpos
    dist_sq = associative_float_sum(dpos * dpos, 0)
    in_bowl = (dist_sq - r * r) ** 2 + r * r * (z + r / 2) ** 2 <= (r ** 4) / 2
    force = torch.zeros((dim, cnt)).cuda()
    force[0] = 4 * x * (dist_sq - r * r)
    force[1] = 4 * y * (dist_sq - r * r)
    force[2] = r ** 3 - 2 * r * r * z + 4 * z * dist_sq
    force *= (r ** 4) / 2 - (dist_sq - r * r) ** 2 - r * r * (z + r / 2) ** 2
    force *= k / (2 * r ** 6)
    return force * in_bowl

# Loss for the above 3D bowl
def static_bowl_loss(center : torch.Tensor, r, eta, poses : torch.Tensor, vels : torch.Tensor):
    assert center.shape == (3,)
    assert poses.shape[0] == 3
    dim      : int          = 3
    cnt      : int          = poses.shape[1]
    center = center.cuda()
    poses = poses.cuda()
    vels = vels.cuda()
    dpos = poses - center.broadcast_to((cnt, dim)).transpose(0, 1)
    x, y, z = dpos
    dist_sq = associative_float_sum(dpos * dpos, 0)
    in_bowl = (dist_sq - r * r) ** 2 + r * r * (z + r / 2) ** 2 <= (r ** 4) / 2
    force_pre = torch.zeros((dim, cnt)).cuda()
    force_pre[0] = 4 * x * (dist_sq - r * r)
    force_pre[1] = 4 * y * (dist_sq - r * r)
    force_pre[2] = r ** 3 - 2 * r * r * z + 4 * z * dist_sq
    dotprd = associative_float_sum(force_pre * vels, 0)
    force = force_pre * ((-eta / 2) * dotprd) / associative_float_sum(force_pre * force_pre, 0)
    return force * in_bowl * (dotprd < 0)


class Acceleration_General(Physics):

    def __init__(self, a, Ep = None):
        super().__init__('a')
        self._a = a
        self._Ep = Ep
    
    def a(self, sim):
        return self._a(sim)
    
    def dv(self, sim):
        return self._a(sim) * sim.dt
    
    def Ep(self, sim):
        if self._Ep is not None:
            return self._Ep(sim)
        return None


class StaticElectricField(Physics):

    def __init__(self, key_charge, E):
        super().__init__('a')
        self.E = E
        self.key_charge = key_charge
    
    def a(self, sim):
        dim      : int          = sim.dim
        pos      : torch.Tensor = sim.pos
        e_val    : torch.Tensor = self.E(pos)
        mass     : torch.Tensor = sim.pr["mass"]
        charge   : torch.Tensor = sim.pr[self.key_charge]
        cnt      : int          = sim.N
        chg_br   : torch.Tensor = torch.broadcast_to(charge, (dim, cnt))
        force    : torch.Tensor = chg_br * e_val
        return force / mass

    def dv(self, sim):
        return self.a(sim) * sim.dt

    def Ep(self, sim):
        pass


class StaticMagneticField3D(Physics):

    # `B` : (3, N) torch.Tensor -> (3, N) torch.Tensor
    def __init__(self, key_charge, B):
        super().__init__('a')
        self.B = B
        self.key_charge = key_charge

    def a(self, sim):
        dim      : int          = sim.dim
        assert dim == 3
        pos      : torch.Tensor = sim.pos
        vel      : torch.Tensor = sim.vel
        bval     : torch.Tensor = self.B(pos)
        cnt      : int          = sim.N
        mass     : torch.Tensor = sim.pr["mass"]
        charge   : torch.Tensor = sim.pr[self.key_charge]
        chg_br   : torch.Tensor = torch.broadcast_to(charge, (dim, cnt))
        force    : torch.Tensor = chg_br * cross_prod(vel, bval)
        return force / mass

    def dv(self, sim):
        return self.a(sim) * sim.dt

    def Ep(self, sim):
        return 0.0