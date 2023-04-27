import torch
from simulator import Simulator
from mathematics.base import associative_float_sum, cross_prod

# distance from `pos` to a line determined by a point `center` and
# direction vector `dir`
def distance_to_line(pos : torch.Tensor, center : torch.Tensor, dir : torch.Tensor):
    diff    = pos - center
    dir_n   = dir / ((dir * dir).sum()) ** 0.5
    difdir  = (diff * dir_n).sum(0)
    dist_sq = (diff * diff).sum(0) - (difdir * difdir)
    return dist_sq ** 0.5

# angular component of velocity with respect to an axis determined by
# 'center' (an arbitrary point on the axis) and `dir` (direction of the axis)
def angular_component_of_velocity(pos : torch.Tensor, vel : torch.Tensor, center : torch.Tensor, dir : torch.Tensor):
    diff    = pos - center
    agdir   = cross_prod(dir, diff)
    sagdir  = (agdir * agdir).sum()
    if sagdir == 0.0:
        return 0.0
    nagdir  = agdir / sagdir ** 0.5
    return float((vel * nagdir).sum())

# radial component of velocity with respect to an axis determined by
# 'center' (an arbitrary point on the axis) and `dir` (direction of the axis)
def radial_component_of_velocity(pos : torch.Tensor, vel : torch.Tensor, center : torch.Tensor, dir : torch.Tensor):
    diff    = pos - center
    rddir   = cross_prod(dir, cross_prod(dir, diff))
    srddir  = (rddir * rddir).sum()
    if srddir == 0.0:
        return 0.0
    nrddir  = rddir / srddir ** 0.5
    return float((vel * nrddir).sum())

# angular velocity with respect to an axis determined by
# 'center' (an arbitrary point on the axis) and `dir` (direction of the axis)
def angular_velocity(pos : torch.Tensor, vel : torch.Tensor, center : torch.Tensor, dir : torch.Tensor):
    sdiff   = distance_to_line(pos, center, dir)
    if sdiff == 0.0:
        return 0.0
    return angular_component_of_velocity(pos, vel, center, dir) / float(sdiff)


class Physics:

    # type
    #   b  -> phys_base
    #   a  -> acceleration based (dv = a dt)
    #   v  -> velocity change based (dv, e.g. collision)
    #   x  -> displacement based (dx, e.g. )
    #   pp -> prescribed position
    def __init__(self, type):
        assert type == 'b' or type == 'a' or type == 'v' or type == 'x' or type == 'pp'
        self.type = type
    
    # acceleration
    def a(self, sim : Simulator):
        return None

    # delta v
    def dv(self, sim : Simulator):
        return None

    # delta x
    def dx(self, sim : Simulator):
        return None
    
    # set v
    def set_v(self, sim : Simulator):
        return None

    # potential energy
    def Ep(self, sim : Simulator):
        pass

    # kinetic energy
    def Ek(self, sim : Simulator):
        vel      : torch.Tensor = sim.vel
        mass     : torch.Tensor = sim.pr["mass"]
        return float((mass * ((vel * vel).sum(0))).sum()) / 2

    # moment of inertia
    def I(self, sim : Simulator, axis : torch.Tensor, center : torch.Tensor):
        axis     : torch.Tensor = axis.cuda()
        center   : torch.Tensor = center.cuda()
        dim      : int          = sim.dim
        pos      : torch.Tensor = sim.pos
        mass     : torch.Tensor = sim.pr["mass"]
        cnt      : int          = sim.N
        # normalized axis
        nax      : torch.Tensor = torch.broadcast_to(axis / ((axis * axis).sum() ** 0.5), (cnt, dim)).transpose(0, 1)
        diff     : torch.Tensor = pos - torch.broadcast_to(center, (cnt, dim)).transpose(0, 1)
        dist_sq  : torch.Tensor = (diff * diff).sum(0) - (nax * diff).sum(0) ** 2
        return float((mass * dist_sq).sum()) / 2
    
    # center of mass
    def xbar(self, sim : Simulator):
        dim      : int          = sim.dim
        pos      : torch.Tensor = sim.pos
        mass     : torch.Tensor = sim.pr["mass"]
        cnt      : int          = sim.N
        mass_br  : torch.Tensor = torch.broadcast_to(mass, (dim, cnt))
        return associative_float_sum(pos * mass_br, -1) / mass.sum()

    # momentum
    def p(self, sim : Simulator):
        dim      : int          = sim.dim
        vel      : torch.Tensor = sim.vel
        mass     : torch.Tensor = sim.pr["mass"]
        cnt      : int          = sim.N
        mass_br  : torch.Tensor = torch.broadcast_to(mass, (dim, cnt))
        return associative_float_sum(vel * mass_br, -1)

    # average velocity
    def vbar(self, sim : Simulator):
        mass     : torch.ndarray = sim.pr["mass"]
        return self.p(sim) / mass.sum()