from physics.base import *


# The Moving Wall is represented by an infinite half (hyper)space
class MovingWall(Physics):

    def __init__(self, index_wall : int, key_r : str, key_rad : str, norm : torch.Tensor):
        super().__init__('v')
        self.index_wall = index_wall
        self.key_r = key_r
        self.key_rad = key_rad
        # Normal vector of the half (hyper)space, points away from the half space that is the wall
        self.norm = (norm / ((norm * norm).sum() ** 0.5)).cuda()
        dim = norm.shape[0]
        # force experienced by the moving wall
        self.transient_I = torch.zeros((dim,))

    def dv(self, sim):
        dim      : int          = sim.dim
        assert sim.pr[self.key_r].shape == (sim.N,)
        relevant : torch.Tensor = torch.nonzero(sim.pr[self.key_r]).flatten()
        cnt      : int          = relevant.shape[0]
        pos      : torch.Tensor = sim.pos[:, relevant]
        vel      : torch.Tensor = sim.vel[:, relevant]
        rad      : torch.Tensor = sim.pr[self.key_rad][relevant]
        mass     : torch.Tensor = sim.pr["mass"][relevant]
        pos_w    : torch.Tensor = sim.pos[:, self.index_wall]
        vel_w    : torch.Tensor = sim.vel[:, self.index_wall]
        mass_w   : torch.Tensor = sim.pr["mass"][self.index_wall]
        diff     : torch.Tensor = pos - pos_w.broadcast_to((cnt, dim)).transpose(0, 1)
        vdiff    : torch.Tensor = vel - vel_w.broadcast_to((cnt, dim)).transpose(0, 1)
        norm_br  : torch.Tensor = torch.broadcast_to(self.norm, (cnt, dim)).transpose(0, 1)
        diffnorm : torch.Tensor = (diff * norm_br).sum(0)
        vdiffnorm: torch.Tensor = (vdiff * norm_br).sum(0)
        dv_base  : torch.Tensor = ((vdiffnorm < 0) * (diffnorm < rad)) * ((-2) * vdiffnorm)
        dv       : torch.Tensor = dv_base * (mass_w / (mass_w + mass)) * norm_br
        ret      : torch.Tensor = torch.zeros((dim, sim.N), dtype=dv.dtype).cuda()
        ret[:, relevant] = dv
        self.transient_I = 0 - (dv * mass).sum(-1)
        ret[:, self.index_wall] = self.transient_I / mass_w
        return ret


class BoundingBox(Physics):

    def __init__(self, key_r : str, key_rad : str, center : torch.Tensor, size : torch.Tensor, p_tau = 0.1):
        super().__init__('v')
        # The array sim.pr[key] decides whether the boundingbox acts upon each particle
        self.key_r = key_r
        self.key_rad = key_rad
        self.center = center.cuda()
        self.size = size.cuda()
        self.upper_bound = self.center + self.size / 2
        self.lower_bound = self.center - self.size / 2
        self.volume = self.size.prod()
        # surface area of each surface
        self.area_of_surface = self.volume / self.size
        dim = size.shape[0]
        # pressure on positive and negative face of the box
        self.transient_I = torch.zeros((dim, 2))
        self.pressure = torch.zeros((dim, 2))
        self.p_tau = p_tau

    def dv(self, sim):
        dim      : int          = sim.dim
        assert sim.pr[self.key_r].shape == (sim.N,)
        relevant : torch.Tensor = torch.nonzero(sim.pr[self.key_r]).flatten()
        cnt      : int          = relevant.shape[0]
        pos      : torch.Tensor = sim.pos[:, relevant]
        vel      : torch.Tensor = sim.vel[:, relevant]
        rad      : torch.Tensor = sim.pr[self.key_rad][relevant]
        mass     : torch.Tensor = sim.pr["mass"][relevant]
        ub_br    : torch.Tensor = torch.broadcast_to(self.upper_bound, (cnt, dim)).transpose(0, 1)
        lb_br    : torch.Tensor = torch.broadcast_to(self.lower_bound, (cnt, dim)).transpose(0, 1)
        too_large: torch.Tensor = (pos + rad > ub_br).to(torch.float64)
        too_small: torch.Tensor = (pos - rad < lb_br).to(torch.float64)
        dv_ub = - 2 * too_large * (vel > 0) * vel
        self.transient_I[:, 0] = 0 - (dv_ub * mass).sum(-1)
        dv_lb = - 2 * too_small * (vel < 0) * vel
        # Transient impulse
        self.transient_I[:, 1] = (dv_lb * mass).sum(-1)
        self.pressure = (1 - sim.dt / self.p_tau) * self.pressure + (1 / self.p_tau) * self.transient_I
        ret      : torch.Tensor = torch.zeros((dim, sim.N), dtype=dv_ub.dtype).cuda()
        ret[:, relevant] = dv_ub + dv_lb
        return ret


# Warning : Don't use PeriodicBox if the objects has some force
#           which tends to infinity when the distance between two objects tends to zero
class PeriodicBox(Physics):

    def __init__(self, key_r : str, key_rad : str, center : torch.Tensor, size : torch.Tensor):
        super().__init__('x')
        self.key_r = key_r
        self.key_rad = key_rad
        self.center = center.cuda()
        self.size = size.cuda()
        self.upper_bound = self.center + self.size / 2
        self.lower_bound = self.center - self.size / 2

    def dx(self, sim):
        dim      : int          = sim.dim
        relevant : torch.Tensor = torch.nonzero(sim.pr[self.key_r]).flatten()
        cnt      : int          = relevant.shape[0]
        pos      : torch.Tensor = sim.pos[:, relevant]
        rad      : torch.Tensor = sim.pr[self.key_rad][relevant]
        ub_br    : torch.Tensor = torch.broadcast_to(self.upper_bound, (cnt, dim)).transpose(0, 1)
        lb_br    : torch.Tensor = torch.broadcast_to(self.lower_bound, (cnt, dim)).transpose(0, 1)
        sz_br    : torch.Tensor = torch.broadcast_to(self.size, (cnt, dim)).transpose(0, 1)
        too_large: torch.Tensor = pos + rad > ub_br
        too_small: torch.Tensor = pos - rad < lb_br
        dx_ub = 0 - too_large * sz_br
        dx_lb = too_small * sz_br
        ret      : torch.Tensor = torch.zeros((dim, sim.N), dtype=dx_ub.dtype).cuda()
        ret[:, relevant] = dx_ub + dx_lb
        return ret


# A planar maxwell daemon, with the direction vector of the plane being `dir`
# The function `region` is an extra condition for a position in space to satisfy
# for it to be considered inside the planar maxwell daemon
class MaxwellDaemon(Physics):

    def __init__(self, key_r : str, key_rad : str, center : torch.Tensor,
                 thickness, dir : torch.Tensor, region, Ek_threshold):
        super().__init__('v')
        self.key_r = key_r
        self.key_rad = key_rad
        self.center = center.cuda()
        self.thickness = thickness
        self.dir = dir.cuda()
        self.region = region
        self.Ek_threshold = Ek_threshold
        dim = center.shape[0]
        self.transient_I = torch.zeros((dim, 2))
        self.force = torch.zeros((dim, 2))
    
    def dv(self, sim):
        dim      : int          = sim.dim
        relevant : torch.Tensor = torch.nonzero(sim.pr[self.key_r]).flatten()
        cnt      : int          = relevant.shape[0]
        pos      : torch.Tensor = sim.pos[:, relevant]
        rad      : torch.Tensor = sim.pr[self.key_rad][relevant]
        mass     : torch.Tensor = sim.pr["mass"][relevant]
        vel      : torch.Tensor = sim.vel
        kinetic  : torch.Tensor = 0.5 * ((vel * vel).sum(0)) * mass
        center_br: torch.Tensor = torch.broadcast_to(self.center, (cnt, dim)).transpose(0, 1)
        dir_n    : torch.Tensor = self.dir / ((self.dir * self.dir).sum() ** 0.5)
        dir_n_br : torch.Tensor = torch.broadcast_to(dir_n, (cnt, dim)).transpose(0, 1)
        verti_v  : torch.Tensor = ((sim.vel * dir_n_br).sum(0)) * dir_n_br
        large    : torch.Tensor = ((pos - center_br) * dir_n_br).sum(0) - rad
        is_in    : torch.Tensor = self.region(pos)
        is_large : torch.Tensor = ((large < self.thickness / 2) * (large > - self.thickness / 2) * is_in * (kinetic > self.Ek_threshold)).to(torch.float64)
        dv_large : torch.Tensor = - 2 * is_large * ((sim.vel * dir_n_br).sum(0) < 0) * verti_v
        small    : torch.Tensor = ((pos - center_br) * dir_n_br).sum(0) + rad
        is_small : torch.Tensor = ((small < self.thickness / 2) * (small > - self.thickness / 2) * is_in * (kinetic < self.Ek_threshold)).to(torch.float64)
        dv_small : torch.Tensor = - 2 * is_small * ((sim.vel * dir_n_br).sum(0) > 0) * verti_v
        ret      : torch.Tensor = torch.zeros((dim, sim.N), dtype=dv_large.dtype).cuda()
        ret[:, relevant] = dv_small + dv_large
        return ret