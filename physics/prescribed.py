import torch
from physics.base import *

# Prescribed position, overrides any other physics
class PrescribedPos(Physics):

    # prescr_fun(pos : torch.Tensor, prop : torch.Tensor, t : float, dt : float)
    #   `pos` is supposed to be the position of this particle at the last time step
    def __init__(self, key_r, key_p, prescr_fun):
        super().__init__('pp')
        self.key_r = key_r
        self.key_p = key_p
        self.prescr_fun = prescr_fun

    def set_x(self, sim : Simulator, pos_last : torch.Tensor):
        relevant : torch.Tensor = torch.nonzero(sim.pr[self.key_r]).flatten()
        pos_last : torch.Tensor = pos_last[:, relevant]
        pr       : torch.Tensor = sim.pr[self.key_p][relevant]
        time     : float        = float(sim.dt * sim.n_step)
        sim.pos[:, relevant] = self.prescr_fun(pos_last, pr, time, float(sim.dt))
        sim.vel[:, relevant] = 0