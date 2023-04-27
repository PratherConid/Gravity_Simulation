from copy import copy
from physics.base import *
from physics.wall import *
from physics.classical_gas import *


# Based on the fact that the kinetic energy of ideal
#   gas is equal to (dim / 2) * k * T
class IdealGasThermometer(Physics):

    # measurand : particle whose temperature is being measured
    # thermometer : particles which are used to measure temperature
    # The measurand particles collides with thermometer particles
    def __init__(self, key_measurand, key_thermometer_particle,
                 key_rad_measurand, key_rad_thermometer_particle,
                 center : torch.Tensor, size : torch.Tensor, p_tau=0.1):
        super().__init__('v')
        self.km = key_measurand
        self.ktp = key_thermometer_particle
        self.krm = key_rad_measurand
        self.krtp = key_rad_thermometer_particle
        self.bbox = BoundingBox(self.ktp, self.krtp, center, size, p_tau)
        self.idmut = IdealGas_Mut_E(self.km, self.ktp, self.krm, self.krtp)
        self.idthm = IdealGas_E(self.ktp, self.krtp)
        self.kT = 0.0
    
    def dv(self, sim):
        therms = torch.nonzero(sim.pr[self.ktp]).flatten()
        self.kT = (sim.pr["mass"][therms] * (sim.vel[:, therms] * sim.vel[:, therms]).sum(0)).sum() / sim.dim
        self.kT = float(self.kT / therms.nelement())
        sim_copy = copy(sim)
        dv_bbox = self.bbox.dv(sim)
        sim_copy.vel = sim_copy.vel + dv_bbox
        dv_idmut = self.idmut.dv(sim_copy)
        sim_copy.vel = sim_copy.vel + dv_idmut
        return dv_bbox + dv_idmut + self.idthm.dv(sim_copy)