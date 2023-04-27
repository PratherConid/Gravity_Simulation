import sys
sys.path.append("../Gravity_Simulation")

from physics.main import *
from renderer.sphereical_renderer import Spherical_Render

# Harmonic potential
def afun(pos : torch.Tensor):
    return - 0.2 * pos

def efun(pos : torch.Tensor):
    return 0.1 * (pos * pos).sum()

sim = Simulator(3, Physics('b'), 0.002)

isqsp = SphericalInverseSquare(1.0, "isq_1", "rad", "mass")
sim.add_physics("SphericalIS", isqsp)
accf = AccelerationField(afun, efun)
sim.add_physics("Harmonic-Potential", accf)
sim.static_cube(2, slen=0.4, center=torch.zeros(3))
sim.static_cube(2, slen=1, center=torch.zeros(3))

# Trajectory in phase space
sim.add_3dof_record(
    "Trajectory in phase space",
    lambda sim : float(sim.pos[0, 0]),
    lambda sim : float(sim.vel[0, 0]),
    lambda sim : float(sim.pos[0, 8])
)

sim.set_property_all("isq_1", torch.zeros(sim.N) + 1)
sim.set_property_all("bbox_1", torch.zeros(sim.N) + 1)
sim.set_property_all("rad", torch.zeros(sim.N) + 0.15)
sim.set_property_all("mass", torch.zeros(sim.N) + 0.1)

spr = Spherical_Render(sim, 1, 4, 4, sim_render_precision=8)
spr.render()