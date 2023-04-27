import sys
sys.path.append("../Gravity_Simulation")

from physics.main import *
from renderer.sphereical_renderer import Spherical_Render

# Friction
def fric(sim : Simulator, a, b, c):
    pos      : torch.tensor = sim.pos
    vel      : torch.tensor = sim.vel
    norm     : torch.tensor = (pos * pos).sum(0) ** 0.5
    vnorm    : torch.tensor = (vel * vel).sum(0) ** 0.5
    vnorm_st : torch.tensor = vnorm + (vnorm < 0.01)
    # Propulsion
    ret      : torch.tensor = a * (norm < 0.6) * (vel / vnorm_st)
    # Friction
    ret += (norm > 0.2) * (-c) * vel
    # Mass Change
    sim.pr['mass'] *= 1 + sim.dt * (pos[0][0] ** 2 > 0.1)
    sim.pr['mass'] *= 1 - sim.dt * (pos[0][0] ** 2 < 0.1) * (sim.pr['mass'] > 0.1)
    return ret

sim = Simulator(3, Physics('b'), 0.01)

isqsp = SphericalInverseSquare(1.0, "isq_1", "rad", "mass")
sim.add_physics("SphericalIS", isqsp)
fricf = Acceleration_General(lambda sim : fric(sim, 0.5, 1.2, 0.4))
sim.add_physics("Friction", fricf)
sim.static_cube(2, slen=1, center=torch.zeros(3))

# Trajectory in phase space
sim.add_3dof_record(
    "Trajectory in phase space",
    lambda sim : float(sim.pos[0, 0]),
    lambda sim : float(sim.vel[0, 0]),
    lambda sim : float(sim.pr['mass'][0])
)

sim.set_property_all("isq_1", torch.zeros(sim.N) + 1)
sim.set_property_all("bbox_1", torch.zeros(sim.N) + 1)
sim.set_property_all("rad", torch.zeros(sim.N) + 0.15)
sim.set_property_all("mass", torch.zeros(sim.N) + 0.1)

spr = Spherical_Render(sim, 1, 32, 1, sim_render_precision=8)
spr.render()