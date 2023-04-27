import sys
sys.path.append("../Gravity_Simulation")

from physics.main import *
from renderer.sphereical_renderer import Spherical_Render

def region_1(pos : torch.Tensor):
    ret = torch.zeros(pos.shape[1]).cuda()
    return ret + 1

sim = Simulator(3, Physics('b'), 0.004)

# electron
sim.static_cube(6, slen=0.4, center=torch.zeros(3))
# nuclei
sim.static_cube(7, slen=0.4, center=torch.zeros(3))

idg = IdealGas("id1", "rad")
sim.add_physics("Ideal-Gas", idg)
bbox_e = BoundingBox("bbox_e", "rad", torch.tensor([0, 0, 0]), torch.tensor([3.5, 3.5, 3.5]), p_tau=1)
sim.add_physics("Bounding-Box-electron", bbox_e)
sim.set_property_multi("bbox_e", 0, 6 ** 3, torch.zeros(6 ** 3) + 1)
bbox_n = BoundingBox("bbox_n", "rad", torch.tensor([0, 0, 0]), torch.tensor([3.5, 3.5, 3.5]), p_tau=100)
sim.add_physics("Bounding-Box-nuclei", bbox_n)
sim.set_property_multi("bbox_n", 6 ** 3, 6 ** 3 + 7 ** 3, torch.zeros(7 ** 3) + 1)

# Record the pressure of the bounding box for electrons
# This will reveal the temperature of the electron gas
sim.add_3dof_record(
    "Pressure of bounding box",
    lambda sim : float(sim.n_step),
    lambda sim : sim.physics["Bounding-Box-electron"].pressure.sum() / 6,
    lambda sim : 0.0
)

# electron
sim.set_property_multi("rad", 0, 6 ** 3, torch.zeros(6 ** 3) + 0.05)
sim.set_property_multi("mass", 0, 6 ** 3, torch.zeros(6 ** 3) + 0.01)
# nuclei
sim.set_property_multi("rad", 6 ** 3, 6 ** 3 + 7 ** 3, torch.zeros(7 ** 3) + 0.12)
sim.set_property_multi("mass", 6 ** 3, 6 ** 3 + 7 ** 3, torch.zeros(7 ** 3) + 19)
sim.set_property_all("id1", torch.zeros(sim.N) + 1)
sim.set_property_all("maxwd_1", torch.zeros(sim.N) + 1)
# heat originally resides in electron gas
sim.vel[:, 0:6 ** 3] = torch.rand((3, 6 ** 3)) - 0.5

spr = Spherical_Render(sim, 1, 8, 20, sim_render_precision=10)
spr.render()