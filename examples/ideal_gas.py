import sys
sys.path.append("../Gravity_Simulation")

from physics.main import *
from renderer.sphereical_renderer import Spherical_Render

def region_1(pos : torch.Tensor):
    ret = torch.zeros(pos.shape[1]).cuda()
    return ret + 1

sim = Simulator(3, Physics('b'), 0.004)

idg = IdealGas_E("id1", "rad")
sim.add_physics("Ideal-Gas", idg)
bbox = BoundingBox("bbox_1", "rad", torch.tensor([0, 0, 0]), torch.tensor([4, 4, 4]), p_tau=1)
sim.add_physics("Bounding-Box", bbox)
sim.static_cube(8, slen=0.4, center=torch.zeros(3))

# Record the pressure of the bounding box
sim.add_3dof_record(
    "Pressure of bounding box",
    lambda sim : float(sim.n_step),
    lambda sim : sim.physics["Bounding-Box"].pressure.sum() / 6,
    lambda sim : sim.phys_base.Ek(sim) * (2 / 3)
)

sim.set_property_all("rad",  torch.zeros(sim.N) + 0.15)
sim.set_property_all("mass", torch.zeros(sim.N) + 0.3)
sim.set_property_all("id1", torch.zeros(sim.N) + 1)
sim.set_property_all("bbox_1", torch.zeros(sim.N) + 1)
sim.uniform_random_vel_box(torch.zeros(3) - 6, torch.zeros(3) + 6)

spr = Spherical_Render(sim, 1, 8, 20, sim_render_precision=10)
spr.render()