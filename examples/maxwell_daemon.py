import sys
sys.path.append("../Gravity_Simulation")

from physics.main import *
from renderer.sphereical_renderer import Spherical_Render

def region_1(pos : torch.Tensor):
    ret = torch.zeros(pos.shape[1]).cuda()
    return ret + 1

sim = Simulator(3, Physics('b'), 0.0004)

vdw = Simple_Van_Der_Walls("vdw_1", "vdw_rad", "vdw_k", 4, 6)
sim.add_physics("Van-Der-Walls", vdw)
bbox = BoundingBox("bbox_1", "rad", torch.tensor([0, 0, 0]), torch.tensor([6, 6, 6]))
sim.add_physics("Bounding-Box", bbox)
mxd = MaxwellDaemon("maxwd_1", "rad", torch.tensor([0, 0, 0]), 0.4, torch.tensor([0, 0, 1]), region_1, 2)
sim.add_physics("Maxwell-Daemon", mxd)
sim.static_cube(8, slen=0.4, center=torch.zeros(3))

# Record the pressure of the bounding box
sim.add_3dof_record(
    "Pressure of bounding box",
    lambda sim : float(sim.n_step),
    lambda sim : sim.physics["Bounding-Box"].pressure.sum() / 6,
    lambda sim : 0.0
)

sim.set_property_all("rad",  torch.zeros(sim.N) + 0.15)
sim.set_property_all("mass", torch.zeros(sim.N) + 0.3)
sim.set_property_all("vdw_1", torch.zeros(sim.N) + 1)
sim.set_property_all("vdw_rad", torch.zeros(sim.N) + 0.1)
sim.set_property_all("vdw_k", torch.zeros(sim.N) + 0.01)
sim.set_property_all("bbox_1", torch.zeros(sim.N) + 1)
sim.set_property_all("maxwd_1", torch.zeros(sim.N) + 1)
sim.uniform_random_vel_box(torch.zeros(3) - 0.1, torch.zeros(3) + 0.1)

spr = Spherical_Render(sim, 1, 8, 20, sim_render_precision=10)
spr.render()