import sys
sys.path.append("../Gravity_Simulation")

from physics.main import *
from renderer.sphereical_renderer import Spherical_Render

sim = Simulator(3, Physics('b'), 0.0008)

isqsp = SphericalInverseSquare(1.0, "isq_1", "rad", "mass")
sim.add_physics("SphericalIS", isqsp)
bbox = BoundingBox("bbox_1", "rad", torch.tensor([0, 0, 0]), torch.tensor([2, 2, 2]))
sim.add_physics("Bounding-Box", bbox)
sim.static_cube(4, slen=0.4, center=torch.zeros(3))

sim.set_property_all("isq_1", torch.zeros(sim.N) + 1)
sim.set_property_all("bbox_1", torch.zeros(sim.N) + 1)
sim.set_property_all("rad", torch.zeros(sim.N) + 0.15)
sim.set_property_all("mass", torch.zeros(sim.N) + 0.1)

spr = Spherical_Render(sim, 1, 4, 20, sim_render_precision=8)
spr.render()