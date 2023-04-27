import sys
sys.path.append("../Gravity_Simulation")

from physics.main import *
from renderer.sphereical_renderer import Spherical_Render

sim = Simulator(3, Physics('b'), 0.0004)

vdw = Simple_Van_Der_Walls("vdw_1", "vdw_rad", "vdw_k", 4, 6)
sim.add_physics("Van-Der-Walls", vdw)
bbox = BoundingBox("bbox_1", "rad", torch.tensor([0, 0, 0]), torch.tensor([7, 7, 7]))
sim.add_physics("Bounding-Box", bbox)
# Van-Der-Walls gas particles
sim.static_cube(16, slen=0.4, center=torch.zeros(3))

# Record the pressure of a side of the bounding box
sim.add_3dof_record(
    "Pressure of bounding box",
    lambda sim : float(sim.n_step),
    lambda sim : sim.physics["Bounding-Box"].pressure[0, 1],
    lambda sim : 0.0
)

sim.set_property_all("rad", torch.zeros(sim.N) + 0.15)
sim.set_property_all("mass", torch.zeros(sim.N) + 0.3)
sim.set_property_all("vdw_1", torch.zeros(sim.N) + 1)
sim.set_property_all("vdw_rad", torch.zeros(sim.N) + 0.1)
sim.set_property_all("vdw_k", torch.zeros(sim.N) + 0.01)
sim.set_property_all("bbox_1", torch.zeros(sim.N) + 1)
sim.uniform_random_vel_box(torch.zeros(3) - 8, torch.zeros(3) + 8)

for i in range(1000):
    if i % 10 == 0:
        print(i)
    sim.update()