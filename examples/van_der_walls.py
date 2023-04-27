import sys
sys.path.append("../Gravity_Simulation")

from physics.main import *
from renderer.sphereical_renderer import Spherical_Render

sim = Simulator(3, Physics('b'), 0.0004)

vdw = Simple_Van_Der_Walls("vdw_1", "vdw_rad", "vdw_k", 4, 6)
sim.add_physics("Van-Der-Walls", vdw)
bbox = BoundingBox("bbox_1", "rad", torch.tensor([0, 0, 0]), torch.tensor([4, 4, 4]))
sim.add_physics("Bounding-Box", bbox)
therm = IdealGasThermometer("vdw_1", "therm_particle", "vdw_rad", "rad",
                            torch.tensor([0, 0, 2]), torch.tensor([2, 2, 2]), 0.5)
sim.add_physics("Thermometer", therm)
# Van-Der-Walls gas particles
sim.static_cube(8, slen=0.4, center=torch.zeros(3))
# Thermometer ideal gas particles
sim.static_cube(5, slen=0.3, center=torch.tensor([0, 0, 2]))

# Record the temperature
sim.add_3dof_record(
    "Temperature",
    lambda sim : float(sim.n_step),
    lambda sim : sim.physics["Thermometer"].kT,
    lambda sim : 0.0
)

# Record the pressure of a side of the bounding box
sim.add_3dof_record(
    "Pressure of bounding box",
    lambda sim : float(sim.n_step),
    lambda sim : sim.physics["Bounding-Box"].pressure[0, 1],
    lambda sim : 0.0
)

sim.set_property_multi("rad", 0, 8 ** 3, torch.zeros(8 ** 3) + 0.15)
sim.set_property_multi("rad", 8 ** 3, 8 ** 3 + 5 ** 3, torch.zeros(5 ** 3) + 0.1)
sim.set_property_multi("mass", 0, 8 ** 3, torch.zeros(8 ** 3) + 0.3)
sim.set_property_multi("mass", 8 ** 3, 8 ** 3 + 5 ** 3, torch.zeros(5 ** 3) + 0.1)
sim.set_property_multi("vdw_1", 0, 8 ** 3, torch.zeros(8 ** 3) + 1)
sim.set_property_multi("vdw_rad", 0, 8 ** 3, torch.zeros(8 ** 3) + 0.1)
sim.set_property_multi("vdw_k", 0, 8 ** 3, torch.zeros(8 ** 3) + 0.01)
sim.set_property_multi("bbox_1", 0, 8 ** 3, torch.zeros(8 ** 3) + 1)
sim.set_property_multi("therm_particle", 8 ** 3, 8 ** 3 + 5 ** 3, torch.zeros(5 ** 3) + 1)
sim.uniform_random_vel_box(torch.zeros(3) - 8, torch.zeros(3) + 8)

spr = Spherical_Render(sim, 1, 8, 20, sim_render_precision=8)
spr.render()