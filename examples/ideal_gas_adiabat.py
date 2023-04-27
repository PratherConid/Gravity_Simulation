import sys
sys.path.append("../Gravity_Simulation")

from physics.main import *
from renderer.sphereical_renderer import Spherical_Render

sim = Simulator(3, Physics('b'), 0.004)

idg = IdealGas_E("id1", "rad")
sim.add_physics("Ideal-Gas", idg)
bbox = BoundingBox("bbox_1", "rad", torch.tensor([20, 0, 0]), torch.tensor([44, 4, 4]), p_tau=0.1)
sim.add_physics("Bounding-Box", bbox)
sim.static_cube(8, slen=0.4, center=torch.zeros(3))
# The particle representing the wall
sim.object_append(torch.Tensor([[0], [0], [0]]))
mvw = MovingWall(8 ** 3, "mvw_1", "rad", torch.tensor([-2, 0, 0]))
sim.add_physics("Moving-Wall", mvw)
accp = AccelOnParticle(8 ** 3, torch.tensor([-8, 0, 0]))
sim.add_physics("Accel-On-Particle", accp)

# Record the pressure of the bounding box
sim.add_3dof_record(
    "Pressure of bounding box",
    lambda sim : float(sim.physics["Bounding-Box"].pressure[0, 1]),
    lambda sim : float(sim.pos[0, 8 ** 3]),
    lambda sim : 0.0
)

sim.set_property_multi("rad",  0, 8 ** 3, torch.zeros(8 ** 3) + 0.15)
sim.set_property_multi("mass", 0, 8 ** 3, torch.zeros(8 ** 3) + 0.3)
sim.set_property_multi("id1", 0, 8 ** 3, torch.zeros(8 ** 3) + 1)
sim.set_property_multi("bbox_1", 0, 8 ** 3, torch.zeros(8 ** 3) + 1)
sim.set_property_multi("mvw_1", 0, 8 ** 3, torch.zeros(8 ** 3) + 1)
sim.uniform_random_vel_box(torch.zeros(3) - 0.3, torch.zeros(3) + 0.3)

# Zero velocity of wall
sim.set_vel(8 ** 3, torch.tensor([0, 0, 0]))
# Set mass of wall
sim.set_property_single("mass", 8 ** 3, 6)
# Position wall
sim.set_pos(8 ** 3, torch.tensor([2, 0, 0]))

spr = Spherical_Render(sim, 1, 8, 20, sim_render_precision=10)

# set force on the moving wall
def set_force(ren : Spherical_Render, f : float):
    ren.simulator.physics["Accel-On-Particle"]._a[0] = -f
# type 'exec self.setf(self, <num>)' to set force on the moving wall
spr.sf = set_force
spr.render()