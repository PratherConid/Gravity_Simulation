import sys
sys.path.append("../Gravity_Simulation")

from physics.main import *
from renderer.sphereical_renderer import Spherical_Render
import math

sim = Simulator(3, Physics('b'), 0.0004)

vdw = Simple_Van_Der_Walls("vdw_1", "vdw_rad", "vdw_k", 4, 6)
sim.add_physics("Van-Der-Walls", vdw)
bbox = BoundingBox("bbox_1", "rad", torch.tensor([20, 0, 0]), torch.tensor([44, 4, 4]), 0.7)
sim.add_physics("Bounding-Box", bbox)
therm = IdealGasThermometer("vdw_1", "therm_particle", "vdw_rad", "rad",
                            torch.tensor([-2, 0, 0]), torch.tensor([2, 2, 2]), 0.5)
sim.add_physics("Thermometer", therm)
mvw = MovingWall(8 ** 3 + 5 ** 3, "mvw_1", "rad", torch.tensor([-1, 0, 0]))
sim.add_physics("Moving-Wall", mvw)
accp = AccelOnParticle(8 ** 3 + 5 ** 3, torch.tensor([-80, 0, 0]))
sim.add_physics("Accel-On-Particle", accp)
# Van-Der-Walls gas particles
sim.static_cube(8, slen=0.4, center=torch.zeros(3))
# Thermometer ideal gas particles
sim.static_cube(5, slen=0.3, center=torch.tensor([-2, 0, 0]))
# The particle representing the wall
sim.object_append(torch.Tensor([[0], [0], [0]]))

# Record the state
sim.add_3dof_record(
    "State",
    # p
    lambda sim : float(sim.physics["Bounding-Box"].pressure[0, 1]),
    # V
    lambda sim : float(sim.pos[0, 8 ** 3 + 5 ** 3]),
    # T
    lambda sim : sim.physics["Thermometer"].kT
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
sim.set_property_multi("mvw_1", 0, 8 ** 3, torch.zeros(8 ** 3) + 1)
sim.uniform_random_vel_box(torch.zeros(3) - 8, torch.zeros(3) + 8)

# Zero velocity of wall
sim.set_vel(8 ** 3 + 5 ** 3, torch.tensor([0, 0, 0]))
# Set mass of wall
sim.set_property_single("mass", 8 ** 3 + 5 ** 3, 6)
# Position wall
sim.set_pos(8 ** 3 + 5 ** 3, torch.tensor([2, 0, 0]))

spr = Spherical_Render(sim, 1, 16, 20, sim_render_precision=8)

# set force on the moving wall
def set_force(ren : Spherical_Render, f : float):
    ren.simulator.physics["Accel-On-Particle"]._a[0] = -f
# type 'exec self.setf(self, <num>)' to set force on the moving wall
spr.sf = set_force

# OpenGL imports for python
try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
    from OpenGL.GLUT import *
except:
    print("OpenGL wrapper for python not found")

import cv2
import numpy as np
from renderer.save_video import create_videowriter, save_scene_to_videowriter
from renderer.two_d import loadTexture, drawQuadWithTexture

def post_display(sr : Spherical_Render):
    if sr.paused:
        return
    # Periodic force applied to the wall， We'll see the entropy of the system increase
    time = sr.simulator.n_step * sr.simulator.dt
    sr.simulator.physics['Accel-On-Particle']._a[0] = - 200 - 120 * math.cos(time + math.pi)

    if not hasattr(sr, "output"):
        sr.output = create_videowriter("output.avi")
    save_scene_to_videowriter(sr.output)
spr.post_display = post_display

def save_video(sr : Spherical_Render):
    sr.output.release()
    cv2.destroyAllWindows()
    print("Video saved")
# type 'exec self.sv(self)' to save video
spr.sv = save_video

def extra_display(sr : Spherical_Render):
    if sr.display_mode != 0:
        return

    glPushMatrix()
    # Set color to wood color
    glColor3f(0.9, 0.6, 0.3)
    lx, ly, lz = sr.lookat
    glTranslatef(-lx, -ly, -lz)
    wall_pos = sr.simulator.pos[:, 8 ** 3 + 5 ** 3].cpu().numpy()
    x, y, z = wall_pos
    cubesize = sr.simulator.physics["Bounding-Box"].size[1].cpu()

    # Draw wall
    glPushMatrix()
    glTranslatef(x + cubesize * 0.02 * 0.5, y, z)
    glScalef(0.02, 1.0, 1.0)
    glutSolidCube(float(cubesize))
    glPopMatrix()
    glPopMatrix()

    # Draw temperature
    temp = sr.simulator.physics["Thermometer"].kT
    sqh = temp / 40
    textId = loadTexture(np.zeros((100, 100, 4), dtype=np.uint8) +
                         np.array([200, 200, 0, 0]))
    glClear(GL_DEPTH_BUFFER_BIT)
    drawQuadWithTexture(0.9, sqh/2 + 0.5, 0.05, sqh, textId)

spr.extra_display = extra_display

spr.render()