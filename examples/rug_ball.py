import sys
sys.path.append("../Gravity_Simulation")

import numpy as np
from physics.main import *
from renderer.sphereical_renderer import Spherical_Render
from mathematics.graph_config import chain

stick_len = 0.5
stick_k = 80
stick_eta = 2
stick_stiff_k = 80
x_num = 30
y_num = 50
grav_g = 0.3

# Elastic force of chain
def elas(post : torch.Tensor, velt : torch.Tensor, propt : torch.Tensor):
    pos1, pos2 = post
    vel1, vel2 = velt
    dist = ((pos2 - pos1) ** 2).sum(0) ** 0.5
    dist_s = dist + 10 * stick_len * (dist < 0.01)
    f1 = stick_k * (pos2 - pos1) * (dist - stick_len) / dist_s
    f2 = stick_k * (pos1 - pos2) * (dist - stick_len) / dist_s
    vel_coef = ((vel2 - vel1) * (pos2 - pos1)).sum(0)
    fric = stick_eta * vel_coef * (pos2 - pos1) / (dist_s ** 2)
    f1 += fric
    f2 -= fric
    return (f1, f2)

# Stiffness of surface
def stiff(post : torch.Tensor, velt : torch.Tensor, propt : torch.Tensor):
    pos1, pos2, pos3 = post
    diff = (pos1 + pos3 - 2 * pos2) / 2
    f1 = - stick_stiff_k * diff
    f2 = 2 * stick_stiff_k * diff
    f3 = - stick_stiff_k * diff
    return (f1, f2, f3)

def grav_force(pos):
    dim, cnt = pos.shape
    gf = torch.tensor([0, 0, -grav_g]).cuda()
    gf = gf.broadcast_to((cnt, dim)).transpose(0, 1)
    return gf

def presc_fun(pos : torch.Tensor, pr : torch.Tensor, t : float, dt : float):
    return pos

def set_up_positions(sim : Simulator):
    x_dir = np.linspace([-x_num / 2, -y_num / 2, 0], [x_num / 2, -y_num / 2, 0], x_num + 1) * stick_len
    surf = np.linspace(x_dir, x_dir + [0, y_num * stick_len, 0], y_num + 1)
    surf = surf.reshape(((x_num + 1) * (y_num + 1), 3)).transpose()
    # Surface
    sim.object_append(torch.tensor(surf))
    sph = np.array([0, 0, 2]).reshape((3, 1))
    # Sphere
    sim.object_append(torch.tensor(sph))

def elas_adj():
    x_dir_base = chain(torch.linspace(0, x_num, x_num + 1, dtype=torch.int64), 2)
    x_dir_base = x_dir_base.broadcast_to((y_num + 1, x_num, 2))
    x_dir_diff = torch.linspace(0, (x_num + 1) * y_num, y_num + 1, dtype=torch.int64)
    x_dir_diff = x_dir_diff.broadcast_to((x_num, 2, y_num + 1)).permute((2, 0, 1))
    x_dir = (x_dir_diff + x_dir_base).reshape((x_num * (y_num + 1), 2))
    y_dir_base = chain(torch.linspace(0, (x_num + 1) * y_num, y_num + 1, dtype=torch.int64), 2)
    y_dir_base = y_dir_base.broadcast_to((x_num + 1, y_num, 2))
    y_dir_diff = torch.linspace(0, x_num, x_num + 1, dtype=torch.int64)
    y_dir_diff = y_dir_diff.broadcast_to((y_num, 2, x_num + 1)).permute((2, 0, 1))
    y_dir = (y_dir_diff + y_dir_base).reshape(((x_num + 1) * y_num, 2))
    return torch.concat([x_dir, y_dir], 0)

def stiff_adj():
    x_dir_base = chain(torch.linspace(0, x_num, x_num + 1, dtype=torch.int64), 3)
    x_dir_base = x_dir_base.broadcast_to((y_num + 1, x_num - 1, 3))
    x_dir_diff = torch.linspace(0, (x_num + 1) * y_num, y_num + 1, dtype=torch.int64)
    x_dir_diff = x_dir_diff.broadcast_to((x_num - 1, 3, y_num + 1)).permute((2, 0, 1))
    x_dir = (x_dir_diff + x_dir_base).reshape(((x_num - 1) * (y_num + 1), 3))
    y_dir_base = chain(torch.linspace(0, (x_num + 1) * y_num, y_num + 1, dtype=torch.int64), 3)
    y_dir_base = y_dir_base.broadcast_to((x_num + 1, y_num - 1, 3))
    y_dir_diff = torch.linspace(0, x_num, x_num + 1, dtype=torch.int64)
    y_dir_diff = y_dir_diff.broadcast_to((y_num - 1, 3, x_num + 1)).permute((2, 0, 1))
    y_dir = (y_dir_diff + y_dir_base).reshape(((x_num + 1) * (y_num - 1), 3))
    return torch.concat([x_dir, y_dir], 0)

sim = Simulator(3, Physics('b'), 0.002)
set_up_positions(sim)

esc = GraphPosVel("dummy", elas, elas_adj())
sim.add_physics("Elastic-Surf", esc)
stf = GraphPosVel("dummy", stiff, stiff_adj())
sim.add_physics("Stiffness", stf)
grav = AccelerationField(grav_force)
sim.add_physics("Gravity", grav)
presc = PrescribedPos("presc", "dummy", presc_fun)
sim.add_physics("Prescribed-Displacement", presc)
idmut = IdealGas_Mut_E("surf", "sphere", "rad", "rad")
sim.add_physics("Collision", idmut)

sim.set_property_multi("rad", 0, (x_num + 1) * (y_num + 1), torch.zeros((x_num + 1) * (y_num + 1)) + 0.15)
sim.set_property_multi("mass", 0, (x_num + 1) * (y_num + 1), torch.zeros((x_num + 1) * (y_num + 1)) + 0.1)
sim.set_property_all("dummy", torch.zeros((x_num + 1) * (y_num + 1) + 1))
sim.set_property_single("rad", (x_num + 1) * (y_num + 1), 0.8)
sim.set_property_single("mass", (x_num + 1) * (y_num + 1), 2.5)
sim.set_property_multi("surf", 0, (x_num + 1) * (y_num + 1), torch.zeros((x_num + 1) * (y_num + 1)) + 1)
sim.set_property_single("sphere", (x_num + 1) * (y_num + 1), 1)
sim.set_property_multi("presc", 0, x_num + 1, torch.zeros(x_num + 1) + 1)
sim.set_property_multi("presc", (x_num + 1) * y_num, (x_num + 1) * (y_num + 1), torch.zeros(x_num + 1) + 1)
sim.pr["presc"][torch.linspace(0, (x_num + 1) * y_num, y_num + 1, dtype=torch.int64)] = 1
sim.pr["presc"][torch.linspace(x_num, (x_num + 1) * y_num + x_num, y_num + 1, dtype=torch.int64)] = 1

# OpenGL imports for python
try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
    from OpenGL.GLUT import *
except:
    print("OpenGL wrapper for python not found")
from renderer.save_video import create_videowriter, save_scene_to_videowriter
import cv2

spr = Spherical_Render(sim, 1, 32, 4, sim_render_precision=8)

def extra_display(sr : Spherical_Render):
    if sr.display_mode == 3:
        lx, ly, lz = sr.lookat
        glTranslatef(-lx, -ly, -lz)

        positions : np.ndarray = sr.simulator.pos.cpu().numpy()
        rugpos = positions.transpose()[:-1].reshape((y_num + 1, x_num + 1, 3))
        rugpos_00 = rugpos[:-1, :-1]
        rugpos_01 = rugpos[1:, :-1]
        rugpos_10 = rugpos[:-1, 1:]
        norms_hlp = np.zeros((y_num + 1, x_num + 1, 3))
        norms_hlp[:-1, :-1] = np.cross(rugpos_10 - rugpos_00, rugpos_01 - rugpos_00)
        norms_hlp[-1, :] = norms_hlp[-2, :]
        norms_hlp[:, -1] = norms_hlp[:, -2]

        glColor3f(0.5, 0.7, 0.6)
        glEnable(GL_LIGHTING)
        for i in range(x_num):
            for j in range(y_num):
                glBegin(GL_QUADS)
                verts = ((0, 0), (0, 1), (1, 1), (1, 0))
                for (di, dj) in verts:
                    x, y, z = rugpos[j + dj, i + di]
                    norm = norms_hlp[j + dj, i + di]
                    nx, ny, nz = norm / np.linalg.norm(norm)
                    glNormal3f(nx, ny, nz)
                    glVertex3f(x, y, z)
                glEnd()
        
        # Draw the sphere
        x, y, z = positions[:, -1]
        glPushMatrix()
        glColor3f(1, 1, 0)
        glTranslatef(x, y, z)
        gluSphere(gluNewQuadric(), sr.simulator.pr["rad"][-1], 15, 15)
        glPopMatrix()

        glTranslatef(lx, ly, lz)
spr.extra_display = extra_display

def post_display(sr : Spherical_Render):
    if sr.paused:
        return
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

spr.render()