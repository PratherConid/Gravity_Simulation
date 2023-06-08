import sys
sys.path.append("../Gravity_Simulation")

import numpy as np
import math
from physics.main import *
from renderer.sphereical_renderer import Spherical_Render
from mathematics.base import cross_prod
from mathematics.graph_config import chain

stick_k = 2000
stick_eta = 2
stick_stiff_k = 1000
bc_r = 10
sc_r = 2.5
bc_num = 60
sc_num = 16
grav_g = 0.3
z_low = -14

# Elastic force of chain
def elas(post : torch.Tensor, velt : torch.Tensor, ini_pos : torch.Tensor):
    pos1, pos2 = post
    vel1, vel2 = velt
    ipos1, ipos2 = ini_pos
    dist = ((pos2 - pos1) ** 2).sum(0) ** 0.5
    ini_len = ((ipos1 - ipos2) ** 2).sum(1) ** 0.5
    dist_s = dist + 10 * ini_len * (dist < 0.01)
    real_k = stick_k / ini_len
    f1 = real_k * (pos2 - pos1) * (dist - ini_len) / dist_s
    f2 = real_k * (pos1 - pos2) * (dist - ini_len) / dist_s
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
    nf = -500.0 * (pos - z_low) * (pos < z_low)
    return gf + nf

def sc_bc():
    bc_ = np.linspace([0, 0], [0, bc_num - 1], bc_num, dtype=np.int64)
    sc_bc = np.linspace(bc_, bc_ + np.array([sc_num - 1, 0], dtype=np.int64), sc_num, dtype=np.int64)
    return sc_bc

def sc_bc_to_id(sc_bc : np.ndarray):
    return (sc_bc[:, :, 0] * bc_num + sc_bc[:, :, 1]).flatten()

def elas_adj():
    bc_dir_base = chain(torch.linspace(0, bc_num, bc_num + 1, dtype=torch.int64), 2)
    bc_dir_base = torch.remainder(bc_dir_base, bc_num)
    bc_dir_base = bc_dir_base.broadcast_to((sc_num, bc_num, 2))
    bc_dir_diff = torch.linspace(0, bc_num * (sc_num - 1), sc_num, dtype=torch.int64)
    bc_dir_diff = bc_dir_diff.broadcast_to((bc_num, 2, sc_num)).permute((2, 0, 1))
    bc_dir = (bc_dir_diff + bc_dir_base).reshape((bc_num * sc_num, 2))
    sc_dir_base = chain(torch.linspace(0, bc_num * sc_num, sc_num + 1, dtype=torch.int64), 2)
    sc_dir_base = torch.remainder(sc_dir_base, bc_num * sc_num)
    sc_dir_base = sc_dir_base.broadcast_to((bc_num, sc_num, 2))
    sc_dir_diff = torch.linspace(0, bc_num - 1, bc_num, dtype=torch.int64)
    sc_dir_diff = sc_dir_diff.broadcast_to((sc_num, 2, bc_num)).permute((2, 0, 1))
    sc_dir = (sc_dir_diff + sc_dir_base).reshape((bc_num * sc_num, 2))
    sh1_src = sc_bc_to_id(sc_bc())
    sh1_dst = sc_bc_to_id(np.roll(sc_bc(), (1, 1), (0, 1)))
    sh1 = torch.tensor(np.stack((sh1_src, sh1_dst), 1))
    sh2_src = sc_bc_to_id(sc_bc())
    sh2_dst = sc_bc_to_id(np.roll(sc_bc(), (1, -1), (0, 1)))
    sh2 = torch.tensor(np.stack((sh2_src, sh2_dst), 1))
    return torch.concat([bc_dir, sc_dir, sh1, sh2], 0)

def stiff_adj():
    bc_dir_base = chain(torch.linspace(0, bc_num + 1, bc_num + 2, dtype=torch.int64), 3)
    bc_dir_base = torch.remainder(bc_dir_base, bc_num)
    bc_dir_base = bc_dir_base.broadcast_to((sc_num, bc_num, 3))
    bc_dir_diff = torch.linspace(0, bc_num * (sc_num - 1), sc_num, dtype=torch.int64)
    bc_dir_diff = bc_dir_diff.broadcast_to((bc_num, 3, sc_num)).permute((2, 0, 1))
    bc_dir = (bc_dir_diff + bc_dir_base).reshape((bc_num * sc_num, 3))
    sc_dir_base = chain(torch.linspace(0, bc_num * (sc_num + 1), sc_num + 2, dtype=torch.int64), 3)
    sc_dir_base = torch.remainder(sc_dir_base, bc_num * sc_num)
    sc_dir_base = sc_dir_base.broadcast_to((bc_num, sc_num, 3))
    sc_dir_diff = torch.linspace(0, bc_num - 1, bc_num, dtype=torch.int64)
    sc_dir_diff = sc_dir_diff.broadcast_to((sc_num, 3, bc_num)).permute((2, 0, 1))
    sc_dir = (sc_dir_diff + sc_dir_base).reshape((bc_num * sc_num, 3))
    return torch.concat([bc_dir, sc_dir], 0)

def set_up_pos_vel(sim : Simulator):
    theta = np.linspace([0, 0], [2 * math.pi, 0], bc_num + 1)[:bc_num]
    theta_phi = np.linspace(theta, theta + np.array([0, 2 * math.pi]), sc_num + 1)[:sc_num]
    theta_phi = theta_phi.reshape((sc_num * bc_num, 2))
    torus = np.zeros((sc_num * bc_num, 3))
    theta = theta_phi[:, 0]; phi = theta_phi[:, 1]
    torus[:, 0] = np.cos(theta) * (bc_r + np.cos(phi) * sc_r)
    torus[:, 1] = np.sin(phi) * sc_r
    torus[:, 2] = np.sin(theta) * (bc_r + np.cos(phi) * sc_r)
    # Position
    sim.object_append(torch.tensor(torus.transpose()))
    sim.set_property_all("initpos", torch.tensor(torus.copy()))
    # Velocity
    rot_omega = torch.tensor([0, 0, 0.5])
    vel = cross_prod(torch.tensor(torus.transpose()), rot_omega)
    sim.set_vel(0, vel)

sim = Simulator(3, Physics('b'), 0.002)
set_up_pos_vel(sim)

esc = GraphPosVel("initpos", elas, elas_adj())
sim.add_physics("Elastic-Surf", esc)
stf = GraphPosVel("dummy", stiff, stiff_adj())
sim.add_physics("Stiffness", stf)
grav = AccelerationField(grav_force)
sim.add_physics("Gravity", grav)

sim.set_property_all("rad", torch.zeros(sim.N) + 0.15)
sim.set_property_all("mass", torch.zeros(sim.N) + 0.1)
sim.set_property_all("dummy", torch.zeros(sim.N))

spr = Spherical_Render(sim, 1, 64, 4, sim_render_precision=8)

# OpenGL imports for python
try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
    from OpenGL.GLUT import *
except:
    print("OpenGL wrapper for python not found")
from renderer.save_video import create_videowriter, save_scene_to_videowriter
import cv2

def extra_display(sr : Spherical_Render):
    if sr.display_mode == 3:
        lx, ly, lz = sr.lookat
        glTranslatef(-lx, -ly, -lz)

        positions : np.ndarray = sr.simulator.pos.cpu().numpy()
        rugpos = positions.transpose().reshape((sc_num, bc_num, 3))
        rugpos_00 = rugpos
        rugpos_01 = np.roll(rugpos, 1, 0)
        rugpos_10 = np.roll(rugpos, 1, 1)
        norms_hlp = np.cross(rugpos_01 - rugpos_00, rugpos_10 - rugpos_00)

        for i in range(bc_num):
            for j in range(sc_num):
                glBegin(GL_QUADS)
                verts = ((0, 0), (0, 1), (1, 1), (1, 0))
                ity = 0.7 + 0.3 * ((i + j) % 2)
                glColor3f(ity, ity, ity)
                for (di, dj) in verts:
                    x, y, z = rugpos[(j + dj) % sc_num, (i + di) % bc_num]
                    norm = norms_hlp[(j + dj) % sc_num, (i + di) % bc_num]
                    nx, ny, nz = norm / np.linalg.norm(norm)
                    glNormal3f(nx, ny, nz)
                    glVertex3f(x, y, z)
                glEnd()
        
        glPushMatrix()
        glTranslatef(0, 0, z_low)
        glScalef(1.0, 1.0, 0.002)
        glColor3f(0.5, 0.3, 0.0)
        glutSolidCube(80.0)
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