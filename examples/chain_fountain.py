import sys
sys.path.append("../Gravity_Simulation")

import numpy as np
import math
from physics.main import *
from renderer.sphereical_renderer import Spherical_Render
from mathematics.graph_config import chain

stick_len = 0.5
cylin_rad = 0.06
stick_k = 800
stick_eta = 20
stick_stiff_k = 800
stick_stiff_eta = 20
bowl_r = 5
bowl_k = 40
bowl_eta = 40
z_low = -40
ci4_r = 2
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

# Stiffness of chain
def stiff(post : torch.Tensor, velt : torch.Tensor, propt : torch.Tensor):
    pos1, pos2, pos3 = post
    diff = (pos1 + pos3 - 2 * pos2) / 2
    dist = (diff * diff).sum(0) ** 0.5
    dist_s = dist + 10 * stick_len * (dist < 0.001 * stick_len)
    coef = (dist > 0.15 * stick_len) * (dist - 0.15 * stick_len)
    f1 = - stick_stiff_k * diff * coef / dist_s
    f2 = 2 * stick_stiff_k * diff * coef / dist_s
    f3 = - stick_stiff_k * diff * coef / dist_s
    vel1, vel2, vel3 = velt
    vdiff = (vel1 + vel3 - 2 * vel2) / 2
    vel_coef = (vdiff * diff).sum(0)
    fric = stick_stiff_eta * vel_coef * diff / (dist_s ** 2)
    f1 -= fric
    f2 += 2 * fric
    f3 -= fric
    return (f1, f2, f3)

# Friction for balls near the edge of the bowl
def fric_force(sim : Simulator):
    pos = sim.pos
    norm = (pos * pos).sum(0) ** 0.5
    vel = sim.vel
    vel_n = (vel * vel).sum(0) ** 0.5 + 0.001
    near_bowl = (norm ** 2 - bowl_r * bowl_r) ** 2 + bowl_r * bowl_r * (pos[2] + bowl_r / 2) ** 2 <= (bowl_r ** 4) / 2 + 0.001
    return - 20.0 * vel * near_bowl * (pos[2] < -bowl_r / 4) / vel_n

def grav_force(pos):
    dim, cnt = pos.shape
    gf = torch.tensor([0, 0, -grav_g]).cuda()
    gf = gf.broadcast_to((cnt, dim)).transpose(0, 1)
    return gf

def presc_fun(pos : torch.Tensor, pr : torch.Tensor, t : float, dt : float):
    return pos

def set_up_positions(sim : Simulator):
    e_circ = np.array([-10.4, 2.8, 5.8])
    circ_r = 5
    circ_n = int(math.pi * circ_r / (2 * stick_len))
    circ_d_ang = 2 * math.asin(stick_len / (circ_r * 2))
    circ_angs = np.linspace(0, circ_d_ang * (circ_n - 1), circ_n)
    circ_dpos = circ_r * np.stack([-np.sin(circ_angs), 0 * circ_angs, np.cos(circ_angs)], 1)
    circ_dpos = np.flip(circ_dpos, 0)
    # Position of spheres on circles
    circ_pos = (circ_dpos + (e_circ - circ_dpos[-1])).transpose()
    b_circ = circ_pos[:, 0]

    e_vert = b_circ - stick_len * np.array([0, 0, 1])
    vert_n = 20
    b_vert = e_vert - stick_len * np.array([0, 0, vert_n - 1])
    # Position of spheres on the vertical line
    vertical = np.linspace(b_vert, e_vert, vert_n).transpose()

    b_hori = e_circ + stick_len * np.array([1, 0, 0])
    hori_n = 6
    e_hori = b_hori + stick_len * np.array([hori_n - 1, 0, 0])
    # Position of spheres on the horizontal line
    horizontal = np.linspace(b_hori, e_hori, hori_n).transpose()

    b_ci2 = e_hori + stick_len * np.array([1, 0, 0])
    ci2_r = 5
    ci2_n = int(math.pi * ci2_r / (2 * stick_len))
    ci2_d_ang = 2 * math.asin(stick_len / (ci2_r * 2))
    ci2_angs = np.linspace(0, ci2_d_ang * (ci2_n - 1), ci2_n)
    ci2_dpos = ci2_r * np.stack([np.sin(ci2_angs), 0 * ci2_angs, np.cos(ci2_angs)], 1)
    # Position of spheres on circle no.2
    ci2_pos = (ci2_dpos + (b_ci2 - ci2_dpos[0])).transpose()
    e_ci2 = ci2_pos[:, -1]

    b_ci3 = e_ci2 + stick_len * np.array([0, 0, -1])
    ci3_r = 3
    ci3_n = int(math.pi * ci3_r / (2 * stick_len))
    ci3_d_ang = 2 * math.asin(stick_len / (ci3_r * 2))
    ci3_angs = np.linspace(0, ci3_d_ang * (ci3_n - 1), ci3_n)
    ci3_dpos = ci3_r * np.stack([0 * ci3_angs, np.cos(ci3_angs), -np.sin(ci3_angs)], 1)
    # Position of spheres on circle no.2
    ci3_pos = (ci3_dpos + (b_ci3 - ci3_dpos[0])).transpose()
    e_ci3 = ci3_pos[:, -1]

    b_ci4 = e_ci3 + stick_len * np.array([0, -1, 0])
    ci4_n = int(12 * math.pi * ci4_r / stick_len)
    ci4_d_ang = 2 * math.asin(stick_len / (ci4_r * 2))
    ci4_angs = np.linspace(0, ci4_d_ang * (ci4_n - 1), ci4_n) + math.pi
    ci4_dpos = ci4_r * np.stack([np.cos(ci4_angs), np.sin(ci4_angs), 0 * ci4_angs], 1)
    # Position of spheres on circle no.2
    ci4_pos = (ci4_dpos + (b_ci4 - ci4_dpos[0])).transpose()
    e_ci4 = ci4_pos[:, -1]

    sim.object_append(torch.tensor(vertical))
    sim.object_append(torch.tensor(circ_pos))
    sim.object_append(torch.tensor(horizontal))
    sim.object_append(torch.tensor(ci2_pos))
    sim.object_append(torch.tensor(ci3_pos))
    sim.object_append(torch.tensor(ci4_pos))

    # We need to append balls, so record the status of ci4
    sim.last_ang = ci4_angs[-1]
    sim.ci4_center = b_ci4 - ci4_dpos[0]

def add_more_to_ci4(sim : Simulator, n : int):
    ci4_d_ang = 2 * math.asin(stick_len / (ci4_r * 2))
    ang_start = sim.last_ang + ci4_d_ang
    ang_end = sim.last_ang + n * ci4_d_ang
    ci4_angs = np.linspace(ang_start, ang_end, n)
    ci4_dpos = ci4_r * np.stack([np.cos(ci4_angs), np.sin(ci4_angs), 0 * ci4_angs], 1)
    ci4_pos = (ci4_dpos + sim.ci4_center).transpose()

    sim.object_append(torch.tensor(ci4_pos))
    sim.set_property_multi("rad", sim.N - n, sim.N, torch.zeros(n) + 0.2)
    sim.set_property_multi("mass", sim.N - n, sim.N, torch.zeros(n) + 0.1)
    sim.set_property_multi("dummy", sim.N - n, sim.N, torch.zeros(n))
    sim.set_property_all("presc", torch.zeros(sim.N))
    sim.set_property_multi("presc", sim.N - 20, sim.N, torch.zeros(20) + 1)
    sim.last_ang = ang_end
    

sim = Simulator(3, Physics('b'), 0.004)
set_up_positions(sim)

esc = GraphPosVel("dummy", elas, chain(torch.linspace(0, sim.N - 1, sim.N, dtype=torch.int64), 2))
sim.add_physics("Elastic-Chain", esc)
stf = GraphPosVel("dummy", stiff, chain(torch.linspace(0, sim.N - 1, sim.N, dtype=torch.int64), 3))
sim.add_physics("Stiffness", stf)
bowl = Acceleration_General(
    lambda sim : static_bowl_accel(torch.zeros(3), bowl_r, bowl_k, sim.pos) + static_bowl_loss(torch.zeros(3), bowl_r, bowl_eta, sim.pos, sim.vel)
)
sim.add_physics("Bowl", bowl)
fric = Acceleration_General(fric_force)
sim.add_physics("Friction", fric)
grav = AccelerationField(grav_force)
sim.add_physics("Gravity", grav)
presc = PrescribedPos("presc", "dummy", presc_fun)
sim.add_physics("Prescribed-Displacement", presc)

sim.set_property_all("rad", torch.zeros(sim.N) + 0.2)
sim.set_property_all("mass", torch.zeros(sim.N) + 0.1)
sim.set_property_all("dummy", torch.zeros(sim.N))
sim.set_property_multi("presc", sim.N - 20, sim.N, torch.zeros(20) + 1)

sim.add_3dof_record(
    "Max Height",
    lambda sim : float(sim.n_step),
    lambda sim : float(torch.max(sim.pos[2]).cpu()),
    lambda sim : 0.0
)

def post_update(sr : Spherical_Render):
    # Find the first ball that has fallen below z_low,
    #   and remove all the balls whose index is less than that ball
    pos      : torch.Tensor = sr.simulator.pos
    outs     : torch.Tensor = torch.nonzero(pos[2] < z_low).flatten()
    n_delete = 0
    if outs.nelement() != 0:
        first_out = int(torch.max(outs).cpu())
        n_delete = first_out + 1
        sr.simulator.object_delete(torch.linspace(0, n_delete - 1, n_delete, dtype=torch.int64))
        add_more_to_ci4(sr.simulator, n_delete)
    return None

# OpenGL imports for python
try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
    from OpenGL.GLUT import *
except:
    print("OpenGL wrapper for python not found")

import renderer.two_d as twoD

def extra_display(sr : Spherical_Render):
    if sr.display_mode == 0:
        lx, ly, lz = sr.lookat
        glTranslatef(-lx, -ly, -lz)

        positions : np.ndarray = sr.simulator.pos.cpu().numpy()

        cylinder_base = positions[:, :-1]
        cylinder_norm = positions[:, 1:] - cylinder_base
        del positions
        dim, cnt = cylinder_norm.shape
        cnorm_norm = np.sum(cylinder_norm * cylinder_norm, 0) ** 0.5
        cylinder_norm = cylinder_norm / cnorm_norm
        cylinder_rot_axis = np.broadcast_to(np.array([0, 0, 1]), (cnt, dim)).transpose() + cylinder_norm

        for i in range(cnt):
            x, y, z = cylinder_base[:, i]
            rx, ry, rz = cylinder_rot_axis[:, i]
            h = cnorm_norm[i]
            x = float(x); y = float(y); z = float(z)
            rx = float(rx); ry = float(ry); rz = float(rz)
            h = float(h)
            glPushMatrix()
            glTranslatef(x, y, z)
            glRotatef(180, rx, ry, rz)
            gluCylinder(gluNewQuadric(), cylin_rad, cylin_rad, h, 5, 5)
            glPopMatrix()

        glPushMatrix()
        # Set color to white
        glColor3f(0.0, 0.0, 0.0)
        glPopMatrix()

        glTranslatef(lx, ly, lz)
    
    if sr.display_mode == 3:
        cnt = sr.simulator.pos.shape[1] - 1
        data = np.zeros((2, cnt))
        data[0] = np.linspace(0, cnt - 1, cnt)
        pos = sr.simulator.pos
        vel = sr.simulator.vel
        dist = ((pos[:, 1:] - pos[:, :-1]) ** 2).sum(0) ** 0.5

        dat1 = ((dist - stick_len) * 8).cpu().numpy()
        dat2 = (vel[0, :-1] / 2).cpu().numpy()
        dat3 = (vel[1, :-1] / 2).cpu().numpy()
        dat4 = (vel[2, :-1] / 2).cpu().numpy()

        if not hasattr(sr, "bar_rec_n"):
            sr.bar_rec_n = 1

        if not hasattr(sr, "dat1_avg"):
            sr.dat1_avg = dat1
        elif not sr.paused:
            sr.dat1_avg = dat1 * (1 / sr.bar_rec_n) + sr.dat1_avg * (1 - 1 / sr.bar_rec_n)
        if not hasattr(sr, "dat2_avg"):
            sr.dat2_avg = dat2
        elif not sr.paused:
            sr.dat2_avg = dat2 * (1 / sr.bar_rec_n) + sr.dat2_avg * (1 - 1 / sr.bar_rec_n)
        if not hasattr(sr, "dat3_avg"):
            sr.dat3_avg = dat3
        elif not sr.paused:
            sr.dat3_avg = dat3 * (1 / sr.bar_rec_n) + sr.dat3_avg * (1 - 1 / sr.bar_rec_n)
        if not hasattr(sr, "dat4_avg"):
            sr.dat4_avg = dat4
        elif not sr.paused:
            sr.dat4_avg = dat4 * (1 / sr.bar_rec_n) + sr.dat4_avg * (1 - 1 / sr.bar_rec_n)
        sr.bar_rec_n += 1

        data[1] = dat1
        colors = np.zeros((cnt, 3))
        colors[:, 0] = dat2
        colors[:, 1] = dat3
        colors[:, 2] = dat4
        twoD.barplot(data, colors, 0.5, 0.65, 0.7, 0.3, y_adapt=False)

        # Averge velocity of moving balls
        vel_sqs = (vel * vel).sum(0)
        vel_mv_avg = (vel_sqs * (vel_sqs > 1)).sum() / (vel_sqs > 1).sum()
        # By energy conservation, v^2 = 2gL
        vel_coef = float((vel_mv_avg / (2 * grav_g * (-z_low - 2))).cpu())
        twoD.renderText(str(vel_coef), 0.4, 0.3, 2, 0.2)

spr = Spherical_Render(sim, 1, 8, 4, sim_render_precision=8)
spr.post_update = post_update
spr.extra_display = extra_display
spr.render()