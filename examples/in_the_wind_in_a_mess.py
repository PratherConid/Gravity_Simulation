# 在风中凌乱

import sys
sys.path.append("../Gravity_Simulation")

import numpy as np
from physics.main import *
from renderer.sphereical_renderer import Spherical_Render
from mathematics.graph_config import chain

stick_len = 0.5
stick_k = 80
stick_stiff_k = 80
x_num = 30
y_num = 50
grav_g = 0.3

# Elastic force of chain
def elas(post : torch.Tensor, velt : torch.Tensor, propt : torch.Tensor):
    pos1, pos2 = post
    dist = ((pos2 - pos1) ** 2).sum(0) ** 0.5
    dist_s = dist + 10 * stick_len * (dist < 0.01)
    f1 = stick_k * (pos2 - pos1) * (dist - stick_len) / dist_s
    f2 = stick_k * (pos1 - pos2) * (dist - stick_len) / dist_s
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
sim.set_property_single("rad", (x_num + 1) * (y_num + 1), 0.6)
sim.set_property_single("mass", (x_num + 1) * (y_num + 1), 0.4)
sim.set_property_multi("surf", 0, (x_num + 1) * (y_num + 1), torch.zeros((x_num + 1) * (y_num + 1)) + 1)
sim.set_property_single("sphere", (x_num + 1) * (y_num + 1), 1)
sim.set_property_multi("presc", 0, x_num + 1, torch.zeros(x_num + 1) + 1)
sim.set_property_multi("presc", (x_num + 1) * y_num, (x_num + 1) * (y_num + 1), torch.zeros(x_num + 1) + 1)
sim.pr["presc"][torch.linspace(0, (x_num + 1) * y_num, y_num + 1, dtype=torch.int64)] = 1
sim.pr["presc"][torch.linspace(x_num, (x_num + 1) * y_num + x_num, y_num + 1, dtype=torch.int64)] = 1

spr = Spherical_Render(sim, 1, 8, 4, sim_render_precision=8)
spr.render()