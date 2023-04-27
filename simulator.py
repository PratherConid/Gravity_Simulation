import torch
import physics.main as main
import mathematics.space_config as space_config

class Simulator:

    def __init__(self, dim, phys_base, dt):
        self.dim = dim
        self.N = 0
        # (3 * N) array
        self.pos = None
        self.vel = None
        # Other properties, dict of (N, shape) or (N,) numpy array
        #   Note that this is different from `pos` and `vel`
        self.pr = {}
        # Physics
        self.physics = {}
        # Basic physics engine, supports
        #   Total momentum
        #   Total angular momentum
        #   Total kinetic energy
        self.phys_base = phys_base
        self.dt = dt
        self.n_step = 0
        # Fields:
        #   name : str
        #   funcs : N list
        #   values : 3 * N) list
        self._3dof_record = []
        # Fields:
        #   name : str
        #   funcs : 6 list
        #           6 : xf, yf, zf, tf, center of slice, direction of slice
        #               "slice" is the 3d slice to the 4d space
        #   values : 4 * N list
        self._4dof_record = []
    
    # p : array of size (dim, n_pts) or (dim,)
    def set_pos(self, id, p : torch.Tensor):
        assert p.shape[0] == self.dim
        if len(p.shape) == 1:
            self.pos[:, id] = p
        elif len(p.shape) == 2:
            self.pos[:, id : id + p.shape[-1]] = p
        else:
            print("set_pos :: invalid dimension of p")

    # v : array of size (dim, n_pts) or (dim,)
    def set_vel(self, id, v : torch.Tensor):
        assert v.shape[0] == self.dim
        if len(v.shape) == 1:
            self.vel[:, id] = v
        elif len(v.shape) == 2:
            self.vel[:, id : id + v.shape[-1]] = v
        else:
            print("set_vel :: invalid dimension of v")
    
    # Set property `pname` of particle `id`
    def set_property_single(self, pname, id, value):
        if not (pname in self.pr):
            self.pr[pname] = torch.zeros(self.N).cuda()
        self.pr[pname][id] = value

    # Set property `pname` of particle `id_begin:id_end`
    def set_property_multi(self, pname, id_begin, id_end, values):
        if not (pname in self.pr):
            self.pr[pname] = torch.zeros(self.N).cuda()
        self.pr[pname][id_begin:id_end] = values

    # Set property `pname` of all particles
    def set_property_all(self, pname, values):
        if not (pname in self.pr):
            self.pr[pname] = torch.zeros(self.N).cuda()
        self.pr[pname][0:self.N] = values

    def add_physics(self, name, physics):
        self.physics[name] = physics

    def add_3dof_record(self, name, xf, yf, zf):
        self._3dof_record.append({"name" : name,
                                  "funcs" : [xf, yf, zf],
                                  "values" : [[], [], []]})

    def add_4dof_record(self, name, xf, yf, zf, tf, center, dir):
        self._4dof_record_funs.append({"name" : name,
                                       "funcs" : [xf, yf, zf, tf, center, dir],
                                       "values" : [[], [], [], []]})

    # Append objects to the end, with prescribed initial position.
    def object_append(self, new_obj_pos : torch.Tensor):
        l = new_obj_pos.shape[-1]
        self.N += l
        if self.pos is None:
            self.pos = new_obj_pos.cuda()
            self.vel = torch.zeros((self.dim, l)).cuda()
            for key in self.pr:
                self.pr[key] = torch.zeros(l).cuda()
        else:
            self.pos = torch.concat([self.pos, new_obj_pos.cuda()], -1)
            self.vel = torch.concat([self.vel, torch.zeros((self.dim, l)).cuda()], -1)
            for key in self.pr:
                shape = self.pr[key].shape[:-1]
                self.pr[key] = torch.concat([self.pr[key], torch.zeros((l,) + shape).cuda()])

    # Delete objects
    def object_delete(self, dels : torch.Tensor):
        remains  : torch.Tensor = torch.zeros(self.N).cuda() + 1
        remains[dels] = 0
        remain_id: torch.Tensor = torch.nonzero(remains).flatten()
        self.N = remain_id.shape[0]
        self.pos = self.pos[:, remain_id]
        self.vel = self.vel[:, remain_id]
        for key in self.pr:
            self.pr[key] = self.pr[key][remain_id]

    def static_cube(self, k, slen, center):
        zr = space_config.cubic_array(self.dim, k, slen)
        self.object_append(space_config.translate(zr, center))
    
    def static_rectangle(self, w, h):
        assert self.dim >= 3
        pos = torch.zeros((self.dim, 4)).cuda()
        pos[0][0] = pos[0][2] = - w / 2
        pos[0][1] = pos[0][3] = w / 2
        pos[2][0] = pos[2][1] = -h / 2
        pos[2][2] = pos[2][3] = h / 2
        self.object_append(pos)
    
    def static_triangle(self, w, h):
        assert self.dim >= 3
        pos = torch.zeros((self.dim, 3)).cuda()
        pos[0][0] = -w / 2
        pos[0][1] = w / 2
        pos[2][0] = pos[2][1] = -h / 3
        pos[2][2] = 2 * h / 3
        self.object_append(pos)

    def uniform_random_pos_box(self, xyzlow, xyzhigh):
        xyzlow = xyzlow.cuda()
        xyzhigh = xyzhigh.cuda()
        self.pos = (xyzlow + torch.rand((self.N, self.dim)).cuda() * (xyzhigh - xyzlow)).transpose(0, 1)
    
    def uniform_random_vel_box(self, xyzlow, xyzhigh):
        xyzlow = xyzlow.cuda()
        xyzhigh = xyzhigh.cuda()
        self.vel = (xyzlow + torch.rand((self.N, self.dim)).cuda() * (xyzhigh - xyzlow)).transpose(0, 1)

    def uniform_random_property(self, name_of_property, low, high):
        self.set_property_multi(name_of_property, 0, self.N, low + torch.rand(self.N) * (high - low))
    
    # omega  : omega_x, omega_y, omega_z
    # center : x, y, z
    def add_3d_rotation(self, center : torch.Tensor, omega : torch.Tensor):
        if self.dim < 3:
            print("add_rotation : Dimension must be greater or equal to 3!")
            return
        self.vel[:3] += main.cross_prod(omega, self.pos[:3] - torch.broadcast_to(center, (self.N, 3)).transpose(0, 1))

    def add_3d_rotation_EkFrac(self, sign, frac):
        thdaxis = torch.tensor([0, 0, 1])
        axis = torch.zeros(self.dim)
        axis[:3] = thdaxis
        mom_inertia = self.phys_base.I(self, axis, torch.zeros(self.dim))
        ek = self.phys_base.Ek(self)
        mag_omega = (ek * frac * 2 / mom_inertia) ** 0.5
        center_of_mass = self.phys_base.xbar(self)
        self.add_3d_rotation(center_of_mass[:3], sign * mag_omega * thdaxis)

    # multiply kinetic energy by factor 'fac'
    def simple_deterministic_heating(self, fac):
        average_velocity = self.phys_base.vbar(self)
        avg_v_br = torch.broadcast_to(average_velocity, (self.N, self.dim)).transpose(0, 1)
        self.vel += (self.vel - avg_v_br) * (fac ** 0.5 - 1)

    # Sympletic Euler
    def update(self):
        if self.N == 0:
            return
        self.n_step += 1
        pos_last = self.pos.clone() # Save position for physics of type `p`

        # Update 'v' type sequentially. This is because
        #   these physics usually models collisions, and
        #   the dv of the collisions should be added sequentially
        for phy in self.physics.values():
            if phy.type == 'v':
                self.vel += phy.dv(self)
        # Note that it is important to calculate all `dv`s
        #   before we update the velocity, because this is
        #   the correct way to do sympletic Euler. Otherwise,
        #   energy will not conserve!
        dvs = [phy.dv(self) for phy in self.physics.values() if phy.type == 'a']
        for dv in dvs:
            if dv is not None:
                self.vel += dv
        self.pos += self.vel * self.dt
        dxs = [phy.dx(self) for phy in self.physics.values() if phy.type == 'x']
        for dx in dxs:
            if dx is not None:
                self.pos += dx
        # Prescribed position
        prescs = [phy for phy in self.physics.values() if phy.type == 'pp']
        for presc in prescs:
            presc.set_x(self, pos_last)

    def record(self):
        cnt = 0
        for rec_3 in self._3dof_record:
            xf, yf, zf = rec_3["funcs"]
            arr = rec_3["values"]
            arr[0].append(float(xf(self)))
            arr[1].append(float(yf(self)))
            arr[2].append(float(zf(self)))
            cnt += 1
        cnt = 0
        for rec_4 in self._4dof_record:
            xf, yf, zf, tf, _, _ = rec_4["funcs"]
            arr = rec_4["values"]
            arr[0].append(float(xf(self)))
            arr[1].append(float(yf(self)))
            arr[2].append(float(zf(self)))
            arr[3].append(float(tf(self)))
            cnt += 1