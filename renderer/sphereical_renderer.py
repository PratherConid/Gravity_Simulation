from simulator import Simulator
from mathematics.base import get_3d_slice_of_4d_line

# Controls: UP/DOWN - turn up/down
#           LEFT/RIGHT - rotate left/right
#           PAGEUP/PAGEDOWN - larger/smaller
#           F1 - Toggle input_mode
#           F2 - Toggle surface as SMOOTH or FLAT
#           F3 - Move 4d slice
#           F5 - Switch display mode
#           F6 - Switch record number to be plotted
#           F9 - Move along axis

#           Space       - Pause/Resume
#           rl <num>    - Add left rotation
#           rr <num>    - Add right rotation
#           h <num>     - Heat up / Cool down (multiply kinetic energy by <num>)
#           d <obj>     - toggle display <obj>
#             d text
#             d axes
#           ma <num>    - Set move along axis to <num>
#           cld         - Set moving direction to +1
#           std         - Set moving direction to -
#           ccr         - Clear current record
#           lo          - Camera look at origin
#           l <num> <num> <num> - Camera look at a specific point in space
#           exec <code> - execute python code

# Python imports
from math import *

# OpenGL imports for python
try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
    from OpenGL.GLUT import *
except:
    print("OpenGL wrapper for python not found")

from renderer.two_d import renderText

# Last time when sphere was re-displayed
last_time = 0


class Spherical_Render:

    # Constructor for the sphere class
    def __init__(self, simulator : Simulator, rad_discount, step_merge = 4, _3d_record_merge = 20, sim_render_precision = 5):

        self.user_phi = pi
        self.user_theta = pi
        self.user_scale = 2.0
        self.rad_discount = rad_discount
        self.paused = True

        # Direction of light
        self.direction = [1.0, 1.0, 1.0, 0.0]

        # Intensity of light
        self.intensity = [0.7, 0.7, 0.7, 1.0]

        # Point to look at
        self.lookat = [0, 0, 0]
        self.move_dir = 1
        self.move_axis = 0

        # Record number to be plotted
        self._3d_rec_n = 0
        self._4d_rec_n = 0
        # 0 -> displaying simulation
        # 1 -> displaying 3d scatter
        # 2 -> displaying 4d scatter
        # 3 -> user defined
        self.display_mode = 0

        # Intensity of ambient light
        self.ambient_intensity = [0.3, 0.3, 0.3, 1.0]

        # The surface type(Flat or Smooth)
        self.surface = GL_SMOOTH

        # Simulator
        self.simulator = simulator
        assert self.simulator.dim >= 3

        self.step_merge = step_merge
        # Does not merge 4d record because there are too few slices
        self._3d_record_merge = _3d_record_merge

        self.axes_size = 0.005

        # Simulation Rendering Precision
        self.sim_render_precision = sim_render_precision

        # Keyboard input facility
        self.input_buffer = ""
        self.last_enter = False
        self.input_mode = False

        # Whether to display object
        self.display_axes_on = True
        self.display_text_on = True
        self.display_spheres_on = True

        # Extra operations to execute during display
        # This should be experiment-specific
        # Please refer to display(self)
        self.pre_display = None
        self.extra_display = None
        self.post_display = None

        # Extra operations to execute before/after update
        self.pre_update = None
        self.post_update = None

    # Initialize
    def init(self):

        # Initialize the OpenGL pipeline
        # glutInit(sys.argv)
        glutInit()

        # Set OpenGL display mode
        glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH)

        # Set the Window size and position
        glutInitWindowSize(900, 900)
        glutInitWindowPosition(50, 100)

        # Create the window with given title
        glutCreateWindow('Sphere')

        # Set background color to grey
        glClearColor(0.6, 0.7, 0.7, 0.0)

        # Set camera
        self.compute_location()

        # Set OpenGL parameters
        glEnable(GL_DEPTH_TEST)

        # Enable lighting
        glEnable(GL_LIGHTING)

        # Set light model
        glLightModelfv(GL_LIGHT_MODEL_AMBIENT, self.ambient_intensity)

        # Enable light number 0
        glEnable(GL_LIGHT0)

        # Set position and intensity of light
        glLightfv(GL_LIGHT0, GL_POSITION, self.direction)
        glLightfv(GL_LIGHT0, GL_DIFFUSE, self.intensity)

        # Setup the material
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE)

    # Compute location
    def compute_location(self):
        x = cos(self.user_phi) * cos(self.user_theta) * self.user_scale
        y = cos(self.user_phi) * sin(self.user_theta) * self.user_scale
        z = sin(self.user_phi) * self.user_scale
        d = sqrt(x * x + y * y + z * z)

        # Set matrix mode
        glMatrixMode(GL_PROJECTION)

        # Reset matrix
        glLoadIdentity()
        glFrustum(-d * 0.5, d * 0.5, -d * 0.5, d * 0.5, d * 0.5, d * 2.5)

        # Set camera
        gluLookAt(x, y, z, 0, 0, 0, 0, 0, 1)

    def display_axes(self):
        lx, ly, lz = self.lookat
        glTranslatef(-lx, -ly, -lz)

        scale = self.user_scale

        # X axis, Set color to red
        glColor3f(1.0, 0.0, 0.0)
        glPushMatrix()
        glTranslatef(-2 * scale, 0, 0)
        glRotatef(90.0, 0.0, 1.0, 0.0)
        gluCylinder(gluNewQuadric(), self.axes_size * scale, self.axes_size * scale, 4 * scale, 32, 32)
        glTranslatef(0, 0, 2.6 * scale)
        glutSolidCone(0.02 * scale, 0.04 * scale, 10, 10)
        glPopMatrix()

        # Y axis, Set color to green
        glColor3f(0.0, 1.0, 0.0)
        glPushMatrix()
        glTranslatef(0, 2 * scale, 0)
        glRotatef(90.0, 1.0, 0.0, 0.0)
        gluCylinder(gluNewQuadric(), self.axes_size * scale, self.axes_size * scale, 4 * scale, 32, 32)
        glTranslatef(0, 0, 1.4 * scale)
        glutSolidCone(0.02 * scale, -0.04 * scale, 10, 10)
        glPopMatrix()

        # Z axis, Set color to blue
        glColor3f(0.0, 0.0, 1.0)
        glPushMatrix()
        glTranslatef(0, 0, -2 * scale)
        glRotatef(90.0, 0.0, 0.0, 1.0)
        gluCylinder(gluNewQuadric(), self.axes_size * scale, self.axes_size * scale, 4 * scale, 32, 32)
        glTranslatef(0, 0, 2.6 * scale)
        glutSolidCone(0.02 * scale, 0.04 * scale, 10, 10)
        glPopMatrix()

        glTranslatef(lx, ly, lz)

    def display_text(self):
        width = glutGet(GLUT_WINDOW_WIDTH)
        height = glutGet(GLUT_WINDOW_HEIGHT)

        # Input Buffer
        glColor3f(0, 0, 0)
        renderText("Input Buffer: " + self.input_buffer, 0.05, 0.95, 2, 0.2)

    def display_spheres(self):
        # Begin plotting spheres
        lx, ly, lz = self.lookat
        glTranslatef(-lx, -ly, -lz)

        if self.display_mode == 1 or self.display_mode == 2:
            # Set color
            glColor3f(1, 0.6, 0)
            # Gather data
            has_data = False
            if self.display_mode == 1 and len(self.simulator._3dof_record) != 0:
                r = self.simulator._3dof_record[self._3d_rec_n]["values"]
                sx, sy, sz = [i[::self._3d_record_merge] for i in r]
                has_data = True
                if len(sx) != 0:
                    xmax, xmin = max(sx), min(sx)
                    ymax, ymin = max(sy), min(sy)
                    zmax, zmin = max(sz), min(sz)
            elif self.display_mode == 2 and len(self.simulator._4dof_record) != 0:
                fx, fy, fz, ft = self.simulator._4dof_record[self._4d_rec_n]["values"]
                _, _, _, _, center, dir = self.simulator._4dof_record_funs[self._4d_rec_n]
                sx, sy, sz, _ = get_3d_slice_of_4d_line(fx, fy, fz, ft, center, dir)
                has_data = True
                if len(fx) != 0:
                    xmax, xmin = max(fx), min(fx)
                    ymax, ymin = max(fy), min(fy)
                    zmax, zmin = max(fz), min(fz)
            # Begin scatter plot
            if has_data and len(sx) != 0:
                xmid = (xmax + xmin) / 2
                ymid = (ymax + ymin) / 2
                zmid = (zmax + zmin) / 2
                for (x, y, z) in zip(sx, sy, sz):
                    x = float((x - xmid) / (xmax - xmin + 2 ** (-500)))
                    y = float((y - ymid) / (ymax - ymin + 2 ** (-500)))
                    z = float((z - zmid) / (zmax - zmin + 2 ** (-500)))
                    glPushMatrix()
                    glTranslatef(x, y, z)
                    gluSphere(gluNewQuadric(), 0.01, 4, 2)
                    glPopMatrix()
        elif self.display_mode == 0:
            # Set color to white
            glColor3f(1.0, 1.0, 1.0)
            # Begin displaying spheres
            positions = self.simulator.pos.cpu().numpy()
            radiuses = self.simulator.pr["rad"].cpu().numpy()
            for i in range(self.simulator.N):
                x, y, z = positions[:3, i]
                x = float(x); y = float(y); z = float(z)
                r = float(radiuses[i] * self.rad_discount)
                glPushMatrix()
                glTranslatef(x, y, z)
                gluSphere(gluNewQuadric(), r, 2 * self.sim_render_precision, self.sim_render_precision)
                glPopMatrix()

        glTranslatef(lx, ly, lz)
        # End Plotting Spheres

    # Display the simulation
    def display(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Set shade mode
        glShadeModel(self.surface)

        if self.pre_display is not None:
            self.pre_display(self)

        # Display axes
        if self.display_axes_on:
            self.display_axes()

        # Display spheres
        if self.display_spheres_on:
            self.display_spheres()

        if self.extra_display is not None:
            self.extra_display(self)

        # Display text
        if self.display_text_on:
            self.display_text()
        
        if self.post_display is not None:
            self.post_display(self)

        glutSwapBuffers()

        # Update simulation
        if not self.paused:
            for _ in range(self.step_merge):
                if self.pre_update is not None:
                    self.pre_update(self)
                self.simulator.update()
                if self.post_update is not None:
                    self.post_update(self)
            print(self.simulator.n_step)
            self.simulator.record()

    def parse_input_buffer(self):
        control_seq = self.input_buffer.split(" ")
        try:
            if control_seq[0] == "h":
                self.simulator.simple_deterministic_heating(float(control_seq[1]))
            elif control_seq[0] == 'rl':
                self.simulator.add_3d_rotation_EkFrac(-1, float(control_seq[1]))
            elif control_seq[0] == 'rr':
                self.simulator.add_3d_rotation_EkFrac(1, float(control_seq[1]))
            elif control_seq[0] == 'd':
                if control_seq[1] == 'text':
                    self.display_text_on = not self.display_text_on
                if control_seq[1] == 'axes':
                    self.display_axes_on = not self.display_axes_on
                if control_seq[1] == 'spheres':
                    self.display_spheres_on = not self.display_spheres_on
            elif control_seq[0] == 'ma':
                ax = int(control_seq[1])
                assert ax >= 0 and ax < 3
                self.move_axis = ax
            elif control_seq[0] == 'std':
                self.move_dir = -1
            elif control_seq[0] == 'cld':
                self.move_dir = 1
            elif control_seq[0] == 'ccr':
                if self.display_mode == 1 and len(self.simulator._3dof_record) != 0:
                    self.simulator._3dof_record[self._3d_rec_n]["values"] = [[], [], []]
                if self.display_mode == 2 and len(self.simulator._4dof_record) != 0:
                    self.simulator._4dof_record[self._4d_rec_n]["values"] = [[], [], [], []]
            elif control_seq[0] == 'l':
                self.lookat = [float(control_seq[i]) for i in range(1, 4)]
            elif control_seq[0] == 'lo':
                self.lookat = [0, 0, 0]
            elif control_seq[0] == 'exec':
                print("executing", self.input_buffer[5:])
                exec(self.input_buffer[5:])
            else:
                pass
        except:
            print("Error executing input buffer " + self.input_buffer)
            self.input_buffer = ""
            self.last_enter = False

    def keyboard(self, ch, x, y):
        # enter
        o = ord(ch)
        if o == 13:
            self.parse_input_buffer()
            self.last_enter = True
        # escape
        elif o == 27:
            if not self.last_enter:
                self.input_buffer = ""
        # backspace
        elif o == 8:
            if not self.last_enter:
                self.input_buffer = self.input_buffer[:-1]
        # Speed up simulation
        elif o == ord('+') and not self.input_mode:
            self.simulator.dt /= 0.95
        # Slow down simulation
        elif o == ord('-') and not self.input_mode:
            self.simulator.dt *= 0.95
        # Pause/Resume
        elif o == ord(' ') and not self.input_mode:
            if self.paused:
                self.paused = False
            else:
                self.paused = True
        else:
            if self.last_enter:
                self.input_buffer = ""
                self.last_enter = False
            self.input_buffer += str(chr(o))

    # Keyboard controller for sphere
    def special(self, key, x, y):

        # Turn the sphere up or down
        if key == GLUT_KEY_UP:
            self.user_phi -= 0.05
        if key == GLUT_KEY_DOWN:
            self.user_phi += 0.05

        # Rotate the cube
        if key == GLUT_KEY_LEFT:
            self.user_theta -= 0.1
        if key == GLUT_KEY_RIGHT:
            self.user_theta += 0.1

        # Scale the cube
        if key == GLUT_KEY_PAGE_UP:
            self.user_scale /= 0.9
        if key == GLUT_KEY_PAGE_DOWN:
            self.user_scale *= 0.9

        # Toggle input_mode
        if key == GLUT_KEY_F1:
            self.input_mode = not self.input_mode

        # Toggle the surface
        if key == GLUT_KEY_F2:
            if self.surface == GL_FLAT:
                self.surface = GL_SMOOTH
            else:
                self.surface = GL_FLAT
        
        # Move 4d slice
        if key == GLUT_KEY_F3:
            if self.display_mode == 2 and len(self.simulator._4dof_record_funs) != 0:
                recf = self.simulator._4dof_record_funs[self._4d_rec_n]
                _, _, _, _, _, dir = recf
                # move `center` along `dir`
                recf[4] = recf[4] + 0.003 * self.move_dir * dir / ((dir * dir).sum() ** 0.5)

        # Switch display mode
        if key == GLUT_KEY_F5:
            self.display_mode = (self.display_mode + 1) % 4

        # Switch record to be displayed
        if key == GLUT_KEY_F6:
            if self.display_mode == 1:
                l = len(self.simulator._3dof_record)
                self._3d_rec_n = (self._3d_rec_n + 1) % l
            if self.display_mode == 2:
                l = len(self.simulator._4dof_record)
                self._3d_rec_n = (self._4d_rec_n + 1) % l

        # Move lookat
        if key == GLUT_KEY_F9:
            self.lookat[self.move_axis] += 0.05 * self.move_dir

        self.compute_location()
        glutPostRedisplay()

    # The idle callback
    # Which will be used to update the simulator
    def idle(self):
        global last_time
        time = glutGet(GLUT_ELAPSED_TIME)

        if last_time == 0 or time >= last_time + 40:
            last_time = time
            glutPostRedisplay()

    # The visibility callback
    def visible(self, vis):
        if vis == GLUT_VISIBLE:
            glutIdleFunc(self.idle)
        else:
            glutIdleFunc(None)
    
    # The main function
    def render(self):

        self.init()
        # Set the callback function for display
        glutDisplayFunc(self.display)

        # Set the callback function for the visibility
        glutVisibilityFunc(self.visible)

        # Set the callback for special function
        glutSpecialFunc(self.special)

        glutKeyboardFunc(self.keyboard)

        # Run the OpenGL main loop
        glutMainLoop()