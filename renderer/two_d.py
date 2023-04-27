# OpenGL imports for python
try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
    from OpenGL.GLUT import *
except:
    print("OpenGL wrapper for python not found")
import numpy as np


# textData should be a W * H * 4 (four channels) array
# Returns the id of the texture
def loadTexture(textData : np.ndarray):
    textID = glGenTextures(1)
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
    glBindTexture(GL_TEXTURE_2D, textID)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, textData.shape[0], textData.shape[1], 0, GL_RGBA, GL_UNSIGNED_BYTE, textData)
    return textID

# Draw a quad at (centerX, cernterY), with size (W, H)
#   centerX ∈ [0, 1], centerY ∈ [0, 1]
#   W ∈ [0, 1], H ∈ [0, 1]
def drawQuad(centerX, centerY, W, H):
    glPushMatrix()
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(0, 1, 0, 1, -1000.0, 1000.0)
    verts = ((1, 1), (1,-1), (-1,-1), (-1,1))
    texts = ((1, 0), (1, 1), (0, 1), (0, 0))
    surf = (0, 1, 2, 3)

    glBegin(GL_QUADS)
    for i in surf:
        glTexCoord2f(texts[i][0], texts[i][1])
        glVertex2f(centerX + verts[i][0] * W / 2, centerY + verts[i][1] * H / 2)
    glEnd()

    glPopMatrix()

# Draw a quad at (centerX, cernterY), with size (W, H)
#   centerX ∈ [0, 1], centerY ∈ [0, 1]
#   W ∈ [0, 1], H ∈ [0, 1]
def drawQuadWithTexture(centerX, centerY, W, H, textureID):
    glPushMatrix()
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(0, 1, 0, 1, -1000.0, 1000.0)
    verts = ((1, 1), (1,-1), (-1,-1), (-1,1))
    texts = ((1, 0), (1, 1), (0, 1), (0, 0))
    surf = (0, 1, 2, 3)

    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, textureID)

    glBegin(GL_QUADS)
    for i in surf:
        glTexCoord2f(texts[i][0], texts[i][1])
        glVertex2f(centerX + verts[i][0] * W / 2, centerY + verts[i][1] * H / 2)
    glEnd()
    
    glDisable(GL_TEXTURE_2D)
    glPopMatrix()

# data : (2, N) numpy array
#   data[0] -> xdata
#   data[1] -> ydata
# colorf : (N, 4) numpy array (rgba), or (N, 3) numpy array (rgb), float-point
def barplot(data : np.ndarray, colorf : np.ndarray, centerX, centerY, W, H, y_adapt=True):
    xdata, ydata = data
    xmin = np.min(xdata); xmax = np.max(xdata); xmid = (xmax + xmin) / 2
    xspan = xmax - xmin
    if xspan == 0:
        xspan = 1
    ymin = np.min(ydata); ymax = np.max(ydata); ymid = (ymax + ymin) / 2
    yspan = ymax - ymin
    if yspan == 0:
        yspan = 1
    l_bound = np.copy(xdata)
    l_bound[:-1] = xdata[1:]
    l_bound = ((l_bound + xdata) / 2 - xmid) * (W / xspan)
    r_bound = np.copy(xdata)
    r_bound[1:] = xdata[:-1]
    r_bound = ((r_bound + xdata) / 2 - xmid) * (W / xspan)
    if y_adapt:
        u_bound = (ydata - ymid) * (H / yspan)
    else:
        u_bound = H * ydata - H / 2
    d = float(-H / 2)
    cnt = xdata.size
    for i in range(cnt):
        l = l_bound[i]; r = r_bound[i]; u = u_bound[i]
        l = float(l); r = float(r); u = float(u)
        colors = colorf[i]
        if colors.shape == (3,):
            cr, cg, cb = colors
            cr = float(cr); cg = float(cg); cb = float(cb)
            glColor3f(cr, cg, cb)
        elif colors.shape ==(4,):
            cr, cg, cb, ca = colors
            cr = float(cr); cg = float(cg); cb = float(cb); ca = float(ca)
            glColor4f(cr, cg, cb, ca)
        drawQuad((l + r) / 2 + centerX, (u + d) / 2 + centerY, r - l, u - d)

# x ∈ [0, 1], y ∈ [0, 1]
def renderText(string : str, x, y, line_width, size):
    width = glutGet(GLUT_WINDOW_WIDTH)
    height = glutGet(GLUT_WINDOW_HEIGHT)
    glPushMatrix()
    glMatrixMode(GL_PROJECTION)
    glClear(GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    glOrtho(0, width, 0, height, -1, 1)
    glLineWidth(line_width)
    glTranslatef(int(x * width), int(y * height), 0)
    glScalef(size, size, size)
    for ch in string:
        glTranslatef(int(40 * size), 0, 0)
        glutStrokeCharacter(GLUT_STROKE_ROMAN, ctypes.c_int(ord(ch)))
    glPopMatrix()