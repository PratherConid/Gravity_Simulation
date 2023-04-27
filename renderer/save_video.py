try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
    from OpenGL.GLUT import *
except:
    print("OpenGL wrapper for python not found")

import numpy as np
import cv2

# Create a MJPG videowriter in openGL environment
def create_videowriter(videopath):
    width = glutGet(GLUT_WINDOW_WIDTH)
    height = glutGet(GLUT_WINDOW_HEIGHT)
    return cv2.VideoWriter(videopath, cv2.VideoWriter_fourcc('M','J','P','G'), 25, (width, height))

# Save the current scene to a MJPG videowriter
def save_scene_to_videowriter(videowriter):
    width = glutGet(GLUT_WINDOW_WIDTH)
    height = glutGet(GLUT_WINDOW_HEIGHT)
    screenshot_byte = glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE)
    screenshot_tr = np.frombuffer(screenshot_byte, np.uint8).reshape((width, height, 4))
    screenshot = np.flip(screenshot_tr, 0)
    videowriter.write(screenshot)