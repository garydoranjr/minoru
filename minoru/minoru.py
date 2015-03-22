"""
Wraper for C++ Minoru interface
"""
import numpy as np

from c_minoru import c_capture

def capture(left_dev, right_dev, width=640, height=480, fps=15):
    return c_capture(left_dev, right_dev, width, height, fps)
