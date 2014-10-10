from mayavi import mlab
from mayavi.mlab import surf
import numpy as np
from numpy import random

@mlab.show

def test_surf():
    """Test surf on regularly spaced co-ordinates like MayaVi."""
    def f(x, y):
        sin, cos = np.sin, np.cos
        return sin(x + y) + sin(2 * x - y) + cos(3 * x + 4 * y)

    x, y = np.mgrid[-7.:7.05:0.1, -5.:5.05:0.05]
    s = surf(x, y, f)
    #cs = contour_surf(x, y, f, contour_z=0)
    return s

test_surf()