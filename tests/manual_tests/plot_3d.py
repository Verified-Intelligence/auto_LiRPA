import numpy as np
from mayavi import mlab
from auto_LiRPA.bound_ops import BoundMul

xl0=-1.
xu0=1.
yl0=-1.
yu0=1.
alpha_l, beta_l, gamma_l, alpha_u, beta_u, gamma_u = BoundMul.get_bound_mul(xl0, xu0, yl0, yu0)

def f(x, y):
    return x * y

def fu(x, y):
    return alpha_l * x + beta_l * y + gamma_l

def fl(x, y):
    return alpha_u * x + beta_u * y + gamma_u

range_min = -1.
range_max = 1.
# Draw 4 points for a rectangular area
mlab.points3d(xl0, yl0, 0, scale_factor=0.2)
mlab.points3d(xu0, yu0, 0, scale_factor=0.2)
mlab.points3d(xl0, yu0, 0, scale_factor=0.2)
mlab.points3d(xu0, yl0, 0, scale_factor=0.2)
x, y = np.mgrid[range_min:range_max:0.01, range_min:range_max:0.01]
# Multiplication function
mlab.surf(x, y, f, color=(0.0, 0.5, 0.))
# Upper bound.
mlab.surf(x, y, fu, color=(0.0, 0.5, 0.5), opacity=0.5)
# # Lower bound
mlab.surf(x, y, fl, color=(0.5, 0.0, 0.5), opacity=0.5)
# z=0 plane
# mlab.surf(x, y, lambda x, y: 0 * x, color=(0.0, 0.0, 0.5), opacity=0.2)
# 4 lines for the rectangular area
mlab.plot3d([xl0, xl0], [yl0, yl0], [2 * range_min, 2 * range_max], tube_radius=0.025, tube_sides=8)
mlab.plot3d([xu0, xu0], [yl0, yl0], [2 * range_min, 2 * range_max], tube_radius=0.025, tube_sides=8)
mlab.plot3d([xu0, xu0], [yu0, yu0], [2 * range_min, 2 * range_max], tube_radius=0.025, tube_sides=8)
mlab.plot3d([xl0, xl0], [yu0, yu0], [2 * range_min, 2 * range_max], tube_radius=0.025, tube_sides=8)
mlab.show()

