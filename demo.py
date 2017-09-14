""" Drive script to test polyaffine.py """

from collections import namedtuple
import numpy as np
import transforms3d as t3d
import matplotlib.pyplot as plt

#local imports
import polyaffine as pa
import gridplotting as gp

def Ttranslation(vec_3d):
    T = np.eye(4)
    T[0, 3] = vec_3d[0]
    T[1, 3] = vec_3d[1]
    T[2, 3] = vec_3d[2]
    return T

def Trotation(axis_3d, angle_rad):
    return t3d.axangles.axangle2aff(axis_3d, angle_rad)

# set up the example parameters
def weight(x, c, sigma=5):
    """ 1d weight function """
    return 1/(1 + ((x[0]-c[0])/sigma)**2)

c1 = np.array([-2, 0, 0])
c2 = np.array([2, 0, 0])
theta = 0.63

T1 = Ttranslation(c1).dot( \
     Trotation([0, 0, 1], theta)).dot (\
     Ttranslation(-c1))

T2 = Ttranslation(c2).dot( \
     Trotation([0, 0, 1], -theta)).dot( \
     Ttranslation(-c2))

# chop 3d transforms down to 2d transforms
T1 = np.delete(T1, (2), axis=0)
T1 = np.delete(T1, (2), axis=1)
T2 = np.delete(T2, (2), axis=0)
T2 = np.delete(T2, (2), axis=1)

Ts = np.array([[T1, T2]])
Vs = pa.velocity_fields(Ts)

AnchoredVelocityTransform = namedtuple("AnchoredVelocityTransform", "transform anchor")

avts = [ AnchoredVelocityTransform(Vs[0][0], c1), \
         AnchoredVelocityTransform(Vs[0][1], c2) ]

#display grid
gx = [ x/3 for x in range(-12,13) ]
gy = [ x/3 for x in range(-12,13) ]
grid_nodes = np.transpose([np.tile(gx, len(gy)), np.repeat(gy, len(gx))])

warped_grid_nodes = pa.warp_approx(grid_nodes, avts, weight)

x = warped_grid_nodes[:,0]
y = warped_grid_nodes[:,1]
plt.plot(x, y, ".")

Tinv = np.array([[ np.linalg.inv(T1), np.linalg.inv(T2) ]])
Vinv = pa.velocity_fields(Tinv)
avts_inv = [ AnchoredVelocityTransform(Vinv[0][0], c1), \
             AnchoredVelocityTransform(Vinv[0][1], c2) ]
inverted_warped_grid= pa.warp_approx(warped_grid_nodes, avts_inv, weight)

x_inv = inverted_warped_grid[:,0]
y_inv = inverted_warped_grid[:,1]
plt.plot(x_inv, y_inv, ".")
plt.show()