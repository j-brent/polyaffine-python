""" This module includes functions to generate a polyaffine transformation. """

import numpy as np
from scipy.linalg import logm
from functools import reduce

def velocity_fields(transformations):
    """" Compute the velocity fields from the given 3d rigid transformations """

    shape = transformations.shape
    dim = shape[:2]
    transform_shape = shape[-2:]
    assert transform_shape == (3, 3), ("Tranformations should be 3x3 rigid 2d transformations")

    assert len(dim) == 2, ("")

    V = np.zeros(shape=shape)
    for i, j in np.ndindex(dim):
        V[i][j] = logm(transformations[i][j])

    return V

def warp_approx(x0s, avts, w):
    """  
        x0s: 2D points (N x 2)
        V: the velocity fields for the cells
        c: the centers of the cells
        w: the weight function, of the form w(x, c{i,j})
            where x and c{i,j} are 2D points
    """
    N = 6
    s = 1/2**N

    def warp(pt):
        sdV = s*dV_avg(pt, avts, w)
        T0 = np.eye(3) + sdV
        # for _ in range(N):
        #     T = T.dot(T)
        T = np.linalg.matrix_power(T0, N)
        x_homogeneous = [pt[0], pt[1], 1]
        x_homogeneous = np.dot(T, x_homogeneous)
        return x_homogeneous[:2] 

    return np.apply_along_axis(warp, 1, x0s)

def dV_avg(pt, avts, w):
    dV = np.zeros(shape=(3,3))
    for avt in avts:
        c = avt.anchor
        V = avt.transform
        dV += w(pt,c)*V
    return dV