import numpy as np


def init_mesh(ndims, npoints):
    axes = [np.linspace(-1, 1, npoints[k]) for k in range(ndims)]
    h = [x[1] - x[0] for x in axes]
    mesh = np.meshgrid(*axes, indexing="ij")
    return axes, h, mesh

