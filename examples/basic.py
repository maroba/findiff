import numpy as np
from findiff import FinDiff


def example_1d():

    # Initialize some array with values to differentiate
    dx, f = initialize_1d()

    # First derivative operator with respect to 0-th axis
    d_dx = FinDiff(0, dx)
    df_dx = d_dx(f)

    # Second derivative operator with respect to 0-th axis
    d2_dx2 = FinDiff(0, dx, 2)
    d2f_dx2 = d2_dx2(f)

    # Sixth derivative operator with respect to 0-th axis
    d6_dx6 = FinDiff((0, dx, 6))
    d6f_dx6 = d6_dx6(f)


def example_2d():

    # Initialize some array with values to differentiate
    [dx, dy], f = initialize_2d()

    # Partial derivative with respect to 0-th axis
    d_dx = FinDiff(0, dx)
    df_dx = d_dx(f)

    # Partial derivative with respect to 1st axis
    d_dy = FinDiff(1, dy)
    df_dy = d_dy(f)

    # Second partial derivative operator with respect to 0-th axis
    d2_dx2 = FinDiff(0, dx, 2)
    d2f_dx2 = d2_dx2(f)

    # Second partial derivative operator with respect to 1-th axis
    d2_dy2 = FinDiff(1, dy, 2)
    d2f_dy2 = d2_dy2(f)

    # Second mixed partial derivative operator
    d2_dxdy = FinDiff((0, dx), (1, dy))
    d2f_dxdy = d2_dxdy(f)


def example_3d():

    # Initialize some array with values to differentiate
    [dx, dy, dz], f = initialize_3d()

    # Partial derivative with respect to 0-th axis
    d_dx = FinDiff(0, dx)
    df_dx = d_dx(f)

    # Partial derivative with respect to 1st axis
    d_dy = FinDiff(1, dy)
    df_dy = d_dy(f)

    # Second partial derivative operator with respect to 0-th axis
    d2_dx2 = FinDiff(0, dx, 2)
    d2f_dx2 = d2_dx2(f)

    # Second partial derivative operator with respect to 2nd axis
    d2_dz2 = FinDiff(2, dz, 2)
    d2f_dz2 = d2_dz2(f)


def initialize_1d():
    num_pts = 100
    L = 10
    x = np.linspace(0, L, num_pts)
    dx = x[1] - x[0]
    f = np.sin(x)
    return dx, f


def initialize_2d():
    num_pts = 100
    L = 10
    x, y = np.linspace(0, L, num_pts), np.linspace(0, L, num_pts)
    dx, dy = x[1] - x[0], y[1] - y[0]
    X, Y = np.meshgrid(x, y, indexing='ij')
    f = np.sin(X) * np.cos(Y)
    return [dx, dy], f


def initialize_3d():
    num_pts = 30
    L = 10
    x, y, z = tuple([np.linspace(0, L, num_pts)]*3)
    dx, dy, dz = x[1] - x[0], y[1] - y[0], z[1] - z[0]
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    f = np.sin(X) * np.cos(Y) * np.sin(Z)
    return [dx, dy, dz], f


example_1d()
example_2d()
example_3d()
