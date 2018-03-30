import numpy as np
from findiff import FinDiff, Laplacian, Coefficient


def example_laplace_3d():

    # Initialize some sample array to differentiate, h is [dx, dy, dz]
    h, f, _ = initialize_3d()

    # Create the Laplace operator
    laplace = FinDiff(h=h, dims=[0,0]) + FinDiff(h=h, dims=[1,1]) + FinDiff(h=h, dims=[2,2])

    # And apply it:
    lap_f = laplace(f)

    # Shortcut:
    laplace = Laplacian(h)
    lap_f = laplace(f)


def example_linear_operator_constant_coefs():

    # Initialize some sample array to differentiate, h is [dx, dy, dz]
    h, f, _ = initialize_3d()

    # Create some linear differential operator with constant coefficients:
    diff_op = Coefficient(2) * FinDiff(h, dims=[0]) + Coefficient(-3) * FinDiff(h, dims=[1,1])

    # Apply it
    result = diff_op(f)


def example_linear_operator_variable_coefs():

    # Initialize some sample array to differentiate,
    # h is [dx, dy, dz]
    # X, Y, Z are numpy arrays with the coordinate values for each grid point
    h, f, [X, Y, Z] = initialize_3d()

    # Create some linear differential operator with variable coefficients:
    diff_op = Coefficient(2*X) * FinDiff(h, dims=[0]) + Coefficient(-3*Y**2) * FinDiff(h, dims=[1,1])

    # Apply it
    result = diff_op(f)


def initialize_3d():
    num_pts = 30
    L = 10
    x, y, z = tuple([np.linspace(0, L, num_pts)] * 3)
    dx, dy, dz = x[1] - x[0], y[1] - y[0], z[1] - z[0]
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    f = np.sin(X) * np.cos(Y) * np.sin(Z)
    return [dx, dy, dz], f, [X, Y, Z]


example_laplace_3d()
example_linear_operator_constant_coefs()
example_linear_operator_variable_coefs()
