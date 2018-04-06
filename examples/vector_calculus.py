import numpy as np
from findiff.vector import Gradient, Divergence, Curl


def example_gradient():

    # Set up the grid, in this case for 3D
    axes, h, [X, Y, Z] = init_mesh([50, 50, 50])

    # We construct a scalar function on an array to take the divergence from
    f = np.sin(X) * np.sin(Y) * np.sin(Z)

    assert f.shape == (50, 50, 50)

    # The exact gradient is:
    grad_f_ex = np.array([np.cos(X) * np.sin(Y) * np.sin(Z),
                         np.sin(X) * np.cos(Y) * np.sin(Z),
                         np.sin(X) * np.sin(Y) * np.cos(Z),
                         ])

    # Our numerical fradient is:
    grad = Gradient(h=h, acc=4)
    grad_f = grad(f)

    # The result should be a 3D vector function of three variables:
    assert grad_f.shape == (3, 50, 50, 50)

    # Should be almost equal (up to tiny numerical differences)
    np.testing.assert_array_almost_equal(grad_f, grad_f_ex)


def example_divergence():

    # Set up the grid, in this case for 3D
    axes, h, [X, Y, Z] = init_mesh([50, 50, 50])

    # We construct a 3D vector function to take the divergence from
    f = np.array([np.sin(X) * np.sin(Y) * np.sin(Z)] * 3)

    # Each of the three components is a scalar field:
    assert f.shape == (3, 50, 50, 50)

    # The exact divergence is:
    div_f_ex = \
        np.cos(X) * np.sin(Y) * np.sin(Z) + \
        np.sin(X) * np.cos(Y) * np.sin(Z) + \
        np.sin(X) * np.sin(Y) * np.cos(Z)

    # Our numerical divergence is:
    div = Divergence(h=h, acc=4)
    div_f = div(f)

    # The result should be a scalar field of three variables
    assert div_f.shape == (50, 50, 50)

    # Should be almost equal (up to tiny numerical differences)
    np.testing.assert_array_almost_equal(div_f, div_f_ex)


def example_curl():

    # Set up the grid, in this case for 3D
    axes, h, [X, Y, Z] = init_mesh([50, 50, 50])

    # We construct a 3D vector function to take the divergence from
    f = np.array([np.sin(X) * np.sin(Y) * np.sin(Z)] * 3)

    # Each of the three components is a scalar field:
    assert f.shape == (3, 50, 50, 50)

    # The exact divergence is:
    curl_f_ex = np.array([
        np.sin(X) * np.cos(Y) * np.sin(Z) - np.sin(X) * np.sin(Y) * np.cos(Z),
        np.sin(X) * np.sin(Y) * np.cos(Z) - np.cos(X) * np.sin(Y) * np.sin(Z),
        np.cos(X) * np.sin(Y) * np.sin(Z) - np.sin(X) * np.cos(Y) * np.sin(Z),
    ])

    # Our numerical curl is:
    curl = Curl(h=h, acc=4)
    curl_f = curl(f)

    # The result should be a 3D vector function of three variables:
    assert curl_f.shape == (3, 50, 50, 50)

    # Should be almost equal (up to tiny numerical differences)
    np.testing.assert_array_almost_equal(curl_f, curl_f_ex)


def init_mesh(npoints):
    ndims = len(npoints)
    axes = [np.linspace(-1, 1, npoints[k]) for k in range(ndims)]
    h = [x[1] - x[0] for x in axes]
    mesh = np.meshgrid(*axes, indexing="ij")
    return axes, h, mesh

example_gradient()
example_divergence()
example_curl()
