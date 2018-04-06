import unittest
from numpy.testing import assert_array_almost_equal
import numpy as np
from findiff.vector import Gradient


class TestGradient(unittest.TestCase):

    def test_2d_gradient_on_scalar_func(self):
        axes, h, [X, Y, Z] = init_mesh(3, (50, 50, 50))
        f = np.sin(X) * np.sin(Y) * np.sin(Z)
        grad_f_ex = np.array([
          np.cos(X) * np.sin(Y) * np.sin(Z),
          np.sin(X) * np.cos(Y) * np.sin(Z),
          np.sin(X) * np.sin(Y) * np.cos(Z),
        ])
        grad = Gradient(h=h, acc=4)
        grad_f = grad(f)
        assert_array_almost_equal(grad_f, grad_f_ex)

    def test_3d_gradient_on_vector_func_should_fail(self):
        axes, h, [X, Y, Z] = init_mesh(3, (50, 50, 50))
        f = np.array([np.sin(X) * np.sin(Y) * np.sin(Z),
                      np.sin(X) * np.sin(Y) * np.sin(Z)
                     ])
        grad = Gradient(h=h, acc=4)
        self.assertRaises(ValueError, grad, f)


def init_mesh(ndims, npoints):
    axes = [np.linspace(-1, 1, npoints[k]) for k in range(ndims)]
    h = [x[1] - x[0] for x in axes]
    mesh = np.meshgrid(*axes, indexing="ij")
    return axes, h, mesh


if __name__ == '__main__':
    unittest.main()