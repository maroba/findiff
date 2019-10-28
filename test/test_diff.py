import sys
sys.path.insert(1, '..')

import unittest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
from findiff.diff import *
#import matplotlib.pyplot as plt
import findiff
from numpy import cos, sin
from findiff.grids import UniformGrid


class DiffTest(unittest.TestCase):

    def test_partial_d_dx(self):
        shape = 101,
        x, dx = make_grid(shape, (0, 1))

        u = x**2
        expected = 2 * x

        fd = Diff(0, 1)
        actual = fd(u, dx)

        assert_array_almost_equal(expected, actual)

    def test_partial_d2_dx2(self):
        shape = 101,
        x, dx = make_grid(shape, (0, 1))

        u = x ** 2
        expected = 2

        fd = Diff(0, 2)
        actual = fd(u, dx)

        assert_array_almost_equal(expected, actual)

    def test_partial_d_dx_acc(self):
        shape = 11,
        x, dx = make_grid(shape, (0, 1))

        u = x ** 3
        expected = 3*x**2

        fd = Diff(0, 1)
        actual = fd(u, dx)
        np.testing.assert_raises(AssertionError, assert_array_almost_equal, expected, actual)

        actual = fd(u, dx, acc=4)
        assert_array_almost_equal(expected, actual)

    def test_partial_d2_dxdy(self):

        shape = 50, 50
        (x, y), (dx, dy), (X, Y) = make_grid(shape, ((0, 1), (0, 1)))
        u = np.sin(X) * np.sin(Y)

        fd = Diff(0, 1) * Diff(1, 1)
        actual = fd(u, h={0: dx, 1: dy})

        expected = np.cos(X) * np.cos(Y)
        assert_array_almost_equal(expected, actual, decimal=3)

        fd = Diff(1, 1) * Diff(0, 1)
        actual = fd(u, h={0: dx, 1: dy})
        assert_array_almost_equal(expected, actual, decimal=3)

    def test_partial_d_dx_on_2d_array(self):

        shape = 50, 50
        (x, y), (dx, dy), (X, Y) = make_grid(shape, ((0, 1), (0, 1)))
        u = np.sin(X) * np.sin(Y)

        fd = Diff(1, 1)
        actual = fd(u, dy)

        expected = np.sin(X) * np.cos(Y)
        assert_array_almost_equal(expected, actual, decimal=3)

    def test_add_two_diffs(self):

        shape = 50, 50
        (x, y), (dx, dy), (X, Y) = make_grid(shape, ((0, 1), (0, 1)))
        u = np.sin(X) * np.sin(Y)

        fd = Diff(0, 2) + Diff(1, 2)
        actual = fd(u, dy)

        expected = -2 * np.sin(X) * np.sin(Y)
        assert_array_almost_equal(expected, actual, decimal=3)

    def test_sub_two_diffs(self):

        shape = 50, 50
        (x, y), (dx, dy), (X, Y) = make_grid(shape, ((0, 1), (0, 1)))
        u = np.sin(X) * np.sin(Y)

        fd = Diff(0, 2) - Diff(1, 2)
        actual = fd(u, dy)

        expected = np.zeros(shape)
        assert_array_almost_equal(expected, actual, decimal=3)

    def test_multiply_with_scalar(self):

        shape = 50, 50
        (x, y), (dx, dy), (X, Y) = make_grid(shape, ((0, 1), (0, 1)))
        u = np.sin(X) * np.sin(Y)

        fd = 3 * Diff(0, 2)
        actual = fd(u, dy, acc=4)

        expected = -3 * np.sin(X) * np.sin(Y)
        assert_array_almost_equal(expected, actual)

    def test_multiply_with_variable_from_left(self):

        shape = 50, 50
        (x, y), (dx, dy), (X, Y) = make_grid(shape, ((0, 1), (0, 1)))
        u = np.sin(X) * np.sin(Y)

        fd = Coef(3 * X * Y) * Diff(0, 2)
        actual = fd(u, dy, acc=4)

        expected = - 3*X *Y* np.sin(X) * np.sin(Y)
        assert_array_almost_equal(expected, actual)

    def test_identity(self):

        shape = 50, 50
        (x, y), (dx, dy), (X, Y) = make_grid(shape, ((0, 1), (0, 1)))
        u = np.sin(X) * np.sin(Y)

        fd = Diff(0, 2) + Id()
        actual = fd(u, dy, acc=4)

        expected = np.zeros_like(u)
        assert_array_almost_equal(expected, actual)

    def test_linear_comb(self):
        shape = 50, 50
        (x, y), (dx, dy), (X, Y) = make_grid(shape, ((0, 1), (0, 1)))
        u = np.sin(X) * np.sin(Y)

        fd = Coef(3*X)*Diff(0, 2) - Coef(Y) * (Diff(1, 2) + Diff(0, 1))
        actual = fd(u, h={0: dx, 1:dy}, acc=4)

        expected = -3*X*u - Y * (-sin(X)*sin(Y) + cos(X)*sin(Y))
        assert_array_almost_equal(expected, actual)


def make_grid(shape, edges):

    if len(shape) == 1:
        x = np.linspace(*edges, shape[0])
        dx = x[1]-x[0]
        return x, dx

    axes = tuple([np.linspace(edges[k][0], edges[k][1], shape[k]) for k in range(len(shape))])
    coords = np.meshgrid(*axes, indexing='ij')
    spacings = [axes[k][1]-axes[k][0] for k in range(len(shape))]
    return axes, spacings, coords
