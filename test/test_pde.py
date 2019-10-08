import sys
sys.path.insert(1, '..')

import unittest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
from findiff.operators import FinDiff, Identity, Coef
from findiff.pde import *
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

class TestPDE(unittest.TestCase):

    def test_1d_dirichlet_hom(self):

        shape = (11,)

        x = np.linspace(0, 1, 11)
        dx = x[1] - x[0]
        L = FinDiff(0, dx, 2)

        bc = BoundaryConditions(shape)

        bc[0] = 1
        bc[-1] = 2

        pde = PDE(L, np.zeros_like(x), bc)
        u = pde.solve()
        expected = x + 1
        np.testing.assert_array_almost_equal(expected, u)

    def test_1d_dirichlet_inhom(self):

        nx = 21
        shape = (nx,)

        x = np.linspace(0, 1, nx)
        dx = x[1] - x[0]
        L = FinDiff(0, dx, 2)

        bc = BoundaryConditions(shape)

        bc[0] = 1
        bc[-1] = 2

        pde = PDE(L, 6*x, bc)

        u = pde.solve()
        expected = x**3 + 1
        np.testing.assert_array_almost_equal(expected, u)

    def test_1d_neumann_hom(self):

        nx = 11
        shape = (nx,)

        x = np.linspace(0, 1, nx)
        dx = x[1] - x[0]
        L = FinDiff(0, dx, 2)

        bc = BoundaryConditions(shape)

        bc[0] = 1
        bc[-1] = FinDiff(0, dx, 1), 2

        pde = PDE(L, np.zeros_like(x), bc)
        u = pde.solve()
        expected = 2*x + 1
        np.testing.assert_array_almost_equal(expected, u)

    def test_2d_dirichlet_hom(self):

        shape = (11, 11)

        x, y = np.linspace(0, 1, shape[0]), np.linspace(0, 1, shape[1])
        dx, dy = x[1] - x[0], y[1] - y[0]
        X, Y = np.meshgrid(x, y, indexing='ij')
        L = FinDiff(0, dx, 2) + FinDiff(1, dy, 2)

        expected = X + 1

        bc = BoundaryConditions(shape)

        bc[0, :] = 1
        bc[-1, :] = 2
        bc[:, 0] = X + 1
        bc[:, -1] = X + 1

        pde = PDE(L, np.zeros_like(X), bc)
        u = pde.solve()

        np.testing.assert_array_almost_equal(expected, u)

    def test_2d_dirichlet_inhom(self):
        shape = (11, 11)

        x, y = np.linspace(0, 1, shape[0]), np.linspace(0, 1, shape[1])
        dx, dy = x[1] - x[0], y[1] - y[0]
        X, Y = np.meshgrid(x, y, indexing='ij')
        L = FinDiff(0, dx, 2) + FinDiff(1, dy, 2)

        expected = X**3 + Y**3 + 1
        f = 6*X + 6*Y

        bc = BoundaryConditions(shape)

        bc[0, :] = expected
        bc[-1, :] = expected
        bc[:, 0] = expected
        bc[:, -1] = expected

        pde = PDE(L, f, bc)
        u = pde.solve()
        np.testing.assert_array_almost_equal(expected, u)

    def test_2d_neumann_hom(self):
        shape = (31, 31)

        x, y = np.linspace(0, 1, shape[0]), np.linspace(0, 1, shape[1])
        dx, dy = x[1] - x[0], y[1] - y[0]
        X, Y = np.meshgrid(x, y, indexing='ij')
        L = FinDiff(0, dx, 2) + FinDiff(1, dy, 2)

        expected = X**2 + Y**2 + 1
        f = 4 * np.ones_like(X)

        bc = BoundaryConditions(shape)

        d_dy = FinDiff(1, dy)

        bc[0, :] = expected
        bc[-1, :] = expected
        bc[:, 0] = d_dy, 2*Y
        bc[:, -1] = d_dy, 2*Y

        pde = PDE(L, f, bc)
        u = pde.solve()
        np.testing.assert_array_almost_equal(expected, u)

    def test_1d_oscillator_free_dirichlet(self):

        n = 300
        shape = n,
        t = np.linspace(0, 5, n)
        dt = t[1] - t[0]
        L = FinDiff(0, dt, 2) + Identity()

        bc = BoundaryConditions(shape)

        bc[0] = 1
        bc[-1] = 2

        eq = PDE(L, np.zeros_like(t), bc)
        u = eq.solve()
        expected = np.cos(t)-(np.cos(5)-2)*np.sin(t)/np.sin(5)
        np.testing.assert_array_almost_equal(expected, u, decimal=4)

    def test_1d_damped_osc_driv_dirichlet(self):
        n = 100
        shape = n,
        t = np.linspace(0, 1, n)
        dt = t[1] - t[0]
        L = FinDiff(0, dt, 2) - FinDiff(0, dt) + Identity()
        f = -3*np.exp(-t)*np.cos(t) + 2*np.exp(-t)*np.sin(t)

        expected = np.exp(-t)*np.sin(t)

        bc = BoundaryConditions(shape)

        bc[0] = expected[0]
        bc[-1] = expected[-1]

        eq = PDE(L, f, bc)
        u = eq.solve()

        np.testing.assert_array_almost_equal(expected, u, decimal=4)

    def test_1d_oscillator_driv_neumann(self):
        n = 200
        shape = n,
        t = np.linspace(0, 1, n)
        dt = t[1] - t[0]
        L = FinDiff(0, dt, 2) - FinDiff(0, dt) + Identity()
        f = -3 * np.exp(-t) * np.cos(t) + 2 * np.exp(-t) * np.sin(t)

        expected = np.exp(-t) * np.sin(t)

        bc = BoundaryConditions(shape)

        bc[0] = FinDiff(0, dt), 1
        bc[-1] = expected[-1]

        eq = PDE(L, f, bc)
        u = eq.solve()

        np.testing.assert_array_almost_equal(expected, u, decimal=4)

    def test_1d_with_coeffs(self):
        n = 200
        shape = n,
        t = np.linspace(0, 1, n)
        dt = t[1] - t[0]
        L = Coef(t) * FinDiff(0, dt, 2)
        f = 6*t**2

        bc = BoundaryConditions(shape)

        bc[0] = 0
        bc[-1] = 1

        eq = PDE(L, f, bc)
        u = eq.solve()
        expected = t**3

        np.testing.assert_array_almost_equal(expected, u, decimal=4)

    def test_mixed_equation__with_coeffs_2d(self):
        shape = (41, 51)

        x, y = np.linspace(0, 1, shape[0]), np.linspace(0, 1, shape[1])
        dx, dy = x[1] - x[0], y[1] - y[0]
        X, Y = np.meshgrid(x, y, indexing='ij')
        L = FinDiff(0, dx, 2) + Coef(X*Y) * FinDiff((0, dx, 1), (1, dy, 1)) + FinDiff(1, dy, 2)

        expected = X ** 3 + Y ** 3 + 1
        f = 6*(X + Y)

        bc = BoundaryConditions(shape)

        bc[0, :] = expected
        bc[-1, :] = expected
        bc[:, 0] = expected
        bc[:, -1] = expected

        pde = PDE(L, f, bc)
        u = pde.solve()
        np.testing.assert_array_almost_equal(expected, u, decimal=4)