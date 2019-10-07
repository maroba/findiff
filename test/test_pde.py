import sys
sys.path.insert(1, '..')

import unittest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
from findiff.operators import FinDiff
from findiff.pde import *


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
        u = pde.solve(shape)
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

        u = pde.solve(shape)
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
        u = pde.solve(shape)
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
        u = pde.solve(shape)

        np.testing.assert_array_almost_equal(expected, u)
