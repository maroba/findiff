import sys
sys.path.insert(1, '..')

import unittest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
from findiff.operators import FinDiff
from findiff.pde import PDE


class TestPDE(unittest.TestCase):


    def test_1d(self):

        shape = (11,)

        x = np.linspace(0, 1, 11)
        dx = x[1] - x[0]
        L = FinDiff(0, dx, 2)

        bc = {}

        bc[0] = 5
        bc[-1] = 1

        pde = PDE(L, np.zeros_like(x), bc)
        u = pde.solve(shape)
        print(u)


    def test_2d(self):

        shape = (3,4)

        x, y = np.linspace(0, 1, 30), np.linspace(0, 1, 40)
        dx, dy = x[1] - x[0], y[1] - y[0]
        X, Y = np.meshgrid(x, y, indexing='ij')
        L = FinDiff(0, dx, 2) + FinDiff(1, dy, 2)

        bc = BoundaryConditions(shape)

        bc[0, None] = 5
        bc[-1, None] = 1
        bc[None, 0] = 4
        bc
