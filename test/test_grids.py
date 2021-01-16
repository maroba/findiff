import unittest

import numpy as np

from findiff.grids import UniformGrid


class TestUniformGrid(unittest.TestCase):

    def test_init_2d(self):
        shape = 30, 30
        spac = 0.1, 0.2
        center = 2, 3
        grid = UniformGrid(shape, spac, center)
        self.assertEqual(0.1, grid.spacing(0))
        np.testing.assert_array_equal(center, grid.center)

    def test_init_1d(self):
        grid = UniformGrid(30, 0.1)
        self.assertEqual(0.1, grid.spacing(0))
        np.testing.assert_array_equal((0,), grid.center)