import unittest

from findiff.stencils import Stencil


class TestStencilOperations(unittest.TestCase):

    def test_solve_laplace_2d_with_5_points(self):
        offsets = [(-1, 0), (0, 0), (1, 0), (0, 1), (0, -1)]

        stencil = Stencil(offsets, {(2, 0): 1, (0, 2): 1})

        expected = {
            (0, 0): -4,
            (-1, 0): 1, (1, 0): 1, (0, 1): 1, (0, -1): 1
        }

        self.assertEqual(expected, stencil.values)
        self.assertEqual(2, stencil.accuracy)

    def test_solve_laplace_2d_with_9_points(self):
        offsets = [(-1, 0), (0, 0), (1, 0), (0, 1), (0, -1), (-2, 0), (2, 0), (0, -2), (0, 2)]

        stencil = Stencil(offsets, {(2, 0): 1, (0, 2): 1})

        expected = {
            (0, 0): -5,
            (-1, 0): 4 / 3., (1, 0): 4 / 3., (0, 1): 4 / 3., (0, -1): 4 / 3.,
            (-2, 0): -1 / 12., (2, 0): -1 / 12., (0, -2): -1 / 12., (0, 2): -1 / 12.
        }

        self.assertEqual(4, stencil.accuracy)
        self.assertEqual(len(expected), len(stencil.values))
        for off, coeff in stencil.values.items():
            self.assertAlmostEqual(coeff, expected[off])

    def test_solve_laplace_2d_with_5_points_times_2(self):
        offsets = [(-1, 0), (0, 0), (1, 0), (0, 1), (0, -1)]

        stencil = Stencil(offsets, {(2, 0): 2, (0, 2): 2})

        expected = {
            (0, 0): -8,
            (-1, 0): 2, (1, 0): 2, (0, 1): 2, (0, -1): 2
        }

        self.assertEqual(expected, stencil.values)
        self.assertEqual(2, stencil.accuracy)

    def test_solve_laplace_2d_with_5_points_times_2_and_spacing(self):
        offsets = [(-1, 0), (0, 0), (1, 0), (0, 1), (0, -1)]

        stencil = Stencil(offsets, {(2, 0): 2, (0, 2): 2}, spacings=(0.1, 0.1))

        expected = {
            (0, 0): -800,
            (-1, 0): 200, (1, 0): 200, (0, 1): 200, (0, -1): 200
        }

        self.assertEqual(len(expected), len(stencil.values))
        for off, coeff in stencil.values.items():
            self.assertAlmostEqual(coeff, expected[off])
        self.assertEqual(2, stencil.accuracy)
