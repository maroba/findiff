import unittest

import numpy as np

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

    def test_apply_laplacian_laplacian(self):

        x = y = np.linspace(0, 1, 21)
        dx = dy = x[1] - x[0]
        X, Y = np.meshgrid(x, y, indexing='ij')

        f = X**3 + Y**3
        expected =  6*X + 6*Y

        offsets = [(-1, 0), (0, 0), (1, 0), (0, 1), (0, -1)]
        stencil = Stencil(offsets, {(2, 0): 1, (0, 2): 1}, spacings=(dx, dy))
        at = (3, 4)
        actual = stencil(f, at)
        self.assertAlmostEqual(expected[at], actual)

    def test_apply_laplacian_laplacian_stencil_x_form(self):

        x = y = np.linspace(0, 1, 21)
        dx = dy = x[1] - x[0]
        X, Y = np.meshgrid(x, y, indexing='ij')

        f = X**3 + Y**3
        expected =  6*X + 6*Y

        offsets = [(0, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]
        stencil = Stencil(offsets, {(2, 0): 1, (0, 2): 1}, spacings=(dx, dy))
        at = (3, 4)
        actual = stencil(f, at)
        self.assertAlmostEqual(expected[at], actual)

    def test_apply_laplacian_laplacian_stencil_outside_grid(self):

        x = y = np.linspace(0, 1, 21)
        dx = dy = x[1] - x[0]
        X, Y = np.meshgrid(x, y, indexing='ij')

        f = X**3 + Y**3

        offsets = [(-1, 0), (0, 0), (1, 0), (0, 1), (0, -1)]
        stencil = Stencil(offsets, {(2, 0): 1, (0, 2): 1}, spacings=(dx, dy))
        at = (0, 1)
        with self.assertRaises(Exception):
            stencil(f, at)

        at = (3, 20)
        with self.assertRaises(Exception):
            stencil(f, at)

    def test_apply_mixed_deriv(self):

        x = y = np.linspace(0, 1, 101)
        dx = dy = x[1] - x[0]
        X, Y = np.meshgrid(x, y, indexing='ij')

        f = np.exp(-X**2-Y**2)
        expected =  4*X*Y*f

        offsets = [(0, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]
        stencil = Stencil(offsets, partials={(1, 1): 1}, spacings=(dx, dy))
        at = (3, 4)
        actual = stencil(f, at)
        self.assertAlmostEqual(expected[at], actual, places=5)

    def test_laplace_1d_9points(self):
        x = np.linspace(0, 1, 101)
        f = x**3
        expected = 6*x
        offsets = list(range(-4, 5))
        stencil = Stencil(offsets, partials={(2,): 1}, spacings=(x[1]-x[0],))
        at = 8,
        actual = stencil(f, at)
        self.assertAlmostEqual(expected[at], actual, places=5)
        print(stencil.values, stencil.accuracy)

    def tests_apply_stencil_on_multislice(self):
        x = y = np.linspace(0, 1, 21)
        dx = dy = x[1] - x[0]
        X, Y = np.meshgrid(x, y, indexing='ij')

        f = X ** 3 + Y ** 3
        expected = 6*X + 6*Y

        offsets = [(-1, 0), (0, 0), (1, 0), (0, 1), (0, -1)]
        stencil = Stencil(offsets, {(2, 0): 1, (0, 2): 1}, spacings=(dx, dy))

        on_slice = slice(1, -1), slice(1, -1)
        actual = stencil(f, on=on_slice)
        np.testing.assert_array_almost_equal(expected[on_slice], actual[on_slice])

    def tests_apply_stencil_on_mask(self):
        x = y = np.linspace(0, 1, 21)
        dx = dy = x[1] - x[0]
        X, Y = np.meshgrid(x, y, indexing='ij')

        f = X ** 3 + Y ** 3
        expected = 6*X + 6*Y

        offsets = [(-1, 0), (0, 0), (1, 0), (0, 1), (0, -1)]
        stencil = Stencil(offsets, {(2, 0): 1, (0, 2): 1}, spacings=(dx, dy))

        mask = np.full_like(f, fill_value=False, dtype=bool)
        mask[1:-1, 1:-1] = True
        actual = stencil(f, on=mask)
        np.testing.assert_array_almost_equal(expected[mask], actual[mask])
