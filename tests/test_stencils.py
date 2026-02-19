import numpy as np
import pytest

from findiff import Identity, FinDiff
from findiff.stencils import Stencil


class TestStencilOperations:

    def test_solve_laplace_2d_with_5_points(self):
        offsets = [(-1, 0), (0, 0), (1, 0), (0, 1), (0, -1)]

        stencil = Stencil(offsets, {(2, 0): 1, (0, 2): 1})

        expected = {(0, 0): -4, (-1, 0): 1, (1, 0): 1, (0, 1): 1, (0, -1): 1}

        assert expected == stencil.values
        assert 2 == stencil.accuracy

    def test_solve_laplace_2d_with_9_points(self):
        offsets = [
            (-1, 0),
            (0, 0),
            (1, 0),
            (0, 1),
            (0, -1),
            (-2, 0),
            (2, 0),
            (0, -2),
            (0, 2),
        ]

        stencil = Stencil(offsets, {(2, 0): 1, (0, 2): 1})

        expected = {
            (0, 0): -5,
            (-1, 0): 4 / 3.0,
            (1, 0): 4 / 3.0,
            (0, 1): 4 / 3.0,
            (0, -1): 4 / 3.0,
            (-2, 0): -1 / 12.0,
            (2, 0): -1 / 12.0,
            (0, -2): -1 / 12.0,
            (0, 2): -1 / 12.0,
        }

        self.assertEqual(4, stencil.accuracy)
        self.assertEqual(len(expected), len(stencil.values))
        for off, coeff in stencil.values.items():
            self.assertAlmostEqual(coeff, expected[off])

    def assertEqual(self, first, second):
        assert first == second

    def assertAlmostEqual(self, first, second, **kwargs):
        assert first == pytest.approx(second, **kwargs)

    def test_solve_laplace_2d_with_5_points_times_2(self):
        offsets = [(-1, 0), (0, 0), (1, 0), (0, 1), (0, -1)]

        stencil = Stencil(offsets, {(2, 0): 2, (0, 2): 2})

        expected = {(0, 0): -8, (-1, 0): 2, (1, 0): 2, (0, 1): 2, (0, -1): 2}

        self.assertEqual(expected, stencil.values)
        self.assertEqual(2, stencil.accuracy)

    def test_solve_laplace_2d_with_5_points_times_2_and_spacing(self):
        offsets = [(-1, 0), (0, 0), (1, 0), (0, 1), (0, -1)]

        stencil = Stencil(offsets, {(2, 0): 2, (0, 2): 2}, spacings=(0.1, 0.1))

        expected = {(0, 0): -800, (-1, 0): 200, (1, 0): 200, (0, 1): 200, (0, -1): 200}

        self.assertEqual(len(expected), len(stencil.values))
        for off, coeff in stencil.values.items():
            self.assertAlmostEqual(coeff, expected[off])
        self.assertEqual(2, stencil.accuracy)

    def test_apply_laplacian_laplacian(self):

        x = y = np.linspace(0, 1, 21)
        dx = dy = x[1] - x[0]
        X, Y = np.meshgrid(x, y, indexing="ij")

        f = X**3 + Y**3
        expected = 6 * X + 6 * Y

        offsets = [(-1, 0), (0, 0), (1, 0), (0, 1), (0, -1)]
        stencil = Stencil(offsets, {(2, 0): 1, (0, 2): 1}, spacings=(dx, dy))
        at = (3, 4)
        actual = stencil(f, at)
        self.assertAlmostEqual(expected[at], actual)

    def test_apply_laplacian_laplacian_stencil_x_form(self):

        x = y = np.linspace(0, 1, 21)
        dx = dy = x[1] - x[0]
        X, Y = np.meshgrid(x, y, indexing="ij")

        f = X**3 + Y**3
        expected = 6 * X + 6 * Y

        offsets = [(0, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]
        stencil = Stencil(offsets, {(2, 0): 1, (0, 2): 1}, spacings=(dx, dy))
        at = (3, 4)
        actual = stencil(f, at)
        self.assertAlmostEqual(expected[at], actual)

    def test_apply_laplacian_laplacian_stencil_outside_grid(self):

        x = y = np.linspace(0, 1, 21)
        dx = dy = x[1] - x[0]
        X, Y = np.meshgrid(x, y, indexing="ij")

        f = X**3 + Y**3

        offsets = [(-1, 0), (0, 0), (1, 0), (0, 1), (0, -1)]
        stencil = Stencil(offsets, {(2, 0): 1, (0, 2): 1}, spacings=(dx, dy))
        at = (0, 1)
        with pytest.raises(Exception):
            stencil(f, at)

        at = (3, 20)
        with pytest.raises(Exception):
            stencil(f, at)

    def test_apply_mixed_deriv(self):

        x = y = np.linspace(0, 1, 101)
        dx = dy = x[1] - x[0]
        X, Y = np.meshgrid(x, y, indexing="ij")

        f = np.exp(-(X**2) - Y**2)
        expected = 4 * X * Y * f

        offsets = [(0, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]
        stencil = Stencil(offsets, partials={(1, 1): 1}, spacings=(dx, dy))
        at = (3, 4)
        actual = stencil(f, at)
        self.assertAlmostEqual(expected[at], actual, abs=1e-6)

    def test_laplace_1d_9points(self):
        x = np.linspace(0, 1, 101)
        f = x**3
        expected = 6 * x
        offsets = list(range(-4, 5))
        stencil = Stencil(offsets, partials={(2,): 1}, spacings=(x[1] - x[0],))
        at = (8,)
        actual = stencil(f, at)
        self.assertAlmostEqual(expected[at], actual, abs=1.0e-6)

    def tests_apply_stencil_on_multislice(self):
        x = y = np.linspace(0, 1, 21)
        dx = dy = x[1] - x[0]
        X, Y = np.meshgrid(x, y, indexing="ij")

        f = X**3 + Y**3
        expected = 6 * X + 6 * Y

        offsets = [(-1, 0), (0, 0), (1, 0), (0, 1), (0, -1)]
        stencil = Stencil(offsets, {(2, 0): 1, (0, 2): 1}, spacings=(dx, dy))

        on_slice = slice(1, -1), slice(1, -1)
        actual = stencil(f, on=on_slice)
        np.testing.assert_array_almost_equal(expected[on_slice], actual[on_slice])

    def tests_apply_stencil_on_mask(self):
        x = y = np.linspace(0, 1, 21)
        dx = dy = x[1] - x[0]
        X, Y = np.meshgrid(x, y, indexing="ij")

        f = X**3 + Y**3
        expected = 6 * X + 6 * Y

        offsets = [(-1, 0), (0, 0), (1, 0), (0, 1), (0, -1)]
        stencil = Stencil(offsets, {(2, 0): 1, (0, 2): 1}, spacings=(dx, dy))

        mask = np.full_like(f, fill_value=False, dtype=bool)
        mask[1:-1, 1:-1] = True
        actual = stencil(f, on=mask)
        np.testing.assert_array_almost_equal(expected[mask], actual[mask])

    def test_helmholtz_stencil_issue_60(self):
        # This is a regression test for issue #60.

        H = Identity() - FinDiff(0, 1, 2)

        stencil_set = H.stencil((10,))

        expected = {
            ("L",): {(0,): -1.0, (1,): 5.0, (2,): -4.0, (3,): 1.0},
            ("C",): {(-1,): -1.0, (0,): 3.0, (1,): -1.0},
            ("H",): {(-3,): 1.0, (-2,): -4.0, (-1,): 5.0, (0,): -1.0},
        }

        actual = stencil_set.data
        self.assertEqual(len(expected), len(actual))
        self.assertEqual(expected.keys(), actual.keys())
        for key, expected_stencil in expected.items():
            actual_stencil = actual[key]

            self.assertEqual(expected_stencil.keys(), actual_stencil.keys())
            for offset, expected_coef in expected_stencil.items():
                actual_coef = actual_stencil[offset]
                self.assertAlmostEqual(expected_coef, actual_coef)


class TestStencilSliceBugs:
    """Test cases for reproducing slice handling bugs in Stencil class. (Issue #99)"""

    def test_negative_slice_indices_bug(self):

        # Create a simple 3D grid
        grid_shape = (3, 3, 3)
        x = y = z = np.linspace(0, 1, 3)

        # Create a level set that varies linearly along x-axis
        level_set = np.zeros(grid_shape)
        for i in range(3):
            level_set[i, :, :] = x[i]

        dx = dy = dz = x[1] - x[0]  # 0.5
        spacings = (dx, dy, dz)

        # Set up forward difference stencil for first derivative along x-axis
        offsets = [(0, 0, 0), (1, 0, 0)]
        partials = {(1, 0, 0): 1}  # First derivative in x direction

        stencil = Stencil(offsets=offsets, partials=partials, spacings=spacings)

        # slice(0, -1) should select all elements except the last along axis 0
        slices = (slice(0, -1), slice(None), slice(None))

        derivative = stencil(level_set, on=slices)

        assert derivative.shape == (3, 3, 3)
        expected_derivative = 1.0
        np.testing.assert_allclose(derivative[0:2, :, :], expected_derivative)

    def test_slice_none_canonicalization_bug(self):
        # Create a simple 3D array
        level_set = np.ones((3, 3, 3))

        # Create a simple stencil
        offsets = [(0, 0, 0), (1, 0, 0)]
        partials = {(1, 0, 0): 1}
        spacings = (0.5, 0.5, 0.5)

        stencil = Stencil(offsets=offsets, partials=partials, spacings=spacings)

        # Test slice(None) handling - this should select all elements
        slices = (slice(0, -1), slice(None), slice(None))

        derivative = stencil(level_set, on=slices)

        # Should return a result with the expected shape
        assert derivative.shape == (3, 3, 3)

    def test_workaround_explicit_slice_bounds_works(self):
        """Test that explicit slice bounds work as expected (user's workaround)."""
        # Create a simple 3D grid
        grid_shape = (3, 3, 3)
        x = y = z = np.linspace(0, 1, 3)

        # Create a level set that varies linearly along x-axis
        level_set = np.zeros(grid_shape)
        for i in range(3):
            level_set[i, :, :] = x[i]

        dx = dy = dz = x[1] - x[0]  # 0.5
        spacings = (dx, dy, dz)

        # Set up forward difference stencil for first derivative along x-axis
        offsets = [(0, 0, 0), (1, 0, 0)]
        partials = {(1, 0, 0): 1}  # First derivative in x direction

        stencil = Stencil(offsets=offsets, partials=partials, spacings=spacings)

        # Workaround: Use explicit bounds instead of slice(None) or slice(0, -1)
        slices = (
            slice(0, level_set.shape[0] - 1),  # Instead of slice(0, -1)
            slice(0, level_set.shape[1]),      # Instead of slice(None)
            slice(0, level_set.shape[2])       # Instead of slice(None)
        )

        # This should work fine
        derivative = stencil(level_set, on=slices)

        # The derivative should be 1.0 everywhere except the last slice along x-axis
        expected_derivative = 1.0
        assert derivative.shape == (3, 3, 3)

        # First two slices should have derivative = 1.0
        np.testing.assert_allclose(derivative[0:2, :, :], expected_derivative)

        # Last slice should be 0 (not computed)
        np.testing.assert_allclose(derivative[2, :, :], 0.0)

    def test_canonic_slice_method_bugs(self):

        # Create a dummy stencil to access the _canonic_slice method
        offsets = [(0,), (1,)]
        partials = {(1,): 1}
        stencil = Stencil(offsets=offsets, partials=partials)

        length = 5

        # slice(None) should become slice(0, length)
        canonical = stencil._canonic_slice(slice(None), length)
        assert canonical.start == 0
        assert canonical.stop == length

        # slice(0, -1) should become slice(0, 4)
        canonical = stencil._canonic_slice(slice(0, -1), length)
        assert canonical.start == 0
        assert canonical.stop == length - 1

        # slice(1, -1) should become slice(1, 4)
        canonical = stencil._canonic_slice(slice(1, -1), length)
        assert canonical.start == 1
        assert canonical.stop == length - 1
