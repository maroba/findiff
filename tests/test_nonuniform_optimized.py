"""Tests for the optimized nonuniform grid implementation.

Covers:
- Batched coefficient computation (calc_coefs_non_uni_batched, _solve_non_uni_batched)
- Vectorized __call__ (_FinDiffNonUniform._apply_nonuni_weights)
- Vectorized write_matrix_entries
- Coefficient reuse (no redundant computation)
- Edge cases and consistency with per-point coefficients
"""

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from findiff import Diff
from findiff.coefs import (
    calc_coefs_non_uni_batched,
    coefficients_non_uni,
    _solve_non_uni_batched,
)
from findiff.findiff import _FinDiffNonUniform


class TestBatchedCoefficients:
    """Tests for calc_coefs_non_uni_batched and _solve_non_uni_batched."""

    def test_batched_matches_per_point_uniform_grid(self):
        """Batched coefficients must match per-point computation on uniform grid."""
        x = np.linspace(0, 10, 50)
        result = calc_coefs_non_uni_batched(deriv=1, acc=2, coords=x)

        for i in range(len(x)):
            expected = coefficients_non_uni(deriv=1, acc=2, coords=x, idx=i)
            if i < result["num_bndry"]:
                actual_coefs = result["forward"]["coefficients"][i]
                actual_offsets = result["forward"]["offsets"]
            elif i >= len(x) - result["num_bndry"]:
                j = i - (len(x) - result["num_bndry"])
                actual_coefs = result["backward"]["coefficients"][j]
                actual_offsets = result["backward"]["offsets"]
            else:
                j = i - result["num_bndry"]
                actual_coefs = result["center"]["coefficients"][j]
                actual_offsets = result["center"]["offsets"]

            assert_array_almost_equal(actual_coefs, expected["coefficients"])
            assert_array_almost_equal(actual_offsets, expected["offsets"])

    def test_batched_matches_per_point_nonuniform_grid(self):
        """Batched coefficients must match per-point computation on nonuniform grid."""
        x = np.sort(np.random.default_rng(42).uniform(0, 10, 30))
        result = calc_coefs_non_uni_batched(deriv=1, acc=2, coords=x)

        for i in range(len(x)):
            expected = coefficients_non_uni(deriv=1, acc=2, coords=x, idx=i)
            if i < result["num_bndry"]:
                actual = result["forward"]["coefficients"][i]
            elif i >= len(x) - result["num_bndry"]:
                j = i - (len(x) - result["num_bndry"])
                actual = result["backward"]["coefficients"][j]
            else:
                j = i - result["num_bndry"]
                actual = result["center"]["coefficients"][j]

            assert_array_almost_equal(actual, expected["coefficients"])

    @pytest.mark.parametrize("deriv", [1, 2, 3])
    @pytest.mark.parametrize("acc", [2, 4])
    def test_batched_various_deriv_acc(self, deriv, acc):
        """Batched results match per-point for different derivative and accuracy orders."""
        x = np.sort(np.random.default_rng(7).uniform(0, 10, 40))
        result = calc_coefs_non_uni_batched(deriv=deriv, acc=acc, coords=x)

        for i in range(len(x)):
            expected = coefficients_non_uni(deriv=deriv, acc=acc, coords=x, idx=i)
            if i < result["num_bndry"]:
                actual = result["forward"]["coefficients"][i]
            elif i >= len(x) - result["num_bndry"]:
                j = i - (len(x) - result["num_bndry"])
                actual = result["backward"]["coefficients"][j]
            else:
                j = i - result["num_bndry"]
                actual = result["center"]["coefficients"][j]
            assert_array_almost_equal(actual, expected["coefficients"], decimal=10)

    def test_batched_result_structure(self):
        """Result dict has the expected keys and array shapes."""
        x = np.linspace(0, 5, 20)
        result = calc_coefs_non_uni_batched(deriv=1, acc=2, coords=x)

        assert "forward" in result
        assert "backward" in result
        assert "center" in result
        assert "num_bndry" in result

        num_bndry = result["num_bndry"]
        assert num_bndry == 1  # for deriv=1, acc=2

        assert result["forward"]["coefficients"].shape[0] == num_bndry
        assert result["backward"]["coefficients"].shape[0] == num_bndry
        assert result["center"]["coefficients"].shape[0] == len(x) - 2 * num_bndry

    def test_batched_even_derivative(self):
        """Even derivatives have forward/backward stencils wider than central."""
        x = np.linspace(0, 5, 20)
        result = calc_coefs_non_uni_batched(deriv=2, acc=2, coords=x)

        central_size = result["center"]["coefficients"].shape[1]
        forward_size = result["forward"]["coefficients"].shape[1]

        # For even derivatives, forward/backward have one more point
        assert forward_size == central_size + 1

    def test_batched_invalid_inputs(self):
        """Invalid deriv/acc raise ValueError."""
        x = np.linspace(0, 5, 20)
        with pytest.raises(ValueError):
            calc_coefs_non_uni_batched(deriv=-1, acc=2, coords=x)
        with pytest.raises(ValueError):
            calc_coefs_non_uni_batched(deriv=1, acc=3, coords=x)
        with pytest.raises(ValueError):
            calc_coefs_non_uni_batched(deriv=1, acc=0, coords=x)

    def test_solve_batched_empty_indices(self):
        """Empty indices produce empty weight array."""
        x = np.linspace(0, 5, 20)
        result = _solve_non_uni_batched(x, np.array([], dtype=int), 1, 1, 1)
        assert result.shape == (0, 3)


class TestVectorizedCall:
    """Tests for the vectorized _FinDiffNonUniform.__call__."""

    def test_1d_first_derivative(self):
        """Vectorized apply computes correct first derivative in 1D."""
        x = np.sort(np.random.default_rng(1).uniform(0, 5, 50))
        f = np.sin(x)
        d_dx = Diff(0, x, acc=4)
        result = d_dx(f)
        assert_array_almost_equal(result, np.cos(x), decimal=3)

    def test_1d_second_derivative(self):
        """Vectorized apply computes correct second derivative in 1D."""
        x = np.sort(np.random.default_rng(2).uniform(0, 5, 80))
        f = np.sin(x)
        d2_dx2 = Diff(0, x, acc=4) ** 2
        result = d2_dx2(f)
        assert_array_almost_equal(result, -np.sin(x), decimal=2)

    def test_2d_partial_x(self):
        """Vectorized apply computes correct partial x-derivative in 2D."""
        x = np.sort(np.random.default_rng(3).uniform(0, np.pi, 40))
        y = np.linspace(0, np.pi, 30)
        X, Y = np.meshgrid(x, y, indexing="ij")
        f = np.sin(X) * np.cos(Y)
        d_dx = Diff(0, x, acc=4)
        result = d_dx(f)
        expected = np.cos(X) * np.cos(Y)
        assert_array_almost_equal(result, expected, decimal=2)

    def test_2d_partial_y(self):
        """Vectorized apply computes correct partial y-derivative in 2D."""
        x = np.linspace(0, np.pi, 30)
        y = np.sort(np.random.default_rng(4).uniform(0, np.pi, 40))
        X, Y = np.meshgrid(x, y, indexing="ij")
        f = np.sin(X) * np.cos(Y)
        d_dy = Diff(1, y, acc=4)
        result = d_dy(f)
        expected = -np.sin(X) * np.sin(Y)
        assert_array_almost_equal(result, expected, decimal=2)

    def test_3d_partial(self):
        """Vectorized apply works in 3D."""
        x = np.sort(np.random.default_rng(5).uniform(0, 2, 15))
        y = np.sort(np.random.default_rng(6).uniform(0, 2, 15))
        z = np.sort(np.random.default_rng(7).uniform(0, 2, 15))
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        f = X ** 3 + Y ** 3 + Z ** 3

        d_dy = Diff(1, y)
        result = d_dy(f)
        expected = 3 * Y ** 2
        assert_array_almost_equal(result, expected, decimal=1)

    def test_integer_input_cast(self):
        """Integer arrays are cast to float before differentiation."""
        x = np.linspace(0, 10, 11)
        f = (x ** 2).astype(int)
        d_dx = Diff(0, x)
        result = d_dx(f)
        assert result.dtype == np.float64

    def test_concatenated_nonuniform_grid(self):
        """Works with concatenated grids of different spacing."""
        x = np.r_[np.arange(0, 4, 0.05), np.arange(4, 10, 1)]
        f = np.exp(-(x ** 2))
        d_dx = Diff(0, x, acc=4)
        result = d_dx(f)
        expected = -2 * x * np.exp(-(x ** 2))
        assert_array_almost_equal(result, expected, decimal=4)


class TestVectorizedMatrix:
    """Tests for the vectorized write_matrix_entries."""

    def test_matrix_1d(self):
        """Matrix representation matches direct application in 1D."""
        x = np.sort(np.random.default_rng(10).uniform(0, 5, 20))
        f = np.sin(x)
        d_dx = Diff(0, x)

        direct = d_dx(f)
        mat = d_dx.matrix(f.shape)
        via_matrix = mat.dot(f)

        assert_array_almost_equal(direct, via_matrix)

    def test_matrix_2d(self):
        """Matrix representation matches direct application in 2D."""
        x = np.sort(np.random.default_rng(11).uniform(0, 3, 12))
        y = np.sort(np.random.default_rng(12).uniform(0, 3, 12))
        X, Y = np.meshgrid(x, y, indexing="ij")
        f = X ** 2 + Y ** 2

        d_dx = Diff(0, x)
        direct = d_dx(f).reshape(-1)
        mat = d_dx.matrix(f.shape)
        via_matrix = mat.dot(f.reshape(-1))

        assert_array_almost_equal(direct, via_matrix)

    def test_matrix_laplacian_2d(self):
        """2D Laplacian matrix matches direct application."""
        x = np.sort(np.random.default_rng(13).uniform(0, 4, 10))
        y = np.sort(np.random.default_rng(14).uniform(0, 4, 10))
        X, Y = np.meshgrid(x, y, indexing="ij")
        f = X ** 2 + Y ** 2

        laplace = Diff(0, x) ** 2 + Diff(1, y) ** 2
        direct = laplace(f).reshape(-1)
        mat = laplace.matrix(f.shape)
        via_matrix = mat.dot(f.reshape(-1))

        assert_array_almost_equal(direct, via_matrix)

    def test_matrix_3d_nonuniform(self):
        """3D nonuniform matrix construction works and gives correct result."""
        x = y = z = np.linspace(0, 4, 15)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        laplace = Diff(0, x) ** 2 + Diff(1, y) ** 2 + Diff(2, z) ** 2
        u = X ** 2 + Y ** 2 + Z ** 2

        mat = laplace.matrix(u.shape)
        result = mat.dot(u.reshape(-1))
        assert_array_almost_equal(6.0 * np.ones(u.size), result)

    def test_matrix_mixed_deriv_2d(self):
        """Mixed partial derivative matrix works for nonuniform grid."""
        x = np.sort(np.random.default_rng(15).uniform(0, 5, 12))
        y = np.sort(np.random.default_rng(16).uniform(0, 5, 12))
        X, Y = np.meshgrid(x, y, indexing="ij")
        d2_dxdy = Diff(0, x) * Diff(1, y)
        u = X ** 2 * Y ** 2

        direct = d2_dxdy(u).reshape(-1)
        mat = d2_dxdy.matrix(u.shape)
        via_matrix = mat.dot(u.reshape(-1))

        assert_array_almost_equal(direct, via_matrix)

    def test_matrix_with_coefficients(self):
        """Matrix with variable coefficients works for nonuniform grid."""
        from findiff.compatible import Coef

        x = np.linspace(0, 10, 20)
        L = Coef(x) * (Diff(0, x) ** 2)
        u = np.random.default_rng(17).random(len(x))

        direct = L(u).reshape(-1)
        mat = L.matrix((len(x),))
        via_matrix = mat.dot(u.reshape(-1))

        assert_array_almost_equal(direct, via_matrix)


class TestStoredCoefficients:
    """Tests that coefficients are stored and reused, not recomputed."""

    def test_no_redundant_coef_computation(self):
        """write_matrix_entries uses stored coefficients, not recomputing."""
        x = np.linspace(0, 10, 20)
        diff = _FinDiffNonUniform(axis=0, order=1, coords=x, acc=2)

        # The differentiator stores center/forward/backward with precomputed weights
        assert "coefficients" in diff.center
        assert "coefficients" in diff.forward
        assert "coefficients" in diff.backward
        assert isinstance(diff.center["coefficients"], np.ndarray)
        assert diff.center["coefficients"].ndim == 2

    def test_coefficient_structure(self):
        """Stored coefficients have correct shapes."""
        x = np.linspace(0, 5, 20)
        diff = _FinDiffNonUniform(axis=0, order=1, coords=x, acc=2)

        n = len(x)
        nb = diff.num_bndry

        assert diff.forward["coefficients"].shape[0] == nb
        assert diff.backward["coefficients"].shape[0] == nb
        assert diff.center["coefficients"].shape[0] == n - 2 * nb

        # Stencil sizes
        assert len(diff.center["offsets"]) == diff.center["coefficients"].shape[1]
        assert len(diff.forward["offsets"]) == diff.forward["coefficients"].shape[1]
        assert len(diff.backward["offsets"]) == diff.backward["coefficients"].shape[1]


class TestApplyNonuniWeightsEdgeCases:
    """Tests for edge cases in _apply_nonuni_weights."""

    def test_empty_weights_skipped(self):
        """_apply_nonuni_weights with zero-length slice is a no-op."""
        x = np.linspace(0, 5, 20)
        diff = _FinDiffNonUniform(axis=0, order=1, coords=x, acc=2)

        y = np.ones(20)
        yd = np.zeros(20)
        # Calling with an empty slice should not modify yd
        diff._apply_nonuni_weights(
            yd, y, dim=0, ndims=1,
            weights=np.empty((0, 3)),
            offsets=np.array([-1, 0, 1]),
            ref_slice=slice(5, 5),
        )
        assert_array_almost_equal(yd, np.zeros(20))

    def test_higher_accuracy_order(self):
        """Higher accuracy order produces wider stencils and more boundary points."""
        x = np.sort(np.random.default_rng(20).uniform(0, 5, 50))
        f = np.exp(-x)

        d_dx_2 = Diff(0, x, acc=2)
        d_dx_6 = Diff(0, x, acc=6)

        # Both should approximate -exp(-x)
        result_2 = d_dx_2(f)
        result_6 = d_dx_6(f)
        expected = -np.exp(-x)

        # Higher accuracy should be at least as good
        err_2 = np.max(np.abs(result_2[5:-5] - expected[5:-5]))
        err_6 = np.max(np.abs(result_6[5:-5] - expected[5:-5]))
        assert err_6 <= err_2

    def test_small_grid(self):
        """Works correctly with a small grid (near minimum size)."""
        x = np.array([0.0, 1.0, 3.0, 6.0, 10.0])
        f = x ** 2
        d_dx = Diff(0, x)
        result = d_dx(f)
        expected = 2 * x
        assert_array_almost_equal(result, expected, decimal=0)
