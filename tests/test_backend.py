"""Tests for the backend dispatch layer and multi-backend support."""

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from findiff.backend import add_at, get_namespace, is_array
from findiff import Diff, Gradient, Divergence, Curl, Laplacian, Identity


# ---------------------------------------------------------------------------
# Unit tests for backend utilities
# ---------------------------------------------------------------------------


class TestGetNamespace:

    def test_numpy_array(self):
        assert get_namespace(np.array([1, 2, 3])) is np

    def test_no_arrays_returns_numpy(self):
        assert get_namespace() is np

    def test_multiple_numpy(self):
        a, b = np.array([1]), np.array([2])
        assert get_namespace(a, b) is np


class TestIsArray:

    def test_numpy_array(self):
        assert is_array(np.array([1, 2]))

    def test_numpy_scalar(self):
        assert is_array(np.float64(3.14))

    def test_number(self):
        assert not is_array(42)
        assert not is_array(3.14)

    def test_list(self):
        assert not is_array([1, 2, 3])

    def test_string(self):
        assert not is_array("hello")

    def test_none(self):
        assert not is_array(None)


class TestAddAt:

    def test_1d_slice(self):
        arr = np.zeros(5)
        result = add_at(arr, slice(1, 4), np.array([1.0, 2.0, 3.0]))
        assert_array_almost_equal(result, [0, 1, 2, 3, 0])
        assert result is arr  # in-place for numpy

    def test_2d_row(self):
        arr = np.zeros((3, 3))
        result = add_at(arr, (0, slice(None)), np.array([1.0, 2.0, 3.0]))
        assert_array_almost_equal(result[0], [1, 2, 3])
        assert result is arr

    def test_tuple_index(self):
        arr = np.zeros((4, 4))
        idx = (slice(1, 3), slice(1, 3))
        val = np.ones((2, 2))
        result = add_at(arr, idx, val)
        assert result[1, 1] == 1.0
        assert result[0, 0] == 0.0

    def test_accumulates(self):
        arr = np.ones(3)
        result = add_at(arr, slice(None), np.array([10.0, 20.0, 30.0]))
        assert_array_almost_equal(result, [11, 21, 31])


# ---------------------------------------------------------------------------
# Numpy regression tests — ensure existing behaviour is preserved
# ---------------------------------------------------------------------------


class TestNumpyRegression:

    def test_diff_1d(self):
        x = np.linspace(0, 2 * np.pi, 100)
        dx = x[1] - x[0]
        f = np.sin(x)
        result = Diff(0, dx)(f)
        assert isinstance(result, np.ndarray)
        assert_array_almost_equal(result, np.cos(x), decimal=2)

    def test_diff_2d(self):
        x = np.linspace(0, 2 * np.pi, 50)
        y = np.linspace(0, 2 * np.pi, 50)
        dx, dy = x[1] - x[0], y[1] - y[0]
        X, Y = np.meshgrid(x, y, indexing="ij")
        f = np.sin(X) * np.cos(Y)
        result = Diff(0, dx)(f)
        assert isinstance(result, np.ndarray)
        assert_array_almost_equal(result, np.cos(X) * np.cos(Y), decimal=2)

    def test_diff_periodic(self):
        x = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        dx = x[1] - x[0]
        f = np.sin(x)
        result = Diff(0, dx, periodic=True)(f)
        assert isinstance(result, np.ndarray)
        assert_array_almost_equal(result, np.cos(x), decimal=2)

    def test_diff_nonuniform(self):
        x = np.linspace(0, np.pi, 100)
        f = np.sin(x)
        result = Diff(0, x)(f)
        assert isinstance(result, np.ndarray)
        assert_array_almost_equal(result, np.cos(x), decimal=2)

    def test_integer_array_promotion(self):
        x = np.linspace(0, 10, 100)
        dx = x[1] - x[0]
        f = np.arange(100)
        result = Diff(0, dx)(f)
        assert result.dtype == np.float64

    def test_higher_order(self):
        x = np.linspace(0, 2 * np.pi, 200)
        dx = x[1] - x[0]
        f = np.sin(x)
        result = (Diff(0, dx) ** 2)(f)
        assert isinstance(result, np.ndarray)
        assert_array_almost_equal(result, -np.sin(x), decimal=2)

    def test_operator_add(self):
        x = np.linspace(0, 2 * np.pi, 200)
        dx = x[1] - x[0]
        f = np.sin(x)
        L = Diff(0, dx) ** 2 + Identity()
        result = L(f)
        assert_array_almost_equal(result, np.zeros_like(x), decimal=2)

    def test_operator_mul_with_ndarray(self):
        x = np.linspace(0, 2 * np.pi, 100)
        dx = x[1] - x[0]
        f = np.sin(x)
        coef = np.ones_like(x)
        L = coef * Diff(0, dx)
        result = L(f)
        assert_array_almost_equal(result, np.cos(x), decimal=2)

    def test_gradient(self):
        x = np.linspace(-1, 1, 50)
        y = np.linspace(-1, 1, 50)
        dx, dy = x[1] - x[0], y[1] - y[0]
        X, Y = np.meshgrid(x, y, indexing="ij")
        f = X ** 2 + Y ** 2
        result = Gradient(h=[dx, dy], acc=2)(f)
        assert isinstance(result, np.ndarray)
        assert_array_almost_equal(result[0], 2 * X, decimal=2)
        assert_array_almost_equal(result[1], 2 * Y, decimal=2)

    def test_divergence(self):
        x = np.linspace(-1, 1, 50)
        y = np.linspace(-1, 1, 50)
        dx, dy = x[1] - x[0], y[1] - y[0]
        X, Y = np.meshgrid(x, y, indexing="ij")
        f = np.array([X, Y])
        result = Divergence(h=[dx, dy], acc=2)(f)
        assert isinstance(result, np.ndarray)
        assert_array_almost_equal(result, 2 * np.ones_like(X), decimal=2)

    def test_curl_3d(self):
        x = np.linspace(-1, 1, 30)
        y = np.linspace(-1, 1, 30)
        z = np.linspace(-1, 1, 30)
        h = [x[1] - x[0], y[1] - y[0], z[1] - z[0]]
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        f = np.array([Y, -X, np.zeros_like(X)])
        result = Curl(h=h, acc=2)(f)
        assert isinstance(result, np.ndarray)
        assert_array_almost_equal(result[2], -2 * np.ones_like(X), decimal=2)

    def test_curl_2d(self):
        x = np.linspace(-1, 1, 50)
        y = np.linspace(-1, 1, 50)
        h = [x[1] - x[0], y[1] - y[0]]
        X, Y = np.meshgrid(x, y, indexing="ij")
        f = np.array([-Y, X])
        result = Curl(h=h, acc=2)(f)
        assert isinstance(result, np.ndarray)
        assert_array_almost_equal(result, 2 * np.ones_like(X), decimal=2)

    def test_laplacian(self):
        x = np.linspace(-1, 1, 50)
        y = np.linspace(-1, 1, 50)
        dx, dy = x[1] - x[0], y[1] - y[0]
        X, Y = np.meshgrid(x, y, indexing="ij")
        f = X ** 2 + Y ** 2
        result = Laplacian(h=[dx, dy])(f)
        assert isinstance(result, np.ndarray)
        assert_array_almost_equal(result, 4 * np.ones_like(X))

    def test_field_operator_str(self):
        from findiff.operators import FieldOperator

        assert str(FieldOperator(np.array([1, 2]))) == "f(x)"
        assert str(FieldOperator(3.0)) == "3.0"

    def test_estimate_error(self):
        x = np.linspace(0, 2 * np.pi, 200)
        dx = x[1] - x[0]
        d_dx = Diff(0, dx)
        est = d_dx.estimate_error(np.sin(x))
        assert est.derivative.shape == (200,)
        assert est.error.shape == (200,)


# ---------------------------------------------------------------------------
# JAX backend tests (skipped when JAX is not installed)
# ---------------------------------------------------------------------------


class TestJAXBackend:

    @pytest.fixture(autouse=True)
    def _require_jax(self):
        jax = pytest.importorskip("jax")
        jax.config.update("jax_enable_x64", True)
        self.jnp = jax.numpy

    def _is_jax(self, arr):
        return type(arr).__module__.split(".")[0] in ("jax", "jaxlib")

    # -- utility tests --

    def test_get_namespace(self):
        arr = self.jnp.array([1, 2, 3])
        assert get_namespace(arr) is self.jnp

    def test_is_array(self):
        assert is_array(self.jnp.array([1, 2]))

    def test_add_at(self):
        arr = self.jnp.zeros(5)
        result = add_at(arr, slice(1, 4), self.jnp.array([1.0, 2.0, 3.0]))
        assert_array_almost_equal(np.asarray(result), [0, 1, 2, 3, 0])
        assert result is not arr  # JAX returns new array

    # -- Diff operator tests --

    def test_diff_1d(self):
        x = np.linspace(0, 2 * np.pi, 100)
        dx = x[1] - x[0]
        f = self.jnp.array(np.sin(x))
        result = Diff(0, dx)(f)
        assert self._is_jax(result)
        assert_array_almost_equal(np.asarray(result), np.cos(x), decimal=2)

    def test_diff_2d(self):
        x = np.linspace(0, 2 * np.pi, 50)
        y = np.linspace(0, 2 * np.pi, 50)
        dx, dy = x[1] - x[0], y[1] - y[0]
        X, Y = np.meshgrid(x, y, indexing="ij")
        f = self.jnp.array(np.sin(X) * np.cos(Y))
        result = Diff(0, dx)(f)
        assert self._is_jax(result)
        assert_array_almost_equal(np.asarray(result), np.cos(X) * np.cos(Y), decimal=2)

    def test_diff_periodic(self):
        x = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        dx = x[1] - x[0]
        f = self.jnp.array(np.sin(x))
        result = Diff(0, dx, periodic=True)(f)
        assert self._is_jax(result)
        assert_array_almost_equal(np.asarray(result), np.cos(x), decimal=2)

    def test_diff_nonuniform(self):
        x = np.linspace(0, np.pi, 100)
        f = self.jnp.array(np.sin(x))
        result = Diff(0, x)(f)
        assert self._is_jax(result)
        assert_array_almost_equal(np.asarray(result), np.cos(x), decimal=2)

    def test_higher_order(self):
        x = np.linspace(0, 2 * np.pi, 200)
        dx = x[1] - x[0]
        f = self.jnp.array(np.sin(x))
        result = (Diff(0, dx) ** 2)(f)
        assert self._is_jax(result)
        assert_array_almost_equal(np.asarray(result), -np.sin(x), decimal=2)

    def test_integer_array_promotion(self):
        x = np.linspace(0, 10, 100)
        dx = x[1] - x[0]
        f = self.jnp.arange(100)
        result = Diff(0, dx)(f)
        assert self._is_jax(result)
        assert result.dtype in (np.float64, np.float32)

    # -- operator composition --

    def test_operator_add(self):
        x = np.linspace(0, 2 * np.pi, 200)
        dx = x[1] - x[0]
        f = self.jnp.array(np.sin(x))
        L = Diff(0, dx) ** 2 + Identity()
        result = L(f)
        assert self._is_jax(result)
        assert_array_almost_equal(np.asarray(result), np.zeros_like(x), decimal=2)

    def test_operator_mul_with_ndarray(self):
        x = np.linspace(0, 2 * np.pi, 100)
        dx = x[1] - x[0]
        f = self.jnp.array(np.sin(x))
        coef = self.jnp.ones_like(f)
        L = coef * Diff(0, dx)
        result = L(f)
        assert self._is_jax(result)
        assert_array_almost_equal(np.asarray(result), np.cos(x), decimal=2)

    def test_estimate_error(self):
        x = np.linspace(0, 2 * np.pi, 200)
        dx = x[1] - x[0]
        d_dx = Diff(0, dx)
        f = self.jnp.array(np.sin(x))
        est = d_dx.estimate_error(f)
        assert self._is_jax(est.derivative)
        assert self._is_jax(est.error)
        assert est.derivative.shape == (200,)

    # -- vector operators --

    def test_gradient(self):
        x = np.linspace(-1, 1, 50)
        y = np.linspace(-1, 1, 50)
        dx, dy = x[1] - x[0], y[1] - y[0]
        X, Y = np.meshgrid(x, y, indexing="ij")
        f = self.jnp.array(X ** 2 + Y ** 2)
        result = Gradient(h=[dx, dy], acc=2)(f)
        assert self._is_jax(result)
        assert_array_almost_equal(np.asarray(result[0]), 2 * X, decimal=2)
        assert_array_almost_equal(np.asarray(result[1]), 2 * Y, decimal=2)

    def test_divergence(self):
        x = np.linspace(-1, 1, 50)
        y = np.linspace(-1, 1, 50)
        dx, dy = x[1] - x[0], y[1] - y[0]
        X, Y = np.meshgrid(x, y, indexing="ij")
        f = self.jnp.array(np.array([X, Y]))
        result = Divergence(h=[dx, dy], acc=2)(f)
        assert self._is_jax(result)
        assert_array_almost_equal(np.asarray(result), 2 * np.ones_like(X), decimal=2)

    def test_curl_3d(self):
        x = np.linspace(-1, 1, 30)
        y = np.linspace(-1, 1, 30)
        z = np.linspace(-1, 1, 30)
        h = [x[1] - x[0], y[1] - y[0], z[1] - z[0]]
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        f = self.jnp.array(np.array([Y, -X, np.zeros_like(X)]))
        result = Curl(h=h, acc=2)(f)
        assert self._is_jax(result)
        result_np = np.asarray(result)
        assert_array_almost_equal(result_np[2], -2 * np.ones_like(X), decimal=2)

    def test_curl_2d(self):
        x = np.linspace(-1, 1, 50)
        y = np.linspace(-1, 1, 50)
        h = [x[1] - x[0], y[1] - y[0]]
        X, Y = np.meshgrid(x, y, indexing="ij")
        f = self.jnp.array(np.array([-Y, X]))
        result = Curl(h=h, acc=2)(f)
        assert self._is_jax(result)
        assert_array_almost_equal(np.asarray(result), 2 * np.ones_like(X), decimal=2)

    def test_laplacian(self):
        x = np.linspace(-1, 1, 50)
        y = np.linspace(-1, 1, 50)
        dx, dy = x[1] - x[0], y[1] - y[0]
        X, Y = np.meshgrid(x, y, indexing="ij")
        f = self.jnp.array(X ** 2 + Y ** 2)
        result = Laplacian(h=[dx, dy])(f)
        assert self._is_jax(result)
        assert_array_almost_equal(np.asarray(result), 4 * np.ones_like(X), decimal=2)

    # -- grid: JAX coords should be accepted and converted --

    def test_nonuniform_grid_with_jax_coords(self):
        x = np.linspace(0, np.pi, 100)
        x_jax = self.jnp.array(x)
        f = self.jnp.array(np.sin(x))
        result = Diff(0, x_jax)(f)
        assert self._is_jax(result)
        assert_array_almost_equal(np.asarray(result), np.cos(x), decimal=2)


# ---------------------------------------------------------------------------
# CuPy backend tests (skipped when CuPy is not installed)
# ---------------------------------------------------------------------------


class TestCuPyBackend:

    @pytest.fixture(autouse=True)
    def _require_cupy(self):
        self.cp = pytest.importorskip("cupy")

    def _to_np(self, arr):
        return self.cp.asnumpy(arr)

    # -- utility tests --

    def test_get_namespace(self):
        arr = self.cp.array([1, 2, 3])
        assert get_namespace(arr) is self.cp

    def test_is_array(self):
        assert is_array(self.cp.array([1, 2]))

    def test_add_at(self):
        arr = self.cp.zeros(5)
        result = add_at(arr, slice(1, 4), self.cp.array([1.0, 2.0, 3.0]))
        assert_array_almost_equal(self._to_np(result), [0, 1, 2, 3, 0])
        assert result is arr  # in-place for CuPy

    # -- Diff operator tests --

    def test_diff_1d(self):
        x = np.linspace(0, 2 * np.pi, 100)
        dx = x[1] - x[0]
        f = self.cp.array(np.sin(x))
        result = Diff(0, dx)(f)
        assert isinstance(result, self.cp.ndarray)
        assert_array_almost_equal(self._to_np(result), np.cos(x), decimal=2)

    def test_diff_periodic(self):
        x = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        dx = x[1] - x[0]
        f = self.cp.array(np.sin(x))
        result = Diff(0, dx, periodic=True)(f)
        assert isinstance(result, self.cp.ndarray)
        assert_array_almost_equal(self._to_np(result), np.cos(x), decimal=2)

    def test_diff_nonuniform(self):
        x = np.linspace(0, np.pi, 100)
        f = self.cp.array(np.sin(x))
        result = Diff(0, x)(f)
        assert isinstance(result, self.cp.ndarray)
        assert_array_almost_equal(self._to_np(result), np.cos(x), decimal=2)

    def test_higher_order(self):
        x = np.linspace(0, 2 * np.pi, 200)
        dx = x[1] - x[0]
        f = self.cp.array(np.sin(x))
        result = (Diff(0, dx) ** 2)(f)
        assert isinstance(result, self.cp.ndarray)
        assert_array_almost_equal(self._to_np(result), -np.sin(x), decimal=2)

    def test_laplacian(self):
        x = np.linspace(-1, 1, 50)
        y = np.linspace(-1, 1, 50)
        dx, dy = x[1] - x[0], y[1] - y[0]
        X, Y = np.meshgrid(x, y, indexing="ij")
        f = self.cp.array(X ** 2 + Y ** 2)
        result = Laplacian(h=[dx, dy])(f)
        assert isinstance(result, self.cp.ndarray)
        assert_array_almost_equal(self._to_np(result), 4 * np.ones_like(X), decimal=2)

    def test_gradient(self):
        x = np.linspace(-1, 1, 50)
        y = np.linspace(-1, 1, 50)
        dx, dy = x[1] - x[0], y[1] - y[0]
        X, Y = np.meshgrid(x, y, indexing="ij")
        f = self.cp.array(X ** 2 + Y ** 2)
        result = Gradient(h=[dx, dy], acc=2)(f)
        assert isinstance(result, self.cp.ndarray)
        assert_array_almost_equal(self._to_np(result[0]), 2 * X, decimal=2)
        assert_array_almost_equal(self._to_np(result[1]), 2 * Y, decimal=2)
