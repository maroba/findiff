"""Performance benchmarks comparing numpy vs JAX backends.

Run with:  pytest -m benchmark -v --no-header
"""

from timeit import timeit

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from findiff import Diff, Laplacian

pytestmark = pytest.mark.benchmark

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
jnp = jax.numpy

REPS_1D = 200
REPS_3D = 20


def _fmt(label, t, reps):
    return f"{label}: {t / reps * 1e3:.2f} ms/call"


# ---------------------------------------------------------------------------
# 1-D benchmarks
# ---------------------------------------------------------------------------


class TestBenchmarkDiff1D:

    @pytest.fixture(autouse=True)
    def setup(self):
        N = 100_000
        x = np.linspace(0, 2 * np.pi, N)
        self.dx = x[1] - x[0]
        self.f_np = np.sin(x)
        self.f_jax = jnp.array(self.f_np)
        self.expected = np.cos(x)
        self.N = N

    def test_diff_numpy(self, capsys):
        d_dx = Diff(0, self.dx)
        result = d_dx(self.f_np)
        assert_array_almost_equal(result, self.expected, decimal=2)
        t = timeit(lambda: d_dx(self.f_np), number=REPS_1D)
        with capsys.disabled():
            print(f"\n  1D Diff (N={self.N}) {_fmt('numpy', t, REPS_1D)}")

    def test_diff_jax_no_jit(self, capsys):
        d_dx = Diff(0, self.dx)
        result = d_dx(self.f_jax)
        assert_array_almost_equal(np.asarray(result), self.expected, decimal=2)
        # warm up
        d_dx(self.f_jax).block_until_ready()
        t = timeit(lambda: d_dx(self.f_jax).block_until_ready(), number=REPS_1D)
        with capsys.disabled():
            print(f"\n  1D Diff (N={self.N}) {_fmt('jax (no jit)', t, REPS_1D)}")

    def test_diff_jax_jit(self, capsys):
        d_dx = Diff(0, self.dx)
        d_dx_jit = jax.jit(d_dx)
        # warm up
        result = d_dx_jit(self.f_jax).block_until_ready()
        assert_array_almost_equal(np.asarray(result), self.expected, decimal=2)
        t = timeit(lambda: d_dx_jit(self.f_jax).block_until_ready(), number=REPS_1D)
        with capsys.disabled():
            print(f"\n  1D Diff (N={self.N}) {_fmt('jax (jit)', t, REPS_1D)}")


class TestBenchmarkPeriodicDiff1D:

    @pytest.fixture(autouse=True)
    def setup(self):
        N = 100_000
        x = np.linspace(0, 2 * np.pi, N, endpoint=False)
        self.dx = x[1] - x[0]
        self.f_np = np.sin(x)
        self.f_jax = jnp.array(self.f_np)
        self.expected = np.cos(x)
        self.N = N

    def test_periodic_numpy(self, capsys):
        d_dx = Diff(0, self.dx, periodic=True)
        result = d_dx(self.f_np)
        assert_array_almost_equal(result, self.expected, decimal=2)
        t = timeit(lambda: d_dx(self.f_np), number=REPS_1D)
        with capsys.disabled():
            print(f"\n  1D Periodic (N={self.N}) {_fmt('numpy', t, REPS_1D)}")

    def test_periodic_jax_jit(self, capsys):
        d_dx = Diff(0, self.dx, periodic=True)
        d_dx_jit = jax.jit(d_dx)
        result = d_dx_jit(self.f_jax).block_until_ready()
        assert_array_almost_equal(np.asarray(result), self.expected, decimal=2)
        t = timeit(lambda: d_dx_jit(self.f_jax).block_until_ready(), number=REPS_1D)
        with capsys.disabled():
            print(f"\n  1D Periodic (N={self.N}) {_fmt('jax (jit)', t, REPS_1D)}")


# ---------------------------------------------------------------------------
# 3-D benchmarks
# ---------------------------------------------------------------------------


class TestBenchmarkLaplacian3D:

    @pytest.fixture(autouse=True)
    def setup(self):
        M = 100
        axes = [np.linspace(0, 2 * np.pi, M)] * 3
        self.h = [a[1] - a[0] for a in axes]
        X, Y, Z = np.meshgrid(*axes, indexing="ij")
        self.f_np = np.sin(X) * np.sin(Y) * np.sin(Z)
        self.f_jax = jnp.array(self.f_np)
        self.expected = -3 * self.f_np
        self.M = M

    def test_laplacian_numpy(self, capsys):
        lap = Laplacian(h=self.h)
        result = lap(self.f_np)
        assert_array_almost_equal(result, self.expected, decimal=1)
        t = timeit(lambda: lap(self.f_np), number=REPS_3D)
        with capsys.disabled():
            print(f"\n  3D Laplacian ({self.M}^3) {_fmt('numpy', t, REPS_3D)}")

    def test_laplacian_jax_no_jit(self, capsys):
        lap = Laplacian(h=self.h)
        result = lap(self.f_jax)
        assert_array_almost_equal(np.asarray(result), self.expected, decimal=1)
        lap(self.f_jax).block_until_ready()
        t = timeit(lambda: lap(self.f_jax).block_until_ready(), number=REPS_3D)
        with capsys.disabled():
            print(f"\n  3D Laplacian ({self.M}^3) {_fmt('jax (no jit)', t, REPS_3D)}")

    def test_laplacian_jax_jit(self, capsys):
        lap = Laplacian(h=self.h)
        lap_jit = jax.jit(lap)
        result = lap_jit(self.f_jax).block_until_ready()
        assert_array_almost_equal(np.asarray(result), self.expected, decimal=1)
        t = timeit(lambda: lap_jit(self.f_jax).block_until_ready(), number=REPS_3D)
        with capsys.disabled():
            print(f"\n  3D Laplacian ({self.M}^3) {_fmt('jax (jit)', t, REPS_3D)}")


class TestBenchmarkDiff3D:

    @pytest.fixture(autouse=True)
    def setup(self):
        M = 100
        axes = [np.linspace(0, 2 * np.pi, M)] * 3
        h = [a[1] - a[0] for a in axes]
        self.dx = h[0]
        X, Y, Z = np.meshgrid(*axes, indexing="ij")
        self.f_np = np.sin(X) * np.sin(Y) * np.sin(Z)
        self.f_jax = jnp.array(self.f_np)
        self.expected = np.cos(X) * np.sin(Y) * np.sin(Z)
        self.M = M

    def test_diff_numpy(self, capsys):
        d_dx = Diff(0, self.dx)
        result = d_dx(self.f_np)
        assert_array_almost_equal(result, self.expected, decimal=1)
        t = timeit(lambda: d_dx(self.f_np), number=REPS_3D)
        with capsys.disabled():
            print(f"\n  3D Diff ({self.M}^3) {_fmt('numpy', t, REPS_3D)}")

    def test_diff_jax_jit(self, capsys):
        d_dx = Diff(0, self.dx)
        d_dx_jit = jax.jit(d_dx)
        result = d_dx_jit(self.f_jax).block_until_ready()
        assert_array_almost_equal(np.asarray(result), self.expected, decimal=1)
        t = timeit(lambda: d_dx_jit(self.f_jax).block_until_ready(), number=REPS_3D)
        with capsys.disabled():
            print(f"\n  3D Diff ({self.M}^3) {_fmt('jax (jit)', t, REPS_3D)}")


# ---------------------------------------------------------------------------
# Second-order derivative benchmark
# ---------------------------------------------------------------------------


class TestBenchmarkSecondOrder3D:

    @pytest.fixture(autouse=True)
    def setup(self):
        M = 100
        axes = [np.linspace(0, 2 * np.pi, M)] * 3
        h = [a[1] - a[0] for a in axes]
        self.dx = h[0]
        X, Y, Z = np.meshgrid(*axes, indexing="ij")
        self.f_np = np.sin(X) * np.sin(Y) * np.sin(Z)
        self.f_jax = jnp.array(self.f_np)
        self.expected = -np.sin(X) * np.sin(Y) * np.sin(Z)
        self.M = M

    def test_d2_numpy(self, capsys):
        d2 = Diff(0, self.dx) ** 2
        result = d2(self.f_np)
        assert_array_almost_equal(result, self.expected, decimal=1)
        t = timeit(lambda: d2(self.f_np), number=REPS_3D)
        with capsys.disabled():
            print(f"\n  3D d²/dx² ({self.M}^3) {_fmt('numpy', t, REPS_3D)}")

    def test_d2_jax_jit(self, capsys):
        d2 = Diff(0, self.dx) ** 2
        d2_jit = jax.jit(d2)
        result = d2_jit(self.f_jax).block_until_ready()
        assert_array_almost_equal(np.asarray(result), self.expected, decimal=1)
        t = timeit(lambda: d2_jit(self.f_jax).block_until_ready(), number=REPS_3D)
        with capsys.disabled():
            print(f"\n  3D d²/dx² ({self.M}^3) {_fmt('jax (jit)', t, REPS_3D)}")
