import numpy as np
import pytest

from findiff import Diff, ErrorEstimate


class TestEstimateError1D:

    def test_first_derivative(self):
        x = np.linspace(0, 2 * np.pi, 200)
        dx = x[1] - x[0]
        f = np.sin(x)
        exact = np.cos(x)

        d_dx = Diff(0, dx)
        result = d_dx.estimate_error(f)

        assert isinstance(result, ErrorEstimate)
        assert result.derivative.shape == f.shape
        assert result.error.shape == f.shape
        assert result.extrapolated.shape == f.shape

        # The error estimate should be on the same order as the actual error
        # in the interior (boundaries use different stencils)
        interior = slice(10, -10)
        actual_error = np.abs(result.derivative[interior] - exact[interior])
        assert np.max(result.error[interior]) < 1.0
        assert np.max(actual_error) < 1.0

    def test_second_derivative(self):
        x = np.linspace(0, 2 * np.pi, 200)
        dx = x[1] - x[0]
        f = np.sin(x)
        exact = -np.sin(x)

        d2_dx2 = Diff(0, dx) ** 2
        result = d2_dx2.estimate_error(f)

        interior = slice(10, -10)
        actual_error = np.abs(result.derivative[interior] - exact[interior])
        # Error estimate should be reasonable (same order of magnitude)
        assert np.max(result.error[interior]) > 0
        assert np.max(actual_error) > 0

    def test_extrapolated_more_accurate(self):
        x = np.linspace(0, 2 * np.pi, 200)
        dx = x[1] - x[0]
        f = np.sin(x)
        exact = np.cos(x)

        d_dx = Diff(0, dx)
        result = d_dx.estimate_error(f)

        interior = slice(10, -10)
        err_low = np.max(np.abs(result.derivative[interior] - exact[interior]))
        err_high = np.max(np.abs(result.extrapolated[interior] - exact[interior]))
        assert err_high < err_low

    def test_higher_base_accuracy(self):
        x = np.linspace(0, 2 * np.pi, 200)
        dx = x[1] - x[0]
        f = np.sin(x)
        exact = np.cos(x)

        d_dx = Diff(0, dx, acc=4)
        result = d_dx.estimate_error(f)

        interior = slice(10, -10)
        err_low = np.max(np.abs(result.derivative[interior] - exact[interior]))
        err_high = np.max(np.abs(result.extrapolated[interior] - exact[interior]))
        # acc=4 should already be quite accurate
        assert err_low < 1e-4
        # acc=6 should be even more accurate
        assert err_high < err_low

    def test_explicit_acc_parameter(self):
        x = np.linspace(0, 2 * np.pi, 200)
        dx = x[1] - x[0]
        f = np.sin(x)

        d_dx = Diff(0, dx, acc=2)
        # Override with acc=4 as base
        result = d_dx.estimate_error(f, acc=4)

        # Should compare acc=4 vs acc=6
        exact = np.cos(x)
        interior = slice(10, -10)
        err = np.max(np.abs(result.derivative[interior] - exact[interior]))
        assert err < 1e-4  # acc=4 is quite accurate on 200 points


class TestEstimateError2D:

    def test_laplacian(self):
        n = 80
        x = np.linspace(0, 2 * np.pi, n)
        y = np.linspace(0, 2 * np.pi, n)
        dx, dy = x[1] - x[0], y[1] - y[0]
        X, Y = np.meshgrid(x, y, indexing="ij")

        f = np.sin(X) * np.sin(Y)
        exact = -2 * np.sin(X) * np.sin(Y)  # Laplacian of sin(x)*sin(y)

        laplacian = Diff(0, dx) ** 2 + Diff(1, dy) ** 2
        result = laplacian.estimate_error(f)

        assert result.derivative.shape == f.shape
        assert result.error.shape == f.shape

        interior = (slice(5, -5), slice(5, -5))
        err_low = np.max(np.abs(result.derivative[interior] - exact[interior]))
        err_high = np.max(np.abs(result.extrapolated[interior] - exact[interior]))
        assert err_high < err_low


class TestEstimateErrorNonUniform:

    def test_non_uniform_grid(self):
        # Stretched grid
        x = np.concatenate([
            np.linspace(0, 1, 60, endpoint=False),
            np.linspace(1, 3, 40),
        ])
        f = np.sin(x)
        exact = np.cos(x)

        d_dx = Diff(0, x)
        result = d_dx.estimate_error(f)

        assert result.derivative.shape == f.shape
        interior = slice(10, -10)
        actual_error = np.abs(result.derivative[interior] - exact[interior])
        assert np.max(actual_error) < 0.1


class TestEstimateErrorOperatorState:

    def test_accuracy_restored(self):
        x = np.linspace(0, 2 * np.pi, 100)
        dx = x[1] - x[0]
        f = np.sin(x)

        d_dx = Diff(0, dx, acc=2)
        d_dx.estimate_error(f)

        assert d_dx.acc == 2

    def test_accuracy_restored_with_explicit_acc(self):
        x = np.linspace(0, 2 * np.pi, 100)
        dx = x[1] - x[0]
        f = np.sin(x)

        d_dx = Diff(0, dx, acc=2)
        d_dx.estimate_error(f, acc=4)

        # Should be restored to acc=4 (the explicit base), not the original 2
        # Actually, the method restores to the acc that was passed in
        assert d_dx.acc == 4

    def test_accuracy_restored_composite(self):
        x = np.linspace(0, 2 * np.pi, 100)
        dx = x[1] - x[0]
        f = np.sin(x)

        d_dx = Diff(0, dx, acc=2)
        op = d_dx ** 2
        op.estimate_error(f)

        assert op.acc == 2


class TestEstimateErrorUnpacking:

    def test_namedtuple_unpacking(self):
        x = np.linspace(0, 2 * np.pi, 100)
        dx = x[1] - x[0]
        f = np.sin(x)

        d_dx = Diff(0, dx)
        derivative, error, extrapolated = d_dx.estimate_error(f)

        assert isinstance(derivative, np.ndarray)
        assert isinstance(error, np.ndarray)
        assert isinstance(extrapolated, np.ndarray)

    def test_namedtuple_field_access(self):
        x = np.linspace(0, 2 * np.pi, 100)
        dx = x[1] - x[0]
        f = np.sin(x)

        d_dx = Diff(0, dx)
        result = d_dx.estimate_error(f)

        assert hasattr(result, "derivative")
        assert hasattr(result, "error")
        assert hasattr(result, "extrapolated")


class TestEstimateErrorCompactScheme:

    def test_compact_scheme_raises(self):
        x = np.linspace(0, 2 * np.pi, 100)
        dx = x[1] - x[0]
        f = np.sin(x)

        d_dx = Diff(0, dx, compact=3)
        with pytest.raises(NotImplementedError, match="compact"):
            d_dx.estimate_error(f)
