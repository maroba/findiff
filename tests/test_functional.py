import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from findiff import Diff

pytestmark = pytest.mark.functional


@pytest.mark.one_dimensional
class TestDiff_1D:

    @pytest.mark.grid_spec(shape=(101,), edges=(0, 1))
    def test_diff_1d(self, grid_data):
        x, dx = grid_data

        u = x**2
        expected = 2 * x

        fd = Diff(0, dx)
        actual = fd(u)

        assert_array_almost_equal(expected, actual)

    @pytest.mark.grid_spec(shape=(101,), edges=(0, 1))
    def test_diff_1d_deferred(self, grid_data):
        x, dx = grid_data

        u = x**2
        expected = 2 * x

        fd = Diff(0)
        fd.set_grid({0: dx})
        actual = fd(u)

        assert_array_almost_equal(expected, actual)

    def test_diff_1d_deferred_called_too_early(self):
        u = np.zeros(10)
        fd = Diff(0)
        with pytest.raises(TypeError, match="Unknown axis type."):
            fd(u)

    @pytest.mark.default_args
    @pytest.mark.grid_spec(shape=(101,), edges=(0, 1))
    def test_diff_1d_defaults(self, grid_data):
        x, dx = grid_data

        u = x**2
        expected = 2 * x

        fd = Diff()
        fd.set_grid({0: dx})
        actual = fd(u)

        assert_array_almost_equal(expected, actual)

    @pytest.mark.invalid_args
    def test_diff_1d_invalid_args(self):

        with pytest.raises(ValueError, match="Dimension must be >= 0"):
            Diff(-1, 1)

        with pytest.raises(ValueError, match="Spacing must be > 0."):
            Diff(0, -1)
