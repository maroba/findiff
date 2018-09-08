import unittest
import numpy as np
from findiff.findiff import FinDiff, Coef, Identity


class TestIdentity(unittest.TestCase):

    def test_identity(self):

        x = np.linspace(-1, 1, 100)
        u = x**2
        identity = Identity()

        np.testing.assert_array_equal(u, identity(u))

        twice_id = Coef(2) * Identity()
        np.testing.assert_array_equal(2 * u, twice_id(u))

        x_id = Coef(x) * Identity()
        np.testing.assert_array_equal(x * u, x_id(u))

    def test_identity_2d(self):

        x = np.linspace(-1, 1, 100)
        y = np.linspace(-1, 1, 100)

        X, Y = np.meshgrid(x, y, indexing='ij')
        u = X ** 2 + Y ** 2
        identity = Identity()

        np.testing.assert_array_equal(u, identity(u))

        twice_id = Coef(2) * Identity()
        np.testing.assert_array_equal(2 * u, twice_id(u))

        x_id = Coef(X) * Identity()
        np.testing.assert_array_equal(X * u, x_id(u))

        dx = x[1] - x[0]
        d_dx = FinDiff(0, 1)
        linop = d_dx + 2 * Identity()
        np.testing.assert_array_almost_equal(2 * X + 2*u, linop(u, spac=[dx]))


if __name__ == '__main__':
    unittest.main()
