import unittest
import numpy as np
from findiff.findiff import FinDiff, Coef


class FinDiffTest(unittest.TestCase):

    def test_partial_diff_1d(self):
        nx = 100
        x = np.linspace(0, np.pi, nx)
        u = np.sin(x)
        ux_ex = np.cos(x)

        fd = FinDiff(0, 1)
        fd.spac = [x[1] - x[0]]
        fd.acc = 4

        ux = fd(u)

        np.testing.assert_array_almost_equal(ux, ux_ex, decimal=5)

        ny = 100
        y = np.linspace(0, np.pi, ny)
        X, Y = np.meshgrid(x, y, indexing='ij')

        u = np.sin(X) * np.sin(Y)
        uxy_ex = np.cos(X) * np.cos(Y)

        fd = FinDiff((0, 1), (1, 1))
        fd.spac = [x[1] - x[0], y[1] - y[0]]
        fd.acc = 4

        uxy = fd(u)

        np.testing.assert_array_almost_equal(uxy, uxy_ex, decimal=5)

    def test_plus(self):

        x, y = [np.linspace(0, 1, 50)] * 2
        X, Y = np.meshgrid(x, y, indexing='ij')

        u = X**2 + Y**2
        d_dx = FinDiff(0, 1)
        d_dy = FinDiff(1, 1)

        d = d_dx + d_dy

        u1 = d(u, spac=[x[1] - x[0]]*2)
        u1_ex = 2*X + 2*Y

        np.testing.assert_array_almost_equal(u1, u1_ex)

    def test_multiply(self):

        x, y = [np.linspace(0, 1, 50)] * 2
        X, Y = np.meshgrid(x, y, indexing='ij')

        u = X**2 + Y**2
        d2_dx2 = FinDiff(0, 2)

        d = Coef(X) * d2_dx2

        h = [x[1] - x[0]]*2
        u1 = d(u, spac=h)
        np.testing.assert_array_almost_equal(u1, 2*X)

    def test_multiply_operators(self):

        x, y = [np.linspace(0, 1, 50)] * 2
        X, Y = np.meshgrid(x, y, indexing='ij')
        h = [x[1] - x[0]] * 2

        u = X**2 + Y**2
        d_dx = FinDiff(0, 1)

        d2_dx2_test = d_dx * d_dx

        uxx = d2_dx2_test(u, spac=h)

        np.testing.assert_array_almost_equal(uxx, np.ones_like(X)*2)

    def test_laplace(self):

        x, y, z = [np.linspace(0, 1, 50)] * 3
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        h = [c[1] - c[0] for c in (x, y, z)]

        u = X**3 + Y**3 + Z**3

        d2_dx2, d2_dy2, d2_dz2 = [FinDiff(i, 2) for i in range(3)]

        laplace = d2_dx2 + d2_dy2 + d2_dz2

        lap_u = laplace(u, spac=h)
        np.testing.assert_array_almost_equal(lap_u, 6*X + 6*Y + 6*Z)

        d_dx, d_dy, d_dz = [FinDiff(i, 1) for i in range(3)]

        d = Coef(X) * d_dx + Coef(Y) * d_dy + Coef(Z) * d_dz

        f = d(lap_u, spac=h)

        d2 = d * laplace
        f2 = d2(u, spac=h)

        np.testing.assert_array_almost_equal(f2, f)
        np.testing.assert_array_almost_equal(f2, 6 * (X + Y + Z))

    def test_non_uniform_3d(self):
        x = np.r_[np.arange(0, 4, 0.05), np.arange(4, 10, 1)]
        y = np.r_[np.arange(0, 4, 0.05), np.arange(4, 10, 1)]
        z = np.r_[np.arange(0, 4.5, 0.05), np.arange(4.5, 10, 1)]
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        f = np.exp(-X**2-Y**2-Z**2)

        d_dy = FinDiff(1, 1, acc=4)
        fy = d_dy(f, coords=[x, y, z])
        fye = - 2 * Y * np.exp(-X**2-Y**2-Z**2)
        np.testing.assert_array_almost_equal(fy, fye, decimal=4)


if __name__ == '__main__':
    unittest.main()