import sys
sys.path.insert(1, '..')

import unittest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
from findiff.operators import FinDiff, Coef, Identity


class FinDiffTest(unittest.TestCase):

    def test_partial_diff_1d(self):
        nx = 11
        x = np.linspace(0, 1, nx)
        u = x**3
        ux_ex = 3*x**2

        fd = FinDiff(0, x[1] - x[0], 1)

        ux = fd(u, acc=4)

        assert_array_almost_equal(ux, ux_ex, decimal=5)


    def test_partial_diff(self):
        nx = 100
        x = np.linspace(0, np.pi, nx)
        u = np.sin(x)
        ux_ex = np.cos(x)

        fd = FinDiff(0, x[1] - x[0], 1)

        ux = fd(u, acc=4)

        assert_array_almost_equal(ux, ux_ex, decimal=5)

        ny = 100
        y = np.linspace(0, np.pi, ny)
        X, Y = np.meshgrid(x, y, indexing='ij')

        u = np.sin(X) * np.sin(Y)
        uxy_ex = np.cos(X) * np.cos(Y)

        fd = FinDiff((0, x[1] - x[0], 1), (1, y[1] - y[0], 1))

        uxy = fd(u, acc=4)

        assert_array_almost_equal(uxy, uxy_ex, decimal=5)

    def test_plus(self):

        (X, Y), _, h = grid(2, 50, 0, 1)

        u = X**2 + Y**2
        d_dx = FinDiff(0, h[0])
        d_dy = FinDiff(1, h[1])

        d = d_dx + d_dy

        u1 = d(u)
        u1_ex = 2*X + 2*Y

        assert_array_almost_equal(u1, u1_ex)

    def test_multiply(self):

        (X, Y), _, h = grid(2, 5, 0, 1)

        u = X**2 + Y**2
        d2_dx2 = FinDiff(0, h[0], 2)

        d = Coef(X) * d2_dx2

        u1 = d(u)
        assert_array_almost_equal(u1, 2*X)

    def test_multiply_operators(self):

        (X, Y), _, h = grid(2, 50, 0, 1)

        u = X**2 + Y**2
        d_dx = FinDiff(0, h[0])

        d2_dx2_test = d_dx * d_dx

        uxx = d2_dx2_test(u)

        assert_array_almost_equal(uxx, np.ones_like(X)*2)

    def test_laplace(self):

        (X, Y, Z), _, h = grid(3, 50, 0, 1)

        u = X**3 + Y**3 + Z**3

        d2_dx2, d2_dy2, d2_dz2 = [FinDiff(i, h[i], 2) for i in range(3)]

        laplace = d2_dx2 + d2_dy2 + d2_dz2

        lap_u = laplace(u)
        assert_array_almost_equal(lap_u, 6*X + 6*Y + 6*Z)

        d_dx, d_dy, d_dz = [FinDiff(i, h[i]) for i in range(3)]

        d = Coef(X) * d_dx + Coef(Y) * d_dy + Coef(Z) * d_dz

        f = d(lap_u)

        d2 = d * laplace
        f2 = d2(u)

        assert_array_almost_equal(f2, f)
        assert_array_almost_equal(f2, 6 * (X + Y + Z))

    def test_identity(self):

        x = np.linspace(-1, 1, 100)
        u = x**2
        identity = Identity()

        assert_array_equal(u, identity(u))

        twice_id = Coef(2) * Identity()
        assert_array_equal(2 * u, twice_id(u))

        x_id = Coef(x) * Identity()
        assert_array_equal(x * u, x_id(u))

    def test_identity_2d(self):

        (X, Y), (x, y), _ = grid(2, 100, -1, 1)

        u = X ** 2 + Y ** 2
        identity = Identity()

        assert_array_equal(u, identity(u))

        twice_id = Coef(2) * Identity()
        assert_array_equal(2 * u, twice_id(u))

        x_id = Coef(X) * Identity()
        assert_array_equal(X * u, x_id(u))

        dx = x[1] - x[0]
        d_dx = FinDiff(0, dx)
        linop = d_dx + 2 * Identity()
        assert_array_almost_equal(2 * X + 2*u, linop(u))

    def test_spac(self):

        (X, Y), _, (dx, dy) = grid(2, 100, -1, 1)

        u = X**2 + Y**2

        d_dx = FinDiff(0, dx)
        d_dy = FinDiff(1, dy)

        assert_array_almost_equal(2*X, d_dx(u))
        assert_array_almost_equal(2*Y, d_dy(u))

        d_dx = FinDiff(0, dx)
        d_dy = FinDiff(1, dy)

        u = X*Y
        d2_dxdy = d_dx * d_dy

        assert_array_almost_equal(np.ones_like(u), d2_dxdy(u))

    def test_mixed_partials(self):

        (X, Y, Z), _, (dx, dy, dz) = grid(3, 50, 0, 1)

        u = X**2 * Y**2 * Z**2

        d3_dxdydz = FinDiff((0, dx), (1, dy), (2, dz))
        diffed = d3_dxdydz(u)
        assert_array_almost_equal(8*X*Y*Z, diffed)

    def test_linear_combinations(self):
        (X, Y, Z), _, (dx, dy, dz) = grid(3, 30, 0, 1)

        u = X ** 2 + Y ** 2 + Z ** 2
        d = Coef(X) * FinDiff(0, dx) + Coef(Y**2) * FinDiff(1, dy, 2)
        assert_array_almost_equal(d(u), 2*X**2 + 2*Y**2)

    def test_minus(self):

        (X, Y), _, h = grid(2, 50, 0, 1)

        u = X**2 + Y**2

        d_dx = FinDiff(0, h[0])
        d_dy = FinDiff(1, h[1])

        d = d_dx - d_dy

        u1 = d(u)
        u1_ex = 2*X - 2*Y

        assert_array_almost_equal(u1, u1_ex)

    def test_local_stencil_single_axis_center_1d(self):

        x = np.linspace(0, 1, 50)
        dx = x[1] - x[0]
        u = x**3
        d2_dx2 = FinDiff((0, dx, 2))

        stl = d2_dx2.stencil(u.shape)
        idx = 5
        actual = stl.apply(u, idx)

        d2u_dx2 = d2_dx2(u)
        expected = d2u_dx2[idx]

        self.assertAlmostEqual(expected, actual)

        actual = stl.apply_all(u)
        expected = d2u_dx2

        np.testing.assert_array_almost_equal(expected, actual)

    def test_local_stencil_single_axis_center_2d_compared_with_findiff(self):
        n = 70
        (X, Y), _, (dx, dy) = grid(2, n, -1, 1)

        u = X ** 3 * Y ** 3

        d4_dx2dy2 = FinDiff(1, dx, 2)
        expected = d4_dx2dy2(u)

        stl = d4_dx2dy2.stencil(u.shape)

        actual = stl.apply_all(u)

        np.testing.assert_array_almost_equal(expected, actual)

    def test_local_stencil_operator_addition(self):

        n = 100
        (X, Y), _, (dx, dy) = grid(2, n, -1, 1)

        u = X**3 + Y**3

        d = FinDiff(0, dx, 2) + FinDiff(1, dy, 2)
        expected = d(u)

        stl = d.stencil(u.shape)

        actual = stl.apply_all(u)
        np.testing.assert_array_almost_equal(expected, actual)


    def dict_almost_equal(self, d1, d2):

        self.assertEqual(len(d1), len(d2))

        for k, v in d1.items():
            self.assertAlmostEqual(v, d2[k])

    def test_matrix_1d(self):

        x = np.linspace(0, 6, 7)
        d2_dx2 = FinDiff(0, x[1]-x[0], 2)
        u = x**2

        mat = d2_dx2.matrix(u.shape)

        np.testing.assert_array_almost_equal(2*np.ones_like(x), mat.dot(u.reshape(-1)))

    def test_matrix_2d(self):
        thr = np.get_printoptions()["threshold"]
        lw = np.get_printoptions()["linewidth"]
        np.set_printoptions(threshold=np.inf)
        np.set_printoptions(linewidth=500)
        x, y = [np.linspace(0, 4, 5)] * 2
        X, Y = np.meshgrid(x, y, indexing='ij')
        laplace = FinDiff(0, x[1]-x[0], 2) + FinDiff(0, y[1]-y[0], 2)
        #d = FinDiff(1, y[1]-y[0], 2)
        u = X**2 + Y**2

        mat = laplace.matrix(u.shape)

        np.testing.assert_array_almost_equal(4 * np.ones_like(X).reshape(-1), mat.dot(u.reshape(-1)))

        np.set_printoptions(threshold=thr)
        np.set_printoptions(linewidth=lw)

    def test_matrix_2d_mixed(self):
        x, y = [np.linspace(0, 5, 6), np.linspace(0, 6, 7)]
        X, Y = np.meshgrid(x, y, indexing='ij')
        d2_dxdy = FinDiff((0, x[1] - x[0]), (1, y[1] - y[0]))
        u = X**2 * Y**2

        mat = d2_dxdy.matrix(u.shape)
        expected = d2_dxdy(u).reshape(-1)

        actual = mat.dot(u.reshape(-1))
        np.testing.assert_array_almost_equal(expected, actual)

    def test_matrix_1d_coeffs(self):
        shape = 11,
        x = np.linspace(0, 10, 11)
        dx = x[1] - x[0]

        L = Coef(x) * FinDiff(0, dx, 2)

        u = np.random.rand(*shape).reshape(-1)

        actual = L.matrix(shape).dot(u)
        expected = L(u).reshape(-1)
        np.testing.assert_array_almost_equal(expected, actual)

    @unittest.skip
    def test_eigs(self):

        shape = 8,
        x = np.linspace(-10, 10, shape[0])
        dx = x[1] - x[0]

        hbar = 1
        m = 1
        omega = 1
        T = Coef(-hbar**2/(2*m)) * FinDiff(0, dx, 2)
        V = Coef(0.5*omega*x**2) * Identity()
        H = T + V
        print("\n",T.matrix(shape).toarray())
        print("\n",V.matrix(shape).toarray())
        print("\n",H.matrix(shape).toarray())

        print(H.matrix(shape).toarray())
        vals, vecs = H.eigs(shape)
        print(vals)
        print(vecs)


class TestFinDiffNonUniform(unittest.TestCase):

    def test_1d_different_accs(self):
        x = np.r_[np.arange(0, 4, 0.05), np.arange(4, 10, 1)]
        f = np.exp(-x**2)

        d_dx = FinDiff(0, x, acc=4)
        f_x = d_dx(f)
        assert_array_almost_equal(- 2*x * np.exp(-x**2), f_x, decimal=4)

    def test_3d_different_accs(self):
        x = np.r_[np.arange(0, 4, 0.05), np.arange(4, 10, 1)]
        y = np.r_[np.arange(0, 4, 0.05), np.arange(4, 10, 1)]
        z = np.r_[np.arange(0, 4.5, 0.05), np.arange(4.5, 10, 1)]
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        f = np.exp(-X**2-Y**2-Z**2)

        d_dy = FinDiff(1, y, acc=4)
        fy = d_dy(f)
        fye = - 2 * Y * np.exp(-X**2-Y**2-Z**2)
        assert_array_almost_equal(fy, fye, decimal=4)

        d_dy = FinDiff(1, y, acc=6)
        fy = d_dy(f)
        assert_array_almost_equal(fy, fye, decimal=4)

    def test_1d_linear_combination(self):
        x = np.sqrt(np.linspace(0, 1, 10))
        d1 = Coef(x) * FinDiff(0, x, acc=4)
        d2 = Coef(x**2) * FinDiff(0, x, 2, acc=4)

        assert_array_almost_equal(3*x**3, d1(x**3))
        assert_array_almost_equal(6*x**3, d2(x**3))
        assert_array_almost_equal(9*x**3, (d1 + d2)(x**3))

    def test_3d_linear_combination(self):
        x, y, z = [np.sqrt(np.linspace(0, 1, 10))] * 3
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

        d1 = Coef(X) * FinDiff(0, x, acc=4)
        d2 = Coef(Y) * FinDiff(1, y, acc=4)

        assert_array_almost_equal(3 * X ** 3, d1(X**3 + Y**3 + Z**3))
        assert_array_almost_equal(3 * Y ** 3, d2(X**3 + Y**3 + Z**3))
        assert_array_almost_equal(3 * (X**3 + Y**3), (d1 + d2)(X**3 + Y**3 + Z**3))

    def test_several_tuples_as_args(self):

        x, y, z = [np.sqrt(np.linspace(0, 1, 10))] * 3
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

        d1 = FinDiff((0, x), (1, y), acc=4)
        assert_array_almost_equal(d1(X*Y), np.ones_like(X))

    def test_accepts_not_more_than_three_args(self):
        x = np.linspace(0, 1, 10)
        self.assertRaises(Exception, lambda: FinDiff(0, x, 2, 3))



def grid(ndim, npts, a, b):

    if not hasattr(a, "__len__"):
        a = [a] * ndim
    if not hasattr(b, "__len__"):
        b = [b] * ndim
    if not hasattr(np, "__len__"):
        npts = [npts] * ndim

    coords = [np.linspace(a[i], b[i], npts[i]) for i in range(ndim)]
    mesh = np.meshgrid(*coords, indexing='ij')
    spac = [coords[i][1] - coords[i][0] for i in range(ndim)]

    return mesh, coords, spac


if __name__ == '__main__':
    unittest.main()