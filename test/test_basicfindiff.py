import findiff as fd
import unittest
import numpy as np


class TestUtilities(object):

    def _prep_1d_func(self, x0=-5, x1=5, nx=100):
        x = np.linspace(x0, x1, nx)
        dx = x[1] - x[0]
        f = x**3
        fxe = 3*x**2
        fxxe = 6*x
        return dx, f, fxe, fxxe

    def _assertAlmostEqual(self, f1, f2, tol=7):
        err = np.max(np.abs(f1-f2))
        self.assertAlmostEqual(0, err, tol)


class TestBasicFinDiffNew(unittest.TestCase):

    def test_diff_1d_ord1_acc2(self):
        x = np.linspace(-3, 3, 50)
        h = x[1] - x[0]
        y = x ** 2
        y1e = 2 * x
        df = fd.diff.BasicFinDiff((0, h))
        y1 = df(y)
        err = np.max(np.abs(y1e - y1))
        self.assertAlmostEqual(0, err)

    def test_diff_1d_ord1_acc4(self):
        x = np.linspace(-3, 3, 50)
        h = x[1] - x[0]
        y = x ** 2
        y1e = 2 * x
        y1 = fd.diff.BasicFinDiff((0, h, 1), acc=4)(y)
        err = np.max(np.abs(y1e - y1))
        self.assertAlmostEqual(0, err)

    def test_diff_1d_ord2_acc2(self):
        x = np.linspace(-3, 3, 50)
        h = x[1] - x[0]
        y = x ** 3
        y2e = 6 * x
        y2 = fd.diff.BasicFinDiff((0, h, 2), acc=2)(y)
        err = np.max(np.abs(y2e - y2))
        self.assertAlmostEqual(0, err)

    def test_diff_1d_ord2_acc4(self):
        x = np.linspace(-3, 3, 50)
        h = x[1] - x[0]
        y = x ** 3
        y2e = 6 * x
        y2 = fd.diff.BasicFinDiff((0, h, 2), acc=4)(y)
        err = np.max(np.abs(y2e - y2))
        self.assertAlmostEqual(0, err)

    def test_diff_2d_ord1_acc2(self):
        x = np.linspace(-1, 1, 500)
        y = np.linspace(-1, 1, 500)
        dx = dy = x[1] - x[0]
        X, Y = np.meshgrid(x, y, indexing='ij')
        f = Y**3 * X**3
        fxe = 3 * X**2 * Y**3
        fx = fd.diff.BasicFinDiff((0, dx), acc=2)(f)
        err = np.max(np.abs(fxe - fx))
        self.assertAlmostEqual(0, err, 4)
        fye = 3 * X**3 * Y**2
        fy = fd.diff.BasicFinDiff((1, dy), acc=2)(f)
        err = np.max(np.abs(fye - fy))
        self.assertAlmostEqual(0, err, 4)

    def test_diff_2d_ord1_acc2_plain(self):
        x = np.linspace(-1, 1, 500)
        y = np.linspace(-1, 1, 500)
        dx = dy = x[1] - x[0]
        X, Y = np.meshgrid(x, y, indexing='ij')
        f = Y**3 * X**3
        fxe = 3 * X**2 * Y**3
        fx = fd.diff.BasicFinDiff(0, dx, acc=2)(f)
        err = np.max(np.abs(fxe - fx))
        self.assertAlmostEqual(0, err, 4)
        fye = 3 * X**3 * Y**2
        fy = fd.diff.BasicFinDiff(1, dy, acc=2)(f)
        err = np.max(np.abs(fye - fy))
        self.assertAlmostEqual(0, err, 4)

    def test_diff_2d_ord2_acc2(self):
        x = np.linspace(-1, 1, 500)
        y = np.linspace(-1, 1, 500)
        dx = dy = x[1] - x[0]
        X, Y = np.meshgrid(x, y, indexing='ij')
        f = Y**3 * X**3
        fxxe = 6 * X * Y**3
        fxx = fd.diff.BasicFinDiff((0, dx, 2), acc=2)(f)
        err = np.max(np.abs(fxxe - fxx))
        self.assertAlmostEqual(0, err, 4)
        fyye = 6 * X**3 * Y
        fyy = fd.diff.BasicFinDiff((1, dy, 2))(f)
        err = np.max(np.abs(fyye - fyy))
        self.assertAlmostEqual(0, err, 6)

    def test_diff_3d_ord1_acc2(self):
        x = np.linspace(-0.5, 0.5, 100)
        y = np.linspace(-0.5, 0.5, 100)
        z = np.linspace(-0.5, 0.5, 100)
        dx = dy = dz = x[1] - x[0]
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        f = Y**3 * X**3 * Z**3
        fxe = 3 * X**2 * Y**3 * Z**3
        fx = fd.diff.BasicFinDiff((0, dx))(f)
        err = np.max(np.abs(fxe - fx))
        self.assertAlmostEqual(0, err, 4)
        fye = 3 * X**3 * Y**2 * Z**3
        fy = fd.diff.BasicFinDiff((1, dy))(f)
        err = np.max(np.abs(fye - fy))
        self.assertAlmostEqual(0, err, 4)
        fze = 3 * X**3 * Y**3 * Z**2
        fz = fd.diff.BasicFinDiff((2, dz))(f)
        err = np.max(np.abs(fze - fz))
        self.assertAlmostEqual(0, err, 4)

    def test_diff_3d_ord2_acc2(self):
        x = np.linspace(-0.5, 0.5, 100)
        y = np.linspace(-0.5, 0.5, 100)
        z = np.linspace(-0.5, 0.5, 100)
        dx = dy = dz = x[1] - x[0]
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        f = Y**3 * X**3 * Z**3
        fxxe = 6 * X * Y**3 * Z**3
        fxx = fd.diff.BasicFinDiff((0, dx, 2))(f)
        err = np.max(np.abs(fxxe - fxx))
        self.assertAlmostEqual(0, err, 4)
        fyye = 6 * X**3 * Y * Z**3
        fyy = fd.diff.BasicFinDiff((1, dy, 2))(f)
        err = np.max(np.abs(fyye - fyy))
        self.assertAlmostEqual(0, err, 6)
        fzze = 6 * X**3 * Y**3 * Z
        fzz = fd.diff.BasicFinDiff((2, dz, 2))(f)
        err = np.max(np.abs(fzze - fzz))
        self.assertAlmostEqual(0, err, 6)

    def test_diff_3d_df_dxdy_acc4(self):
        x = np.linspace(-0.5, 0.5, 100)
        y = np.linspace(-0.5, 0.5, 100)
        z = np.linspace(-0.5, 0.5, 100)
        dx = dy = dz = x[1] - x[0]
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        f = Y**3 * X**3 * Z**3
        fxye = 9 * X**2 * Y**2 * Z**3
        fxy = fd.diff.BasicFinDiff((0, dx), (1, dy), acc=4)(f)
        err = np.max(np.abs(fxye - fxy))
        self.assertAlmostEqual(0, err, 4)

    def test_diff_3d_df_dxdy_acc6(self):
        x = np.linspace(-12.5, 12.5, 25)
        y = np.linspace(-12.5, 12.5, 25)
        z = np.linspace(-12.5, 12.5, 25)
        dx = dy = dz = x[1] - x[0]
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        f = Y**3 * X**3 * Z**3
        fxye = 9 * X**2 * Y**2 * Z**3
        fxy = fd.diff.BasicFinDiff((0, dx), (1, dy), acc=4)(f)
        err = np.max(np.abs(fxye - fxy))
        self.assertAlmostEqual(0, err, 4)


class TestFinDiffNonUniform(unittest.TestCase, TestUtilities):

    def test_BasicFinDiffNonUni_equi(self):
        x = np.linspace(-3, 3, 100)
        f = x**2

        d_dx = fd.diff.BasicFinDiffNonUniform((0, x))
        fx = d_dx(f)
        fxe = 2*x
        self._assertAlmostEqual(fx, fxe)

    def test_BasicFinDiffNonUni(self):
        x = np.r_[np.arange(0, 4, 0.005), np.arange(4, 10, 1)]
        f = np.exp(-x**2)

        d_dx = fd.diff.BasicFinDiffNonUniform((0, x, 1))
        fx = d_dx(f)
        fxe = - 2 * x * np.exp(-x**2)
        self._assertAlmostEqual(fx, fxe, 4)

    def test_BasicFinDiffNonUni_2d(self):
        x = np.r_[np.arange(0, 4, 0.005), np.arange(4, 10, 1)]
        y = np.r_[np.arange(0, 4, 0.005), np.arange(4, 10, 1)]
        X, Y = np.meshgrid(x, y, indexing='ij')
        f = np.exp(-X**2-Y**2)

        d_dx = fd.diff.BasicFinDiffNonUniform((0, x))
        fx = d_dx(f)
        fxe = - 2 * X * np.exp(-X**2-Y**2)
        self._assertAlmostEqual(fx, fxe, 4)

    def test_BasicFinDiffNonUni_3d(self):
        x = np.r_[np.arange(0, 4, 0.05), np.arange(4, 10, 1)]
        y = np.r_[np.arange(0, 4, 0.05), np.arange(4, 10, 1)]
        z = np.r_[np.arange(0, 4.5, 0.05), np.arange(4.5, 10, 1)]
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        f = np.exp(-X**2-Y**2-Z**2)

        d_dy = fd.diff.BasicFinDiffNonUniform((1, y), acc=4)
        fy = d_dy(f)
        fye = - 2 * Y * np.exp(-X**2-Y**2-Z**2)
        self._assertAlmostEqual(fy, fye, 4)

if __name__ == '__main__':
    unittest.main()