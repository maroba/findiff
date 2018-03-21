import findiff as fd #.diff as fd
import unittest
import numpy as np


class TestFinDiff(unittest.TestCase):

    def test_diff_1d_ord1_acc2(self):
        x = np.linspace(-3, 3, 50)
        h = x[1] - x[0]
        y = x ** 2
        y1e = 2 * x
        df = fd.FinDiff(h, dims=[0], acc=2)
        y1 = df(y)
        err = np.max(np.abs(y1e - y1))
        self.assertAlmostEqual(0, err)

    def test_diff_1d_ord1_acc4(self):
        x = np.linspace(-3, 3, 50)
        h = x[1] - x[0]
        y = x ** 2
        y1e = 2 * x
        y1 = fd.FinDiff(h=h, dims=[0], acc=4)(y)
        err = np.max(np.abs(y1e - y1))
        self.assertAlmostEqual(0, err)

    def test_diff_1d_ord2_acc2(self):
        x = np.linspace(-3, 3, 50)
        h = x[1] - x[0]
        y = x ** 3
        y2e = 6 * x
        y2 = fd.FinDiff(h=h, dims=[0, 0], acc=2)(y)
        err = np.max(np.abs(y2e - y2))
        self.assertAlmostEqual(0, err)

    def test_diff_1d_ord2_acc4(self):
        x = np.linspace(-3, 3, 50)
        h = x[1] - x[0]
        y = x ** 3
        y2e = 6 * x
        y2 = fd.FinDiff(h=h, dims=[0, 0], acc=4)(y)
        err = np.max(np.abs(y2e - y2))
        self.assertAlmostEqual(0, err)

    def test_diff_2d_ord1_acc2(self):
        x = np.linspace(-1, 1, 500)
        y = np.linspace(-1, 1, 500)
        dx = dy = x[1] - x[0]
        X, Y = np.meshgrid(x, y, indexing='ij')
        f = Y**3 * X**3
        fxe = 3 * X**2 * Y**3
        fx = fd.FinDiff(h=[dx, dy], dims=[0], acc=2)(f)
        err = np.max(np.abs(fxe - fx))
        self.assertAlmostEqual(0, err, 4)
        fye = 3 * X**3 * Y**2
        fy = fd.FinDiff(h=[dx, dy], dims=[1], acc=2)(f)
        err = np.max(np.abs(fye - fy))
        self.assertAlmostEqual(0, err, 4)

    def test_diff_2d_ord2_acc2(self):
        x = np.linspace(-1, 1, 500)
        y = np.linspace(-1, 1, 500)
        dx = dy = x[1] - x[0]
        X, Y = np.meshgrid(x, y, indexing='ij')
        f = Y**3 * X**3
        fxxe = 6 * X * Y**3
        fxx = fd.FinDiff(h=[dx, dy], dims=[0, 0], acc=2)(f)
        err = np.max(np.abs(fxxe - fxx))
        self.assertAlmostEqual(0, err, 4)
        fyye = 6 * X**3 * Y
        fyy = fd.FinDiff(acc=2, h=[dx, dy], dims=[1, 1])(f)
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
        fx = fd.FinDiff(acc=2, h=[dx, dy, dz], dims=[0])(f)
        err = np.max(np.abs(fxe - fx))
        self.assertAlmostEqual(0, err, 4)
        fye = 3 * X**3 * Y**2 * Z**3
        fy = fd.FinDiff(acc=2, h=[dx, dy, dz], dims=[1])(f)
        err = np.max(np.abs(fye - fy))
        self.assertAlmostEqual(0, err, 4)
        fze = 3 * X**3 * Y**3 * Z**2
        fz = fd.FinDiff(acc=2, h=[dx, dy, dz], dims=[2])(f)
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
        fxx = fd.FinDiff(acc=2, h=[dx, dy, dz], dims=[0, 0])(f)
        err = np.max(np.abs(fxxe - fxx))
        self.assertAlmostEqual(0, err, 4)
        fyye = 6 * X**3 * Y * Z**3
        fyy = fd.FinDiff(acc=2, h=[dx, dy, dz], dims=[1, 1])(f)
        err = np.max(np.abs(fyye - fyy))
        self.assertAlmostEqual(0, err, 6)
        fzze = 6 * X**3 * Y**3 * Z
        fzz = fd.FinDiff(acc=2, h=[dx, dy, dz], dims=[2, 2])(f)
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
        fxy = fd.FinDiff(acc=4, h=[dx, dy, dz], dims=[0, 1])(f)
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
        fxy = fd.FinDiff(acc=4, h=[dx, dy, dz], dims=[0, 1])(f)
        err = np.max(np.abs(fxye - fxy))
        self.assertAlmostEqual(0, err, 4)

    def test_FinDiff_1d(self):
        h, f, fxe, fxxe = self._prep_1d_func()
        d_dx = fd.FinDiff(h, dims=[0], acc=4)
        d2_dx2 = fd.FinDiff(h, dims=[0, 0], acc=4)
        fx = d_dx(f)
        self._assertAlmostEqual(fx, fxe)
        fxx = d2_dx2(f)
        self._assertAlmostEqual(fxx, fxxe)

    def test_FinDiff_2d(self):
        xy0 = [-5, -5]
        xy1 = [5, 5]
        nxy = [100, 100]
        x = np.linspace(xy0[0], xy1[0], nxy[0])
        y = np.linspace(xy0[1], xy1[1], nxy[1])
        dxy = [x[1] - x[0], y[1] - y[0]]
        X, Y = np.meshgrid(x, y, indexing='ij')
        f = X ** 3 * Y ** 3
        fxe = 3 * X ** 2 * Y ** 3
        fyye = 6 * X ** 3 * Y
        d_dx = fd.FinDiff(dxy, dims=[0], acc=4)
        fx = d_dx(f)
        d2_dy2 = fd.FinDiff(dxy, dims=[1, 1], acc=4)
        fyy = d2_dy2(f)
        self._assertAlmostEqual(fxe, fx)
        self._assertAlmostEqual(fyye, fyy)

    def _assertAlmostEqual(self, f1, f2, tol=7):
        err = np.max(np.abs(f1-f2))
        self.assertAlmostEqual(0, err, tol)

    def _prep_1d_func(self, x0=-5, x1=5, nx=100):
        x = np.linspace(x0, x1, nx)
        dx = x[1] - x[0]
        f = x**3
        fxe = 3*x**2
        fxxe = 6*x
        return dx, f, fxe, fxxe


if __name__ == '__main__':
    unittest.main()
