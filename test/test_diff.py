import findiff as fd #.diff as fd
import unittest
import numpy as np


class TestFinDiff(unittest.TestCase):

    def test_diff_1d_ord1_acc2(self):
        x = np.linspace(-3, 3, 50)
        h = x[1] - x[0]
        y = x ** 2
        y1e = 2 * x
        y1 = fd.diff(y, h=h, dims=[0], acc=2)
        err = np.max(np.abs(y1e - y1))
        self.assertAlmostEqual(0, err)

    def test_diff_1d_ord1_acc4(self):
        x = np.linspace(-3, 3, 50)
        h = x[1] - x[0]
        y = x ** 2
        y1e = 2 * x
        y1 = fd.diff(y, h=h, dims=[0], acc=4)
        err = np.max(np.abs(y1e - y1))
        self.assertAlmostEqual(0, err)

    def test_diff_1d_ord2_acc2(self):
        x = np.linspace(-3, 3, 50)
        h = x[1] - x[0]
        y = x ** 3
        y2e = 6 * x
        y2 = fd.diff(y, h=h, dims=[0, 0], acc=2)
        err = np.max(np.abs(y2e - y2))
        self.assertAlmostEqual(0, err)

    def test_diff_1d_ord2_acc4(self):
        x = np.linspace(-3, 3, 50)
        h = x[1] - x[0]
        y = x ** 3
        y2e = 6 * x
        y2 = fd.diff(y, h=h, dims=[0, 0], acc=4)
        err = np.max(np.abs(y2e - y2))
        self.assertAlmostEqual(0, err)

    def test_diff_2d_ord1_acc2(self):
        x = np.linspace(-1, 1, 500)
        y = np.linspace(-1, 1, 500)
        dx = dy = x[1] - x[0]
        X, Y = np.meshgrid(x, y, indexing='ij')
        f = Y**3 * X**3
        fxe = 3 * X**2 * Y**3
        fx = fd.diff(f, h=[dx, dy], dims=[0], acc=2)
        err = np.max(np.abs(fxe - fx))
        self.assertAlmostEqual(0, err, 4)
        fye = 3 * X**3 * Y**2
        fy = fd.diff(f, h=[dx, dy], dims=[1], acc=2)
        err = np.max(np.abs(fye - fy))
        self.assertAlmostEqual(0, err, 4)

    def test_diff_2d_ord2_acc2(self):
        x = np.linspace(-1, 1, 500)
        y = np.linspace(-1, 1, 500)
        dx = dy = x[1] - x[0]
        X, Y = np.meshgrid(x, y, indexing='ij')
        f = Y**3 * X**3
        fxxe = 6 * X * Y**3
        fxx = fd.diff(f, h=[dx, dy], dims=[0, 0], acc=2)
        err = np.max(np.abs(fxxe - fxx))
        self.assertAlmostEqual(0, err, 4)
        fyye = 6 * X**3 * Y
        fyy = fd.diff(f, acc=2, h=[dx, dy], dims=[1, 1])
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
        fx = fd.diff(f, acc=2, h=[dx, dy, dz], dims=[0])
        err = np.max(np.abs(fxe - fx))
        self.assertAlmostEqual(0, err, 4)
        fye = 3 * X**3 * Y**2 * Z**3
        fy = fd.diff(f, acc=2, h=[dx, dy, dz], dims=[1])
        err = np.max(np.abs(fye - fy))
        self.assertAlmostEqual(0, err, 4)
        fze = 3 * X**3 * Y**3 * Z**2
        fz = fd.diff(f, acc=2, h=[dx, dy, dz], dims=[2])
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
        fxx = fd.diff(f, acc=2, h=[dx, dy, dz], dims=[0, 0])
        err = np.max(np.abs(fxxe - fxx))
        self.assertAlmostEqual(0, err, 4)
        fyye = 6 * X**3 * Y * Z**3
        fyy = fd.diff(f, acc=2, h=[dx, dy, dz], dims=[1, 1])
        err = np.max(np.abs(fyye - fyy))
        self.assertAlmostEqual(0, err, 6)
        fzze = 6 * X**3 * Y**3 * Z
        fzz = fd.diff(f, acc=2, h=[dx, dy, dz], dims=[2, 2])
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
        fxy = fd.diff(f, acc=4, h=[dx, dy, dz], dims=[0, 1])
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
        fxy = fd.diff(f, acc=4, h=[dx, dy, dz], dims=[0, 1])
        err = np.max(np.abs(fxye - fxy))
        self.assertAlmostEqual(0, err, 4)


if __name__ == '__main__':
    unittest.main()
