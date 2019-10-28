import sys
sys.path.insert(1, '..')

from math import log
import unittest
import numpy as np
from findiff.operators import FinDiff

class TestScaling(unittest.TestCase):

    def fit_1d(self, acc):
        nx_list = 10 ** np.linspace(1.75, 2., 10)
        Lx = np.pi

        log_err_list = []
        log_dx_list = []

        for nx in nx_list:
            x = np.linspace(0., Lx, nx)
            dx = x[1] - x[0]
            f = np.sin(x)
            d_dx = FinDiff(0, dx)
            fx = d_dx(f, acc=acc)
            fxe = np.cos(x)
            err = np.max(np.abs(fxe - fx))
            log_dx_list.append(log(dx))
            log_err_list.append(log(err))

        fit = np.polyfit(log_dx_list, log_err_list, deg=1)
        return fit[0]

    def fit_2d(self, acc):
        nx_list = [10, 30, 100, 300]
        ny_list = [10, 30, 100, 300]
        Lx, Ly = 3, 3

        log_err_list = []
        log_dx_list = []

        for nx, ny in zip(nx_list, ny_list):
            x = np.linspace(0, Lx, nx)
            y = np.linspace(0, Ly, ny)
            dx, dy = x[1] - x[0], y[1] - y[0]
            X, Y = np.meshgrid(x, y, indexing='ij')
            f = np.sin(X) * np.sin(Y)
            d_dx = FinDiff(0, dx)
            fx = d_dx(f, acc=acc)
            fxe = np.cos(X) * np.sin(Y)
            err = np.max(np.abs(fxe - fx))
            log_dx_list.append(log(dx))
            log_err_list.append(log(err))

        fit = np.polyfit(log_dx_list, log_err_list, deg=1)
        return fit[0]

    def test_1d_acc2(self):
        self.assertAlmostEqual(2, self.fit_1d(acc=2), 1)

    def test_1d_acc4(self):
        self.assertAlmostEqual(4, self.fit_1d(acc=4), 1)

    def test_1d_acc6(self):
        self.assertAlmostEqual(6, self.fit_1d(acc=6), 1)

    def test_2d_acc2(self):
        self.assertAlmostEqual(2, self.fit_2d(acc=2), 1)

    def test_2d_acc4(self):
        self.assertAlmostEqual(4, self.fit_2d(acc=4), 1)


if __name__ == '__main__':
    unittest.main()