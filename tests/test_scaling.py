from math import log

import numpy as np
import pytest

from findiff import FinDiff


class TestScaling:

    def fit_1d(self, acc):
        nx_list = 10 ** np.linspace(1.75, 2.0, 10)
        Lx = np.pi

        log_err_list = []
        log_dx_list = []

        for nx in nx_list:
            nx = int(nx)
            x = np.linspace(0.0, Lx, nx)
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
            X, Y = np.meshgrid(x, y, indexing="ij")
            f = np.sin(X) * np.sin(Y)
            d_dx = FinDiff(0, dx)
            fx = d_dx(f, acc=acc)
            fxe = np.cos(X) * np.sin(Y)
            err = np.max(np.abs(fxe - fx))
            log_dx_list.append(log(dx))
            log_err_list.append(log(err))

        fit = np.polyfit(log_dx_list, log_err_list, deg=1)
        return fit[0]

    @pytest.mark.parametrize("acc", [2, 4, 6])
    def test_1d_acc(self, acc):
        assert self.fit_1d(acc=acc) == pytest.approx(acc, 0.1)

    @pytest.mark.parametrize("acc", [2, 4])
    def test_2d_acc2(self, acc):
        assert self.fit_2d(acc=acc) == pytest.approx(acc, 0.1)

    def assertAlmostEqual(self, first, second, acc=None):
        if acc:
            assert first == pytest.approx(second, abs=acc)
        assert first == pytest.approx(second)
