import numpy as np

from findiff import Diff


def test_iterative_accuracy():

    ns = np.logspace(2, 3, 10)

    def compute_errs(acc, plot=False):
        # import matplotlib.pyplot as plt

        errs_a = []
        errs_b = []
        errs_c = []

        for n in ns:
            x = np.linspace(0.1, 1, int(n))
            dx = x[1] - x[0]

            D = Diff(0, dx, acc=acc)
            D2 = D**2
            DD = D * D

            f = x**8

            exact = 8 * 7 * x**6
            actual_a = D2(f)
            actual_b = D(D(f))
            actual_c = DD(f)

            err_a = np.max(abs((actual_a - exact) / exact))
            err_b = np.max(abs((actual_b - exact) / exact))
            err_c = np.max(abs((actual_c - exact) / exact))
            errs_a.append(err_a)
            errs_b.append(err_b)
            errs_c.append(err_c)

        # if plot:
        #     plt.loglog(ns, errs_a, "-x", label="D^2f")
        #     plt.loglog(ns, errs_b, "-", label="D(D(f))")
        #     plt.loglog(ns, errs_c, "-", label="D*D*f")
        #     plt.grid()
        #     plt.show()

        slope_a = abs(loglog_slope(ns, errs_a))
        slope_b = abs(loglog_slope(ns, errs_b))
        slope_c = abs(loglog_slope(ns, errs_c))
        return slope_a, slope_b, slope_c

    slope_a, slope_b, slope_c = compute_errs(2, plot=False)

    assert abs(slope_a - 2) < 0.2
    # applying operators iteratively should reduce the order by one at a time:
    assert abs(slope_b - 1) < 0.2
    assert abs(slope_c - 2) < 0.2

    slope_a, slope_b, slope_c = compute_errs(4, plot=False)
    assert abs(slope_a - 4) < 0.2
    # applying operators iteratively should reduce the order by one at a time:
    assert abs(slope_b - 3) < 0.4
    assert abs(slope_c - 4) < 0.2


def loglog_slope(x, y):
    slope, intercept = np.polyfit(np.log(x), np.log(y), 1)
    return slope


if __name__ == "__main__":
    test_iterative_accuracy()
