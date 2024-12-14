import unittest

import numpy as np
import pytest
from sympy import Rational

from findiff import coefficients
from findiff.coefs import calc_coefs
from findiff.coefs import coefficients_non_uni


class TestCoefs(unittest.TestCase):

    def test_order2_acc2(self):

        for analytic_inv in [True, False]:
            c = coefficients(deriv=2, acc=2, analytic_inv=analytic_inv)
            with self.subTest():

                coefs = c["center"]["coefficients"]
                np.testing.assert_array_almost_equal(np.array([1.0, -2.0, 1.0]), coefs)
                offs = c["center"]["offsets"]
                np.testing.assert_array_almost_equal(np.array([-1, 0, 1]), offs)

                coefs = c["forward"]["coefficients"]
                np.testing.assert_array_almost_equal(np.array([2, -5, 4, -1]), coefs)
                offs = c["forward"]["offsets"]
                np.testing.assert_array_almost_equal(np.array([0, 1, 2, 3]), offs)

                coefs = c["backward"]["coefficients"]
                np.testing.assert_array_almost_equal(
                    np.array(([2, -5, 4, -1])[::-1]), coefs
                )
                offs = c["backward"]["offsets"]
                np.testing.assert_array_almost_equal(np.array([-3, -2, -1, 0]), offs)

    def test_order1_acc2(self):

        for analytic_inv in [True, False]:
            c = coefficients(deriv=1, acc=2, analytic_inv=analytic_inv)
            with self.subTest():

                coefs = c["center"]["coefficients"]
                np.testing.assert_array_almost_equal(np.array([-0.5, 0, 0.5]), coefs)
                offs = c["center"]["offsets"]
                np.testing.assert_array_almost_equal(np.array([-1, 0, 1]), offs)

                coefs = c["forward"]["coefficients"]
                np.testing.assert_array_almost_equal(np.array([-1.5, 2, -0.5]), coefs)
                offs = c["forward"]["offsets"]
                np.testing.assert_array_almost_equal(np.array([0, 1, 2]), offs)

                coefs = c["backward"]["coefficients"]
                np.testing.assert_array_almost_equal(
                    -np.array([-1.5, 2, -0.5])[::-1], coefs
                )
                offs = c["backward"]["offsets"]
                np.testing.assert_array_almost_equal(np.array([-2, -1, 0]), offs)

    def test_order1_acc4(self):

        for analytic_inv in [True, False]:
            c = coefficients(deriv=1, acc=4, analytic_inv=analytic_inv)
            with self.subTest():
                coefs = c["center"]["coefficients"]
                np.testing.assert_array_almost_equal(
                    np.array([1 / 12, -2 / 3, 0, 2 / 3, -1 / 12]), coefs
                )
                offs = c["center"]["offsets"]
                np.testing.assert_array_almost_equal(np.array([-2, -1, 0, 1, 2]), offs)

                coefs = c["forward"]["coefficients"]
                np.testing.assert_array_almost_equal(
                    np.array([-25 / 12, 4, -3, 4 / 3, -1 / 4]), coefs
                )
                offs = c["forward"]["offsets"]
                np.testing.assert_array_almost_equal(np.array([0, 1, 2, 3, 4]), offs)

                coefs = c["backward"]["coefficients"]
                np.testing.assert_array_almost_equal(
                    -np.array([-25 / 12, 4, -3, 4 / 3, -1 / 4])[::-1], coefs
                )
                offs = c["backward"]["offsets"]
                np.testing.assert_array_almost_equal(
                    -np.array([0, 1, 2, 3, 4])[::-1], offs
                )

    def test_order2_acc4(self):

        for analytic_inv in [True, False]:
            c = coefficients(deriv=2, acc=4, analytic_inv=analytic_inv)
            with self.subTest():
                coefs = c["center"]["coefficients"]
                np.testing.assert_array_almost_equal(
                    np.array([-1 / 12, 4 / 3, -2.5, 4 / 3, -1 / 12]), coefs
                )
                offs = c["center"]["offsets"]
                np.testing.assert_array_almost_equal(np.array([-2, -1, 0, 1, 2]), offs)

                coefs = c["forward"]["coefficients"]
                np.testing.assert_array_almost_equal(
                    np.array([15 / 4, -77 / 6, 107 / 6, -13, 61 / 12, -5 / 6]), coefs
                )
                offs = c["forward"]["offsets"]
                np.testing.assert_array_almost_equal(np.array([0, 1, 2, 3, 4, 5]), offs)

                coefs = c["backward"]["coefficients"]
                np.testing.assert_array_almost_equal(
                    np.array([15 / 4, -77 / 6, 107 / 6, -13, 61 / 12, -5 / 6])[::-1],
                    coefs,
                )
                offs = c["backward"]["offsets"]
                np.testing.assert_array_almost_equal(
                    -np.array([0, 1, 2, 3, 4, 5])[::-1], offs
                )

    def test_calc_accuracy_central_deriv2_acc2(self):
        for analytic_inv in [True, False]:
            coefs = calc_coefs(2, [-1, 0, 1], analytic_inv=analytic_inv)
            with self.subTest():
                self.assertEqual(2, coefs["accuracy"])

    def test_calc_accuracy_central_deriv1_acc2(self):
        for analytic_inv in [True, False]:
            coefs = calc_coefs(1, [-1, 0, 1], analytic_inv=analytic_inv)
            with self.subTest():
                self.assertEqual(2, coefs["accuracy"])

    def test_calc_accuracy_left1_right0_deriv1_acc1(self):
        for analytic_inv in [True, False]:
            coefs = calc_coefs(1, [-1, 0], analytic_inv=analytic_inv)
            with self.subTest():
                self.assertEqual(1, coefs["accuracy"])

    def test_calc_accuracy_left0_right3_deriv1_acc3(self):
        for analytic_inv in [True, False]:
            coefs = calc_coefs(2, [0, 1, 2, 3], analytic_inv=analytic_inv)
            with self.subTest():
                self.assertEqual(2, coefs["accuracy"])

    def test_calc_accuracy_from_offsets_symbolic(self):
        for analytic_inv in [True, False]:
            coefs = calc_coefs(
                2, [0, 1, 2, 3], symbolic=True, analytic_inv=analytic_inv
            )
            with self.subTest():
                self.assertEqual(2, coefs["accuracy"])

    def test_calc_accuracy_from_offsets_symbolic(self):
        for analytic_inv in [True, False]:
            coefs = calc_coefs(
                2, [-4, -2, 0, 2, 4], symbolic=True, analytic_inv=analytic_inv
            )
            with self.subTest():
                self.assertEqual(4, coefs["accuracy"])

    def test_calc_coefs_from_offsets(self):
        for analytic_inv in [True, False]:
            coefs = calc_coefs(1, [-2, 0, 1], analytic_inv=analytic_inv)
            with self.subTest():
                np.testing.assert_array_almost_equal(
                    coefs["coefficients"], [-1.0 / 6, -0.5, 2.0 / 3]
                )

    def test_calc_coefs_from_offsets_no_central_point(self):
        for analytic_inv in [True, False]:
            coefs = calc_coefs(1, [-2, 1, 2], analytic_inv=analytic_inv)
            with self.subTest():
                np.testing.assert_array_almost_equal(
                    coefs["coefficients"], [-1.0 / 4, 0, 1.0 / 4]
                )

    def test_calc_coefs_from_offsets_not_enough_points(self):
        with pytest.raises(ValueError):
            coefficients(2, offsets=[-2, 2], analytic_inv=False)

    def test_calc_coefs_symbolic(self):
        for analytic_inv in [True, False]:
            coefs = calc_coefs(1, [-2, 0, 1], symbolic=True, analytic_inv=analytic_inv)
            expected = [Rational(-1, 6), Rational(-1, 2), Rational(2, 3)]
            with self.subTest():
                np.testing.assert_array_almost_equal(coefs["coefficients"], expected)

    def test_non_uniform(self):

        x = np.linspace(0, 10, 100)
        dx = x[1] - x[0]

        for analytic_inv in [True, False]:
            c_uni = coefficients(deriv=2, acc=2, analytic_inv=analytic_inv)
            coefs_uni = c_uni["center"]["coefficients"] / dx**2

            c_non_uni = coefficients_non_uni(deriv=2, acc=2, coords=x, idx=5)
            coefs_non_uni = c_non_uni["coefficients"]

            with self.subTest():
                np.testing.assert_array_almost_equal(coefs_non_uni, coefs_uni)

    def test_invalid_acc_raises_exception(self):
        with self.assertRaises(ValueError):
            coefficients(deriv=1, acc=3)
        with self.assertRaises(ValueError):
            coefficients(deriv=1, acc=0)
        with self.assertRaises(ValueError):
            coefficients_non_uni(1, 3, None, None)
        with self.assertRaises(ValueError):
            coefficients_non_uni(1, 0, None, None)

    def test_invalid_deriv_raises_exception(self):
        with self.assertRaises(ValueError):
            coefficients(-1, 2)
        with self.assertRaises(ValueError):
            coefficients_non_uni(-1, 2, None, None)


if __name__ == "__main__":
    unittest.main()
