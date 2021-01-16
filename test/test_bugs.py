import sys

sys.path.insert(1, '..')

import unittest
import numpy as np
from findiff import FinDiff
import findiff


class TestOldBugs(unittest.TestCase):

    def test_findiff_should_raise_exception_when_applied_to_unevaluated_function(self):
        def f(x, y):
            return 5 * x ** 2 - 5 * x + 10 * y ** 2 - 10 * y

        d_dx = FinDiff(1, 0.01)
        self.assertRaises(ValueError, lambda ff: d_dx(ff), f)

    def test_matrix_representation_doesnt_work_for_order_greater_2_issue_24(self):
        x = np.zeros((10))
        d3_dx3 = FinDiff((0, 1, 3))
        mat = d3_dx3.matrix(x.shape)
        self.assertAlmostEqual(-2.5, mat[0, 0])
        self.assertAlmostEqual(-2.5, mat[1, 1])
        self.assertAlmostEqual(-0.5, mat[2, 0])

    def test_high_accuracy_results_in_type_error(self):
        # in issue 25 the following line resulted in a TypeError
        findiff.coefficients(deriv=1, acc=16)

    def test_matrix_repr_with_different_accs(self):
        # issue 28
        shape = (11,)
        d1 = findiff.FinDiff(0, 1, 2).matrix(shape)
        d2 = findiff.FinDiff(0, 1, 2, acc=4).matrix(shape)

        self.assertTrue(np.max(np.abs((d1 - d2).toarray())) > 1)

        x = np.linspace(0, 10, 11)
        f = x ** 2
        df = d2.dot(f)
        np.testing.assert_almost_equal(2 * np.ones_like(f), df)

    def test_accuracy_should_be_passed_down_to_stencil(self):
        # issue 31

        shape = 11, 11
        dx = 1.
        d1x = FinDiff(0, dx, 1, acc=4)
        stencil1 = d1x.stencil(shape)

        expected = {
            ('L', 'L'): {(0, 0): -2.083333333333331, (1, 0): 3.9999999999999916, (2, 0): -2.999999999999989,
                         (3, 0): 1.3333333333333268, (4, 0): -0.24999999999999858},
            ('L', 'C'): {(0, 0): -2.083333333333331, (1, 0): 3.9999999999999916, (2, 0): -2.999999999999989,
                         (3, 0): 1.3333333333333268, (4, 0): -0.24999999999999858},
            ('L', 'H'): {(0, 0): -2.083333333333331, (1, 0): 3.9999999999999916, (2, 0): -2.999999999999989,
                         (3, 0): 1.3333333333333268, (4, 0): -0.24999999999999858},
            ('C', 'L'): {(-2, 0): 0.08333333333333333, (-1, 0): -0.6666666666666666,
                         (1, 0): 0.6666666666666666, (2, 0): -0.08333333333333333},
            ('C', 'C'): {(-2, 0): 0.08333333333333333, (-1, 0): -0.6666666666666666,
                         (1, 0): 0.6666666666666666, (2, 0): -0.08333333333333333},
            ('C', 'H'): {(-2, 0): 0.08333333333333333, (-1, 0): -0.6666666666666666,
                         (1, 0): 0.6666666666666666, (2, 0): -0.08333333333333333},
            ('H', 'L'): {(-4, 0): 0.24999999999999958, (-3, 0): -1.3333333333333313, (-2, 0): 2.9999999999999956,
                         (-1, 0): -3.999999999999996, (0, 0): 2.0833333333333317},
            ('H', 'C'): {(-4, 0): 0.24999999999999958, (-3, 0): -1.3333333333333313, (-2, 0): 2.9999999999999956,
                         (-1, 0): -3.999999999999996, (0, 0): 2.0833333333333317},
            ('H', 'H'): {(-4, 0): 0.24999999999999958, (-3, 0): -1.3333333333333313, (-2, 0): 2.9999999999999956,
                         (-1, 0): -3.999999999999996, (0, 0): 2.0833333333333317},
        }

        for char_pt in stencil1.data:
            stl = stencil1.data[char_pt]
            self.assert_dict_almost_equal(expected[char_pt], stl)

        d1x = FinDiff(0, dx, 1)
        stencil1 = d1x.stencil(shape, acc=4)
        for char_pt in stencil1.data:
            stl = stencil1.data[char_pt]
            self.assert_dict_almost_equal(expected[char_pt], stl)

    def test_order_as_numpy_integer(self):

        order = np.ones(3, dtype=np.int32)[0]
        d_dx = FinDiff(0, 0.1, order) # raised an AssertionError with the bug

        np.testing.assert_allclose(d_dx(np.linspace(0, 1, 11)), np.ones(11))


    def assert_dict_almost_equal(self, actual, expected, places=7):
        if len(actual) != len(expected):
            return False

        for key, value in actual.items():
            self.assertAlmostEqual(actual[key], expected[key], places=places)


if __name__ == '__main__':
    unittest.main()
