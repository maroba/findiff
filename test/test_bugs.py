import sys
sys.path.insert(1, '..')

import unittest
import numpy as np
from findiff import FinDiff
import findiff


class TestOldBugs(unittest.TestCase):

    def test_findiff_should_raise_exception_when_applied_to_unevaluated_function(self):

        def f(x, y):
            return 5*x**2 - 5*x + 10*y**2 -10*y

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
        findiff.coefficients(deriv=1, acc=15)


if __name__ == '__main__':
    unittest.main()