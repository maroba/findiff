import sys
sys.path.insert(1, '..')

import unittest
from findiff import FinDiff


class TestOldBugs(unittest.TestCase):

    def test_findiff_should_raise_exception_when_applied_to_unevaluated_function(self):

        def f(x, y):
            return 5*x**2 - 5*x + 10*y**2 -10*y

        d_dx = FinDiff(1, 0.01)
        self.assertRaises(ValueError, lambda ff: d_dx(ff), f)


if __name__ == '__main__':
    unittest.main()