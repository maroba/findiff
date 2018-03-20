import unittest
from findiff import coefficients
import numpy as np


class TestCoefs(unittest.TestCase):

    def test_order2_acc2(self):

        c = coefficients(deriv=2, acc=2)

        coefs = c["center"]["coefficients"]
        np.testing.assert_array_almost_equal(np.array([1., -2., 1.]), coefs)
        offs = c["center"]["offsets"]
        np.testing.assert_array_almost_equal(np.array([-1, 0, 1]), offs)

        coefs = c["forward"]["coefficients"]
        np.testing.assert_array_almost_equal(np.array([2, -5, 4, -1]), coefs)
        offs = c["forward"]["offsets"]
        np.testing.assert_array_almost_equal(np.array([0, 1, 2, 3]), offs)

        coefs = c["backward"]["coefficients"]
        np.testing.assert_array_almost_equal(np.array(([2, -5, 4, -1])[::-1]), coefs)
        offs = c["backward"]["offsets"]
        np.testing.assert_array_almost_equal(np.array([-3, -2, -1, 0]), offs)

    def test_order1_acc2(self):

        c = coefficients(deriv=1, acc=2)

        coefs = c["center"]["coefficients"]
        np.testing.assert_array_almost_equal(np.array([-0.5, 0, 0.5]), coefs)
        offs = c["center"]["offsets"]
        np.testing.assert_array_almost_equal(np.array([-1, 0, 1]), offs)

        coefs = c["forward"]["coefficients"]
        np.testing.assert_array_almost_equal(np.array([-1.5, 2, -0.5]), coefs)
        offs = c["forward"]["offsets"]
        np.testing.assert_array_almost_equal(np.array([0, 1, 2]), offs)

        coefs = c["backward"]["coefficients"]
        np.testing.assert_array_almost_equal(-np.array([-1.5, 2, -0.5])[::-1], coefs)
        offs = c["backward"]["offsets"]
        np.testing.assert_array_almost_equal(np.array([-2, -1, 0]), offs)

    def test_order1_acc4(self):

        c = coefficients(deriv=1, acc=4)
        coefs = c["center"]["coefficients"]
        np.testing.assert_array_almost_equal(np.array([1/12, -2/3, 0, 2/3, -1/12]), coefs)
        offs = c["center"]["offsets"]
        np.testing.assert_array_almost_equal(np.array([-2, -1, 0, 1, 2]), offs)

        coefs = c["forward"]["coefficients"]
        np.testing.assert_array_almost_equal(np.array([-25/12, 4, -3, 4/3, -1/4]), coefs)
        offs = c["forward"]["offsets"]
        np.testing.assert_array_almost_equal(np.array([0, 1, 2, 3, 4]), offs)

        coefs = c["backward"]["coefficients"]
        np.testing.assert_array_almost_equal(-np.array([-25/12, 4, -3, 4/3, -1/4])[::-1], coefs)
        offs = c["backward"]["offsets"]
        np.testing.assert_array_almost_equal(-np.array([0, 1, 2, 3, 4])[::-1], offs)

    def test_order2_acc4(self):

        c = coefficients(deriv=2, acc=4)
        coefs = c["center"]["coefficients"]
        np.testing.assert_array_almost_equal(np.array([-1/12, 4/3, -2.5, 4/3, -1/12]), coefs)
        offs = c["center"]["offsets"]
        np.testing.assert_array_almost_equal(np.array([-2, -1, 0, 1, 2]), offs)

        coefs = c["forward"]["coefficients"]
        np.testing.assert_array_almost_equal(np.array([15/4, -77/6, 107/6, -13, 61/12, -5/6]), coefs)
        offs = c["forward"]["offsets"]
        np.testing.assert_array_almost_equal(np.array([0, 1, 2, 3, 4, 5]), offs)

        coefs = c["backward"]["coefficients"]
        np.testing.assert_array_almost_equal(np.array([15/4, -77/6, 107/6, -13, 61/12, -5/6])[::-1], coefs)
        offs = c["backward"]["offsets"]
        np.testing.assert_array_almost_equal(-np.array([0, 1, 2, 3, 4, 5])[::-1], offs)


if __name__ == '__main__':
    unittest.main()