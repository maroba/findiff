import unittest

from findiff.utils import *


class TestUtils(unittest.TestCase):

    def test_all_index_tuples_for_shape(self):

        shape = 2, 3

        expected = []
        for ix in range(shape[0]):
            for iy in range(shape[1]):
                expected.append((ix, iy))

        actual = all_index_tuples_as_list(shape)

        for a, e in zip(actual, expected):
            self.assertEqual(e, a)

    def test_to_long_index(self):

        shape = 3, 4, 5
        all_tuples = all_index_tuples_as_list(shape)

        expected = list(range(np.prod(shape)))

        for i, idx in enumerate(all_tuples):
            actual = to_long_index(idx, shape)
            self.assertEqual(expected[i], actual)

    def test_to_index_tuple(self):
        shape = 3, 4, 5

        all_tuples = all_index_tuples_as_list(shape)

        for long_idx in range(np.prod(shape)):
            expected = all_tuples[long_idx]
            actual = to_index_tuple(long_idx, shape)
            np.testing.assert_array_equal(expected, actual)

    def test_interior_mask_2d(self):
        shape = 5, 5
        expected = np.zeros(shape, dtype=bool)
        expected[1:-1, 1:-1] = True

        np.testing.assert_array_equal(expected, interior_mask_as_ndarray(shape))
