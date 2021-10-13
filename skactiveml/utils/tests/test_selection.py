import unittest

import numpy as np

from skactiveml.utils import rand_argmin, rand_argmax


class TestSelection(unittest.TestCase):

    def setUp(self):
        self.a = [2, 5, 6, -1]
        self.b = [[2, 1], [3, 5]]
        self.c = [2, 2, 1, 1]
        self.d = [[2, 2], [1, 1]]
        self.e = [np.nan, 1]

    def test_rand_argmin(self):
        np.testing.assert_array_equal([3], rand_argmin(self.a))
        np.testing.assert_array_equal([1, 0], rand_argmin(self.b, axis=1))
        np.testing.assert_array_equal([0, 1], rand_argmin(self.b))
        np.testing.assert_array_equal([2],
                                      rand_argmin(self.c, random_state=42))
        np.testing.assert_array_equal([1, 0], rand_argmin(self.d, axis=1,
                                                          random_state=42))
        np.testing.assert_array_equal([1, 0],
                                      rand_argmin(self.d, random_state=42))
        np.testing.assert_array_equal([3], rand_argmin(self.c, random_state=1))
        np.testing.assert_array_equal([1, 1], rand_argmin(self.d, axis=1,
                                                          random_state=1))
        np.testing.assert_array_equal([1, 1],
                                      rand_argmin(self.d, random_state=1))
        np.testing.assert_array_equal([1], rand_argmin(self.e))

    def test_rand_argmax(self):
        np.testing.assert_array_equal([2], rand_argmax(self.a))
        np.testing.assert_array_equal([0, 1], rand_argmax(self.b, axis=1))
        np.testing.assert_array_equal([1, 1], rand_argmax(self.b))
        np.testing.assert_array_equal([1],
                                      rand_argmax(self.c, random_state=42))
        np.testing.assert_array_equal([1, 0], rand_argmax(self.d, axis=1,
                                                          random_state=42))
        np.testing.assert_array_equal([0, 1],
                                      rand_argmax(self.d, random_state=42))
        np.testing.assert_array_equal([0],
                                      rand_argmax(self.c, random_state=10))
        np.testing.assert_array_equal([1, 1], rand_argmax(self.d, axis=1,
                                                          random_state=1))
        np.testing.assert_array_equal([0, 0],
                                      rand_argmax(self.d, random_state=10))
        np.testing.assert_array_equal([1], rand_argmax(self.e))
