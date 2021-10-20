import unittest

import numpy as np

from skactiveml.utils import simple_batch, call_func


class TestFunctions(unittest.TestCase):

    def test_simple_batch(self):
        utils = [4, 2, 5, 3, 1, 0]
        expected_indices = np.array([2, 0, 3, 1, 4, 5])
        expected_batches = np.array([[4, 2, 5, 3, 1, 0],
                                     [4, 2, np.nan, 3, 1, 0],
                                     [np.nan, 2, np.nan, 3, 1, 0],
                                     [np.nan, 2, np.nan, np.nan, 1, 0],
                                     [np.nan, np.nan, np.nan, np.nan, 1, 0],
                                     [np.nan, np.nan, np.nan, np.nan, np.nan,
                                      0]])
        self.assertRaises(TypeError, simple_batch,
                          utils, random_state=42, batch_size='invalid')
        self.assertRaises(ValueError, simple_batch,
                          utils, random_state=42, batch_size=0)
        indices, batches = simple_batch(utils, random_state=42,
                                        batch_size=len(utils) + 1,
                                        return_utilities=True)
        np.testing.assert_array_equal(indices, expected_indices)
        np.testing.assert_array_equal(batches, expected_batches)

        indices, batches = simple_batch(utils, random_state=42, batch_size=3,
                                        return_utilities=True)
        np.testing.assert_array_equal(indices[0:3], expected_indices[0:3])
        np.testing.assert_array_equal(batches[0:3], expected_batches[0:3])
        indices, batches = simple_batch([[np.nan, np.nan], [np.nan, np.nan]], random_state=42, batch_size=1,
                                        return_utilities=True)
        np.testing.assert_equal((0, 2), indices.shape)
        np.testing.assert_array_equal((0, 2, 2), batches.shape)

    def test_call_func(self):
        def dummy_function(a, b=2, c=3):
            return a * b * c

        result = call_func(dummy_function, a=2, b=5, c=5)
        self.assertEqual(result, 50)
        result = call_func(dummy_function, only_mandatory=True, a=2, b=5, c=5)
        self.assertEqual(result, 12)
