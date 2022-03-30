import itertools
import unittest

import numpy as np

from skactiveml.utils import simple_batch, call_func
from skactiveml.utils._functions import rank_utilities, update_X_y


class TestFunctions(unittest.TestCase):
    def test_call_func(self):
        def dummy_function(a, b=2, c=3):
            return a * b * c

        result = call_func(dummy_function, a=2, b=5, c=5)
        self.assertEqual(result, 50)
        result = call_func(dummy_function, only_mandatory=True, a=2, b=5, c=5)
        self.assertEqual(result, 12)

    def test_simple_batch(self):
        utils = [4, 2, 5, 3, 1, 0]
        expected_indices = np.array([2, 0, 3, 1, 4, 5])
        expected_batches = np.array(
            [
                [4, 2, 5, 3, 1, 0],
                [4, 2, np.nan, 3, 1, 0],
                [np.nan, 2, np.nan, 3, 1, 0],
                [np.nan, 2, np.nan, np.nan, 1, 0],
                [np.nan, np.nan, np.nan, np.nan, 1, 0],
                [np.nan, np.nan, np.nan, np.nan, np.nan, 0],
            ]
        )
        self.assertRaises(
            TypeError, simple_batch, utils, random_state=42, batch_size="invalid"
        )
        self.assertRaises(
            ValueError, simple_batch, utils, random_state=42, batch_size=0
        )
        indices, batches = simple_batch(
            utils, random_state=42, batch_size=len(utils) + 1, return_utilities=True
        )
        np.testing.assert_array_equal(indices, expected_indices)
        np.testing.assert_array_equal(batches, expected_batches)

        indices, batches = simple_batch(
            utils, random_state=42, batch_size=3, return_utilities=True
        )
        np.testing.assert_array_equal(indices[0:3], expected_indices[0:3])
        np.testing.assert_array_equal(batches[0:3], expected_batches[0:3])
        indices, batches = simple_batch(
            [[np.nan, np.nan], [np.nan, np.nan]],
            random_state=42,
            batch_size=1,
            return_utilities=True,
        )
        np.testing.assert_equal((0, 2), indices.shape)
        np.testing.assert_array_equal((0, 2, 2), batches.shape)

    def test_update_X_y(self):

        X = np.arange(7 * 2).reshape(7, 2)
        y = np.arange(7)

        x_pot = np.array([3, 4])
        y_pot = 5

        X_new, y_new = update_X_y(X, y, y_pot, X_update=x_pot)

        self.assertEqual(X_new.shape, (8, 2))
        self.assertEqual(y_new.shape, (8,))
        np.testing.assert_equal(X_new[7], x_pot)
        self.assertEqual(y_new[7], y_pot)

    def test_combine_utilities(self):

        utilities_one = np.arange(2 * 3).reshape((2, 3, 1)) / (3 * 2)
        utilities_two = np.arange(2 * 5).reshape((2, 1, 5)) / (5 * 2)

        combined_utilities = rank_utilities(
            utilities_two, utilities_one, rank_per_batch=True
        )

        for j, i in itertools.product(range(2), range(4)):
            self.assertGreater(
                np.min(combined_utilities[j, :, i + 1]),
                np.max(combined_utilities[j, :, i]),
            )

        for i, k, j in itertools.product(range(2), range(2), range(5)):
            self.assertGreater(
                combined_utilities[i, k + 1, j], combined_utilities[i, k, j]
            )

        combined_utilities = rank_utilities(
            utilities_two, utilities_one, rank_per_batch=False
        )

        for i, j in itertools.product(range(2), range(4)):
            self.assertGreater(
                np.min(combined_utilities[i, :, j + 1]),
                np.max(combined_utilities[i, :, j]),
            )

        for i in range(5):
            self.assertGreater(
                np.min(combined_utilities[1, :, i]),
                np.max(combined_utilities[0, :, i]),
            )

        for i, k, j in itertools.product(range(2), range(2), range(5)):
            self.assertGreater(
                combined_utilities[i, k + 1, j], combined_utilities[i, k, j]
            )

        a = np.array([0, 1, 0, 1])
        b = np.array([1, 2, 3, 4])

        c = rank_utilities(a, b)

        self.assertGreater(c[3], c[1])
        self.assertGreater(c[1], c[2])
        self.assertGreater(c[2], c[0])
