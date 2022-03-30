import itertools
import unittest

from skactiveml.utils import call_func


class TestFunctions(unittest.TestCase):
    def test_call_func(self):
        def dummy_function(a, b=2, c=3):
            return a * b * c

        result = call_func(dummy_function, a=2, b=5, c=5)
        self.assertEqual(result, 50)
        result = call_func(dummy_function, only_mandatory=True, a=2, b=5, c=5)
        self.assertEqual(result, 12)

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
