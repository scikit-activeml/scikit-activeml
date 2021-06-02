import unittest

import numpy as np

from skactiveml.evaluation._scores import minimum_aggregated_cost


class TestFunctions(unittest.TestCase):

    def test_minimum_aggregated_cost(self):
        n_acquired_labels = [1, 2, 3, 4, 5]
        error = [10, 8, 7, 3, 3]
        annotation_cost = [0.2, 0.4, 0.6, 0.8]
        error_cost = [0.8, 0.6, 0.4, 0.2]

        a = minimum_aggregated_cost(n_acquired_labels, error, annotation_cost,
                                    error_cost)
        np.testing.assert_array_equal(a, [3.2, 3.4, 3.6, 2.8])

        a = minimum_aggregated_cost(1, 10, annotation_cost, error_cost)
        np.testing.assert_array_equal(a, [8.2, 6.4, 4.6, 2.8])

        a = minimum_aggregated_cost(n_acquired_labels, error, 1, 2)
        np.testing.assert_equal(a, 10)

        a = minimum_aggregated_cost(n_acquired_labels, error, [1], [2])
        np.testing.assert_equal(a, [10])
