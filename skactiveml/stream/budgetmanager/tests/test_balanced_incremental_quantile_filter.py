import unittest

import numpy as np

from skactiveml.stream.budgetmanager import BalancedIncrementalQuantileFilter


class TestBalancedIncrementalQuantileFilter(unittest.TestCase):
    def setUp(self):
        # initialise var for sampled var tests
        self.utilities = np.array([True, False])

    def test_init_param_budget(self):
        # budget must be defined as a float with a range of: 0 < budget <= 1
        budget_manager = BalancedIncrementalQuantileFilter(budget="string")
        self.assertRaises(
            TypeError, budget_manager.query_by_utility, self.utilities
        )
        budget_manager = BalancedIncrementalQuantileFilter(budget=1.1)
        self.assertRaises(
            ValueError, budget_manager.query_by_utility, self.utilities
        )
        budget_manager = BalancedIncrementalQuantileFilter(budget=-1.0)
        self.assertRaises(
            ValueError, budget_manager.query_by_utility, self.utilities
        )

    def test_init_param_w(self):
        # w must be defined as an int with a range of w > 0
        budget_manager = BalancedIncrementalQuantileFilter(w="string")
        self.assertRaises(
            TypeError, budget_manager.query_by_utility, self.utilities
        )
        budget_manager = BalancedIncrementalQuantileFilter(w=None)
        self.assertRaises(
            TypeError, budget_manager.query_by_utility, self.utilities
        )
        budget_manager = BalancedIncrementalQuantileFilter(w=1.1)
        self.assertRaises(
            TypeError, budget_manager.query_by_utility, self.utilities
        )
        budget_manager = BalancedIncrementalQuantileFilter(w=0)
        self.assertRaises(
            ValueError, budget_manager.query_by_utility, self.utilities
        )
        budget_manager = BalancedIncrementalQuantileFilter(w=-1)
        self.assertRaises(
            ValueError, budget_manager.query_by_utility, self.utilities
        )

    def test_init_param_w_tol(self):
        # w must be defined as an int with a range of w_tol > 0
        budget_manager = BalancedIncrementalQuantileFilter(w_tol="string")
        self.assertRaises(
            TypeError, budget_manager.query_by_utility, self.utilities
        )
        budget_manager = BalancedIncrementalQuantileFilter(w_tol=None)
        self.assertRaises(
            TypeError, budget_manager.query_by_utility, self.utilities
        )
        budget_manager = BalancedIncrementalQuantileFilter(w_tol=0)
        self.assertRaises(
            ValueError, budget_manager.query_by_utility, self.utilities
        )
        budget_manager = BalancedIncrementalQuantileFilter(w_tol=-1)
        self.assertRaises(
            ValueError, budget_manager.query_by_utility, self.utilities
        )

    def test_query_param_utilities(self):
        # s must be defined as a float ndarray
        budget_manager = BalancedIncrementalQuantileFilter()
        self.assertRaises(
            TypeError, budget_manager.query_by_utility, utilities="string"
        )
        self.assertRaises(
            TypeError, budget_manager.query_by_utility, utilities=None
        )

    def test_update_without_query(self):
        budget_manager = BalancedIncrementalQuantileFilter()
        budget_manager.update(
            np.array([[0], [1], [2]]),
            np.array([0, 2]),
            utilities=[0.9, 0.1, 0.8],
        )
        self.assertRaises(
            TypeError,
            budget_manager.update,
            np.array([[0], [1], [2]]),
            np.array([0, 2]),
        )
