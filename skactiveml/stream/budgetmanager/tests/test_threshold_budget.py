import unittest

import numpy as np

from skactiveml.stream.budgetmanager import DensityBasedSplitBudgetManager


class TestDensityBasedSplitBudgetManager(unittest.TestCase):
    def setUp(self):
        # initialise var for sampled var tests
        self.utilities = np.array([True, False])

    def get_budget_manager(self):
        return DensityBasedSplitBudgetManager

    def test_init_param_budget(self):
        # budget must be defined as a float with a range of: 0 < budget <= 1
        budget_manager = self.get_budget_manager()(budget="string")
        self.assertRaises(
            TypeError, budget_manager.query_by_utility, self.utilities
        )
        budget_manager = self.get_budget_manager()(budget=1.1)
        self.assertRaises(
            ValueError, budget_manager.query_by_utility, self.utilities
        )
        budget_manager = self.get_budget_manager()(budget=-1.0)
        self.assertRaises(
            ValueError, budget_manager.query_by_utility, self.utilities
        )

    def test_init_param_theta(self):
        # theta must be defined as a float
        budget_manager = self.get_budget_manager()(theta="string")
        self.assertRaises(
            TypeError, budget_manager.query_by_utility, self.utilities
        )

    def test_init_param_random_state(self):
        # v must be defined as an float with a range of: 0 < v < 1
        budget_manager = self.get_budget_manager()(random_state="string")
        self.assertRaises(
            ValueError, budget_manager.query_by_utility, self.utilities
        )

    def test_init_param_s(self):
        # s must be defined as a float with a range of: 0 < s <= 1
        budget_manager = self.get_budget_manager()(s="string")
        self.assertRaises(
            TypeError, budget_manager.query_by_utility, self.utilities
        )
        budget_manager = self.get_budget_manager()(s=1.1)
        self.assertRaises(
            ValueError, budget_manager.query_by_utility, self.utilities
        )
        budget_manager = self.get_budget_manager()(s=0.0)
        self.assertRaises(
            ValueError, budget_manager.query_by_utility, self.utilities
        )
        budget_manager = self.get_budget_manager()(s=-1.0)
        self.assertRaises(
            ValueError, budget_manager.query_by_utility, self.utilities
        )

    def test_init_param_delta(self):
        # v must be defined as an float with a range of: 0 < delta
        budget_manager = self.get_budget_manager()(delta="string")
        self.assertRaises(
            TypeError, budget_manager.query_by_utility, self.utilities
        )
        budget_manager = self.get_budget_manager()(delta=0.0)
        self.assertRaises(
            ValueError, budget_manager.query_by_utility, self.utilities
        )
        budget_manager = self.get_budget_manager()(delta=-1.0)
        self.assertRaises(
            ValueError, budget_manager.query_by_utility, self.utilities
        )

    def test_query_param_utilities(self):
        # s must be defined as a float ndarray
        budget_manager = self.get_budget_manager()()
        self.assertRaises(
            TypeError, budget_manager.query_by_utility, utilities="string"
        )
        self.assertRaises(
            TypeError, budget_manager.query_by_utility, utilities=None
        )
        self.assertRaises(
            TypeError, budget_manager.query_by_utility, utilities=[10, 10]
        )

    def test_update_without_query(self):
        bm = self.get_budget_manager()()
        bm.update(np.array([[0], [1], [2]]), np.array([0, 2]))
