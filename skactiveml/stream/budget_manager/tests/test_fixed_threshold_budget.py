import unittest
import numpy as np

from skactiveml.stream.budget_manager import FixedThresholdBudget


class TestFixedThresholdBudget(unittest.TestCase):
    def setUp(self):
        # initialise var for sampled var tests
        self.utilities = np.array([True, False])

    def test_init_param_budget(self):
        # budget must be defined as a float with a range of: 0 < budget <= 1
        budget_manager = FixedThresholdBudget(budget="string")
        self.assertRaises(
            TypeError, budget_manager.query_by_utility, self.utilities
        )
        budget_manager = FixedThresholdBudget(budget=1.1)
        self.assertRaises(
            ValueError, budget_manager.query_by_utility, self.utilities
        )
        budget_manager = FixedThresholdBudget(budget=-1.0)
        self.assertRaises(
            ValueError, budget_manager.query_by_utility, self.utilities
        )

    def test_init_param_allow_exceeding_budget(self):
        # allow_exceeding_budget must be defined as a bool
        budget_manager = FixedThresholdBudget(allow_exceeding_budget="string")
        self.assertRaises(
            TypeError, budget_manager.query_by_utility, self.utilities
        )

        budget_manager = FixedThresholdBudget(allow_exceeding_budget=1)
        self.assertRaises(
            TypeError, budget_manager.query_by_utility, self.utilities
        )

        budget_manager = FixedThresholdBudget(allow_exceeding_budget=None)
        self.assertRaises(
            TypeError, budget_manager.query_by_utility, self.utilities
        )

    def test_query_param_utilities(self):
        # s must be defined as a float ndarray
        budget_manager = FixedThresholdBudget()
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
        bm = FixedThresholdBudget()
        bm.update(np.array([[0], [1], [2]]), np.array([0, 2]))
