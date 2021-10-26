import unittest
import numpy as np

from skactiveml.stream.budget_manager import FixedThresholdBudget


class TestFixedBudget(unittest.TestCase):
    def setUp(self):
        # initialise var for sampled var tests
        self.utilities = np.array([True, False])

    def test_fixed_budget(self):
        # init param test
        self._test_init_param_budget(FixedThresholdBudget)

        # init param test
        self._test_init_param_allow_exceeding_budget(FixedThresholdBudget)

        # sampled param test
        self._test_sampled_param_utilities(FixedThresholdBudget)

        # update test
        self._test_update_without_query(FixedThresholdBudget)

    def _test_init_param_budget(self, budget_manager_name):
        # budget must be defined as a float with a range of: 0 < budget <= 1
        budget_manager = budget_manager_name(budget="string")
        self.assertRaises(
            TypeError, budget_manager.query_by_utility, self.utilities
        )
        budget_manager = budget_manager_name(budget=1.1)
        self.assertRaises(
            ValueError, budget_manager.query_by_utility, self.utilities
        )
        budget_manager = budget_manager_name(budget=-1.0)
        self.assertRaises(
            ValueError, budget_manager.query_by_utility, self.utilities
        )

    def _test_init_param_allow_exceeding_budget(self, budget_manager_name):
        # allow_exceeding_budget must be defined as a bool
        budget_manager = budget_manager_name(allow_exceeding_budget="string")
        self.assertRaises(
            TypeError, budget_manager.query_by_utility, self.utilities
        )

        budget_manager = budget_manager_name(allow_exceeding_budget=1)
        self.assertRaises(
            TypeError, budget_manager.query_by_utility, self.utilities
        )

        budget_manager = budget_manager_name(allow_exceeding_budget=None)
        self.assertRaises(
            TypeError, budget_manager.query_by_utility, self.utilities
        )

    def _test_sampled_param_utilities(self, budget_manager_name):
        # s must be defined as a float ndarray
        budget_manager = budget_manager_name()
        self.assertRaises(
            TypeError, budget_manager.query_by_utility, utilities="string"
        )
        self.assertRaises(
            TypeError, budget_manager.query_by_utility, utilities=None
        )
        self.assertRaises(
            TypeError, budget_manager.query_by_utility, utilities=[10, 10]
        )

    def _test_update_without_query(self, query_strategy_name):
        qs = query_strategy_name()
        qs.update(np.array([[0], [1], [2]]), np.array([0, 2]))
