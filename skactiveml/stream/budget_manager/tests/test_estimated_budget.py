import unittest
import numpy as np

from .._estimated_budget import (
    FixedUncertaintyBudget,
    VarUncertaintyBudget,
    SplitBudget,
)


class TestEstimatedBudget(unittest.TestCase):
    def setUp(self):
        # initialise var for sampled var tests
        self.utilities = np.array([True, False])

    def test_estimated_budget_FixedUncertaintyBudget(self):
        # init param test
        self._test_init_param_budget(FixedUncertaintyBudget)
        self._test_init_param_w(FixedUncertaintyBudget)
        self._test_init_param_num_classes(FixedUncertaintyBudget)

        # sampled param test
        self._test_sampled_param_utilities(FixedUncertaintyBudget)

        self._test_update_without_query(FixedUncertaintyBudget)

        # functinality test
        # self._test_sampled_utilities(FixedUncertaintyBudget)

    def test_estimated_budget_VarUncertaintyBudget(self):
        # init param test
        self._test_init_param_budget(VarUncertaintyBudget)
        self._test_init_param_w(VarUncertaintyBudget)
        self._test_init_param_theta(VarUncertaintyBudget)
        self._test_init_param_s(VarUncertaintyBudget)

        # sampled param test
        self._test_sampled_param_utilities(VarUncertaintyBudget)

        self._test_update_without_query(VarUncertaintyBudget)

        # functinality test
        # self._test_sampled_utilities(VarUncertaintyBudget)

    def test_estimated_budget_SplitBudget(self):
        # init param test
        self._test_init_param_budget(SplitBudget)
        self._test_init_param_w(SplitBudget)
        self._test_init_param_theta(SplitBudget)
        self._test_init_param_s(SplitBudget)
        self._test_init_param_v(SplitBudget)

        # sampled param test
        self._test_sampled_param_utilities(SplitBudget)

        self._test_update_without_query(SplitBudget)

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

    def _test_init_param_w(self, budget_manager_name):
        # w must be defined as an int with a range of w > 0
        budget_manager = budget_manager_name(w="string")
        self.assertRaises(
            TypeError, budget_manager.query_by_utility, self.utilities
        )
        budget_manager = budget_manager_name(w=None)
        self.assertRaises(
            TypeError, budget_manager.query_by_utility, self.utilities
        )
        budget_manager = budget_manager_name(w=1.1)
        self.assertRaises(
            TypeError, budget_manager.query_by_utility, self.utilities
        )
        budget_manager = budget_manager_name(w=0)
        self.assertRaises(
            ValueError, budget_manager.query_by_utility, self.utilities
        )
        budget_manager = budget_manager_name(w=-1)
        self.assertRaises(
            ValueError, budget_manager.query_by_utility, self.utilities
        )

    def _test_init_param_num_classes(self, budget_manager_name):
        # num_classes must be defined as an int and greater than 0
        budget_manager = budget_manager_name(num_classes="string")
        self.assertRaises(
            TypeError, budget_manager.query_by_utility, self.utilities
        )
        budget_manager = budget_manager_name(num_classes=-1)
        self.assertRaises(
            ValueError, budget_manager.query_by_utility, self.utilities
        )
        budget_manager = budget_manager_name(num_classes=0)
        self.assertRaises(
            ValueError, budget_manager.query_by_utility, self.utilities
        )

    def _test_init_param_theta(self, budget_manager_name):
        # theta must be defined as a float
        budget_manager = budget_manager_name(theta="string")
        self.assertRaises(
            TypeError, budget_manager.query_by_utility, self.utilities
        )

    def _test_init_param_s(self, budget_manager_name):
        # s must be defined as a float with a range of: 0 < s <= 1
        budget_manager = budget_manager_name(s="string")
        self.assertRaises(
            TypeError, budget_manager.query_by_utility, self.utilities
        )
        budget_manager = budget_manager_name(s=1.1)
        self.assertRaises(
            ValueError, budget_manager.query_by_utility, self.utilities
        )
        budget_manager = budget_manager_name(s=0.0)
        self.assertRaises(
            ValueError, budget_manager.query_by_utility, self.utilities
        )
        budget_manager = budget_manager_name(s=-1.0)
        self.assertRaises(
            ValueError, budget_manager.query_by_utility, self.utilities
        )

    def _test_init_param_v(self, budget_manager_name):
        # v must be defined as an float with a range of: 0 < v < 1
        budget_manager = budget_manager_name(v="string")
        self.assertRaises(
            TypeError, budget_manager.query_by_utility, self.utilities
        )
        budget_manager = budget_manager_name(v=1.1)
        self.assertRaises(
            ValueError, budget_manager.query_by_utility, self.utilities
        )
        budget_manager = budget_manager_name(v=0.0)
        self.assertRaises(
            ValueError, budget_manager.query_by_utility, self.utilities
        )
        budget_manager = budget_manager_name(v=-1.0)
        self.assertRaises(
            ValueError, budget_manager.query_by_utility, self.utilities
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
