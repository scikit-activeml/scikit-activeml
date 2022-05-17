import unittest

import numpy as np

from skactiveml.stream.budgetmanager import (
    FixedUncertaintyBudgetManager,
    VariableUncertaintyBudgetManager,
    SplitBudgetManager,
    RandomVariableUncertaintyBudgetManager,
)


class TemplateTestEstimatedBudgetZliobaite:
    def setUp(self):
        # initialise var for sampled var tests
        self.utilities = np.array([True, False])

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

    def test_init_param_w(self):
        # w must be defined as an int with a range of w > 0
        budget_manager = self.get_budget_manager()(w="string")
        self.assertRaises(
            TypeError, budget_manager.query_by_utility, self.utilities
        )
        budget_manager = self.get_budget_manager()(w=None)
        self.assertRaises(
            TypeError, budget_manager.query_by_utility, self.utilities
        )
        budget_manager = self.get_budget_manager()(w=1.1)
        self.assertRaises(
            TypeError, budget_manager.query_by_utility, self.utilities
        )
        budget_manager = self.get_budget_manager()(w=0)
        self.assertRaises(
            ValueError, budget_manager.query_by_utility, self.utilities
        )
        budget_manager = self.get_budget_manager()(w=-1)
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


class TestFixedUncertaintyBudgetManager(
    TemplateTestEstimatedBudgetZliobaite, unittest.TestCase
):
    def get_budget_manager(self):
        return FixedUncertaintyBudgetManager

    def test_init_param_num_classes(self):
        # num_classes must be defined as an int and greater than 0
        budget_manager = self.get_budget_manager()(num_classes="string")
        self.assertRaises(
            TypeError, budget_manager.query_by_utility, self.utilities
        )

        budget_manager = self.get_budget_manager()(num_classes=-1)
        self.assertRaises(
            ValueError, budget_manager.query_by_utility, self.utilities
        )

        budget_manager = self.get_budget_manager()(num_classes=0)
        self.assertRaises(
            ValueError, budget_manager.query_by_utility, self.utilities
        )


class TestVariableUncertaintyBudgetManager(
    TemplateTestEstimatedBudgetZliobaite, unittest.TestCase
):
    def get_budget_manager(self):
        return VariableUncertaintyBudgetManager

    def test_init_param_theta(self):
        # theta must be defined as a float
        budget_manager = self.get_budget_manager()(theta="string")
        self.assertRaises(
            TypeError, budget_manager.query_by_utility, self.utilities
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


class TestRandomVariableUncertaintyBudgetManager(
    TemplateTestEstimatedBudgetZliobaite, unittest.TestCase
):
    def get_budget_manager(self):
        return RandomVariableUncertaintyBudgetManager

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


class TestSplitBudgetManager(TestVariableUncertaintyBudgetManager):
    def get_budget_manager(self):
        return SplitBudgetManager

    def test_init_param_random_state(self):
        # v must be defined as an float with a range of: 0 < v < 1
        budget_manager = self.get_budget_manager()(random_state="string")
        self.assertRaises(
            ValueError, budget_manager.query_by_utility, self.utilities
        )

    def test_init_param_v(self):
        # v must be defined as an float with a range of: 0 < v < 1
        budget_manager = self.get_budget_manager()(v="string")
        self.assertRaises(
            TypeError, budget_manager.query_by_utility, self.utilities
        )
        budget_manager = self.get_budget_manager()(v=1.1)
        self.assertRaises(
            ValueError, budget_manager.query_by_utility, self.utilities
        )
        budget_manager = self.get_budget_manager()(v=0.0)
        self.assertRaises(
            ValueError, budget_manager.query_by_utility, self.utilities
        )
        budget_manager = self.get_budget_manager()(v=-1.0)
        self.assertRaises(
            ValueError, budget_manager.query_by_utility, self.utilities
        )
