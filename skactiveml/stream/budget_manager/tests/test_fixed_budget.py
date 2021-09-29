import unittest
import numpy as np

from .._fixed_budget import FixedBudget


class TestFixedBudget(unittest.TestCase):
    def setUp(self):
        # initialise var for sampled var tests
        self.utilities = np.array([True, False])

    def test_fixed_budget(self):
        # init param test
        self._test_init_param_budget(FixedBudget)

        # sampled param test
        self._test_sampled_param_utilities(FixedBudget)

    def _test_init_param_budget(self, budget_manager_name):
        # budget must be defined as a float with a range of: 0 < budget <= 1
        budget_manager = budget_manager_name(budget="string")
        self.assertRaises(TypeError, budget_manager.query, self.utilities)
        budget_manager = budget_manager_name(budget=1.1)
        self.assertRaises(ValueError, budget_manager.query, self.utilities)
        budget_manager = budget_manager_name(budget=-1.0)
        self.assertRaises(ValueError, budget_manager.query, self.utilities)

    def _test_sampled_param_utilities(self, budget_manager_name):
        # s must be defined as a float ndarray
        budget_manager = budget_manager_name()
        self.assertRaises(TypeError, budget_manager.query, utilities="string")
        self.assertRaises(TypeError, budget_manager.query, utilities=None)
        self.assertRaises(TypeError, budget_manager.query, utilities=[10, 10])
