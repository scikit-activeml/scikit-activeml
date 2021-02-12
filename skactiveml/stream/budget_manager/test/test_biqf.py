import unittest
import numpy as np

from .._biqf import BIQF


class TestBIQF(unittest.TestCase):
    def setUp(self):
        # initialise var for sampled var tests
        self.utilities = np.array([True, False])

    def test_biqf(self):
        # init param test
        self._test_init_param_budget(BIQF)
        self._test_init_param_w(BIQF)
        self._test_init_param_w_tol(BIQF)

        # sample param test
        self._test_sampled_param_utilities(BIQF)

    def _test_init_param_budget(self, budget_manager_name):
        # budget must be defined as a float with a range of: 0 < budget <= 1
        budget_manager = budget_manager_name(budget="string")
        self.assertRaises(TypeError, budget_manager.sample, self.utilities)
        budget_manager = budget_manager_name(budget=1.1)
        self.assertRaises(ValueError, budget_manager.sample, self.utilities)
        budget_manager = budget_manager_name(budget=-1.0)
        self.assertRaises(ValueError, budget_manager.sample, self.utilities)
    
    def _test_init_param_w(self, budget_manager_name):
        # w must be defined as an int with a range of w > 0
        budget_manager = budget_manager_name(w="string")
        self.assertRaises(TypeError, budget_manager.sample, self.utilities)
        budget_manager = budget_manager_name(w=None)
        self.assertRaises(TypeError, budget_manager.sample, self.utilities)
        budget_manager = budget_manager_name(w=1.1)
        self.assertRaises(TypeError, budget_manager.sample, self.utilities)
        budget_manager = budget_manager_name(w=0)
        self.assertRaises(ValueError, budget_manager.sample, self.utilities)
        budget_manager = budget_manager_name(w=-1)
        self.assertRaises(ValueError, budget_manager.sample, self.utilities)

    def _test_init_param_w_tol(self, budget_manager_name):
        # w must be defined as an int with a range of w_tol > 0
        budget_manager = budget_manager_name(w="string")
        self.assertRaises(TypeError, budget_manager.sample, self.utilities)
        budget_manager = budget_manager_name(w=None)
        self.assertRaises(TypeError, budget_manager.sample, self.utilities)
        budget_manager = budget_manager_name(w=0)
        self.assertRaises(ValueError, budget_manager.sample, self.utilities)
        budget_manager = budget_manager_name(w=-1)
        self.assertRaises(ValueError, budget_manager.sample, self.utilities)

    def _test_sampled_param_utilities(self, budget_manager_name):
        # s must be defined as a float ndarray
        budget_manager = budget_manager_name()
        self.assertRaises(TypeError, budget_manager.sample, utilities="string")
        self.assertRaises(TypeError, budget_manager.sample, utilities=None)