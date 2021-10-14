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
        self._test_init_param_save_utilities(BIQF)

        # sample param test
        self._test_query_param_utilities(BIQF)

        # update test
        self._test_update_without_query(BIQF)

    def _test_init_param_budget(self, budget_manager_name):
        # budget must be defined as a float with a range of: 0 < budget <= 1
        budget_manager = budget_manager_name(budget="string")
        self.assertRaises(TypeError, budget_manager.query, self.utilities)
        budget_manager = budget_manager_name(budget=1.1)
        self.assertRaises(ValueError, budget_manager.query, self.utilities)
        budget_manager = budget_manager_name(budget=-1.0)
        self.assertRaises(ValueError, budget_manager.query, self.utilities)

    def _test_init_param_w(self, budget_manager_name):
        # w must be defined as an int with a range of w > 0
        budget_manager = budget_manager_name(w="string")
        self.assertRaises(TypeError, budget_manager.query, self.utilities)
        budget_manager = budget_manager_name(w=None)
        self.assertRaises(TypeError, budget_manager.query, self.utilities)
        budget_manager = budget_manager_name(w=1.1)
        self.assertRaises(TypeError, budget_manager.query, self.utilities)
        budget_manager = budget_manager_name(w=0)
        self.assertRaises(ValueError, budget_manager.query, self.utilities)
        budget_manager = budget_manager_name(w=-1)
        self.assertRaises(ValueError, budget_manager.query, self.utilities)

    def _test_init_param_w_tol(self, budget_manager_name):
        # w must be defined as an int with a range of w_tol > 0
        budget_manager = budget_manager_name(w_tol="string")
        self.assertRaises(TypeError, budget_manager.query, self.utilities)
        budget_manager = budget_manager_name(w_tol=None)
        self.assertRaises(TypeError, budget_manager.query, self.utilities)
        budget_manager = budget_manager_name(w_tol=0)
        self.assertRaises(ValueError, budget_manager.query, self.utilities)
        budget_manager = budget_manager_name(w_tol=-1)
        self.assertRaises(ValueError, budget_manager.query, self.utilities)

    def _test_init_param_save_utilities(self, budget_manager_name):
        # w must be defined as an int with a range of w_tol > 0
        budget_manager = budget_manager_name(save_utilities="string")
        self.assertRaises(TypeError, budget_manager.query, self.utilities)
        budget_manager = budget_manager_name(save_utilities=None)
        self.assertRaises(TypeError, budget_manager.query, self.utilities)
        budget_manager = budget_manager_name(save_utilities=0)
        self.assertRaises(TypeError, budget_manager.query, self.utilities)
        budget_manager = budget_manager_name(save_utilities=-1)
        self.assertRaises(TypeError, budget_manager.query, self.utilities)

    def _test_query_param_utilities(self, budget_manager_name):
        # s must be defined as a float ndarray
        budget_manager = budget_manager_name()
        self.assertRaises(TypeError, budget_manager.query, utilities="string")
        self.assertRaises(TypeError, budget_manager.query, utilities=None)

    def _test_update_without_query(self, budget_manager_name):
        budget_manager = budget_manager_name(save_utilities=False)
        budget_manager.update(
            np.array([[0], [1], [2]]),
            np.array([0, 2]),
            utilities=[0.9, 0.1, 0.8]
        )
        self.assertRaises(
            ValueError,
            budget_manager.update,
            np.array([[0], [1], [2]]),
            np.array([0, 2])
        )
