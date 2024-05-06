import unittest

import numpy as np

from skactiveml.stream.budgetmanager import BalancedIncrementalQuantileFilter

from skactiveml.tests.template_budget_manager import (
    TemplateBudgetManager,
)


class TestBalancedIncrementalQuantileFilter(
    TemplateBudgetManager, unittest.TestCase
):
    def setUp(self):
        query_by_utility_params = {
            "utilities": np.array([[0.5]]),
        }
        super().setUp(
            bm_class=BalancedIncrementalQuantileFilter,
            init_default_params={},
            query_by_utility_params=query_by_utility_params,
        )

    def test_init_param_w(self, test_cases=None):
        # w must be defined as an int with a range of w > 0
        test_cases = [] if test_cases is None else test_cases
        test_cases += [
            (10, None),
            (1.1, TypeError),
            (0, ValueError),
            (-1, ValueError),
            ("string", TypeError),
        ]
        self._test_param("init", "w", test_cases)

    def test_init_param_w_tol(self, test_cases=None):
        # w must be defined as an int with a range of w_tol > 0
        test_cases = [] if test_cases is None else test_cases
        test_cases += [
            (10, None),
            (None, TypeError),
            (0, ValueError),
            (-1, ValueError),
            ("string", TypeError),
        ]
        self._test_param("init", "w_tol", test_cases)

    def test_query_by_utility(
        self,
    ):
        expected_output = [0, 1, 7, 8, 20]
        return super().test_query_by_utility(expected_output)
