import unittest

import numpy as np

from skactiveml.stream.budgetmanager import DensityBasedSplitBudgetManager
from skactiveml.tests.template_budget_manager import (
    TemplateBudgetManager,
)


class TestDensityBasedSplitBudgetManager(
    TemplateBudgetManager, unittest.TestCase
):
    def setUp(self):
        query_by_utility_params = {
            "utilities": np.array([[0.5]]),
        }
        super().setUp(
            bm_class=DensityBasedSplitBudgetManager,
            init_default_params={},
            query_by_utility_params=query_by_utility_params,
        )

    def test_init_param_theta(self, test_cases=None):
        test_cases = [] if test_cases is None else test_cases
        test_cases += [(1.0, None), ("string", TypeError)]
        self._test_param("init", "theta", test_cases)

    def test_init_param_delta(self, test_cases=None):
        test_cases = [] if test_cases is None else test_cases
        test_cases += [(1.0, None), (0.0, ValueError), ("string", TypeError)]
        self._test_param("init", "delta", test_cases)

    def test_init_param_s(self, test_cases=None):
        test_cases = [] if test_cases is None else test_cases
        test_cases += [
            (1.0, None),
            (1.1, ValueError),
            (0.0, ValueError),
            (-1.0, ValueError),
            ("string", TypeError),
        ]
        self._test_param("init", "s", test_cases)

    def test_query_by_utility(
        self,
    ):
        expected_output = [0, 10, 20, 30, 40]
        return super().test_query_by_utility(expected_output)
