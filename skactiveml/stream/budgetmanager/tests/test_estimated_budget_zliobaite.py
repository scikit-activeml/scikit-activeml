import unittest
import inspect
import numpy as np

from skactiveml.stream.budgetmanager import (
    FixedUncertaintyBudgetManager,
    VariableUncertaintyBudgetManager,
    SplitBudgetManager,
    RandomVariableUncertaintyBudgetManager,
    RandomBudgetManager,
)
from skactiveml.tests.template_budget_manager import (
    TemplateBudgetManager,
)


class TemplateEstimatedBudgetManager(TemplateBudgetManager):
    def setUp(
        self,
        bm_class,
        init_default_params,
        query_by_utility_params,
    ):
        super().setUp(
            bm_class=bm_class,
            init_default_params=init_default_params,
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

    def test_init_param_theta(self, test_cases=None):
        # theta must be defined as a float
        init_params_list = inspect.signature(self.bm_class.__init__).parameters
        if "theta" in init_params_list:
            test_cases = [] if test_cases is None else test_cases
            test_cases += [(1.0, None), ("string", TypeError)]
            self._test_param("init", "theta", test_cases)

    def test_init_param_s(self, test_cases=None):
        # s must be defined as a float with a range of: 0 < s <= 1
        init_params_list = inspect.signature(self.bm_class.__init__).parameters
        if "s" in init_params_list:
            test_cases = [] if test_cases is None else test_cases
            test_cases += [
                (1.0, None),
                (0.01, None),
                (1.1, ValueError),
                (0.0, ValueError),
                (-1.0, ValueError),
                ("string", TypeError),
            ]
            self._test_param("init", "s", test_cases)


class TestFixedUncertaintyBudgetManager(
    TemplateEstimatedBudgetManager, unittest.TestCase
):
    def setUp(self):
        init_default_params = {
            "classes": np.array([0, 1]),
        }
        query_by_utility_params = {
            "utilities": np.array([[0.5]]),
        }
        super().setUp(
            bm_class=FixedUncertaintyBudgetManager,
            init_default_params=init_default_params,
            query_by_utility_params=query_by_utility_params,
        )

    def test_init_param_classes(self):
        # classes must be defined
        test_cases = []
        test_cases += [
            ([0, 1], None),
            (["False", "True"], None),
            (2, TypeError),
            ("string", None),
            (-1, TypeError),
            (0, TypeError),
        ]
        self._test_param("init", "classes", test_cases)

    def test_query_by_utility(
        self,
    ):
        expected_output = [0, 1, 2, 3, 5, 7, 8, 10, 11, 12, 13, 18, 27, 37]
        return super().test_query_by_utility(expected_output)


class TestVariableUncertaintyBudgetManager(
    TemplateEstimatedBudgetManager, unittest.TestCase
):
    def setUp(self):
        query_by_utility_params = {
            "utilities": np.array([[0.5]]),
        }
        super().setUp(
            bm_class=VariableUncertaintyBudgetManager,
            init_default_params={},
            query_by_utility_params=query_by_utility_params,
        )

    def test_query_by_utility(
        self,
    ):
        expected_output = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 17, 26, 35, 45]
        return super().test_query_by_utility(expected_output)


class TestRandomVariableUncertaintyBudgetManager(
    TemplateEstimatedBudgetManager, unittest.TestCase
):
    def setUp(self):
        query_by_utility_params = {
            "utilities": np.array([[0.5]]),
        }
        super().setUp(
            bm_class=RandomVariableUncertaintyBudgetManager,
            init_default_params={},
            query_by_utility_params=query_by_utility_params,
        )

    def test_init_param_delta(self, test_cases=None):
        test_cases = [] if test_cases is None else test_cases
        test_cases += [(1.0, None), (0.0, ValueError), ("string", TypeError)]
        self._test_param("init", "delta", test_cases)

    def test_query_by_utility(
        self,
    ):
        expected_output = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 17, 26, 36, 45]
        return super().test_query_by_utility(expected_output)


class TestSplitBudgetManager(
    TemplateEstimatedBudgetManager, unittest.TestCase
):
    def setUp(self):
        query_by_utility_params = {
            "utilities": np.array([[0.5]]),
        }
        super().setUp(
            bm_class=SplitBudgetManager,
            init_default_params={},
            query_by_utility_params=query_by_utility_params,
        )

    def test_init_param_v(self, test_cases=None):
        # v must be defined as an float with a range of: 0 < v < 1
        test_cases = [] if test_cases is None else test_cases
        test_cases += [
            (0.2, None),
            (1.0, ValueError),
            (1.1, ValueError),
            (0.0, ValueError),
            (-1.0, ValueError),
            ("string", TypeError),
        ]
        self._test_param("init", "v", test_cases)

    def test_query_by_utility(
        self,
    ):
        expected_output = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 17, 26, 35, 46]
        return super().test_query_by_utility(expected_output)


class TestRandomBudgetManager(
    TemplateEstimatedBudgetManager, unittest.TestCase
):
    def setUp(self):
        query_by_utility_params = {
            "utilities": np.array([[0.5]]),
        }
        super().setUp(
            bm_class=RandomBudgetManager,
            init_default_params={},
            query_by_utility_params=query_by_utility_params,
        )

    def test_query_by_utility(
        self,
    ):
        expected_output = [14, 15, 16, 34, 43]
        return super().test_query_by_utility(expected_output)
