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

    def test_update_param_utilities(self, test_cases=None):
        test_cases = [] if test_cases is None else test_cases
        test_cases += [
            ([], ValueError),
            ([0.1], ValueError),
            ([0.1, 0.2, 0.3], ValueError),
            ([[0.1], [0.2]], ValueError),
            (0.1, ValueError),
            (None, ValueError),
            ([0.1, 0.2], None),
            (np.array([0.1, 0.2]), None),
        ]
        self._test_param("update", "utilities", test_cases)

    def test_tied_utilities_obey_budget(self):
        expected_output = np.arange(0, 1000, 10)
        candidates = np.zeros((1000, 1))

        for utility_value in [0.0, 1.0]:
            with self.subTest(utility_value=utility_value):
                utilities = np.full(1000, utility_value)
                bm = BalancedIncrementalQuantileFilter(budget=0.1)

                queried_indices = bm.query_by_utility(utilities)
                np.testing.assert_array_equal(queried_indices, expected_output)
                np.testing.assert_array_equal(
                    bm.query_by_utility(utilities), expected_output
                )
                self.assertEqual(bm.observed_samples_, 0)
                self.assertEqual(bm.queried_samples_, 0)
                self.assertEqual(list(bm.history_sorted_), [])

                bm.update(candidates, queried_indices, utilities)
                history_before_query = list(bm.history_sorted_)
                queried_indices = bm.query_by_utility(utilities)
                np.testing.assert_array_equal(queried_indices, expected_output)
                self.assertEqual(bm.observed_samples_, 1000)
                self.assertEqual(bm.queried_samples_, 100)
                self.assertEqual(
                    list(bm.history_sorted_), history_before_query
                )

                bm.update(candidates, queried_indices, utilities)
                self.assertEqual(bm.observed_samples_, 2000)
                self.assertEqual(bm.queried_samples_, 200)

    def test_tied_utilities_batch_matches_sequential_queries(self):
        utilities = np.ones(100)
        expected_output = np.arange(0, 100, 10)

        batch_bm = BalancedIncrementalQuantileFilter(budget=0.1)
        batch_output = batch_bm.query_by_utility(utilities)

        sequential_bm = BalancedIncrementalQuantileFilter(budget=0.1)
        sequential_output = []
        for i, utility in enumerate(utilities):
            queried_indices = sequential_bm.query_by_utility(
                np.array([utility])
            )
            if queried_indices:
                sequential_output.append(i)
            sequential_bm.update(
                np.zeros((1, 1)), queried_indices, np.array([utility])
            )

        np.testing.assert_array_equal(batch_output, expected_output)
        np.testing.assert_array_equal(sequential_output, expected_output)

    def test_tied_utilities_with_full_budget(self):
        utilities = np.ones(100)
        bm = BalancedIncrementalQuantileFilter(budget=1.0)
        np.testing.assert_array_equal(
            bm.query_by_utility(utilities), np.arange(100)
        )
