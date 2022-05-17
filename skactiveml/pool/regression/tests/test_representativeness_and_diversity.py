import unittest

import numpy as np

from skactiveml.pool.regression import (
    RepresentativenessDiversity,
    QueryByCommittee,
)
from skactiveml.pool.regression.tests.provide_test_pool_regression import (
    provide_test_regression_query_strategy_init_random_state,
    provide_test_regression_query_strategy_init_missing_label,
    provide_test_regression_query_strategy_query_X,
    provide_test_regression_query_strategy_query_y,
    provide_test_regression_query_strategy_query_candidates,
    provide_test_regression_query_strategy_query_batch_size,
    provide_test_regression_query_strategy_query_return_utilities,
)
from skactiveml.regressor import NICKernelRegressor


class TestRepresentativenessDiversity(unittest.TestCase):
    def setUp(self):
        self.random_state = 1
        self.candidates = np.array([[8, 1], [9, 1], [5, 1]])
        self.X = np.array([[1, 2], [5, 8], [8, 4], [5, 4]])
        self.y = np.array([0, 1, 2, -2])
        self.reg = NICKernelRegressor()
        self.query_dict = dict(X=self.X, y=self.y, candidates=self.candidates)

    def test_init_param_random_state(self):
        provide_test_regression_query_strategy_init_random_state(
            self, RepresentativenessDiversity
        )

    def test_init_param_missing_label(self):
        provide_test_regression_query_strategy_init_missing_label(
            self, RepresentativenessDiversity
        )

    def test_init_param_inner_qs(self):
        for illegal_qs in ["illegal", dict]:
            qs = RepresentativenessDiversity(inner_qs=illegal_qs)
            self.assertRaises(
                (ValueError, TypeError), qs.query, **self.query_dict
            )

        inner_qs = QueryByCommittee()
        qs = RepresentativenessDiversity(inner_qs=inner_qs)
        self.query_dict["inner_qs_dict"] = dict(ensemble=NICKernelRegressor())
        indices, utilities = qs.query(**self.query_dict, return_utilities=True)
        self.assertEqual(indices.shape, (1,))
        self.assertEqual(utilities.shape, (1, len(self.candidates)))

    def test_query_param_X(self):
        provide_test_regression_query_strategy_query_X(
            self, RepresentativenessDiversity
        )

    def test_query_param_y(self):
        provide_test_regression_query_strategy_query_y(
            self, RepresentativenessDiversity
        )

    def test_query_param_sample_weight(self):
        for illegal_sample_weight in ["illegal", dict]:
            self.query_dict["sample_weight"] = illegal_sample_weight
            qs = RepresentativenessDiversity()
            self.assertRaises(
                (ValueError, TypeError), qs.query, **self.query_dict
            )

    def test_query_param_inner_qs_dict(self):
        for qs_dict in ["illegal", dict]:
            self.query_dict["inner_qs_dict"] = qs_dict
            qs = RepresentativenessDiversity()
            self.assertRaises(
                (ValueError, TypeError), qs.query, **self.query_dict
            )

    def test_query_param_candidates(self):
        provide_test_regression_query_strategy_query_candidates(
            self, RepresentativenessDiversity
        )

    def test_query_param_batch_size(self):
        provide_test_regression_query_strategy_query_batch_size(
            self, RepresentativenessDiversity
        )

    def test_query_param_return_utilities(self):
        provide_test_regression_query_strategy_query_return_utilities(
            self, RepresentativenessDiversity
        )
