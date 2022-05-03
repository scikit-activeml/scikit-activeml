import unittest
from itertools import product

import numpy as np
from sklearn.ensemble import BaggingRegressor, VotingRegressor, AdaBoostRegressor
from sklearn.gaussian_process import GaussianProcessRegressor

from skactiveml.classifier import ParzenWindowClassifier
from skactiveml.pool.regression import QueryByCommittee
from skactiveml.pool.regression.tests.provide_test_pool_regression import (
    provide_test_regression_query_strategy_init_random_state,
    provide_test_regression_query_strategy_init_missing_label,
    provide_test_regression_query_strategy_query_X,
    provide_test_regression_query_strategy_query_y,
    provide_test_regression_query_strategy_query_candidates,
    provide_test_regression_query_strategy_query_batch_size,
    provide_test_regression_query_strategy_query_return_utilities,
)
from skactiveml.regressor import NICKernelRegressor, SklearnRegressor


class TestQueryByCommittee(unittest.TestCase):
    def setUp(self):
        self.random_state = 1
        self.candidates = np.array([[8, 1], [9, 1], [5, 1]])
        self.X = np.array([[1, 2], [5, 8], [8, 4], [5, 4]])
        self.y = np.array([0, 1, 2, -2])
        self.ensemble = NICKernelRegressor()
        self.query_dict_regression_test = dict(ensemble=self.ensemble)
        self.query_dict = dict(
            ensemble=self.ensemble, X=self.X, y=self.y, candidates=self.candidates
        )

    def test_init_param_random_state(self):
        provide_test_regression_query_strategy_init_random_state(
            self, QueryByCommittee, query_dict=self.query_dict_regression_test
        )

    def test_init_param_missing_label(self):
        provide_test_regression_query_strategy_init_missing_label(
            self, QueryByCommittee, query_dict=self.query_dict_regression_test
        )

    def test_init_param_k_boostrap(self):
        for illegal_k_boostrap in [7.4, "illegal"]:
            self.query_dict["ensemble"] = illegal_k_boostrap
            qs = QueryByCommittee()
            self.assertRaises((ValueError, TypeError), qs.query, **self.query_dict)

    def test_init_param_n_train(self):
        for illegal_k_boostrap in [6.0, "illegal"]:
            self.query_dict["ensemble"] = illegal_k_boostrap
            qs = QueryByCommittee()
            self.assertRaises((ValueError, TypeError), qs.query, **self.query_dict)

    def test_query_param_X(self):
        provide_test_regression_query_strategy_query_X(
            self, QueryByCommittee, query_dict=self.query_dict_regression_test
        )

    def test_query_param_y(self):
        provide_test_regression_query_strategy_query_y(
            self, QueryByCommittee, query_dict=self.query_dict_regression_test
        )

    def test_query_param_ensemble(self):
        for illegal_ensemble in [ParzenWindowClassifier(), "illegal"]:
            self.query_dict["ensemble"] = illegal_ensemble
            qs = QueryByCommittee()
            self.assertRaises((ValueError, TypeError), qs.query, **self.query_dict)

    def test_query_param_fit_ensemble(self):
        for illegal_fit_ensemble in ["illegal", dict]:
            self.query_dict["fit_ensemble"] = illegal_fit_ensemble
            qs = QueryByCommittee()
            self.assertRaises((ValueError, TypeError), qs.query, **self.query_dict)

    def test_query_param_sample_weight(self):
        for illegal_sample_weight in ["illegal", dict]:
            self.query_dict["sample_weight"] = illegal_sample_weight
            qs = QueryByCommittee()
            self.assertRaises((ValueError, TypeError), qs.query, **self.query_dict)

    def test_query_param_candidates(self):
        provide_test_regression_query_strategy_query_candidates(
            self, QueryByCommittee, query_dict=self.query_dict_regression_test
        )

    def test_query_param_batch_size(self):
        provide_test_regression_query_strategy_query_batch_size(
            self, QueryByCommittee, query_dict=self.query_dict_regression_test
        )

    def test_query_param_return_utilities(self):
        provide_test_regression_query_strategy_query_return_utilities(
            self, QueryByCommittee, query_dict=self.query_dict_regression_test
        )

    def test_query_ensemble_fit_ensemble(self):
        ensemble_regressors = [
            SklearnRegressor(estimator=GaussianProcessRegressor()),
            SklearnRegressor(estimator=GaussianProcessRegressor()),
            SklearnRegressor(estimator=GaussianProcessRegressor()),
        ]
        nic_reg = NICKernelRegressor()
        ensemble_bagging = SklearnRegressor(BaggingRegressor(base_estimator=nic_reg))
        ensemble_voting = SklearnRegressor(
            VotingRegressor(
                estimators=[
                    (f"gpr{i}", reg) for i, reg in enumerate(ensemble_regressors)
                ]
            )
        )
        ensemble_adaboost = SklearnRegressor(AdaBoostRegressor(n_estimators=3))
        ensemble_list = [
            self.ensemble,
            ensemble_regressors,
            ensemble_bagging,
            ensemble_voting,
            ensemble_adaboost,
        ]

        for ensemble_regressors in ensemble_list:
            if isinstance(ensemble_regressors, list):
                for ensemble_regressor in ensemble_regressors:
                    ensemble_regressor.fit(self.X, self.y)
            else:
                ensemble_regressors.fit(self.X, self.y)

        self.query_dict["return_utilities"] = True
        for ensemble_regressors, fit_ensemble in product(ensemble_list, [True, False]):
            qs = QueryByCommittee()
            self.query_dict["ensemble"] = ensemble_regressors
            self.query_dict["fit_ensemble"] = fit_ensemble

            indices, utilities = qs.query(**self.query_dict)
            self.assertEqual(indices.shape, (1,))
            self.assertEqual(utilities.shape, (1, len(self.candidates)))
