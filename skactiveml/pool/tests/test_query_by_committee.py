import unittest
from itertools import product

import numpy as np
from sklearn.ensemble import (
    BaggingClassifier,
    RandomForestClassifier,
    VotingClassifier,
    BaggingRegressor,
    VotingRegressor,
    AdaBoostRegressor, RandomForestRegressor,
)
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process import GaussianProcessRegressor

from skactiveml.classifier import ParzenWindowClassifier, SklearnClassifier
from skactiveml.pool._query_by_committee import (
    QueryByCommittee,
    average_kl_divergence,
    vote_entropy,
)
from skactiveml.pool.tests.provide_test_pool_regression import (
    provide_test_regression_query_strategy_init_random_state,
    provide_test_regression_query_strategy_init_missing_label,
    provide_test_regression_query_strategy_query_X,
    provide_test_regression_query_strategy_query_y,
    provide_test_regression_query_strategy_query_candidates,
    provide_test_regression_query_strategy_query_batch_size,
    provide_test_regression_query_strategy_query_return_utilities,
)
from skactiveml.regressor import NICKernelRegressor, SklearnRegressor
from skactiveml.tests.template_query_strategy import \
    TemplateSingleAnnotatorPoolQueryStrategy
from skactiveml.utils import MISSING_LABEL


class TestQueryByCommittee(TemplateSingleAnnotatorPoolQueryStrategy,
                           unittest.TestCase):
    def setUp(self):
        self.classes = [0, 1]
        self.ensemble_clf = SklearnClassifier(
            estimator=RandomForestClassifier(random_state=42),
            classes=self.classes,
            random_state=42,
        )
        self.ensemble_reg = SklearnRegressor(
            estimator=RandomForestRegressor(random_state=42),
            random_state=42,
        )
        query_default_params_clf = {
            'X': np.array([[1, 2], [5, 8], [8, 4], [5, 4]]),
            'y': np.array([0, 1, MISSING_LABEL, MISSING_LABEL]),
            'ensemble': self.ensemble_clf,
        }
        query_default_params_reg = {
            'X': np.array([[1, 2], [5, 8], [8, 4], [5, 4]]),
            'y': np.array([0, 1, MISSING_LABEL, MISSING_LABEL]),
            'ensemble': self.ensemble_reg,
        }
        super().setUp(qs_class=QueryByCommittee, init_default_params={},
                      query_default_params_clf=query_default_params_clf,
                      query_default_params_reg=query_default_params_reg)

    def test_init_param_method(self, test_cases=None):
        test_cases = [] if test_cases is None else test_cases
        test_cases += [(1, TypeError), ("string", TypeError)]
        self._test_param("init", "method", test_cases)

    def test_query_param_ensemble(self, test_cases=None):
        test_cases = [] if test_cases is None else test_cases
        test_cases += [
            (None, TypeError), ("test", TypeError), (1, TypeError),
            (ParzenWindowClassifier(classes=self.classes), TypeError),
            (GaussianProcessRegressor(), TypeError),
            (RandomForestRegressor(), TypeError),
            (RandomForestClassifier(), TypeError),
            (self.ensemble_clf, None),
            (self.ensemble_reg, None),
        ]
        self._test_param("query", "ensemble", test_cases)

    def test_query_param_sample_weight(self, test_cases=None):
        test_cases = [] if test_cases is None else test_cases
        X = self.query_default_params_clf['X']
        test_cases += [("string", ValueError), (X, ValueError),
                       (np.empty((len(X) - 1)), ValueError)]
        self._test_param("query", "sample_weight", test_cases)

    def test_query_param_fit_ensemble(self, test_cases=None):
        test_cases = [] if test_cases is None else test_cases
        test_cases += [("string", TypeError), (None, TypeError)]
        self._test_param("query", "fit_ensemble", test_cases)

    def test_query(self):
        ensemble_classifiers = [
            SklearnClassifier(
                classes=self.classes, estimator=GaussianProcessClassifier()
            ),
            SklearnClassifier(
                classes=self.classes, estimator=GaussianProcessClassifier()
            ),
            SklearnClassifier(
                classes=self.classes, estimator=GaussianProcessClassifier()
            ),
        ]
        gpc = ParzenWindowClassifier(classes=self.classes)
        ensemble_bagging = SklearnClassifier(
            estimator=BaggingClassifier(base_estimator=gpc),
            classes=self.classes,
        )
        ensemble_voting = SklearnClassifier(
            VotingClassifier(estimators=ensemble_classifiers, voting="soft")
        )
        ensemble_list = [
            self.ensemble_clf,
            self.ensemble_reg,
            ensemble_classifiers,
            ensemble_bagging,
            ensemble_voting,
        ]
        for ensemble in ensemble_list:
            query_params = self.query_default_params_clf
            query_params['ensemble'] = ensemble
            query_params['return_utilities'] = True
            for method in ["KL_divergence", "vote_entropy"]:
                qs = QueryByCommittee(method=method)
                idx, u = qs.query(**query_params)
                self.assertEqual(len(idx), 1)
                self.assertEqual(len(u), 1)


class TestAverageKlDivergence(unittest.TestCase):
    def setUp(self):
        self.probas = np.array(
            [[[0.3, 0.7], [0.4, 0.6]], [[0.2, 0.8], [0.5, 0.5]]]
        )
        self.scores = np.array([0.00670178182226764, 0.005059389928987596])

    def test_param_probas(self):
        self.assertRaises(ValueError, average_kl_divergence, "string")
        self.assertRaises(ValueError, average_kl_divergence, 1)
        self.assertRaises(ValueError, average_kl_divergence, np.ones((1,)))
        self.assertRaises(ValueError, average_kl_divergence, np.ones((1, 1)))
        self.assertRaises(
            ValueError, average_kl_divergence, np.ones((1, 1, 1, 1))
        )

    def test_average_kl_divergence(self):
        average_kl_divergence(np.full((10, 10, 10), 0.5))
        average_kl_divergence(np.zeros((10, 10, 10)))
        scores = average_kl_divergence(self.probas)
        np.testing.assert_allclose(scores, self.scores)


class TestVoteEntropy(unittest.TestCase):
    def setUp(self):
        self.classes = np.array([0, 1, 2])
        self.votes = np.array([[0, 0, 2], [1, 0, 2], [2, 1, 2]]).T
        self.scores = np.array(
            [-np.log(1 / 3), -2 / 3 * np.log(2 / 3) - 1 / 3 * np.log(1 / 3), 0]
        )

    def test_param_votes(self):
        self.assertRaises(
            ValueError, vote_entropy, votes="string", classes=self.classes
        )
        self.assertRaises(
            ValueError, vote_entropy, votes=1, classes=self.classes
        )
        self.assertRaises(
            ValueError, vote_entropy, votes=[1], classes=self.classes
        )
        self.assertRaises(
            ValueError, vote_entropy, votes=[[[1]]], classes=self.classes
        )
        self.assertRaises(
            ValueError,
            vote_entropy,
            votes=np.array([[10]]),
            classes=self.classes,
        )
        self.assertRaises(
            ValueError,
            vote_entropy,
            votes=np.full((9, 9), np.nan),
            classes=self.classes,
        )

    def test_param_classes(self):
        self.assertRaises(
            ValueError, vote_entropy, votes=self.votes, classes="string"
        )
        self.assertRaises(
            ValueError, vote_entropy, votes=self.votes, classes="class"
        )
        self.assertRaises(TypeError, vote_entropy, votes=self.votes, classes=1)
        self.assertRaises(
            TypeError, vote_entropy, votes=self.votes, classes=[[1]]
        )
        self.assertRaises(
            ValueError,
            vote_entropy,
            votes=self.votes,
            classes=[MISSING_LABEL, 1],
        )

    def test_vote_entropy(self):
        scores = vote_entropy(votes=self.votes, classes=self.classes)
        np.testing.assert_array_equal(scores.round(10), self.scores.round(10))


class TestQueryByCommitteeForRegression(unittest.TestCase):
    def setUp(self):
        self.random_state = 1
        self.candidates = np.array([[8, 1], [9, 1], [5, 1]])
        self.X = np.array([[1, 2], [5, 8], [8, 4], [5, 4]])
        self.y = np.array([0, 1, 2, -2])
        self.ensemble = SklearnRegressor(
            BaggingRegressor(
                NICKernelRegressor(), random_state=self.random_state
            ),
            random_state=self.random_state,
        )
        self.query_dict_regression_test = dict(ensemble=self.ensemble)
        self.query_dict = dict(
            ensemble=self.ensemble,
            fit_ensemble=True,
            X=self.X,
            y=self.y,
            candidates=self.candidates,
        )

    def test_init_param_random_state(self):
        provide_test_regression_query_strategy_init_random_state(
            self, QueryByCommittee, query_dict=self.query_dict_regression_test
        )

    def test_init_param_missing_label(self):
        provide_test_regression_query_strategy_init_missing_label(
            self,
            QueryByCommittee,
            init_dict={"missing_label": MISSING_LABEL},
            query_dict=self.query_dict_regression_test,
            missing_label_params_query_dict=["ensemble"],
        )

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
            self.assertRaises(
                (ValueError, TypeError), qs.query, **self.query_dict
            )

    def test_query_param_fit_ensemble(self):
        for illegal_fit_ensemble in ["illegal", dict]:
            self.query_dict["fit_ensemble"] = illegal_fit_ensemble
            qs = QueryByCommittee()
            self.assertRaises(
                (ValueError, TypeError), qs.query, **self.query_dict
            )

    def test_query_param_sample_weight(self):
        for illegal_sample_weight in ["illegal", dict]:
            self.query_dict["sample_weight"] = illegal_sample_weight
            qs = QueryByCommittee()
            self.assertRaises(
                (ValueError, TypeError), qs.query, **self.query_dict
            )

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
        ensemble_bagging = SklearnRegressor(
            BaggingRegressor(base_estimator=nic_reg)
        )
        ensemble_voting = SklearnRegressor(
            VotingRegressor(
                estimators=[
                    (f"gpr{i}", reg)
                    for i, reg in enumerate(ensemble_regressors)
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
        for ensemble_regressors, fit_ensemble in product(
            ensemble_list, [True, False]
        ):
            qs = QueryByCommittee()
            self.query_dict["ensemble"] = ensemble_regressors
            self.query_dict["fit_ensemble"] = fit_ensemble

            indices, utilities = qs.query(**self.query_dict)
            self.assertEqual(indices.shape, (1,))
            self.assertEqual(utilities.shape, (1, len(self.candidates)))
