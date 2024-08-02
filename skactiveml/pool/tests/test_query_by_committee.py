import numpy as np
import unittest

from copy import deepcopy

from sklearn import clone
from sklearn.ensemble import (
    BaggingClassifier,
    RandomForestClassifier,
    VotingClassifier,
    RandomForestRegressor,
)
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.utils.validation import NotFittedError

from skactiveml.classifier import ParzenWindowClassifier, SklearnClassifier
from skactiveml.pool._query_by_committee import (
    QueryByCommittee,
    average_kl_divergence,
    vote_entropy,
    variation_ratios,
)
from skactiveml.regressor import NICKernelRegressor, SklearnRegressor
from skactiveml.tests.template_query_strategy import (
    TemplateSingleAnnotatorPoolQueryStrategy,
)
from skactiveml.utils import MISSING_LABEL


class TestQueryByCommittee(
    TemplateSingleAnnotatorPoolQueryStrategy, unittest.TestCase
):
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
            "X": np.array([[1, 2], [5, 8], [8, 4], [5, 4]]),
            "y": np.array([0, 1, MISSING_LABEL, MISSING_LABEL]),
            "ensemble": self.ensemble_clf,
            "fit_ensemble": True,
        }
        query_default_params_reg = {
            "X": np.array([[1, 2], [5, 8], [8, 4], [5, 4]]),
            "y": np.array([0, 1, MISSING_LABEL, MISSING_LABEL]),
            "ensemble": self.ensemble_reg,
            "fit_ensemble": True,
        }
        super().setUp(
            qs_class=QueryByCommittee,
            init_default_params={},
            query_default_params_clf=query_default_params_clf,
            query_default_params_reg=query_default_params_reg,
        )

    def test_init_param_method(self, test_cases=None):
        test_cases = [] if test_cases is None else test_cases
        test_cases += [(1, TypeError), ("string", TypeError)]
        self._test_param("init", "method", test_cases)

    def test_init_param_eps(self):
        test_cases = [
            (0, ValueError),
            (1e-3, None),
            (0.1, None),
            ("1", TypeError),
            (1, ValueError),
        ]
        self._test_param("init", "eps", test_cases, exclude_reg=True)

    def test_init_param_sample_predictions_method_name(self):
        # Fails as the default ensemble from `setup` does not support sampling.
        test_cases = [
            (0, TypeError),
            (0.1, TypeError),
            ("Test", ValueError),
        ]
        self._test_param("init", "sample_predictions_method_name", test_cases)
        test_cases = [
            ("predict_proba", ValueError),
        ]
        self._test_param(
            "init",
            "sample_predictions_method_name",
            test_cases,
            replace_query_params={
                "ensemble": [
                    ParzenWindowClassifier(),
                    ParzenWindowClassifier(),
                ]
            },
        )
        test_cases = [
            ("sample_proba", None),
        ]
        self._test_param(
            "init",
            "sample_predictions_method_name",
            test_cases,
            replace_query_params={
                "ensemble": ParzenWindowClassifier(),
                "fit_ensemble": True,
            },
        )
        test_cases = [
            ("sample_y", None),
        ]
        self._test_param(
            "init",
            "sample_predictions_method_name",
            test_cases,
            replace_query_params={
                "ensemble": SklearnRegressor(GaussianProcessRegressor()),
                "fit_ensemble": True,
            },
        )

    def test_init_param_sample_predictions_dict(self):
        test_cases = [
            (None, None),
            ({}, None),
            ({"n_samples": 1000}, None),
            ("Test", ValueError),
            ({"Test": 2}, TypeError),
        ]
        self._test_param(
            "init",
            "sample_predictions_dict",
            test_cases,
            replace_init_params={
                "sample_predictions_method_name": "sample_proba",
            },
            replace_query_params={
                "ensemble": ParzenWindowClassifier(),
                "fit_ensemble": True,
            },
        )
        self._test_param(
            "init",
            "sample_predictions_dict",
            test_cases,
            replace_init_params={
                "sample_predictions_method_name": "sample_y",
            },
            replace_query_params={
                "ensemble": SklearnRegressor(GaussianProcessRegressor()),
                "fit_ensemble": True,
            },
        )
        test_cases = [
            (None, None),
            ({}, ValueError),
            ({"n_samples": 1000}, ValueError),
            ("Test", ValueError),
            ({"Test": 2}, ValueError),
        ]
        self._test_param(
            "init",
            "sample_predictions_dict",
            test_cases,
            replace_init_params={
                "sample_predictions_method_name": None,
            },
        )

    def test_query_param_ensemble(self):
        estimators = [
            ("pwc1", ParzenWindowClassifier()),
            ("pwc2", ParzenWindowClassifier()),
        ]
        vote = SklearnClassifier(
            VotingClassifier(estimators=estimators, voting="soft"),
            classes=[0, 1],
        )
        test_cases = [(vote, None)]
        self._test_param(
            "query",
            "ensemble",
            test_cases,
            replace_query_params={
                "fit_ensemble": True,
                "y": np.full(4, np.nan),
            },
        )
        test_cases = [] if test_cases is None else test_cases
        test_cases += [
            (None, TypeError),
            ("test", TypeError),
            (1, TypeError),
            (ParzenWindowClassifier(), TypeError),
            (GaussianProcessRegressor(), TypeError),
            (RandomForestRegressor(), TypeError),
            (RandomForestClassifier(), TypeError),
            (
                [GaussianProcessRegressor(), GaussianProcessRegressor()],
                TypeError,
            ),
            (
                [GaussianProcessClassifier(), GaussianProcessRegressor()],
                TypeError,
            ),
            ([NICKernelRegressor(), ParzenWindowClassifier()], TypeError),
            (self.ensemble_clf, None),
            (self.ensemble_reg, None),
            ([NICKernelRegressor(), NICKernelRegressor()], None),
            ([ParzenWindowClassifier(), ParzenWindowClassifier()], None),
            (vote, None),
        ]
        self._test_param("query", "ensemble", test_cases)
        X = self.query_default_params_clf["X"]
        y = self.query_default_params_clf["y"]
        vote = vote.fit(X=X, y=y)
        pwc_list = [ParzenWindowClassifier(), ParzenWindowClassifier()]
        test_cases = [(vote, None), (pwc_list, NotFittedError)]
        self._test_param(
            "query",
            "ensemble",
            test_cases,
            replace_query_params={"fit_ensemble": False},
        )

    def test_query_param_y(self, test_cases=None):
        y = self.query_default_params_clf["y"]
        test_cases = [(y, None), (np.vstack([y, y]), ValueError)]
        self._test_param("query", "y", test_cases, exclude_reg=True)

        for ml, classes, t, err in [
            (np.nan, [1.0, 2.0], float, None),
            (0, [1, 2], int, None),
            (None, [1, 2], object, None),
            (None, ["A", "B"], object, None),
            ("", ["A", "B"], str, None),
        ]:
            print(ml, classes, t, err)
            replace_init_params = {"missing_label": ml}

            ensemble = clone(self.query_default_params_clf["ensemble"])
            ensemble.missing_label = ml
            ensemble.classes = classes
            replace_query_params = {"ensemble": ensemble}

            replace_y = np.full_like(y, ml, dtype=t)
            replace_y[0] = classes[0]
            replace_y[1] = classes[1]
            test_cases = [(replace_y, err)]
            self._test_param(
                "query",
                "y",
                test_cases,
                replace_init_params=replace_init_params,
                replace_query_params=replace_query_params,
                exclude_reg=True,
            )

    def test_query_param_fit_ensemble(self, test_cases=None):
        test_cases = [] if test_cases is None else test_cases
        test_cases += [("string", TypeError), (None, TypeError)]
        self._test_param("query", "fit_ensemble", test_cases)

    def test_query_param_sample_weight(self, test_cases=None):
        super().test_query_param_sample_weight(test_cases)

        clf = SklearnClassifier(RandomForestClassifier(random_state=42))
        replace_clf = {"ensemble": [clf] * 5}
        reg = SklearnRegressor(RandomForestRegressor(random_state=42))
        replace_reg = {"ensemble": [reg] * 5}

        for exclude_clf, exclude_reg, query_params, replace_query_params in [
            (False, True, self.query_default_params_clf, replace_clf),
            (True, False, self.query_default_params_reg, replace_reg),
        ]:
            if query_params is not None:
                y = query_params["y"]
                test_cases = [
                    (np.ones(len(y)), None),
                    (np.ones(len(y) + 1), ValueError),
                ]
                self._test_param(
                    "query",
                    "sample_weight",
                    test_cases,
                    exclude_clf=exclude_clf,
                    exclude_reg=exclude_reg,
                    replace_query_params=replace_query_params,
                )

    def test_query(self):
        voting_classifiers = [
            (
                "gp1",
                SklearnClassifier(
                    classes=self.classes, estimator=GaussianProcessClassifier()
                ),
            ),
            (
                "gp2",
                SklearnClassifier(
                    classes=self.classes, estimator=GaussianProcessClassifier()
                ),
            ),
            (
                "gp3",
                SklearnClassifier(
                    classes=self.classes, estimator=GaussianProcessClassifier()
                ),
            ),
        ]
        ensemble_classifiers = [member[1] for member in voting_classifiers]
        gpc = ParzenWindowClassifier(classes=self.classes)
        ensemble_bagging = SklearnClassifier(
            estimator=BaggingClassifier(estimator=gpc),
            classes=self.classes,
        )
        ensemble_voting = SklearnClassifier(
            VotingClassifier(estimators=voting_classifiers, voting="soft")
        )
        ensemble_array_reg = [NICKernelRegressor(), NICKernelRegressor()]
        ensemble_array_clf = [
            ParzenWindowClassifier(classes=self.classes),
            ParzenWindowClassifier(classes=self.classes),
        ]
        ensemble_list = [
            self.ensemble_clf,
            self.ensemble_reg,
            ensemble_classifiers,
            ensemble_bagging,
            ensemble_voting,
            ensemble_array_reg,
            ensemble_array_clf,
        ]
        for ensemble in ensemble_list:
            query_params = deepcopy(self.query_default_params_clf)
            query_params["ensemble"] = ensemble
            query_params["return_utilities"] = True
            for m in ["KL_divergence", "vote_entropy", "variation_ratios"]:
                qs = QueryByCommittee(method=m)
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
        np.testing.assert_almost_equal(scores, self.scores)


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
        np.testing.assert_almost_equal(scores, self.scores)


class TestVariationRatios(unittest.TestCase):
    def setUp(self):
        self.votes = np.array(
            [
                [0, 0, 2],
                [1, 0, 2],
                [2, 1, 2],
                [0, 0, 0],
            ],
        )
        self.scores = np.array([1.0 / 3.0, 2.0 / 3.0, 1.0 / 3.0, 0.0])

    def test_param_votes(self):
        self.assertRaises(ValueError, variation_ratios, votes="string")
        self.assertRaises(ValueError, variation_ratios, votes=1)
        self.assertRaises(ValueError, variation_ratios, votes=[1])
        self.assertRaises(ValueError, variation_ratios, votes=[[[1]]])
        self.assertRaises(
            ValueError,
            variation_ratios,
            votes=np.full((9, 9), np.nan),
        )

    def test_variation_ratios(self):
        scores = variation_ratios(votes=self.votes)
        np.testing.assert_almost_equal(scores, self.scores)
