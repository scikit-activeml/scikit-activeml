import unittest

import numpy as np
from sklearn.ensemble import (
    BaggingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.gaussian_process import GaussianProcessClassifier

from skactiveml.classifier import ParzenWindowClassifier, SklearnClassifier
from skactiveml.pool._query_by_committee import (
    QueryByCommittee,
    average_kl_divergence,
    vote_entropy,
)
from skactiveml.utils import MISSING_LABEL


class TestQueryByCommittee(unittest.TestCase):
    def setUp(self):
        self.random_state = 41
        self.candidates = [[8, 1, 6, 8], [9, 1, 6, 5], [5, 1, 6, 5]]
        self.X = [[1, 2, 5, 9], [5, 8, 4, 6], [8, 4, 5, 9], [5, 4, 8, 5]]
        self.y = [0.0, 0.0, 1.0, 1.0]
        self.classes = [0, 1]
        self.ensemble = SklearnClassifier(
            estimator=RandomForestClassifier(random_state=0),
            classes=self.classes,
            random_state=self.random_state,
        )

    def test_init_param_method(self):
        selector = QueryByCommittee()
        self.assertTrue(hasattr(selector, "method"))
        for method in ["test", 0]:
            selector = QueryByCommittee(method=method)
            self.assertRaises(
                ValueError,
                selector.query,
                candidates=self.candidates,
                X=self.X,
                y=self.y,
                ensemble=self.ensemble,
            )

    def test_query_param_ensemble(self):
        selector = QueryByCommittee()
        ensemble_list = [
            None,
            "test",
            1,
            GaussianProcessClassifier(),
            SklearnClassifier(GaussianProcessClassifier, classes=self.classes),
            ParzenWindowClassifier(classes=self.classes),
        ]
        for ensemble in ensemble_list:
            self.assertRaises(
                TypeError,
                selector.query,
                X=self.X,
                y=self.y,
                ensemble=ensemble,
                candidates=self.candidates,
            )

    def test_query_param_X(self):
        selector = QueryByCommittee()
        for X in [None, np.nan]:
            self.assertRaises(
                ValueError,
                selector.query,
                X=X,
                y=self.y,
                ensemble=self.ensemble,
                candidates=self.candidates,
            )
        for X in [[], self.X[:3]]:
            self.assertRaises(
                ValueError,
                selector.query,
                X=X,
                y=self.y,
                ensemble=self.ensemble,
                candidates=self.candidates,
            )

    def test_query_param_y(self):
        selector = QueryByCommittee()
        for y in [None, np.nan]:
            self.assertRaises(
                TypeError,
                selector.query,
                X=self.X,
                y=y,
                ensemble=self.ensemble,
                candidates=self.candidates,
            )
        for y in [[], self.y[:3]]:
            self.assertRaises(
                ValueError,
                selector.query,
                X=self.X,
                y=y,
                ensemble=self.ensemble,
                candidates=self.candidates,
            )

    def test_query_param_sample_weight(self):
        selector = QueryByCommittee()
        sample_weight_list = [
            "test",
            self.candidates,
            np.empty((len(self.X) - 1)),
            np.empty((len(self.X) + 1)),
            np.ones((len(self.X) + 1)),
        ]
        for sample_weight in sample_weight_list:
            self.assertRaises(
                ValueError,
                selector.query,
                X=self.X,
                y=self.y,
                ensemble=self.ensemble,
                sample_weight=sample_weight,
                candidates=self.candidates,
            )

    def test_query_param_fit_ensemble(self):
        selector = QueryByCommittee()
        self.assertRaises(
            TypeError,
            selector.query,
            candidates=self.candidates,
            X=self.X,
            y=self.y,
            ensemble=self.ensemble,
            fit_ensemble="string",
        )
        self.assertRaises(
            TypeError,
            selector.query,
            candidates=self.candidates,
            X=self.X,
            y=self.y,
            ensemble=self.ensemble,
            fit_ensemble=self.candidates,
        )
        self.assertRaises(
            TypeError,
            selector.query,
            candidates=self.candidates,
            X=self.X,
            y=self.y,
            ensemble=self.ensemble,
            fit_ensemble=None,
        )

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
            self.ensemble,
            ensemble_classifiers,
            ensemble_bagging,
            ensemble_voting,
        ]
        for ensemble in ensemble_list:
            for method in ["KL_divergence", "vote_entropy"]:
                selector = QueryByCommittee(method=method)
                idx, u = selector.query(
                    candidates=self.candidates,
                    ensemble=ensemble,
                    X=self.X,
                    y=self.y,
                    return_utilities=True,
                )
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
        self.scores = np.array([1, 0.5793801643, 0])

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
        vote_entropy(votes=np.full((10, 10), 0), classes=self.classes)
        scores = vote_entropy(votes=self.votes, classes=self.classes)
        np.testing.assert_array_equal(scores.round(10), self.scores.round(10))
