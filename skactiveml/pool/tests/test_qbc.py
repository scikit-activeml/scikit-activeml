import unittest

import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier

from skactiveml.classifier import PWC, SklearnClassifier
from skactiveml.pool._qbc import QBC, average_kl_divergence, vote_entropy
from skactiveml.utils import MISSING_LABEL


class TestQBC(unittest.TestCase):

    def setUp(self):
        self.random_state = 41
        self.X_cand = [[8, 1, 6, 8], [9, 1, 6, 5], [5, 1, 6, 5]]
        self.X = [[1, 2, 5, 9], [5, 8, 4, 6], [8, 4, 5, 9], [5, 4, 8, 5]]
        self.y = [0., 0., 1., 1.]
        self.classes = [0, 1]
        self.ensemble = SklearnClassifier(
            estimator=RandomForestClassifier(random_state=0),
            classes=self.classes, random_state=self.random_state
        )

    def test_init_param_method(self):
        selector = QBC()
        self.assertTrue(hasattr(selector, 'method'))
        for method in ['test', 0]:
            selector = QBC(method=method)
            self.assertRaises(
                ValueError, selector.query, X_cand=self.X_cand, X=self.X,
                y=self.y, ensemble=self.ensemble
            )

    def test_query_param_ensemble(self):
        selector = QBC()
        ensemble_list = [
            None, 'test', 1, GaussianProcessClassifier(),
            SklearnClassifier(GaussianProcessClassifier, classes=self.classes),
            PWC(classes=self.classes)
        ]
        for ensemble in ensemble_list:
            self.assertRaises(
                TypeError, selector.query, X_cand=self.X_cand, X=self.X,
                y=self.y, ensemble=ensemble
            )

    def test_query_param_X(self):
        selector = QBC()
        for X in [None, np.nan]:
            self.assertRaises(
                TypeError, selector.query, X_cand=self.X_cand, X=X, y=self.y,
                ensemble=self.ensemble
            )
        for X in [[], self.X[:3]]:
            self.assertRaises(
                ValueError, selector.query, X_cand=self.X_cand, X=X, y=self.y,
                ensemble=self.ensemble
            )

    def test_query_param_y(self):
        selector = QBC()
        for y in [None, np.nan]:
            self.assertRaises(
                TypeError, selector.query, X_cand=self.X_cand, X=self.X, y=y,
                ensemble=self.ensemble
            )
        for y in [[], self.y[:3]]:
            self.assertRaises(
                ValueError, selector.query, X_cand=self.X_cand, X=self.X, y=y,
                ensemble=self.ensemble
            )

    def test_query_param_sample_weight(self):
        selector = QBC()
        sample_weight_list = [
            'test', self.X_cand, np.empty((len(self.X) - 1)),
            np.empty((len(self.X) + 1))
        ]
        for sample_weight in sample_weight_list:
            self.assertRaises(
                ValueError, selector.query, X_cand=self.X_cand, X=self.X,
                y=self.y, ensemble=self.ensemble, sample_weight=sample_weight
            )

    def test_query(self):
        ensemble_classifiers = [
            SklearnClassifier(classes=self.classes,
                              estimator=GaussianProcessClassifier()
                              ),
            SklearnClassifier(classes=self.classes,
                              estimator=GaussianProcessClassifier()
                              ),
            SklearnClassifier(classes=self.classes,
                              estimator=GaussianProcessClassifier()
                              ),
        ]
        gpc = PWC(classes=self.classes)
        ensemble_bagging = SklearnClassifier(
            estimator=BaggingClassifier(base_estimator=gpc),
            classes=self.classes
        )
        ensemble_list = [self.ensemble, ensemble_classifiers, ensemble_bagging]
        for ensemble in ensemble_list:
            for method in ['KL_divergence', 'vote_entropy']:
                selector = QBC(method=method)
                idx, u = selector.query(
                    X_cand=self.X_cand, ensemble=ensemble, X=self.X, y=self.y,
                    return_utilities=True
                )
                self.assertEqual(len(idx), 1)
                self.assertEqual(len(u), 1)


class TestAverageKlDivergence(unittest.TestCase):

    def setUp(self):
        self.probas = np.array([[[0.3, 0.7], [0.4, 0.6]],
                                [[0.2, 0.8], [0.5, 0.5]]])
        self.scores = np.array([0.00670178182226764, 0.005059389928987596])

    def test_param_probas(self):
        self.assertRaises(ValueError, average_kl_divergence, 'string')
        self.assertRaises(ValueError, average_kl_divergence, 1)
        self.assertRaises(ValueError, average_kl_divergence,
                          np.ones((1,)))
        self.assertRaises(ValueError, average_kl_divergence,
                          np.ones((1, 1)))
        self.assertRaises(ValueError, average_kl_divergence,
                          np.ones((1, 1, 1, 1)))

    def test_average_kl_divergence(self):
        average_kl_divergence(np.full((10, 10, 10), 0.5))
        average_kl_divergence(np.zeros((10, 10, 10)))
        scores = average_kl_divergence(self.probas)
        np.testing.assert_allclose(scores, self.scores)


class TestVoteEntropy(unittest.TestCase):

    def setUp(self):
        self.classes = np.array([0, 1, 2])
        self.votes = np.array([[0, 0, 2],
                               [1, 0, 2],
                               [2, 1, 2]]).T
        self.scores = np.array([1, 0.5793801643, 0])

    def test_param_votes(self):
        self.assertRaises(ValueError, vote_entropy, votes='string',
                          classes=self.classes)
        self.assertRaises(ValueError, vote_entropy, votes=1,
                          classes=self.classes)
        self.assertRaises(ValueError, vote_entropy, votes=[1],
                          classes=self.classes)
        self.assertRaises(ValueError, vote_entropy, votes=[[[1]]],
                          classes=self.classes)
        self.assertRaises(ValueError, vote_entropy, votes=np.array([[10]]),
                          classes=self.classes)
        self.assertRaises(
            ValueError, vote_entropy, votes=np.full((9, 9), np.nan),
            classes=self.classes
        )

    def test_param_classes(self):
        self.assertRaises(ValueError, vote_entropy, votes=self.votes,
                          classes='string')
        self.assertRaises(ValueError, vote_entropy, votes=self.votes,
                          classes='class')
        self.assertRaises(TypeError, vote_entropy, votes=self.votes,
                          classes=1)
        self.assertRaises(TypeError, vote_entropy, votes=self.votes,
                          classes=[[1]])
        self.assertRaises(ValueError, vote_entropy, votes=self.votes,
                          classes=[MISSING_LABEL, 1])

    def test_vote_entropy(self):
        vote_entropy(votes=np.full((10, 10), 0), classes=self.classes)
        scores = vote_entropy(votes=self.votes, classes=self.classes)
        np.testing.assert_array_equal(scores.round(10), self.scores.round(10))

