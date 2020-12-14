import numpy as np
import unittest

from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessRegressor, \
    GaussianProcessClassifier
from sklearn.ensemble import BaggingClassifier

from skactiveml.base import SkactivemlClassifier
from skactiveml.classifier import PWC, SklearnClassifier
from skactiveml.utils import MISSING_LABEL
from skactiveml.pool._qbc import QBC, average_kl_divergence, vote_entropy


class TestQBC(unittest.TestCase):

    def setUp(self):
        self.random_state = 41
        self.X_cand = np.array([[8, 1, 6, 8], [9, 1, 6, 5], [5, 1, 6, 5]])
        self.X = np.array(
            [[1, 2, 5, 9], [5, 8, 4, 6], [8, 4, 5, 9], [5, 4, 8, 5]])
        self.y = np.array([0, 0, 1, 1])
        self.classes = np.array([0, 1])
        self.clf = PWC(classes=self.classes, random_state=self.random_state)
        self.kwargs = dict(X_cand=self.X_cand, X=self.X, y=self.y)

    def test_init_param_clf(self):
        selector = QBC(clf=PWC(), random_state=self.random_state)
        selector.query(**self.kwargs)
        self.assertTrue(hasattr(selector, 'clf'))

        selector = QBC(clf=GaussianProcessClassifier())
        self.assertRaises(TypeError, selector.query, **self.kwargs)
        selector = QBC(clf='string')
        self.assertRaises(TypeError, selector.query, **self.kwargs)
        selector = QBC(clf=None)
        self.assertRaises(TypeError, selector.query, **self.kwargs)
        selector = QBC(clf=1)
        self.assertRaises(TypeError, selector.query, **self.kwargs)

    def test_init_param_ensemble(self):
        selector = QBC(clf=self.clf, ensemble=None)
        self.assertTrue(hasattr(selector, 'ensemble'))

        selector = QBC(clf=self.clf, ensemble='String')
        self.assertRaises(TypeError, selector.query, **self.kwargs)

    def test_init_param_method(self):
        selector = QBC(clf=self.clf)
        self.assertTrue(hasattr(selector, 'method'))
        selector = QBC(clf=self.clf, method='String')
        self.assertRaises(ValueError, selector.query, **self.kwargs)
        selector = QBC(clf=self.clf, method=1)
        self.assertRaises(TypeError, selector.query, **self.kwargs)

        selector = QBC(clf=GaussianProcessRegressor, method='KL_divergence')
        self.assertRaises(TypeError, selector.query, **self.kwargs)

    def test_init_param_random_state(self):
        selector = QBC(clf=self.clf, random_state='string')
        self.assertRaises(ValueError, selector.query, **self.kwargs)
        selector = QBC(clf=self.clf, random_state=self.random_state)
        self.assertTrue(hasattr(selector, 'random_state'))
        self.assertRaises(ValueError, selector.query, X_cand=[[1]], X=self.X,
                          y=self.y)

    def test_init_param_ensemble_dict(self):
        selector = QBC(clf=self.clf, ensemble_dict='string')
        self.assertRaises(TypeError, selector.query, **self.kwargs)
        selector = QBC(clf=self.clf, ensemble_dict=dict(kwargg='kwargg'))
        self.assertRaises(TypeError, selector.query, **self.kwargs)

    def test_query_param_X_cand(self):
        selector = QBC(clf=self.clf)
        self.assertRaises(ValueError, selector.query, X_cand=[], X=self.X,
                          y=self.y)
        self.assertRaises(ValueError, selector.query, X_cand=None, X=self.X,
                          y=self.y)
        self.assertRaises(ValueError, selector.query, X_cand=np.nan, X=self.X,
                          y=self.y)

    def test_query_param_X(self):
        selector = QBC(clf=self.clf)
        self.assertRaises(ValueError, selector.query, X_cand=self.X_cand,
                          X=None, y=self.y)
        self.assertRaises(ValueError, selector.query, X_cand=self.X_cand,
                          X='string', y=self.y)
        self.assertRaises(ValueError, selector.query, X_cand=self.X_cand,
                          X=[], y=self.y)
        self.assertRaises(ValueError, selector.query, X_cand=self.X_cand,
                          X=self.X[0:-1], y=self.y)

    def test_query_param_y(self):
        selector = QBC(clf=self.clf)
        self.assertRaises(TypeError, selector.query, X_cand=self.X_cand,
                          X=self.X, y=None)
        self.assertRaises(TypeError, selector.query, X_cand=self.X_cand,
                          X=self.X, y='string')
        self.assertRaises(ValueError, selector.query, X_cand=self.X_cand,
                          X=self.X, y=[])
        self.assertRaises(ValueError, selector.query, X_cand=self.X_cand,
                          X=self.X, y=self.y[0:-1])

    def test_query_param_sample_weight(self):
        selector = QBC(clf=self.clf)
        self.assertRaises(TypeError, selector.query, **self.kwargs,
                          sample_weight='string')
        self.assertRaises(ValueError, selector.query, **self.kwargs,
                          sample_weight=self.X_cand)
        self.assertRaises(ValueError, selector.query, **self.kwargs,
                          sample_weight=np.empty((len(self.X) - 1)))
        self.assertRaises(ValueError, selector.query, **self.kwargs,
                          sample_weight=np.empty((len(self.X) + 1)))

    def test_query_param_batch_size(self):
        selector = QBC(clf=self.clf)
        self.assertRaises(TypeError, selector.query, **self.kwargs,
                          batch_size=1.2)
        self.assertRaises(TypeError, selector.query, **self.kwargs,
                          batch_size='string')
        self.assertRaises(ValueError, selector.query, **self.kwargs,
                          batch_size=0)
        self.assertRaises(ValueError, selector.query, **self.kwargs,
                          batch_size=-10)

    def test_query_param_return_utilities(self):
        selector = QBC(clf=self.clf)
        self.assertRaises(TypeError, selector.query, **self.kwargs,
                          return_utilities=None)
        self.assertRaises(TypeError, selector.query, **self.kwargs,
                          return_utilities=[])
        self.assertRaises(TypeError, selector.query, **self.kwargs,
                          return_utilities=0)

    def test_query(self):
        # clf
        selector = QBC(clf=SklearnClassifier(GaussianProcessClassifier(),
                                                classes=self.classes))
        selector.query(**self.kwargs)
        selector = QBC(clf=SklearnClassifier(BaggingClassifier(),
                                                classes=self.classes))
        selector.query(**self.kwargs)

        # ensemble
        selector = QBC(clf=self.clf, ensemble=BaggingClassifier,
                       ensemble_dict=dict(n_estimators=10))
        selector.query(**self.kwargs)
        self.assertTrue(isinstance(selector._clf.estimator, BaggingClassifier))
        selector = QBC(clf=self.clf, ensemble=RandomForestClassifier)
        selector.query(**self.kwargs)

        # ensemble_dict
        selector = QBC(clf=self.clf, ensemble=RandomForestClassifier,
                       ensemble_dict=dict(n_estimators=5))
        selector.query(**self.kwargs)
        self.assertTrue(selector._clf.estimator.n_estimators == 5)

        # return_utilities
        L = list(selector.query(**self.kwargs, return_utilities=True))
        self.assertTrue(len(L) == 2)
        L = list(selector.query(**self.kwargs, return_utilities=False))
        self.assertTrue(len(L) == 1)

        # batch_size
        bs = 3
        selector = QBC(clf=self.clf)
        best_idx = selector.query(**self.kwargs, batch_size=bs)
        self.assertEqual(bs, len(best_idx))

        # query
        selector = QBC(clf=self.clf, method='vote_entropy')
        selector.query(**self.kwargs)
        selector.query(X_cand=[[1]], X=[[1]], y=[MISSING_LABEL])

        selector = QBC(clf=self.clf, method='KL_divergence')
        selector.query(**self.kwargs)

        selector = QBC(clf=self.clf, random_state=self.random_state)
        best_indices1, utilities1 = selector.query(**self.kwargs,
                                                   return_utilities=True)
        self.assertEqual(utilities1.shape, (1, len(self.X_cand)))
        self.assertEqual(best_indices1.shape, (1,))
        best_indices2, utilities2 = selector.query(**self.kwargs,
                                                   return_utilities=True)
        np.testing.assert_array_equal(utilities1, utilities2)
        np.testing.assert_array_equal(best_indices1, best_indices2)


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
                               [2, 1, 2]])
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
        self.assertRaises(TypeError, vote_entropy, votes=self.votes,
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
        vote_entropy(votes=[['s', 't']], classes='string')
        scores = vote_entropy(votes=self.votes, classes=self.classes)
        np.testing.assert_array_equal(scores.round(10), self.scores.round(10))


if __name__ == '__main__':
    unittest.main()
