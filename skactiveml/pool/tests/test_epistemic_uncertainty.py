import numpy as np
import unittest

import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

from sklearn.gaussian_process import GaussianProcessClassifier, \
    GaussianProcessRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from skactiveml.pool import EpistemicUncertainty
from skactiveml.pool._epistemic_uncertainty import _interpolate, \
    _epistemic_pwc_sup_1, _epistemic_pwc_sup_0, epistemic_uncertainty_pwc
from skactiveml.utils import rand_argmax, MISSING_LABEL
from skactiveml.classifier import SklearnClassifier, PWC


class TestEpistemicUncertainty(unittest.TestCase):

    def setUp(self):
        self.random_state = 1
        self.X_cand = np.array([[8, 1], [9, 1], [5, 1]])
        self.X = np.array([[1, 2], [5, 8], [8, 4], [5, 4]])
        self.y = np.array([0, 0, 1, 1])
        self.classes = np.array([0, 1])
        self.clf = PWC(classes=self.classes, random_state=self.random_state)
        self.kwargs = dict(X_cand=self.X_cand, X=self.X, y=self.y)

    def test_init_param_clf(self):
        selector = EpistemicUncertainty(clf=PWC(),
                                        random_state=self.random_state)
        selector.query(**self.kwargs)
        self.assertTrue(hasattr(selector, 'clf'))
        # selector = QBC(clf=GaussianProcessClassifier(
        #    random_state=self.random_state), random_state=self.random_state)
        # selector.query(**self.kwargs)

        selector = EpistemicUncertainty(clf='string')
        self.assertRaises(TypeError, selector.query, **self.kwargs)
        selector = EpistemicUncertainty(clf=None)
        self.assertRaises(TypeError, selector.query, **self.kwargs)
        selector = EpistemicUncertainty(clf=1)
        self.assertRaises(TypeError, selector.query, **self.kwargs)
        selector = EpistemicUncertainty(
            clf=SklearnClassifier(DecisionTreeClassifier())
        )
        self.assertRaises(TypeError, selector.query, **self.kwargs)

    def test_init_param_precompute(self):
        selector = EpistemicUncertainty(clf=self.clf, precompute=None)
        self.assertRaises(TypeError, selector.query, **self.kwargs)

        selector = EpistemicUncertainty(clf=self.clf, precompute=[])
        self.assertRaises(TypeError, selector.query, **self.kwargs)

        selector = EpistemicUncertainty(clf=self.clf, precompute=0)
        self.assertRaises(TypeError, selector.query, **self.kwargs)

    def test_init_param_random_state(self):
        selector = EpistemicUncertainty(clf=self.clf, random_state='string')
        self.assertRaises(ValueError, selector.query, **self.kwargs)
        selector = EpistemicUncertainty(clf=self.clf,
                                        random_state=self.random_state)
        self.assertTrue(hasattr(selector, 'random_state'))
        self.assertRaises(ValueError, selector.query, X_cand=[[1]], X=self.X,
                          y=self.y)

    def test_query_param_X_cand(self):
        selector = EpistemicUncertainty(clf=self.clf)
        self.assertRaises(ValueError, selector.query, X_cand=[], X=self.X,
                          y=self.y)
        self.assertRaises(ValueError, selector.query, X_cand=None, X=self.X,
                          y=self.y)
        self.assertRaises(ValueError, selector.query, X_cand=np.nan, X=self.X,
                          y=self.y)

    def test_query_param_X(self):
        selector = EpistemicUncertainty(clf=self.clf)
        self.assertRaises(ValueError, selector.query, X_cand=self.X_cand,
                          X=None, y=self.y)
        self.assertRaises(ValueError, selector.query, X_cand=self.X_cand,
                          X='string', y=self.y)
        self.assertRaises(ValueError, selector.query, X_cand=self.X_cand,
                          X=[], y=self.y)
        self.assertRaises(ValueError, selector.query, X_cand=self.X_cand,
                          X=self.X[0:-1], y=self.y)

    def test_query_param_y(self):
        selector = EpistemicUncertainty(clf=self.clf)
        self.assertRaises(TypeError, selector.query, X_cand=self.X_cand,
                          X=self.X, y=None)
        self.assertRaises(TypeError, selector.query, X_cand=self.X_cand,
                          X=self.X, y='string')
        self.assertRaises(ValueError, selector.query, X_cand=self.X_cand,
                          X=self.X, y=[])
        self.assertRaises(ValueError, selector.query, X_cand=self.X_cand,
                          X=self.X, y=self.y[0:-1])

    def test_query_param_sample_weight(self):
        selector = EpistemicUncertainty(clf=self.clf)
        self.assertRaises(TypeError, selector.query, **self.kwargs,
                          sample_weight='string')
        self.assertRaises(ValueError, selector.query, **self.kwargs,
                          sample_weight=self.X_cand)
        self.assertRaises(ValueError, selector.query, **self.kwargs,
                          sample_weight=np.empty((len(self.X) - 1)))
        self.assertRaises(ValueError, selector.query, **self.kwargs,
                          sample_weight=np.empty((len(self.X) + 1)))

    def test_query_param_batch_size(self):
        selector = EpistemicUncertainty(clf=self.clf)
        self.assertRaises(TypeError, selector.query, **self.kwargs,
                          batch_size=1.2)
        self.assertRaises(TypeError, selector.query, **self.kwargs,
                          batch_size='string')
        self.assertRaises(ValueError, selector.query, **self.kwargs,
                          batch_size=0)
        self.assertRaises(ValueError, selector.query, **self.kwargs,
                          batch_size=-10)

    def test_query_param_return_utilities(self):
        selector = EpistemicUncertainty(clf=self.clf)
        self.assertRaises(TypeError, selector.query, **self.kwargs,
                          return_utilities=None)
        self.assertRaises(TypeError, selector.query, **self.kwargs,
                          return_utilities=[])
        self.assertRaises(TypeError, selector.query, **self.kwargs,
                          return_utilities=0)

    def test_interpolate(self):
        interpolated = _interpolate(np.array([[0, 0], [1, 1]]),
                                    np.array([[0.5, 0.5]]))
        np.testing.assert_array_equal(interpolated, np.array([0.5]))

    def test_epistemic_pwc_sup_1(self):
        self.assertEqual(1.0, -_epistemic_pwc_sup_1(None, 0.0, 0.0))
        self.assertEqual(0.0, -_epistemic_pwc_sup_1(1, 0.5, 0.8))
        self.assertAlmostEqual(0.36, -_epistemic_pwc_sup_1(0.9, 1, 1))
        self.assertEqual(-0.8, -_epistemic_pwc_sup_1(0.1, 1, 1))


    def test_epistemic_pwc_sup_0(self):
        self.assertEqual(1.0, -_epistemic_pwc_sup_0(None, 0.0, 0.0))
        self.assertEqual(-1.0, -_epistemic_pwc_sup_0(1, 0.5, 0.8))
        self.assertAlmostEqual(0.36, -_epistemic_pwc_sup_0(0.1, 1, 1))
        self.assertEqual(-0.8, -_epistemic_pwc_sup_0(0.9, 1, 1))

    def test_epistemic_uncertainty_pwc(self):
        freq = np.empty((121, 2))
        for n in range(11):
         for p in range(11):
             freq[n * 11 + p] = n, p

        indices = [39, 27, 18, 68, 20]
        expected = np.array([0.23132135217407046,
                             0.22057583593855598,
                             0.056099946963575974,
                             0.16316360415548017,
                             0.021220951860586187])

        utilities, arr = epistemic_uncertainty_pwc(freq, None)
        self.assertEqual(utilities.shape, (121,))
        np.testing.assert_allclose(expected, utilities[indices])
        epistemic_uncertainty_pwc(np.array([[2.5, 1.5]]), None)

        val_utilities = utilities
        precompute_array = np.full((1, 1), np.nan)

        utilities, precompute_array = epistemic_uncertainty_pwc(
            freq, precompute_array)
        np.testing.assert_array_equal(val_utilities, utilities)
        np.testing.assert_array_equal(
            val_utilities, precompute_array[:11, :11].flatten())

    def test_query(self):
        pass
        # TODO has to be revised
        selector = EpistemicUncertainty(clf=self.clf)

        # return_utilities
        L = list(selector.query(**self.kwargs, return_utilities=True))
        self.assertTrue(len(L) == 2)
        L = list(selector.query(**self.kwargs, return_utilities=False))
        self.assertTrue(len(L) == 1)

        # batch_size
        bs = 3
        selector = EpistemicUncertainty(clf=self.clf)
        best_idx = selector.query(**self.kwargs, batch_size=bs)
        self.assertEqual(bs, len(best_idx))

        # query
        selector = EpistemicUncertainty(
            clf=PWC(classes=self.classes, random_state=self.random_state)
        )
        selector.query(X_cand=[[1]], X=[[1]], y=[MISSING_LABEL])

        #selector = EpistemicUncertainty(clf=LogisticRegression)
        #selector.query(X_cand=[[1]], X=[[1]], y=[MISSING_LABEL])


if __name__ == '__main__':
    unittest.main()
