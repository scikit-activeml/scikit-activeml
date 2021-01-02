import numpy as np
import unittest

from sklearn.gaussian_process import GaussianProcessClassifier, \
    GaussianProcessRegressor
from sklearn.linear_model import LogisticRegression

from skactiveml.pool import EpistemicUncertainty
from skactiveml.utils import rand_argmax
from skactiveml.classifier import SklearnClassifier, PWC


class TestEpistemicUncertainty(unittest.TestCase):

    def setUp(self):
        self.random_state = 1
        self.X_cand = np.array([[8, 1], [9, 1], [5, 1]])
        self.X = np.array([[1, 2], [5, 8], [8, 4], [5, 4]])
        self.y = np.array([0, 0, 1, 1])
        self.classes = np.array([0, 1])
        self.clf = PWC()
        self.kwargs = dict(X_cand=self.X_cand, X=self.X, y=self.y)

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

    def test_query(self):
        # TODO has to be revised
        compare_list = []
        # epistemic_uncertainty - pwc


#        clf = dummyPWC()
#        freq = np.zeros((121, 2))
#         for n in range(11):
#             for p in range(11):
#                 freq[n * 11 + p] = n, p
#         uncertainty = UncertaintySampling(clf, method='epistemic', precompute=True, random_state=self.random_state)
#         best_indices, utilities = uncertainty.query(freq, self.X, self.y, return_utilities=True)
#
#         pi0 = np.zeros((11, 11))
#         pi1 = np.zeros((11, 11))
#         for n in range(11):
#             for p in range(11):
#                 if (n == 0 | p == 0):
#                     pi0[n, p] = 1
#                     pi1[n, p] = 1
#                 else:
#                     pi0[n, p] = -epistemic_pwc_sup_1(
#                         minimize_scalar(epistemic_pwc_sup_1, method='Bounded', bounds=(0.0, 1.0), args=(n, p)).x, n, p)
#                     pi1[n, p] = -epistemic_pwc_sup_0(
#                         minimize_scalar(epistemic_pwc_sup_0, method='Bounded', bounds=(0.0, 1.0), args=(n, p)).x, n, p)
#         pi = np.min(np.array([pi0, pi1]), axis=0)
#         val_utilities = pi
#         np.testing.assert_array_equal(utilities, [val_utilities.flatten()])
#         print(utilities.shape)
#         self.assertEqual(utilities.shape, (1, 121))
#         self.assertEqual(best_indices.shape, (1,))
#         compare_list.append(best_indices)

# epistemic_uncertainty - logistic regression#
# clf = LogisticRegression()
# uncertainty = UncertaintySampling(clf, method='epistemic', random_state=self.random_state)
# best_indices, utilities = uncertainty.query(self.X_cand, self.X, self.y, return_utilities=True)



if __name__ == '__main__':
    unittest.main()
