import numpy as np
import unittest

from sklearn.gaussian_process import GaussianProcessClassifier, \
    GaussianProcessRegressor
from sklearn.linear_model import LogisticRegression

from skactiveml.utils import rand_argmax
from skactiveml.classifier import SklearnClassifier, PWC
from skactiveml.pool import UncertaintySampling, expected_average_precision
from skactiveml.base import ClassFrequencyEstimator
from scipy.optimize import minimize_scalar


class dummyPWC(ClassFrequencyEstimator):
    def __init__(self):
        super().__init__(np.array([0, 1]))
        pass

    def predict_freq(self, freq):
        return freq

    def predict_proba(self, X_cand):
        pass

    def fit(self, X, y):
        pass


class TestUncertaintySampling(unittest.TestCase):

    def setUp(self):
        self.random_state = 1
        self.X_cand = np.array([[8, 1], [9, 1], [5, 1]])
        self.X = np.array([[1, 2], [5, 8], [8, 4], [5, 4]])
        self.y = np.array([0, 0, 1, 1])
        self.classes = np.array([0, 1])
        self.clf = PWC()
        self.kwargs = dict(X_cand=self.X_cand, X=self.X, y=self.y)

    def test_init_param_clf(self):
        selector = UncertaintySampling(clf=PWC(),
                                       random_state=self.random_state)
        selector.query(**self.kwargs)
        self.assertTrue(hasattr(selector, 'clf'))
        # selector = QBC(clf=GaussianProcessClassifier(
        #    random_state=self.random_state), random_state=self.random_state)
        # selector.query(**self.kwargs)

        selector = UncertaintySampling(clf='string')
        self.assertRaises(TypeError, selector.query, **self.kwargs)
        selector = UncertaintySampling(clf=None)
        self.assertRaises(TypeError, selector.query, **self.kwargs)
        selector = UncertaintySampling(clf=1)
        self.assertRaises(TypeError, selector.query, **self.kwargs)

    def test_init_param_classes(self):
        # TODO Not required if classes are removed
        return
        clf = GaussianProcessRegressor()
        select = UncertaintySampling(clf=clf)
        self.assertRaises(TypeError, select.query, self.X_cand, self.X, self.y)
        select = UncertaintySampling(clf=clf, classes=None, method='epistemic')
        self.assertRaises(TypeError, select.query, self.X_cand, self.X, self.y)
        clf = SklearnClassifier(estimator=GaussianProcessClassifier(),
                                random_state=self.random_state)
        select = UncertaintySampling(clf=clf, classes=None,
                                     method='expected_average_precision')
        self.assertRaises(ValueError, select.query, self.X_cand,
                          self.X, self.y)

    def test_init_param_method(self):
        selector = UncertaintySampling(clf=self.clf)
        self.assertTrue(hasattr(selector, 'method'))
        selector = UncertaintySampling(clf=self.clf, method='String')
        self.assertRaises(ValueError, selector.query, **self.kwargs)
        selector = UncertaintySampling(clf=self.clf, method=1)
        self.assertRaises(TypeError, selector.query, **self.kwargs)

    def test_init_param_precompute(self):
        selector = UncertaintySampling(clf=self.clf, method='epistemic',
                                       precompute=None)
        self.assertRaises(TypeError, selector.query, **self.kwargs)

        selector = UncertaintySampling(clf=self.clf, method='epistemic',
                                       precompute=[])
        self.assertRaises(TypeError, selector.query, **self.kwargs)

        selector = UncertaintySampling(clf=self.clf, method='epistemic',
                                       precompute=0)
        self.assertRaises(TypeError, selector.query, **self.kwargs)

    def test_init_param_missing_label(self):
        selector = UncertaintySampling(clf=self.clf, missing_label='string')
        self.assertTrue(hasattr(selector, 'missing_label'))

    def test_init_param_random_state(self):
        selector = UncertaintySampling(clf=self.clf, random_state='string')
        self.assertRaises(ValueError, selector.query, **self.kwargs)
        selector = UncertaintySampling(clf=self.clf,
                                       random_state=self.random_state)
        self.assertTrue(hasattr(selector, 'random_state'))
        self.assertRaises(ValueError, selector.query, X_cand=[[1]], X=self.X,
                          y=self.y)

    def test_query_param_X_cand(self):
        selector = UncertaintySampling(clf=self.clf)
        self.assertRaises(ValueError, selector.query, X_cand=[], X=self.X,
                          y=self.y)
        self.assertRaises(ValueError, selector.query, X_cand=None, X=self.X,
                          y=self.y)
        self.assertRaises(ValueError, selector.query, X_cand=np.nan, X=self.X,
                          y=self.y)

    def test_query_param_X(self):
        selector = UncertaintySampling(clf=self.clf)
        self.assertRaises(ValueError, selector.query, X_cand=self.X_cand,
                          X=None, y=self.y)
        self.assertRaises(ValueError, selector.query, X_cand=self.X_cand,
                          X='string', y=self.y)
        self.assertRaises(ValueError, selector.query, X_cand=self.X_cand,
                          X=[], y=self.y)
        self.assertRaises(ValueError, selector.query, X_cand=self.X_cand,
                          X=self.X[0:-1], y=self.y)

    def test_query_param_y(self):
        selector = UncertaintySampling(clf=self.clf)
        self.assertRaises(ValueError, selector.query, X_cand=self.X_cand,
                          X=self.X, y=None)
        self.assertRaises(ValueError, selector.query, X_cand=self.X_cand,
                          X=self.X, y='string')
        self.assertRaises(ValueError, selector.query, X_cand=self.X_cand,
                          X=self.X, y=[])
        self.assertRaises(ValueError, selector.query, X_cand=self.X_cand,
                          X=self.X, y=self.y[0:-1])

    def test_query_param_batch_size(self):
        selector = UncertaintySampling(clf=self.clf)
        self.assertRaises(TypeError, selector.query, **self.kwargs,
                          batch_size=1.2)
        self.assertRaises(TypeError, selector.query, **self.kwargs,
                          batch_size='string')
        self.assertRaises(ValueError, selector.query, **self.kwargs,
                          batch_size=0)
        self.assertRaises(ValueError, selector.query, **self.kwargs,
                          batch_size=-10)

    def test_query_param_return_utilities(self):
        selector = UncertaintySampling(clf=self.clf)
        self.assertRaises(TypeError, selector.query, **self.kwargs,
                          return_utilities=None)
        self.assertRaises(TypeError, selector.query, **self.kwargs,
                          return_utilities=[])
        self.assertRaises(TypeError, selector.query, **self.kwargs,
                          return_utilities=0)

    def test_query(self):
        # TODO has to be revised
        compare_list = []
        clf = SklearnClassifier(estimator=GaussianProcessClassifier(),
                                random_state=self.random_state)
        # entropy
        uncertainty = UncertaintySampling(clf=clf, method='entropy')
        best_indices, utilities = uncertainty.query(self.X_cand, self.X,
                                                    self.y,
                                                    return_utilities=True,
                                                    random_state=self.random_state)

        clf.fit(self.X, self.y)
        probas = clf.predict_proba(self.X_cand)
        val_utilities = np.array([-np.sum(probas * np.log(probas), axis=1)])
        val_best_indices = rand_argmax(val_utilities, axis=1,
                                       random_state=self.random_state)

        np.testing.assert_array_equal(utilities, val_utilities)
        np.testing.assert_array_equal(best_indices, val_best_indices)
        self.assertEqual(utilities.shape, (1, len(self.X_cand)))
        self.assertEqual(best_indices.shape, (1,))
        compare_list.append(best_indices)

        # margin_sampling
        uncertainty = UncertaintySampling(clf=clf, method='margin_sampling')
        best_indices, utilities = uncertainty.query(self.X_cand, self.X,
                                                    self.y,
                                                    return_utilities=True,
                                                    random_state=self.random_state)

        clf.fit(self.X, self.y)
        probas = clf.predict_proba(self.X_cand)
        sort_probas = np.sort(probas, axis=1)
        val_utilities = np.array([1 + sort_probas[:, -2] - sort_probas[:, -1]])
        val_best_indices = rand_argmax(val_utilities, axis=1,
                                       random_state=self.random_state)

        np.testing.assert_array_equal(utilities, val_utilities)
        np.testing.assert_array_equal(best_indices, val_best_indices)
        self.assertEqual(utilities.shape, (1, len(self.X_cand)))
        self.assertEqual(best_indices.shape, (1,))
        compare_list.append(best_indices)

        # least_confident
        uncertainty = UncertaintySampling(clf=clf, method='least_confident')
        best_indices, utilities = uncertainty.query(self.X_cand, self.X,
                                                    self.y,
                                                    return_utilities=True,
                                                    random_state=self.random_state)

        clf.fit(self.X, self.y)
        probas = clf.predict_proba(self.X_cand)
        val_utilities = np.array([1 - np.max(probas, axis=1)])
        val_best_indices = rand_argmax(val_utilities, axis=1,
                                       random_state=self.random_state)

        np.testing.assert_array_equal(utilities, val_utilities)
        np.testing.assert_array_equal(best_indices, val_best_indices)
        self.assertEqual(utilities.shape, (1, len(self.X_cand)))
        self.assertEqual(best_indices.shape, (1,))
        compare_list.append(best_indices)

        for x in compare_list:
            self.assertEqual(compare_list[0], x)

        # expected_average_precision
        uncertainty = UncertaintySampling(clf=clf, classes=self.classes,
                                          method='expected_average_precision')
        best_indices, utilities = uncertainty.query(self.X_cand, self.X,
                                                    self.y,
                                                    return_utilities=True,
                                                    random_state=self.random_state)

        clf.fit(self.X, self.y)
        probas = clf.predict_proba(self.X_cand)
        val_utilities = np.array(
            [expected_average_precision(self.classes, probas)])
        val_best_indices = rand_argmax(val_utilities, axis=1,
                                       random_state=self.random_state)

        np.testing.assert_array_equal(utilities, val_utilities)
        np.testing.assert_array_equal(best_indices, val_best_indices)
        self.assertEqual(utilities.shape, (1, len(self.X_cand)))
        self.assertEqual(best_indices.shape, (1,))
        compare_list.append(best_indices)

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


class TestExpectedAveragePrecision(unittest.TestCase):
    def setUp(self):
        self.classes = np.array([0, 1])
        self.probas = np.array([[0.4, 0.6], [0.3, 0.7]])

    def test_param_classes(self):
        self.assertRaises(ValueError, expected_average_precision,
                          classes=[], probas=self.probas)
        self.assertRaises(ValueError, expected_average_precision,
                          classes='string', probas=self.probas)
        self.assertRaises(ValueError, expected_average_precision,
                          classes=[0], probas=self.probas)
        self.assertRaises(ValueError, expected_average_precision,
                          classes=[0, 1, 2], probas=self.probas)

    def test_param_probas(self):
        self.assertRaises(ValueError, expected_average_precision,
                          classes=self.classes, probas=[1])
        self.assertRaises(ValueError, expected_average_precision,
                          classes=self.classes, probas=[[[1]]])
        self.assertRaises(ValueError, expected_average_precision,
                          classes=self.classes, probas=[[0.7, 0.1, 0.2]])
        self.assertRaises(ValueError, expected_average_precision,
                          classes=self.classes, probas=[[0.6, 0.1, 0.2]])
        self.assertRaises(ValueError, expected_average_precision,
                          classes=self.classes, probas='string')

    def test_expected_average_precision(self):
        expected_average_precision(classes=self.classes, probas=[[0.0, 1.0]])
        scores = expected_average_precision(
            classes=self.classes, probas=self.probas)
        print(scores.shape)
        self.assertTrue(scores.shape == (len(self.probas),))
        scores_val = _expected_average_precision(
            classes=self.classes, probas=self.probas)
        print(scores, scores_val)
        np.testing.assert_array_equal(scores, scores_val)


def _expected_average_precision(classes, probas):
    score = np.zeros(len(probas))
    for i in range(len(classes)):
        for n in range(len(probas)):
            # The i-th column of p without p[n,i]
            p = probas[:, i]
            p = np.delete(p, [n])
            # Sort p in descending order
            p = np.flipud(np.sort(p, axis=0))
            for t in range(len(p)):
                score[n] += f(p, len(p), t + 1) / (t + 1)
    return score


def g(p, n, t):
    if t > n or (t == 0 and n > 0):
        return 0
    if t == 0 and n == 0:
        return 1
    return p[n - 1] * g(p, n - 1, t - 1) + (1 - p[n - 1]) * g(p, n - 1, t)


def f(p, n, t):
    if t > n or (t == 0 and n > 0):
        return 0
    if t == 0 and n == 0:
        return 1
    return p[n - 1] * f(p, n - 1, t - 1) + (1 - p[n - 1]) * f(p, n - 1, t) + (
                p[n - 1] * t * g(p, n - 1, t - 1)) / n


def epistemic_pwc_sup_1(t, n, p):
    if ((n == 0.0) and (p == 0.0)):
        return -1.0
    piH = ((t ** p) * ((1 - t) ** n)) / (
                ((p / (n + p)) ** p) * ((n / (n + p)) ** n))
    return -np.minimum(piH, 2 * t - 1)


def epistemic_pwc_sup_0(t, n, p):
    if ((n == 0.0) and (p == 0.0)):
        return -1.0
    piH = ((t ** p) * ((1 - t) ** n)) / (
                ((p / (n + p)) ** p) * ((n / (n + p)) ** n))
    return -np.minimum(piH, 1 - 2 * t)


if __name__ == '__main__':
    unittest.main()
