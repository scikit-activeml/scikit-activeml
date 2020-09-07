import numpy as np
import unittest

from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor

from skactiveml.utils import rand_argmax
from skactiveml.classifier import SklearnClassifier
from skactiveml.pool import UncertaintySampling

class TestUncertainty(unittest.TestCase):

    def setUp(self):
        self.random_state = 1
        self.X_cand = np.array([[8,1,6,8],[9,1,6,5],[5,1,6,5]])
        self.X = np.array([[1,2,5,9],[5,8,4,6],[8,4,5,9],[5,4,8,5]])
        self.y = np.array([0,0,1,1])
        self.classes = np.array([0,1])
        pass

    def test_init(self):
        clf = SklearnClassifier(estimator=GaussianProcessClassifier(), random_state=self.random_state)
        self.assertRaises(ValueError, UncertaintySampling, clf=clf, classes=None, method ='expected_average_precision')
        clf = GaussianProcessRegressor()
        self.assertRaises(TypeError, UncertaintySampling, clf=clf)

    def test_query(self):
        compare_list = []
        clf = SklearnClassifier(estimator=GaussianProcessClassifier(), random_state=self.random_state)
        # entropy
        uncertainty = UncertaintySampling(clf=clf, method='entropy')
        best_indices, utilities = uncertainty.query(self.X_cand, self.X, self.y, return_utilities=True, random_state=self.random_state)

        clf.fit(self.X, self.y)
        probas = clf.predict_proba(self.X_cand)
        val_utilities = np.array([-np.sum(probas * np.log(probas), axis=1)])
        val_best_indices = rand_argmax(val_utilities, axis=1, random_state=self.random_state)

        np.testing.assert_array_equal(utilities, val_utilities)
        np.testing.assert_array_equal(best_indices, val_best_indices)
        self.assertEqual(utilities.shape, (1, len(self.X_cand)))
        self.assertEqual(best_indices.shape, (1,))
        compare_list.append(best_indices)

        # margin_sampling
        uncertainty = UncertaintySampling(clf=clf, method='margin_sampling')
        best_indices, utilities = uncertainty.query(self.X_cand, self.X, self.y, return_utilities=True, random_state=self.random_state)

        clf.fit(self.X, self.y)
        probas = clf.predict_proba(self.X_cand)
        sort_probas = np.sort(probas, axis=1)
        val_utilities = np.array([sort_probas[:,-2] - sort_probas[:,-1]])
        val_best_indices = rand_argmax(val_utilities, axis=1, random_state=self.random_state)

        np.testing.assert_array_equal(utilities, val_utilities)
        np.testing.assert_array_equal(best_indices, val_best_indices)
        self.assertEqual(utilities.shape, (1, len(self.X_cand)))
        self.assertEqual(best_indices.shape, (1,))
        compare_list.append(best_indices)

        # least_confident
        uncertainty = UncertaintySampling(clf=clf, method='least_confident')
        best_indices, utilities = uncertainty.query(self.X_cand, self.X, self.y, return_utilities=True, random_state=self.random_state)

        clf.fit(self.X, self.y)
        probas = clf.predict_proba(self.X_cand)
        val_utilities = np.array([-np.max(probas, axis=1)])
        val_best_indices = rand_argmax(val_utilities, axis=1, random_state=self.random_state)

        np.testing.assert_array_equal(utilities, val_utilities)
        np.testing.assert_array_equal(best_indices, val_best_indices)
        self.assertEqual(utilities.shape, (1, len(self.X_cand)))
        self.assertEqual(best_indices.shape, (1,))
        compare_list.append(best_indices)

        for x in compare_list:
            self.assertEqual(compare_list[0], x)

        # expected_average_precision
        uncertainty = UncertaintySampling(clf=clf, classes=self.classes, method='expected_average_precision')
        best_indices, utilities = uncertainty.query(self.X_cand, self.X, self.y, return_utilities=True, random_state=self.random_state)

        clf.fit(self.X, self.y)
        probas = clf.predict_proba(self.X_cand)
        val_utilities = np.array([expected_average_precision(self.X_cand, self.classes, probas)])
        val_best_indices = rand_argmax(val_utilities, axis=1, random_state=self.random_state)

        np.testing.assert_array_equal(utilities, val_utilities)
        np.testing.assert_array_equal(best_indices, val_best_indices)
        self.assertEqual(utilities.shape, (1, len(self.X_cand)))
        self.assertEqual(best_indices.shape, (1,))
        compare_list.append(best_indices)


def expected_average_precision(X_cand, classes, probas):
    score = np.zeros(len(X_cand))
    for i in range(len(classes)):
        for n in range(len(X_cand)):
            # The i-th column of p without p[n,i]
            p = probas[:,i]
            p = np.delete(p,[n])
            # Sort p in descending order
            p = np.flipud(np.sort(p, axis=0))
            for t in range(len(p)):
                score[n] += f(p,len(p), t+1)/(t+1)
    return score


def g(p,n,t):
    if t>n or (t==0 and n>0):
        return 0
    if t==0 and n==0:
        return 1
    return p[n-1]*g(p,n-1,t-1)+(1-p[n-1])*g(p,n-1,t)


def f(p,n,t):
    if t>n or (t==0 and n>0):
        return 0
    if t==0 and n==0:
        return 1
    return p[n-1]*f(p,n-1,t-1) + (1-p[n-1])*f(p,n-1,t) + (p[n-1]*t*g(p,n-1,t-1))/n


if __name__ == '__main__':
    unittest.main()
