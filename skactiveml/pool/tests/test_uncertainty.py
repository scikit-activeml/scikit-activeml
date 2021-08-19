import unittest

import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier, \
    GaussianProcessRegressor

from skactiveml.classifier import SklearnClassifier, PWC
from skactiveml.pool import UncertaintySampling, expected_average_precision
from skactiveml.pool._uncertainty import uncertainty_scores
from skactiveml.utils import MISSING_LABEL


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
        selector = UncertaintySampling(clf=GaussianProcessClassifier())
        self.assertRaises(TypeError, selector.query, **self.kwargs)
        selector = UncertaintySampling(
            clf=SklearnClassifier(GaussianProcessRegressor())
        )
        self.assertRaises(TypeError, selector.query, **self.kwargs)

    def test_init_param_method(self):
        selector = UncertaintySampling(clf=self.clf)
        self.assertTrue(hasattr(selector, 'method'))
        selector = UncertaintySampling(clf=self.clf, method='String')
        self.assertRaises(ValueError, selector.query, **self.kwargs)
        selector = UncertaintySampling(clf=self.clf, method=1)
        self.assertRaises(TypeError, selector.query, **self.kwargs)

    def test_init_param_cost_matrix(self):
        selector = UncertaintySampling(clf=self.clf,
                                       cost_matrix=np.ones((2, 3)))
        self.assertRaises(ValueError, selector.query, **self.kwargs)

        selector = UncertaintySampling(clf=self.clf,
                                       cost_matrix='string')
        self.assertRaises(ValueError, selector.query, **self.kwargs)

        selector = UncertaintySampling(clf=self.clf,
                                       cost_matrix=np.ones((3, 3)))
        self.assertRaises(ValueError, selector.query, **self.kwargs)


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
        self.assertRaises(TypeError, selector.query, X_cand=self.X_cand,
                          X=None, y=self.y)
        self.assertRaises(TypeError, selector.query, X_cand=self.X_cand,
                          X='string', y=self.y)
        self.assertRaises(ValueError, selector.query, X_cand=self.X_cand,
                          X=[], y=self.y)
        self.assertRaises(ValueError, selector.query, X_cand=self.X_cand,
                          X=self.X[0:-1], y=self.y)

    def test_query_param_y(self):
        selector = UncertaintySampling(clf=self.clf)
        self.assertRaises(TypeError, selector.query, X_cand=self.X_cand,
                          X=self.X, y=None)
        self.assertRaises(TypeError, selector.query, X_cand=self.X_cand,
                          X=self.X, y='string')
        self.assertRaises(ValueError, selector.query, X_cand=self.X_cand,
                          X=self.X, y=[])
        self.assertRaises(ValueError, selector.query, X_cand=self.X_cand,
                          X=self.X, y=self.y[0:-1])

    def test_query_param_sample_weight(self):
        selector = UncertaintySampling(clf=self.clf)
        self.assertRaises(ValueError, selector.query, **self.kwargs,
                          sample_weight='string')
        self.assertRaises(ValueError, selector.query, **self.kwargs,
                          sample_weight=self.X_cand)
        self.assertRaises(ValueError, selector.query, **self.kwargs,
                          sample_weight=np.empty((len(self.X) - 1)))
        self.assertRaises(ValueError, selector.query, **self.kwargs,
                          sample_weight=np.empty((len(self.X) + 1)))

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
        compare_list = []
        clf = SklearnClassifier(estimator=GaussianProcessClassifier(),
                                random_state=self.random_state,
                                classes=self.classes)

        selector = UncertaintySampling(clf=clf)

        # return_utilities
        L = list(selector.query(**self.kwargs, return_utilities=True))
        self.assertTrue(len(L) == 2)
        L = list(selector.query(**self.kwargs, return_utilities=False))
        self.assertTrue(len(L) == 1)

        # batch_size
        bs = 3
        selector = UncertaintySampling(clf=clf)
        best_idx = selector.query(**self.kwargs, batch_size=bs)
        self.assertEqual(bs, len(best_idx))

        # query
        selector = UncertaintySampling(clf=clf, method='entropy')
        selector.query(X_cand=[[1]], X=[[1]], y=[MISSING_LABEL])
        compare_list.append(selector.query(**self.kwargs))

        selector = UncertaintySampling(clf=clf, method='margin_sampling')
        selector.query(X_cand=[[1]], X=[[1]], y=[MISSING_LABEL])
        compare_list.append(selector.query(**self.kwargs))

        selector = UncertaintySampling(clf=clf, method='least_confident')
        selector.query(X_cand=[[1]], X=[[1]], y=[MISSING_LABEL])
        compare_list.append(selector.query(**self.kwargs))

        selector = UncertaintySampling(clf=clf, method='margin_sampling',
                                       cost_matrix=[[0, 1], [1, 0]])
        selector.query(X_cand=[[1]], X=[[1]], y=[MISSING_LABEL])

        selector = UncertaintySampling(clf=clf, method='least_confident',
                                       cost_matrix=[[0, 1], [1, 0]])
        selector.query(X_cand=[[1]], X=[[1]], y=[MISSING_LABEL])

        for x in compare_list:
            self.assertEqual(compare_list[0], x)

        selector = UncertaintySampling(clf=clf,
                                       method='expected_average_precision')
        selector.query(X_cand=[[1]], X=[[1]], y=[MISSING_LABEL])
        best_indices, utilities = selector.query(**self.kwargs,
                                                 return_utilities=True)
        self.assertEqual(utilities.shape, (1, len(self.X_cand)))
        self.assertEqual(best_indices.shape, (1,))


class TestExpectedAveragePrecision(unittest.TestCase):
    def setUp(self):
        self.classes = np.array([0, 1])
        self.probas = np.array([[0.4, 0.6], [0.3, 0.7]])
        self.scores_val = np.array([2.0, 2.0])

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
                          classes=self.classes, probas=[[0.6, 0.2]])
        self.assertRaises(ValueError, expected_average_precision,
                          classes=self.classes, probas='string')

    def test_expected_average_precision(self):
        expected_average_precision(classes=self.classes, probas=[[0.0, 1.0]])
        scores = expected_average_precision(
            classes=self.classes, probas=self.probas)
        self.assertTrue(scores.shape == (len(self.probas),))
        np.testing.assert_array_equal(scores, self.scores_val)


class TestUncertaintyScores(unittest.TestCase):
    def setUp(self):
        self.probas = np.array([[0.2, 0.5, 0.3], [0.1, 0.7, 0.2]])
        self.classes = np.array([0, 1, 2])
        self.cost_matrix = np.ones((3, 3))

    def test_param_probas(self):
        self.assertRaises(ValueError, uncertainty_scores, probas=[1])
        self.assertRaises(ValueError, uncertainty_scores, probas=[[[1]]])
        self.assertRaises(ValueError, uncertainty_scores,
                          probas=[[0.6, 0.1, 0.2]])
        self.assertRaises(ValueError, uncertainty_scores, probas='string')

    def test_init_param_method(self):
        self.assertRaises(ValueError, uncertainty_scores, self.probas,
                          method='String')
        self.assertRaises(ValueError, uncertainty_scores, self.probas,
                          method=1)

    def test_param_cost_matrix(self):
        self.assertRaises(ValueError, uncertainty_scores, self.probas,
                          cost_matrix=np.ones((2, 3)))
        self.assertRaises(ValueError, uncertainty_scores, self.probas,
                          cost_matrix='string')
        self.assertRaises(ValueError, uncertainty_scores, self.probas,
                          cost_matrix=np.ones((2, 2)))

    def test_uncertainty_scores(self):
        # least_confident
        val_scores = np.array([0.5, 0.3])
        scores = uncertainty_scores(self.probas, method='least_confident')
        np.testing.assert_allclose(val_scores, scores)
        # entropy
        val_scores = np.array([1.029653014, 0.8018185525])
        scores = uncertainty_scores(self.probas, method='entropy')
        np.testing.assert_allclose(val_scores, scores)
        # margin_sampling
        val_scores = np.array([0.8, 0.5])
        scores = uncertainty_scores(self.probas, method='margin_sampling')
        np.testing.assert_allclose(val_scores, scores)

if __name__ == '__main__':
    unittest.main()
