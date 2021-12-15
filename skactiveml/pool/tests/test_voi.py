import unittest
import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB

from skactiveml.base import SkactivemlClassifier
from skactiveml.classifier import SklearnClassifier
from skactiveml.pool import VOI
from skactiveml.pool._voi import _total_labeled_risk, \
    _total_unlabeled_risk
from skactiveml.utils import MISSING_LABEL, labeled_indices


class TestVOI(unittest.TestCase):

    def setUp(self):
        self.classes = [0, 1]
        self.random_state = 1
        self.X_cand = np.array([[8, 1], [9, 1], [5, 1]])
        self.X = np.array([[1, 2], [5, 8], [8, 4], [5, 4]])
        self.y = np.array([0, 0, 1, 1])
        self.clf = SklearnClassifier(
            GaussianProcessClassifier(), classes=self.classes)
        self.cost_matrix = np.eye(2)
        self.kwargs = dict(X_cand=self.X_cand, clf=self.clf,
                           X=self.X, y=self.y)

    def test_init_param_cost_matrix(self):
        qs = VOI(cost_matrix=np.ones((2, 3)))
        self.assertRaises(ValueError, qs.query, **self.kwargs)

        qs = VOI(cost_matrix='string')
        self.assertRaises(ValueError, qs.query, **self.kwargs)

        qs = VOI(cost_matrix=np.ones((3, 3)))
        self.assertRaises(ValueError, qs.query, **self.kwargs)

    def test_init_param_labeling_cost(self):
        #qs = VOI(labeling_cost=None)
        #self.assertRaises(ValueError, qs.query, **self.kwargs)

        labeling_cost = np.ones((len(self.X_cand)-1, 1))
        qs = VOI(labeling_cost=labeling_cost)
        self.assertRaises(ValueError, qs.query, **self.kwargs)

        labeling_cost = np.ones((1, len(self.classes) - 1))
        qs = VOI(labeling_cost=labeling_cost)
        self.assertRaises(ValueError, qs.query, **self.kwargs)

        labeling_cost = np.ones((len(self.X_cand)-1, len(self.classes) - 1))
        qs = VOI(labeling_cost=labeling_cost)
        self.assertRaises(ValueError, qs.query, **self.kwargs)

        qs = VOI(labeling_cost='string')
        self.assertRaises(ValueError, qs.query, **self.kwargs)

    def test_init_param_ignore_partial_fit(self):
        qs = VOI(ignore_partial_fit=None)
        self.assertRaises(
            TypeError, qs.query, X_cand=self.X_cand, X=self.X, y=self.y,
            clf=self.clf
        )

    # Test query parameters
    def test_query_param_clf(self):
        qs = VOI()
        self.assertRaises(TypeError, qs.query, X_cand=self.X_cand,
                          clf=GaussianProcessClassifier(), X=self.X,
                          y=self.y)

    def test_query_param_X(self):
        qs = VOI()
        self.assertRaises(ValueError, qs.query, X_cand=self.X_cand,
                          clf=self.clf, X=None, y=self.y)
        self.assertRaises(ValueError, qs.query, X_cand=self.X_cand,
                          clf=self.clf, X='string', y=self.y)
        self.assertRaises(ValueError, qs.query, X_cand=self.X_cand,
                          clf=self.clf, X=[], y=self.y)
        self.assertRaises(ValueError, qs.query, X_cand=self.X_cand,
                          clf=self.clf, X=self.X[0:-1], y=self.y)

    def test_query_param_y(self):
        qs = VOI()
        self.assertRaises(TypeError, qs.query, X_cand=self.X_cand,
                          clf=self.clf, X=self.X, y=None)
        self.assertRaises(ValueError, qs.query, X_cand=self.X_cand,
                          clf=self.clf, X=self.X, y='string')
        self.assertRaises(ValueError, qs.query, X_cand=self.X_cand,
                          clf=self.clf, X=self.X, y=[])
        self.assertRaises(ValueError, qs.query, X_cand=self.X_cand,
                          clf=self.clf, X=self.X, y=self.y[0:-1])
        
    def test_query_param_sample_weight(self):
        qs = VOI()
        sample_weight_list = [
            'string', self.X_cand, self.X_cand[:-2],
            np.empty((len(self.X) - 1)), np.empty((len(self.X) + 1))
        ]
        for sample_weight in sample_weight_list:
            self.assertRaises(
                ValueError, qs.query, **self.kwargs,
                sample_weight=sample_weight,
                sample_weight_cand=np.ones(shape=len(self.X_cand))
            )

    def test_query_param_sample_weight_cand(self):
        qs = VOI()
        sample_weight_cand_list = [
            'string', self.X_cand, self.X_cand[:-2],
            np.empty((len(self.X_cand) - 1)), np.empty((len(self.X_cand) + 1))
        ]
        for sample_weight in sample_weight_cand_list:
            self.assertRaises(
                ValueError, qs.query, **self.kwargs,
                sample_weight=np.ones(shape=len(self.X)),
                sample_weight_cand=sample_weight
            )

    def test_query(self):
        classes = [0, 1]
        X_cand = np.array([[8, 1], [9, 1]])
        X = np.array([[1, 2], [5, 8], [8, 4], [5, 4]])
        y = np.array([MISSING_LABEL, 0, 1, MISSING_LABEL])
        cost_matrix = 1 - np.eye(2)
        clf_partial = SklearnClassifier(
            GaussianNB(), classes=classes
        ).fit(X, y)
        clf = SklearnClassifier(estimator=GaussianProcessClassifier(),
                                random_state=self.random_state,
                                classes=classes)
        qs = VOI()

        # return_utilities
        L = list(qs.query(**self.kwargs, return_utilities=True))
        self.assertTrue(len(L) == 2)
        L = list(qs.query(**self.kwargs, return_utilities=False))
        self.assertTrue(len(L) == 1)

        # batch_size
        bs = 2
        best_idx = qs.query(**self.kwargs, batch_size=bs)
        self.assertEqual(bs, len(best_idx))

        # query
        qs.query(X_cand=X_cand, clf=clf_partial, X=X, y=y)
        qs = VOI()
        qs.query(X_cand=X_cand, clf=clf_partial, X=X, y=y)
        class DummyClf(SkactivemlClassifier):
            def fit(self, X, y, sample_weight=None):
                self.classes_ = np.unique(y[labeled_indices(y)])
                return self

            def predict_proba(self, X):
                return np.full(shape=(len(X), len(self.classes_)), fill_value=0.5)

        labeling_cost = 2.345
        qs = VOI(cost_matrix=cost_matrix,
                 labeling_cost=labeling_cost)
        idxs, utils = qs.query(X_cand=X_cand, clf=DummyClf(), X=X, y=y,
                               return_utilities=True)
        np.testing.assert_array_equal(utils[0], [-labeling_cost, -labeling_cost])

        labeling_cost = np.array([2.346, 6.234])
        qs = VOI(cost_matrix=cost_matrix,
                 labeling_cost=labeling_cost)
        idxs, utils = qs.query(X_cand=X_cand, clf=DummyClf(), X=X, y=y,
                               return_utilities=True)
        np.testing.assert_array_equal(utils[0], -labeling_cost)

        labeling_cost = np.array([[2.346, 6.234]])
        expected = [-labeling_cost.mean(), -labeling_cost.mean()]
        qs = VOI(cost_matrix=cost_matrix,
                 labeling_cost=labeling_cost)
        idxs, utils = qs.query(X_cand=X_cand, clf=DummyClf(), X=X, y=y,
                               return_utilities=True)
        np.testing.assert_array_equal(utils[0], expected)

        labeling_cost = np.array([[2.346, 6.234],
                                  [3.876, 3.568]])
        expected = -labeling_cost.mean(axis=1)
        qs = VOI(cost_matrix=cost_matrix,
                 labeling_cost=labeling_cost)
        idxs, utils = qs.query(X_cand=X_cand, clf=DummyClf(), X=X, y=y,
                               return_utilities=True)
        np.testing.assert_array_equal(utils[0], expected)

    def test_total_labeled_risk(self):
        cost_matrix = np.array([[0, 1], [2, 0]])
        y_labeled = [0, 0, 1, 1]
        probas = [[0.6, 0.4],
                  [0.7, 0.3],
                  [0.2, 0.8],
                  [0.1, 0.9]]
        expected = 1.3
        risk = _total_labeled_risk(y_labeled, probas, cost_matrix)
        np.testing.assert_equal(risk, expected)

    def test_total_unlabeled_risk(self):
        cost_matrix = np.array([[0, 1], [2, 0]])
        probas = np.array([[0.6, 0.4],
                           [0.7, 0.3],
                           [0.2, 0.8],
                           [0.1, 0.9]])
        expected = 2.1
        risk = _total_unlabeled_risk(probas, cost_matrix)
        np.testing.assert_equal(risk, expected)

        cost_matrix = np.array([[0, 1, 2],
                                [4, 0, 3],
                                [5, 6, 0]])
        probas = np.array([[0.3, 0.3, 0.4],
                           [0.2, 0.2, 0.6]])
        expected = 4.49
        risk = _total_unlabeled_risk(probas, cost_matrix)
        np.testing.assert_equal(risk, expected)
