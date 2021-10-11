import unittest

import numpy as np
from sklearn.naive_bayes import GaussianNB

from skactiveml.classifier import PWC, SklearnClassifier
from skactiveml.pool import ExpectedErrorReduction as EER
from skactiveml.utils import MISSING_LABEL


class TestExpectedErrorReduction(unittest.TestCase):

    def setUp(self):
        self.X_cand = np.zeros((100, 2))
        self.X = np.zeros((6, 2))
        self.y = [0, 1, 1, 0, 2, 1]
        self.classes = [0, 1, 2]
        self.cost_matrix = np.eye(3)
        self.clf = PWC(classes=self.classes)
        self.clf_partial = SklearnClassifier(
            GaussianNB(), classes=self.classes
        ).fit(self.X, self.y)

    # Test init parameters
    def test_init_param_method(self):
        eer = EER(method='wrong_method')
        self.assertTrue(hasattr(eer, 'method'))
        self.assertRaises(
            ValueError, eer.query, X_cand=self.X_cand, clf=self.clf,
            X=self.X, y=self.y
        )

        for method in ['emr', 'csl', 'log_loss']:
            eer = EER(method=method)
            self.assertTrue(hasattr(eer, 'method'))
            self.assertEqual(eer.method, method)

    def test_init_param_cost_matrix(self):
        for cost_matrix in [np.ones((2, 3)), 'string', np.ones((2, 2))]:
            eer = EER(cost_matrix=cost_matrix)
            self.assertRaises(
                ValueError, eer.query, X_cand=self.X_cand, X=self.X, y=self.y,
                clf=self.clf
            )

    def test_init_param_ignore_partial_fit(self):
        eer = EER(ignore_partial_fit=None)
        self.assertRaises(
            TypeError, eer.query, X_cand=self.X_cand, X=self.X, y=self.y,
            clf=self.clf
        )

    # Test query parameters
    def test_query_param_clf(self):
        eer = EER()
        self.assertRaises(
            TypeError, eer.query, X_cand=self.X_cand, X=self.X, y=self.y,
            clf='test'
        )

    def test_query_param_X(self):
        eer = EER(cost_matrix=self.cost_matrix)
        for X in [None, np.ones((5, 3))]:
            self.assertRaises(
                ValueError, eer.query, X_cand=self.X_cand, X=X, y=self.y,
                clf=self.clf
            )
        eer = EER(cost_matrix=self.cost_matrix, method='csl')
        self.assertRaises(
            ValueError, eer.query, X_cand=self.X_cand, X=None, y=self.y,
            clf=self.clf_partial
        )

    def test_query_param_y(self):
        eer = EER(cost_matrix=self.cost_matrix)
        y_list = [None, [0, 1, 4, 0, 2, 1]]
        for y in y_list:
            self.assertRaises(
                ValueError, eer.query, X_cand=self.X_cand, X=self.X, y=y,
                clf=self.clf
            )
        eer = EER(cost_matrix=self.cost_matrix, method='csl')
        self.assertRaises(
            ValueError, eer.query, X_cand=self.X_cand, X=self.X, y=None,
            clf=self.clf_partial
        )

    def test_query_param_sample_weight(self):
        eer = EER()
        for sample_weight in ['string', np.ones(3)]:
            self.assertRaises(
                ValueError, eer.query, X_cand=self.X_cand, X=self.X, y=self.y,
                sample_weight=sample_weight, clf=self.clf
            )

    def test_query_param_sample_weight_cand(self):
        eer = EER()
        for sample_weight_cand in ['string', np.ones(3)]:
            self.assertRaises(
                ValueError, eer.query, X_cand=self.X_cand, X=self.X, y=self.y,
                sample_weight_cand=sample_weight_cand, clf=self.clf
            )

    def test_query(self):
        # Test methods.
        X = [[0], [1], [2]]
        clf = PWC(classes=[0, 1])
        clf_partial = SklearnClassifier(GaussianNB(), classes=[0, 1])
        for method in ['emr', 'csl', 'log_loss']:
            eer = EER(method=method)
            y = [MISSING_LABEL, MISSING_LABEL, MISSING_LABEL]
            clf_partial.fit(X, y)
            _, utilities = eer.query(
                X_cand=X, X=X, y=y,
                sample_weight=np.ones_like(y),
                sample_weight_cand=np.ones_like(y),
                clf=clf, return_utilities=True
            )
            self.assertEqual(utilities.shape, (1, len(X)))
            self.assertEqual(len(np.unique(utilities)), 1)

            if method != 'csl':
                for sample_weight_cand in [None, np.ones_like(y)]:
                    _, utilities = eer.query(
                        X_cand=X, clf=clf_partial,
                        sample_weight_cand=sample_weight_cand,
                        return_utilities=True
                    )
                    self.assertEqual(utilities.shape, (1, len(X)))
                    self.assertEqual(len(np.unique(utilities)), 1)

            _, utilities = eer.query(
                X_cand=X, X=X, y=[0, 1, MISSING_LABEL], clf=self.clf,
                return_utilities=True
            )
            self.assertGreater(utilities[0, 2], utilities[0, 1])
            self.assertGreater(utilities[0, 2], utilities[0, 0])

        # Test scenario.
        X_cand = [[0], [1], [2], [5]]
        eer = EER()

        _, utilities = eer.query(
            X_cand=X_cand, X=[[1]], y=[0], clf=PWC(classes=[0, 1]),
            return_utilities=True
        )
        np.testing.assert_array_equal(utilities, np.zeros((1, len(X_cand))))
        query_indices = eer.query(
            X_cand=[[0], [100], [200]], X=[[0], [200]], y=[0, 1], clf=self.clf
        )
        np.testing.assert_array_equal(query_indices, [1])
