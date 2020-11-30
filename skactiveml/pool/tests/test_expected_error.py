import unittest

import numpy as np

from skactiveml.classifier import PWC
from skactiveml.pool import ExpectedErrorReduction as EER
from skactiveml.utils import MISSING_LABEL


class ExpectedErrorReductionTest(unittest.TestCase):

    def setUp(self):
        self.X_cand = np.zeros((100, 2))
        self.X = np.zeros((6, 2))
        self.y = [0, 1, 1, 0, 2, 1]
        self.classes = [0, 1, 2]
        self.cost_matrix = np.eye(3)
        self.clf = PWC(classes=self.classes)

    # Test init parameters
    def test_init_param_clf(self):
        eer = EER(clf=3, classes=self.classes)
        self.assertTrue(hasattr(eer, 'clf'))
        self.assertRaises(TypeError, eer.query, self.X_cand, self.X, self.y)

    def test_init_param_classes(self):
        eer = EER(clf=self.clf, classes=self.clf)
        self.assertTrue(hasattr(eer, 'classes'))
        self.assertRaises(TypeError, eer.query, self.X_cand, self.X, self.y)

        eer = EER(clf=self.clf, classes=[0, 1, 2, 3])
        self.assertTrue(hasattr(eer, 'classes'))
        self.assertRaises(ValueError, eer.query, self.X_cand, self.X, self.y)

    def test_init_param_method(self):
        eer = EER(clf=self.clf, classes=self.classes, method='wrong_method')
        self.assertTrue(hasattr(eer, 'method'))
        self.assertRaises(ValueError, eer.query, self.X_cand, self.X, self.y)

        for method in ['emr', 'csl', 'log_loss']:
            eer = EER(clf=self.clf, classes=self.classes, method=method)
            self.assertTrue(hasattr(eer, 'method'))
            self.assertEqual(eer.method, method)

    def test_init_param_cost_matrix(self):
        eer = EER(clf=self.clf, classes=self.classes,
                  cost_matrix=np.ones((2, 3)))
        self.assertRaises(ValueError, eer.query, self.X_cand, self.X, self.y)

        eer = EER(clf=self.clf, classes=self.classes, cost_matrix='string')
        self.assertRaises(ValueError, eer.query, self.X_cand, self.X, self.y)

        eer = EER(clf=self.clf, classes=self.classes,
                  cost_matrix=np.ones((2, 2)))
        self.assertRaises(ValueError, eer.query, self.X_cand, self.X, self.y)

    def test_init_param_random_state(self):
        eer = EER(clf=self.clf, classes=self.classes, random_state='string')
        self.assertRaises(ValueError, eer.query, self.X_cand, self.X, self.y)

    def test_init_param_missing_label(self):
        eer = EER(clf=self.clf, classes=self.classes, missing_label=[1, 2, 3])
        self.assertRaises(TypeError, eer.query, self.X_cand, self.X, self.y)

    # Test query parameters
    def test_query_param_X_cand(self):
        eer = EER(self.clf, self.classes, cost_matrix=self.cost_matrix)
        self.assertRaises(ValueError, eer.query, X_cand=[], X=[], y=[])
        self.assertRaises(ValueError, eer.query, X_cand=[], X=self.X, y=self.y)

    def test_query_param_X(self):
        eer = EER(self.clf, self.classes, cost_matrix=self.cost_matrix)
        self.assertRaises(ValueError, eer.query, X_cand=self.X_cand,
                          X=np.ones((5, 3)), y=self.y)

    def test_query_param_y(self):
        eer = EER(self.clf, self.classes, cost_matrix=self.cost_matrix)
        self.assertRaises(ValueError, eer.query, X_cand=self.X_cand,
                          X=self.X, y=[0, 1, 4, 0, 2, 1])

    def test_query_param_batch_size(self):
        eer = EER(self.clf, self.classes)
        self.assertRaises(TypeError, eer.query, self.X_cand, self.X, self.y,
                          batch_size=1.0)
        self.assertRaises(ValueError, eer.query, self.X_cand, self.X, self.y,
                          batch_size=0)

    def test_query_param_return_utilities(self):
        eer = EER(self.clf, self.classes, cost_matrix=self.cost_matrix)
        self.assertRaises(TypeError, eer.query, X_cand=self.X_cand,
                          return_utilities=None)
        self.assertRaises(TypeError, eer.query, X_cand=self.X_cand,
                          return_utilities=[])
        self.assertRaises(TypeError, eer.query, X_cand=self.X_cand,
                          return_utilities=0)

    def test_query(self):
        # Test methods
        X_cand = [[0], [1], [2]]
        for method in ['emr', 'csl', 'log_loss']:
            eer = EER(clf=PWC(classes=[0, 1]), classes=[0, 1], method=method)
            _, utilities = eer.query(X_cand, [[1]], [MISSING_LABEL],
                                     return_utilities=True)
            self.assertEqual(utilities.shape, (1, len(X_cand)))
            self.assertEqual(len(np.unique(utilities)), 1)

            _, utilities = eer.query(X_cand, X=[[0], [1], [2]],
                                     y=[0, 1, MISSING_LABEL],
                                     return_utilities=True)
            self.assertGreater(utilities[0, 2], utilities[0, 1])
            self.assertGreater(utilities[0, 2], utilities[0, 0])

        # Test scenario
        X_cand = [[0], [1], [2], [5]]
        eer = EER(clf=PWC(classes=[0, 1]), classes=[0, 1])

        _, utilities = eer.query(X_cand, [[1]], [0], return_utilities=True)
        np.testing.assert_array_equal(utilities, np.zeros((1, len(X_cand))))

        query_indices = eer.query([[0], [100], [200]], [[0], [200]], [0, 1])
        np.testing.assert_array_equal(query_indices, [1])


if __name__ == '__main__':
    unittest.main()
