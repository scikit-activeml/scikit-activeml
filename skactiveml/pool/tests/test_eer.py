import numpy as np
import unittest

from skactiveml.pool import ExpectedErrorReduction as EER
from skactiveml.classifier import PWC


class TestEER(unittest.TestCase):

    def setUp(self):
        self.X = np.zeros((6, 2))
        self.X_cand = np.zeros((2, 2))
        self.y = [0, 1, 1, 0, 2, 1]
        self.classes = [0, 1, 2]
        self.C = np.eye(3)
        self.clf = PWC(classes=self.classes)

    def test_init(self):
        self.assertRaises(TypeError, EER)
        eer = EER(clf=3, classes=self.classes)
        self.assertRaises(TypeError, eer.query, self.X_cand, self.X, self.y)
        eer = EER(clf=self.clf, classes=self.clf)
        self.assertRaises(TypeError, eer.query, self.X_cand, self.X, self.y)
        eer = EER(clf=self.clf, classes=self.classes, method='wrong_method')
        self.assertRaises(ValueError, eer.query, self.X_cand, self.X, self.y)
        eer = EER(clf=self.clf, classes=self.classes, C=np.ones((2, 3)))
        self.assertRaises(ValueError, eer.query, self.X_cand, self.X, self.y)
        eer = EER(clf=self.clf, classes=self.classes, C=np.ones((3, 2)))
        self.assertRaises(ValueError, eer.query, self.X_cand, self.X, self.y)
        eer = EER(clf=self.clf, classes=self.classes, C='wrong_string')
        self.assertRaises(TypeError, eer.query, self.X_cand, self.X, self.y)
        eer = EER(clf=self.clf, classes=self.classes, C=np.ones((2, 2)))
        self.assertRaises(ValueError, eer.query, self.X_cand, self.X, self.y)
        eer = EER(clf=self.clf, classes=[0, 1, 2, 3])
        self.assertRaises(ValueError, eer.query, self.X_cand, self.X, self.y)
        self.assertRaises(ValueError, EER, self.clf, self.classes,
                          random_state='string')
        eer = EER(clf=self.clf, classes=self.classes, missing_label=[1, 2, 3])
        self.assertRaises(TypeError, eer.query, self.X_cand, self.X, self.y)

        eer = EER(clf=self.clf, classes=self.classes)
        self.assertIsNotNone(eer.classes)
        for method in ['emr', 'csl', 'log_loss']:
            eer = EER(clf=self.clf, classes=self.classes, method=method)
            self.assertEqual(eer.method, method)

    def test_query(self):
        eer = EER(clf=self.clf, classes=self.classes, C=self.C)
        eer.query(self.X_cand, self.X, self.y)
        self.assertRaises(ValueError, eer.query, X_cand=[], X=[], y=[])
        self.assertRaises(ValueError, eer.query, X_cand=[], X=self.X, y=self.y)
        self.assertRaises(ValueError, eer.query, X_cand=self.X_cand,
                          X=self.X, y=[0, 1, 4, 0, 2, 1])
        self.assertRaises(ValueError, eer.query, X_cand=np.zeros((2, 3)),
                          X=self.X, y=self.y)
        # eer = EER(self.clf, self.classes, method='csl')
        # eer.query(X_cand=np.array([[0], [1], [2], [3]]),
        #           X=np.array([[0], [1], [2], [3]]),
        #           y=np.array([np.nan, np.nan, np.nan, np.nan]))

    def test_scenario(self):
        clf = PWC(classes=[0, 1])
        X_cand = [[0], [1], [2]]
        for method in ['emr', 'csl', 'log_loss']:
            eer = EER(clf=clf, classes=[0, 1], method=method)
            utilities = eer.query(X_cand, X=[[1]], y=[0],
                                  return_utilities=True)[1]
            self.assertEqual(utilities.shape, (1, len(X_cand)))

        X_cand = [[0], [1], [2], [5]]
        eer = EER(clf=PWC(classes=[0, 1]), classes=[0, 1])

        utilities = eer.query(X_cand, [[1]], [0], return_utilities=True)[1]
        np.testing.assert_array_equal(utilities, np.zeros((1, len(X_cand))))

        query_indices = eer.query(X_cand=[[0], [100], [200]],
                                  X=[[0], [200]], y=[0, 1])
        np.testing.assert_array_equal(query_indices, [1])


if __name__ == '__main__':
    unittest.main()
