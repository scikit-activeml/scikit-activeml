import unittest

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier

from skactiveml.classifier import SklearnClassifier
from skactiveml.pool import UncertaintySampling
from skactiveml.pool._wrapper import MultiAnnotWrapper


class TestMultiAnnotWrapper(unittest.TestCase):

    def setUp(self):
        self.random_state = 1
        self.X_cand = np.array([[8, 1], [9, 1], [5, 1]])
        self.X = np.array([[1, 2], [5, 8], [8, 4], [5, 4]])
        self.y = np.array([0, 0, 1, 1])
        self.classes = np.array([0, 1])
        self.A_cand = np.array([[True, True], [False, True], [True, False]])
        pass

    def test_query(self):
        clf = SklearnClassifier(estimator=GaussianProcessClassifier(), random_state=self.random_state)
        # entropy
        uncertainty = UncertaintySampling(clf=clf, method='entropy')

        wrapper = MultiAnnotWrapper(uncertainty, self.random_state)

        best_indices, utilities = wrapper.query(self.X_cand, self.X, self.y, A_cand=self.A_cand, return_utilities=True,
                                                random_state=self.random_state)
