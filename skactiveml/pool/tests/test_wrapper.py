import unittest

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor, \
    GaussianProcessClassifier

from skactiveml.classifier import SklearnClassifier, CMM
from skactiveml.pool import UncertaintySampling, RandomSampler
from skactiveml.pool._wrapper import MultiAnnotWrapper


class TestMultiAnnotWrapper(unittest.TestCase):

    def setUp(self):
        self.random_state = 1
        self.X_cand = np.array([[7, 1], [9, 1], [5, 1]])
        self.X = np.array([[1, 2], [5, 8], [8, 4], [5, 4]])
        self.y = np.array([0, 0, 1, 1])
        self.classes = np.array([0, 1])
        self.A_cand = np.array([[False, True],
                                [True, True],
                                [True, False]])

    def test_query(self):

        # test Exception
        wrapper = MultiAnnotWrapper(CMM())

        self.assertRaises(TypeError, wrapper.query, self.X_cand,
                          self.X, self.y)

        # test functionality with uncertainty sampling
        clf = SklearnClassifier(estimator=GaussianProcessClassifier(),
                                random_state=self.random_state)

        uncertainty = UncertaintySampling(clf=clf, method='entropy')

        wrapper = MultiAnnotWrapper(uncertainty, self.random_state)

        y = np.array([[[1, 0], [0, 1], [1, 1], [0, 0]]])

        self.assertRaises(ValueError, wrapper.query, self.X_cand,
                          self.X, y, A_cand=self.A_cand,
                          return_utilities=True)

        y = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])

        re_val = wrapper.query(self.X_cand, self.X, self.y,
                               A_cand=self.A_cand, return_utilities=True)
        best_cand_indices, utilities = re_val

        self.check_availability(best_cand_indices)

        # test functionality with random sampler and larger batch size
        random = RandomSampler(self.random_state)

        wrapper = MultiAnnotWrapper(random, self.random_state)

        best_cand_indices = wrapper.query(self.X_cand, A_cand=self.A_cand,
                                          batch_size=3,
                                          return_utilities=False)

        self.check_availability(best_cand_indices)

    def check_availability(self, best_cand_indices):

        best_value_indices = best_cand_indices[:, 0]
        best_annotator_indices = best_cand_indices[:, 1]

        self.assertEqual(best_value_indices.shape[0],
                         best_annotator_indices.shape[0])

        for i in range(best_value_indices.shape[0]):
            self.assertTrue(self.A_cand[best_value_indices[i],
                                        best_annotator_indices[i]])
