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
        self.X_cand = np.array([[7, 1], [9, 1], [5, 1], [3, 4], [9, 12]])
        self.X = np.array([[1, 2], [5, 8], [8, 4], [5, 4]])
        self.y = np.array([0, 0, 1, 1])
        self.classes = np.array([0, 1])
        self.A_cand = np.array([[False, True,  True],
                                [True,  True,  False],
                                [True,  False, False],
                                [True,  True,  True],
                                [True, False, False]])

    def test_init_param(self):
        wrapper = MultiAnnotWrapper(CMM())

        self.assertRaises(TypeError, wrapper.query, self.X_cand,
                          self.X, self.y, A_cand=self.A_cand)

        y = np.array([[[1, 0], [0, 1], [1, 1], [0, 0]]])

        random = RandomSampler(self.random_state)

        wrapper = MultiAnnotWrapper(random, self.random_state)

        self.assertRaises(ValueError, wrapper.query, self.X_cand,
                          self.X, y, A_cand=self.A_cand, return_utilities=True)

        X_cand = np.array([[7, 1], [9, 1]])

        A_cand = np.array([[True, True, True, True],
                           [False, False, False, False]])

        wrapper = MultiAnnotWrapper(random, self.random_state)

        self.assertRaises(ValueError, wrapper.query, X_cand, A_cand=A_cand,
                          batch_size=5, return_utilities=False)

    def test_query_one_annotator_per_sample_batch_size_one(self):
        # test functionality with uncertainty sampling
        clf = SklearnClassifier(estimator=GaussianProcessClassifier(),
                                random_state=self.random_state)

        uncertainty = UncertaintySampling(clf=clf, method='entropy')

        wrapper = MultiAnnotWrapper(uncertainty, self.random_state)

        re_val = wrapper.query(self.X_cand, self.X, self.y,
                               A_cand=self.A_cand, return_utilities=True)
        best_cand_indices, utilities = re_val
        self.check_availability(best_cand_indices, self.A_cand)

        # test functionality with random sampler and larger batch size
        random = RandomSampler(self.random_state)

        wrapper = MultiAnnotWrapper(random, self.random_state)

        best_cand_indices = wrapper.query(self.X_cand, A_cand=self.A_cand,
                                          return_utilities=False)
        self.check_availability(best_cand_indices, self.A_cand)

    def test_query_one_annotator_per_sample_batch_size_five(self):
        random = RandomSampler(self.random_state)

        wrapper = MultiAnnotWrapper(random, self.random_state)

        best_cand_indices = wrapper.query(self.X_cand, A_cand=self.A_cand,
                                          batch_size=5,
                                          return_utilities=False)
        self.check_availability(best_cand_indices, self.A_cand)

    def test_query_three_annotators_per_sample_batch_size_five(self):
        random = RandomSampler(self.random_state)

        wrapper = MultiAnnotWrapper(random, self.random_state)

        best_cand_indices = wrapper.query(self.X_cand, A_cand=self.A_cand,
                                          batch_size=5,
                                          pref_annotators_per_sample=3,
                                          return_utilities=False)
        self.check_availability(best_cand_indices, self.A_cand)

    def test_query_three_annotators_per_sample_batch_size_five_vert(self):
        random = RandomSampler(self.random_state)

        X_cand = np.array([[7, 1], [9, 1]])

        A_cand = np.array([[True, True,  True,  True],
                           [True, False, False, False]])

        wrapper = MultiAnnotWrapper(random, self.random_state)

        best_cand_indices = wrapper.query(X_cand, A_cand=A_cand,
                                          batch_size=5,
                                          pref_annotators_per_sample=3,
                                          return_utilities=False)
        self.check_availability(best_cand_indices, A_cand)

    def check_availability(self, best_cand_indices, A_cand):
        best_value_indices = best_cand_indices[:, 0]
        best_annotator_indices = best_cand_indices[:, 1]

        self.assertEqual(best_value_indices.shape[0],
                         best_annotator_indices.shape[0])

        for i in range(best_value_indices.shape[0]):
            self.assertTrue(A_cand[best_value_indices[i],
                                   best_annotator_indices[i]])
