import unittest

import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier

from skactiveml.classifier import SklearnClassifier, CMM
from skactiveml.pool import UncertaintySampling, RandomSampler
from skactiveml.pool.multi._wrapper import MultiAnnotWrapper
from skactiveml.utils._aggregation import majority_vote


class TestMultiAnnotWrapper(unittest.TestCase):

    def setUp(self):
        self.random_state = 1
        self.X_cand = np.array([[7, 1], [9, 1], [5, 1], [3, 4], [9, 12]])
        self.X = np.array([[1, 2], [5, 8], [8, 4], [5, 4]])
        self.y = np.array([0, 0, 1, 1])
        self.classes = np.array([0, 1])
        self.A_cand = np.array([[False, True, True],
                                [True, True, False],
                                [True, False, False],
                                [True, True, True],
                                [True, False, False]])

    def test_init_param_strategy(self):
        wrapper = MultiAnnotWrapper(CMM())

        query_params_dict = {'X': self.X, 'y': self.y}
        self.assertRaises(TypeError, wrapper.query, self.X_cand,
                          query_params_dict, A_cand=self.A_cand)

        wrapper = MultiAnnotWrapper(0)

        query_params_dict = {'X': self.X, 'y': self.y}
        self.assertRaises(TypeError, wrapper.query, self.X_cand,
                          query_params_dict, A_cand=self.A_cand)

    def test_init_param_n_annotators(self):
        random = RandomSampler(self.random_state)
        wrapper = MultiAnnotWrapper(random, random_state=self.random_state,
                                    n_annotators='string')

        self.assertRaises(TypeError, wrapper.query, self.X_cand,
                          A_cand=None, return_utilities=True)

    def test_init_param_y_aggregate(self):
        random = RandomSampler(self.random_state)
        wrapper = MultiAnnotWrapper(random, y_aggregate='string',
                                    random_state=self.random_state)

        query_params_dict = {'X': self.X, 'y': self.y}
        self.assertRaises(TypeError, wrapper.query, self.X_cand,
                          query_params_dict=query_params_dict,
                          A_cand=self.A_cand, return_utilities=True)

        dummy_function = lambda x, y: majority_vote(x)

        random = RandomSampler(self.random_state)
        wrapper = MultiAnnotWrapper(random, y_aggregate=dummy_function,
                                    random_state=self.random_state)

        query_params_dict = {'X': self.X, 'y': self.y}
        self.assertRaises(TypeError, wrapper.query, self.X_cand,
                          query_params_dict=query_params_dict,
                          A_cand=self.A_cand, return_utilities=True)

    def test_init_param_random_state(self):
        random = RandomSampler(self.random_state)
        wrapper = MultiAnnotWrapper(random, random_state='string')

        self.assertRaises(ValueError, wrapper.query, self.X_cand,
                          A_cand=self.A_cand, return_utilities=True)

    def test_query_param_X_cand(self):
        random = RandomSampler(self.random_state)
        X_cand = [1, 0, 2, 4]
        A_cand = [True, True, True]
        wrapper = MultiAnnotWrapper(random, self.random_state)

        self.assertRaises(ValueError, wrapper.query, X_cand,
                          A_cand=A_cand, return_utilities=False)

        self.assertRaises(ValueError, wrapper.query, 0, A_cand=self.A_cand,
                          return_utilities=False)

    def test_query_param_query_params_dict(self):
        clf = SklearnClassifier(estimator=GaussianProcessClassifier(),
                                random_state=self.random_state)
        uncertainty = UncertaintySampling(method='entropy')
        wrapper = MultiAnnotWrapper(uncertainty, self.random_state)

        y = np.array([[[1, 0], [0, 1], [1, 1], [0, 0]]])

        query_params_dict = {'X': self.X, 'y': y, 'clf': clf}
        self.assertRaises(ValueError, wrapper.query, self.X_cand,
                          query_params_dict, A_cand=self.A_cand,
                          return_utilities=True)

        self.assertRaises(TypeError, wrapper.query, self.X_cand,
                          "string", A_cand=self.A_cand)

    def test_query_param_A_cand(self):
        random = RandomSampler(self.random_state)
        wrapper = MultiAnnotWrapper(random, random_state=self.random_state)
        self.assertRaises(TypeError, wrapper.query, self.X_cand,
                          A_cand=None, batch_size=5, return_utilities=False)

    def test_query_param_batch_size(self):
        random = RandomSampler(self.random_state)
        wrapper = MultiAnnotWrapper(random, self.random_state)
        self.assertRaises(TypeError, wrapper.query, self.X_cand,
                          A_cand=self.A_cand, batch_size=None,
                          return_utilities=False)

    def test_query_param_return_utilities(self):
        random = RandomSampler(self.random_state)
        wrapper = MultiAnnotWrapper(random, self.random_state)
        self.assertRaises(TypeError, wrapper.query, self.X_cand,
                          A_cand=self.A_cand, batch_size=5,
                          return_utilities=None)

    def test_query_param_n_annotators_per_sample(self):
        random = RandomSampler(self.random_state)
        wrapper = MultiAnnotWrapper(random, self.random_state)
        self.assertRaises(TypeError, wrapper.query, self.X_cand,
                          A_cand=self.A_cand, batch_size=5,
                          return_utilities=False,
                          n_annotators_per_sample=None)
        n_annotators_per_sample = np.array([[0, 1], [0, 2]])
        self.assertRaises(ValueError, wrapper.query, self.X_cand,
                          A_cand=self.A_cand, batch_size=5,
                          return_utilities=False,
                          n_annotators_per_sample=n_annotators_per_sample)

    def test_query_param_A_perf(self):
        random = RandomSampler(self.random_state)
        wrapper = MultiAnnotWrapper(random, self.random_state)
        self.assertRaises(TypeError, wrapper.query, self.X_cand,
                          A_cand=self.A_cand, batch_size=5,
                          return_utilities=False,
                          A_perf=3)
        self.assertRaises(ValueError, wrapper.query, self.X_cand,
                          A_cand=self.A_cand, batch_size=5,
                          return_utilities=False,
                          A_perf=np.array([[0, ], ]))

    def test_query_one_annotator_per_sample_batch_size_one(self):
        # test functionality with uncertainty sampling
        clf = SklearnClassifier(estimator=GaussianProcessClassifier(),
                                random_state=self.random_state)

        uncertainty = UncertaintySampling(method='entropy')

        def y_aggregate(y_t):
            w = np.repeat(np.arange(y.shape[1]).reshape(1, -1), y.shape[0],
                          axis=0)
            return majority_vote(y_t, w=w)

        wrapper = MultiAnnotWrapper(uncertainty, y_aggregate=y_aggregate,
                                    random_state=self.random_state)

        X = np.array([[1, 2], [5, 8], [8, 4], [5, 4], [3, 4]])
        y = np.array([[1, 0], [0, 1], [1, 1], [0, 0], [0, 1]])

        query_params_dict = {'X': X, 'y': y, 'clf': clf}
        re_val = wrapper.query(self.X_cand, query_params_dict,
                               A_cand=self.A_cand, return_utilities=True,
                               batch_size='adaptive')

        best_cand_indices, utilities = re_val
        self.assertEqual((1, 2), best_cand_indices.shape)
        self.assertEqual((1, 5, 3), utilities.shape)
        self.check_max(best_cand_indices, utilities)
        self.check_availability(best_cand_indices, self.A_cand)

        # test functionality with random sampler, larger batch size,
        # return_utilities set to false
        random = RandomSampler(self.random_state)

        wrapper = MultiAnnotWrapper(random, self.random_state)

        best_cand_indices = wrapper.query(self.X_cand, A_cand=self.A_cand,
                                          return_utilities=False)
        self.assertEqual((1, 2), best_cand_indices.shape)
        self.check_availability(best_cand_indices, self.A_cand)

    def test_query_one_annotator_per_sample_batch_size_five(self):
        random = RandomSampler(self.random_state)

        wrapper = MultiAnnotWrapper(random, self.random_state)

        re_val = wrapper.query(self.X_cand, A_cand=self.A_cand,
                               batch_size=5, return_utilities=True)
        best_cand_indices, utilities = re_val
        self.assertEqual((5, 2), best_cand_indices.shape)
        self.assertEqual((5, 5, 3), utilities.shape)
        self.check_max(best_cand_indices, utilities)
        self.check_availability(best_cand_indices, self.A_cand)

    def test_query_n_annotators_is_set(self):
        random = RandomSampler(self.random_state)

        wrapper = MultiAnnotWrapper(random, random_state=self.random_state,
                                    n_annotators=3)

        re_val = wrapper.query(self.X_cand, A_cand=None, batch_size=5,
                               return_utilities=True)
        best_cand_indices, utilities = re_val
        self.assertEqual((5, 2), best_cand_indices.shape)
        self.assertEqual((5, 5, 3), utilities.shape)
        self.check_max(best_cand_indices, utilities)

    def test_query_three_n_annotators_per_sample_batch_size_five(self):
        random = RandomSampler(self.random_state)

        wrapper = MultiAnnotWrapper(random, self.random_state)

        re_val = wrapper.query(self.X_cand, A_cand=self.A_cand, batch_size=5,
                               n_annotators_per_sample=3,
                               return_utilities=True)
        best_cand_indices, utilities = re_val
        self.assertEqual((5, 2), best_cand_indices.shape)
        self.assertEqual((5, 5, 3), utilities.shape)
        self.check_max(best_cand_indices, utilities)
        self.check_availability(best_cand_indices, self.A_cand)

    def test_query_three_n_annotators_per_sample_batch_size_five_mismatch(self):
        random = RandomSampler(self.random_state)

        X_cand = np.array([[7, 1], [9, 1]])

        A_cand = np.array([[True, True, True, True],
                           [True, False, False, False]])

        wrapper = MultiAnnotWrapper(random, self.random_state)

        re_val = wrapper.query(X_cand, A_cand=A_cand, batch_size=5,
                               n_annotators_per_sample=3,
                               return_utilities=True)
        best_cand_indices, utilities = re_val
        self.assertEqual((5, 2), best_cand_indices.shape)
        self.assertEqual((5, 2, 4), utilities.shape)
        self.check_max(best_cand_indices, utilities)
        self.check_availability(best_cand_indices, A_cand)

    def test_query_varying_n_annotators_per_sample_batch_size_five(self):
        random = RandomSampler(self.random_state)

        wrapper = MultiAnnotWrapper(random, n_annotators=3,
                                    random_state=self.random_state)

        pref = np.array([3, 2])

        re_val = wrapper.query(self.X_cand, A_cand=None,
                               batch_size=5, n_annotators_per_sample=pref,
                               return_utilities=True)
        best_cand_indices, utilities = re_val
        self.assertEqual((5, 2), best_cand_indices.shape)
        self.assertEqual(best_cand_indices[0, 0], best_cand_indices[1, 0])
        self.assertEqual(best_cand_indices[1, 0], best_cand_indices[2, 0])
        self.assertEqual(best_cand_indices[3, 0], best_cand_indices[4, 0])
        self.assertEqual((5, 5, 3), utilities.shape)
        self.check_max(best_cand_indices, utilities)

    def test_query_per_sample_too_large(self):
        random = RandomSampler(self.random_state)
        wrapper = MultiAnnotWrapper(random, n_annotators=3,
                                    random_state=self.random_state)

        pref = np.array([3, 2, 1, 1, 1, 1])

        re_val = wrapper.query(self.X_cand, A_cand=None,
                               batch_size=1, n_annotators_per_sample=pref,
                               return_utilities=True)

        best_cand_indices, utilities = re_val
        self.assertEqual((1, 2), best_cand_indices.shape)
        self.assertEqual((1, 5, 3), utilities.shape)
        self.check_max(best_cand_indices, utilities)

    def test_query_unavailable_annotators(self):
        random = RandomSampler(self.random_state)

        X_cand = np.array([[7, 1], [9, 1], [3, 5], [2, 7]])

        A_cand = np.array([[False, False, False, False],
                           [True, True, True, True],
                           [False, False, False, False],
                           [True, False, False, False]])

        wrapper = MultiAnnotWrapper(random, self.random_state)

        re_val = wrapper.query(X_cand, A_cand=A_cand, batch_size=5,
                               return_utilities=True)

        best_cand_indices, utilities = re_val
        self.assertEqual((5, 2), best_cand_indices.shape)
        self.assertEqual((5, 4, 4), utilities.shape)
        self.check_max(best_cand_indices, utilities)
        self.check_availability(best_cand_indices, A_cand)

    def test_query_custom_annotator_special_preference(self):
        random = RandomSampler(self.random_state)

        wrapper = MultiAnnotWrapper(random, n_annotators=3,
                                    random_state=self.random_state)

        X_cand = np.array([[7, 1], [9, 1]])

        A_perf = np.array([[1, 2, 3],
                            [3, 2, 1]])

        re_val = wrapper.query(X_cand=X_cand, batch_size=6,
                               n_annotators_per_sample=3,
                               A_perf=A_perf,
                               return_utilities=True)

        best_cand_indices, utilities = re_val
        # assert the utilities fit A_perf
        self.assertFalse(np.any(utilities[:, 0, 2] < utilities[:, 0, 1]))
        self.assertFalse(np.any(utilities[:, 0, 1] < utilities[:, 0, 0]))

        self.assertFalse(np.any(utilities[:, 1, 0] < utilities[:, 1, 1]))
        self.assertFalse(np.any(utilities[:, 1, 1] < utilities[:, 1, 2]))

        self.check_max(best_cand_indices, utilities)

    def test_query_custom_annotator_general_equal_preference(self):
        random = RandomSampler(self.random_state)

        wrapper = MultiAnnotWrapper(random, n_annotators=3,
                                    random_state=self.random_state)

        X_cand = np.array([[7, 1], [9, 1]])
        A_perf = np.array([1, 1, 1])

        re_val = wrapper.query(X_cand=X_cand, batch_size=6,
                               n_annotators_per_sample=3,
                               A_perf=A_perf,
                               return_utilities=True)

        best_cand_indices, utilities = re_val
        # assert the utilities fit A_perf
        self.assertTrue(np.all((utilities[:, :, 2] == utilities[:, :, 1])
                               | np.isnan(utilities[:, :, 2])
                               | np.isnan(utilities[:, :, 1])))
        self.assertTrue(np.all((utilities[:, :, 1] == utilities[:, :, 0])
                               | np.isnan(utilities[:, :, 1])
                               | np.isnan(utilities[:, :, 0])))

    def check_availability(self, best_cand_indices, A_cand):
        best_value_indices = best_cand_indices[:, 0]
        best_annotator_indices = best_cand_indices[:, 1]

        self.assertEqual(best_value_indices.shape[0],
                         best_annotator_indices.shape[0])

        for i in range(best_value_indices.shape[0]):
            self.assertTrue(A_cand[best_value_indices[i],
                                   best_annotator_indices[i]])

    def check_max(self, best_cand_indices, utilities):
        for i in range(best_cand_indices.shape[0]):
            a = np.nanargmax(utilities[i])
            b = best_cand_indices[i, 0] * utilities.shape[2] + \
                best_cand_indices[i, 1]
            self.assertEqual(a, b)
