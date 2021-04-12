import itertools

import numpy as np
import unittest
from itertools import product

from numpy.linalg import LinAlgError
from sklearn.metrics import pairwise_kernels

from skactiveml.pool import McPAL, XPAL
from skactiveml.classifier import PWC
from skactiveml.pool._probal import probabilistic_gain, f1_score_func, \
    _reduce_candlist_set, _calc_sim, _get_nonmyopic_cand_set, to_int_labels, \
    _dependent_cand_prob, _get_y_sim_list, _transform_scoring, _dperf, \
    estimate_bandwidth, score_recall, macro_accuracy_func, score_accuracy, \
    score_precision, calculate_optimal_prior, to_int_labels
from skactiveml.utils import MISSING_LABEL


class TestMcPAL(unittest.TestCase):

    def setUp(self):
        self.X = np.zeros((6, 2))
        self.utility_weight = np.ones(len(self.X)) / len(self.X)
        self.X_cand = np.zeros((2, 2))
        self.y = [0, 1, 1, 0, 2, 1]
        self.classes = [0, 1, 2]
        self.C = np.eye(3)
        self.clf = PWC(classes=self.classes)

    # Test init parameters
    def test_init_param_clf(self):
        pal = McPAL(clf=3)
        self.assertTrue(hasattr(pal, 'clf'))
        self.assertRaises(TypeError, pal.query, self.X_cand, self.X, self.y)

    def test_init_param_prior(self):
        pal = McPAL(self.clf, prior=0)
        self.assertTrue(hasattr(pal, 'prior'))
        self.assertRaises(ValueError, pal.query, self.X_cand, self.X, self.y)

        pal = McPAL(self.clf, prior='string')
        self.assertTrue(hasattr(pal, 'prior'))
        self.assertRaises(TypeError, pal.query, self.X_cand, self.X, self.y)

    def test_init_param_m_max(self):
        pal = McPAL(self.clf, m_max=-2)
        self.assertTrue(hasattr(pal, 'm_max'))
        self.assertRaises(ValueError, pal.query, self.X_cand, self.X, self.y)

        pal = McPAL(self.clf, m_max=1.5)
        self.assertTrue(hasattr(pal, 'm_max'))
        self.assertRaises(TypeError, pal.query, self.X_cand, self.X, self.y)

    def test_init_param_random_state(self):
        pal = McPAL(self.clf, random_state='string')
        self.assertTrue(hasattr(pal, 'random_state'))
        self.assertRaises(ValueError, pal.query, self.X_cand, self.X, self.y)

    # Test query parameters
    def test_query_param_X_cand(self):
        pal = McPAL(self.clf)
        self.assertRaises(ValueError, pal.query, X_cand=[], X=[], y=[])
        self.assertRaises(ValueError, pal.query, X_cand=[], X=self.X, y=self.y)

    def test_query_param_X(self):
        pal = McPAL(self.clf)
        self.assertRaises(ValueError, pal.query, X_cand=self.X_cand,
                          X=np.ones((5, 3)), y=self.y)

    def test_query_param_y(self):
        pal = McPAL(self.clf)
        self.assertRaises(ValueError, pal.query, X_cand=self.X_cand,
                          X=self.X, y=[0, 1, 4, 0, 2, 1])

    def test_query_param_sample_weight(self):
        pal = McPAL(self.clf)
        self.assertRaises(TypeError, pal.query, X_cand=self.X_cand,
                          X=self.X, y=self.y, sample_weight='string')
        self.assertRaises(ValueError, pal.query, X_cand=self.X_cand,
                          X=self.X, y=self.y, sample_weight=np.ones(3))

    def test_query_param_utility_weight(self):
        pal = McPAL(self.clf)
        self.assertRaises(TypeError, pal.query, X_cand=self.X_cand,
                          X=self.X, y=self.y, utility_weight='string')
        self.assertRaises(ValueError, pal.query, X_cand=self.X_cand,
                          X=self.X, y=self.y, utility_weight=np.ones(3))

    def test_query_param_batch_size(self):
        pal = McPAL(self.clf)
        self.assertRaises(TypeError, pal.query, self.X_cand, self.X, self.y,
                          batch_size=1.0)
        self.assertRaises(ValueError, pal.query, self.X_cand, self.X, self.y,
                          batch_size=0)

    def test_query_param_return_utilities(self):
        pal = McPAL(self.clf)
        self.assertRaises(TypeError, pal.query, X_cand=self.X_cand,
                          return_utilities=None)
        self.assertRaises(TypeError, pal.query, X_cand=self.X_cand,
                          return_utilities=[])
        self.assertRaises(TypeError, pal.query, X_cand=self.X_cand,
                          return_utilities=0)

    def test_query(self):
        mcpal = McPAL(clf=self.clf)
        self.assertRaises(ValueError, mcpal.query, X_cand=[], X=[], y=[])
        self.assertRaises(ValueError, mcpal.query, X_cand=[], X=self.X,
                          y=self.y)
        self.assertRaises(ValueError, mcpal.query, X_cand=self.X_cand,
                          X=self.X, y=[0, 1, 4, 0, 2, 1])

        # Test missing labels
        X_cand = [[0], [1], [2], [3]]
        mcpal = McPAL(clf=PWC(classes=[0, 1]))
        _, utilities = mcpal.query(X_cand, [[1]], [MISSING_LABEL],
                                   return_utilities=True)
        self.assertEqual(utilities.shape, (1, len(X_cand)))
        self.assertEqual(len(np.unique(utilities)), 1)

        _, utilities = mcpal.query(X_cand, X=[[0], [1], [2]],
                                   y=[0, 1, MISSING_LABEL],
                                   return_utilities=True)
        self.assertGreater(utilities[0, 2], utilities[0, 1])
        self.assertGreater(utilities[0, 2], utilities[0, 0])

        # Test scenario
        X_cand = [[0], [1], [2], [5]]
        mcpal = McPAL(clf=PWC(classes=[0, 1]))

        best_indices = mcpal.query(X_cand, X=[[1]], y=[0])
        np.testing.assert_array_equal(best_indices, np.array([3]))

        _, utilities = mcpal.query(X_cand, X=[[1]], y=[0],
                                   return_utilities=True)
        min_utilities = np.argmin(utilities)
        np.testing.assert_array_equal(min_utilities, np.array([1]))

        best_indices = mcpal.query(X_cand=[[0], [1], [2]], X=[[0], [2]],
                                   y=[0, 1])
        np.testing.assert_array_equal(best_indices, [1])


class TestXPAL(unittest.TestCase):

    def setUp(self):
        self.random_state = 1
        self.X_cand = np.array([[8, 1], [9, 1], [5, 1]])
        self.X_eval = np.array([[5, 2], [3, 7]])
        self.X = np.array([[1, 2], [5, 8], [8, 4], [5, 4], [5, 8], [9, 8]])
        self.y = np.array([0, 0, 1, 1, 0, 0])
        self.classes = np.array([0, 1])
        self.clf = PWC(classes=self.classes)
        self.args = dict(X_cand=self.X_cand, X=self.X, y=self.y)

    def test_init_param_clf(self):
        selector = XPAL(clf=None)
        self.assertRaises(TypeError, selector.query, **self.args)

        selector = XPAL(clf='string')
        self.assertRaises(TypeError, selector.query, **self.args)

        self.assertTrue(hasattr(selector, 'clf'))

    def test_init_param_scoring(self):
        selector = XPAL(clf=self.clf, scoring=None)
        self.assertRaises(ValueError, selector.query, **self.args)

        selector = XPAL(clf=self.clf, scoring=2)
        self.assertRaises(ValueError, selector.query, **self.args)

        selector = XPAL(clf=self.clf, scoring='String')
        self.assertRaises(ValueError, selector.query, **self.args)

        self.assertTrue(hasattr(selector, 'scoring'))

    def test_init_param_cost_vector(self):
        selector = XPAL(clf=self.clf, scoring='cost-vector',
                        cost_vector='string')
        self.assertRaises(ValueError, selector.query, **self.args)

        selector = XPAL(clf=self.clf, scoring='cost-vector',
                        cost_vector=[3, 2, 1])  # clf.n_classes=2
        self.assertRaises(ValueError, selector.query, **self.args)

        selector = XPAL(clf=self.clf, scoring='cost-vector',
                        cost_vector=None)  # clf.n_classes=2
        self.assertRaises(ValueError, selector.query, **self.args)

        self.assertTrue(hasattr(selector, 'cost_vector'))

    def test_init_param_cost_matrix(self):
        selector = XPAL(clf=self.clf, scoring='misclassification-loss',
                        cost_matrix=None)
        self.assertRaises(ValueError, selector.query, **self.args)

        selector = XPAL(clf=self.clf, cost_matrix=np.ones((2, 3)))
        self.assertRaises(ValueError, selector.query, **self.args)

        selector = XPAL(clf=self.clf, cost_matrix='string')
        self.assertRaises(AttributeError, selector.query, **self.args)

        selector = XPAL(clf=self.clf, cost_matrix=np.ones((3, 3)))
        self.assertRaises(ValueError, selector.query, **self.args)

        # TODO cost_vector and cost_matrix together?
        selector = XPAL(clf=self.clf, cost_matrix=np.ones((2, 2)),
                        cost_vector=[1, 1])
        self.assertRaises(ValueError, selector.query, **self.args)

        self.assertTrue(hasattr(selector, 'cost_matrix'))

    def test_init_param_custom_perf_func(self):
        # TODO custom not implemented
        # selector = XPAL(clf=self.clf, scoring='custom',
        #                custom_perf_func='string')
        # self.assertRaises(TypeError, selector.query, **self.args)

        # selector = XPAL(clf=self.clf, scoring='custom',
        #                 custom_perf_func=42)
        # self.assertRaises(TypeError, selector.query, **self.args)
        #
        # selector = XPAL(clf=self.clf, scoring='custom')
        # self.assertRaises(TypeError, selector.query, **self.args)
        #
        # self.assertTrue(hasattr(selector, 'custom_perf_func'))
        pass

    def test_init_param_prior_cand(self):
        selector = XPAL(clf=self.clf, prior_cand='string')
        self.assertRaises(TypeError, selector.query, **self.args)

        selector = XPAL(clf=self.clf, prior_cand=None)
        self.assertRaises((TypeError, ValueError), selector.query, **self.args)

        selector = XPAL(clf=self.clf, prior_cand=[[1, 2], [1, 2]])
        self.assertRaises((TypeError, ValueError), selector.query, **self.args)

        selector = XPAL(clf=self.clf, prior_cand=[1, 2, 3])  # clf.n_classes=2
        self.assertRaises((TypeError, ValueError), selector.query, **self.args)

        self.assertTrue(hasattr(selector, 'prior_cand'))

    def test_init_param_prior_eval(self):
        selector = XPAL(clf=self.clf, prior_eval='string')
        self.assertRaises(TypeError, selector.query, **self.args)

        selector = XPAL(clf=self.clf, prior_eval=None)
        self.assertRaises((TypeError, ValueError), selector.query, **self.args)

        selector = XPAL(clf=self.clf, prior_eval=[[1, 2], [1, 2]])
        self.assertRaises((TypeError, ValueError), selector.query, **self.args)

        selector = XPAL(clf=self.clf, prior_eval=[1, 2, 3])  # clf.n_classes=2
        self.assertRaises((TypeError, ValueError), selector.query, **self.args)

        self.assertTrue(hasattr(selector, 'prior_eval'))

    def test_init_param_estimator_metric(self):
        selector = XPAL(clf=self.clf, estimator_metric=False)
        self.assertRaises(ValueError, selector.query, **self.args)

        selector = XPAL(clf=self.clf, estimator_metric='String')
        self.assertRaises(ValueError, selector.query, **self.args)

        selector = XPAL(clf=self.clf, estimator_metric=None)
        self.assertRaises(ValueError, selector.query, **self.args)

        selector = XPAL(clf=self.clf, estimator_metric='precomputed')
        self.assertRaises(ValueError, selector.query, **self.args)

        self.assertTrue(hasattr(selector, 'estimator_metric'))

    def test_init_param_estimator_metric_dict(self):
        selector = XPAL(clf=self.clf, estimator_metric_dict='String')
        self.assertRaises(TypeError, selector.query, **self.args)

        selector = XPAL(clf=self.clf, estimator_metric_dict=2)
        self.assertRaises(TypeError, selector.query, **self.args)

        selector = XPAL(clf=self.clf, estimator_metric_dict=True)
        self.assertRaises(TypeError, selector.query, **self.args)

        def func(X1, X2):
            return 0

        selector = XPAL(clf=self.clf, estimator_metric=func,
                        estimator_metric_dict={'arg': 'arg'})
        self.assertRaises(TypeError, selector.query, **self.args)

        self.assertTrue(hasattr(selector, 'estimator_metric_dict'))

    def test_init_param_batch_mode(self):
        selector = XPAL(clf=self.clf, batch_mode='string')
        self.assertRaises(ValueError, selector.query, **self.args)

        selector = XPAL(clf=self.clf, batch_mode=None)
        self.assertRaises(ValueError, selector.query, **self.args)

        selector = XPAL(clf=self.clf, batch_mode=False)
        self.assertRaises(ValueError, selector.query, **self.args)

        self.assertTrue(hasattr(selector, 'batch_mode'))

    def test_init_param_batch_labels_equal(self):
        selector = XPAL(clf=self.clf, batch_labels_equal="string")
        self.assertRaises(TypeError, selector.query, **self.args)

        selector = XPAL(clf=self.clf, batch_labels_equal=None)
        self.assertRaises(TypeError, selector.query, **self.args)

        selector = XPAL(clf=self.clf, batch_labels_equal=[])
        self.assertRaises(TypeError, selector.query, **self.args)

        selector = XPAL(clf=self.clf, batch_labels_equal=2)
        self.assertRaises(TypeError, selector.query, **self.args)

        self.assertTrue(hasattr(selector, 'nonmyopic_labels_equal'))

    def test_init_param_nonmyopic_max_cand(self):
        selector = XPAL(clf=self.clf, nonmyopic_max_cand=2,
                        batch_mode='full')
        self.assertRaises(NotImplementedError, selector.query, **self.args)

        selector = XPAL(clf=self.clf, nonmyopic_max_cand=0,
                        batch_mode='greedy')
        self.assertRaises(ValueError, selector.query, **self.args)

        selector = XPAL(clf=self.clf, nonmyopic_max_cand=1.5,
                        batch_mode='greedy')
        self.assertRaises(TypeError, selector.query, **self.args)

        selector = XPAL(clf=self.clf, nonmyopic_max_cand=-5,
                        batch_mode='greedy')
        self.assertRaises(ValueError, selector.query, **self.args)

        selector = XPAL(clf=self.clf, nonmyopic_max_cand='string',
                        batch_mode='greedy')
        self.assertRaises(TypeError, selector.query, **self.args)

        selector = XPAL(clf=self.clf, nonmyopic_max_cand=None,
                        batch_mode='greedy')
        self.assertRaises(TypeError, selector.query, **self.args)

        self.assertTrue(hasattr(selector, 'nonmyopic_max_cand'))

    def test_init_param_nonmyopic_neighbors(self):
        # nonmyopic_max_cand = 2 (nonmyopic method)
        selector = XPAL(clf=self.clf, nonmyopic_max_cand=2,
                        nonmyopic_neighbors="string", batch_mode='greedy')
        self.assertRaises(ValueError, selector.query, **self.args)

        selector = XPAL(clf=self.clf, nonmyopic_max_cand=2,
                        nonmyopic_neighbors=None, batch_mode='greedy')
        self.assertRaises(ValueError, selector.query, **self.args)

        self.assertTrue(hasattr(selector, 'nonmyopic_labels_equal'))

    def test_init_param_nonmyopic_labels_equal(self):
        # nonmyopic_max_cand = 2 (nonmyopic method)
        selector = XPAL(clf=self.clf, nonmyopic_max_cand=2,
                        nonmyopic_labels_equal=None, batch_mode='greedy')
        self.assertRaises(TypeError, selector.query, **self.args)

        self.assertTrue(hasattr(selector, 'nonmyopic_labels_equal'))

        selector = XPAL(clf=self.clf, nonmyopic_max_cand=2,
                        nonmyopic_labels_equal=[], batch_mode='greedy')
        self.assertRaises(TypeError, selector.query, **self.args)

        selector = XPAL(clf=self.clf, nonmyopic_max_cand=2,
                        nonmyopic_labels_equal=0, batch_mode='greedy')
        self.assertRaises(TypeError, selector.query, **self.args)

        selector = XPAL(clf=self.clf, nonmyopic_max_cand=2,
                        nonmyopic_labels_equal='string', batch_mode='greedy')
        self.assertRaises(TypeError, selector.query, **self.args)

    def test_init_param_nonmyopic_independent_probs(self):
        # nonmyopic_max_cand = 2 (nonmyopic method)
        selector = XPAL(clf=self.clf, nonmyopic_max_cand=2,
                        nonmyopic_independent_probs=0, batch_mode='greedy')
        self.assertRaises(TypeError, selector.query, **self.args)

        self.assertTrue(hasattr(selector, 'nonmyopic_independent_probs'))

        selector = XPAL(clf=self.clf, nonmyopic_max_cand=2,
                        nonmyopic_independent_probs=None, batch_mode='greedy')
        self.assertRaises(TypeError, selector.query, **self.args)

        selector = XPAL(clf=self.clf, nonmyopic_max_cand=2,
                        nonmyopic_independent_probs=[], batch_mode='greedy')
        self.assertRaises(TypeError, selector.query, **self.args)

        selector = XPAL(clf=self.clf, nonmyopic_max_cand=2,
                        nonmyopic_independent_probs='string',
                        batch_mode='greedy')
        self.assertRaises(TypeError, selector.query, **self.args)

    def test_init_param_random_state(self):
        selector = XPAL(clf=self.clf, random_state='string')
        self.assertRaises(ValueError, selector.query, **self.args)

        self.assertTrue(hasattr(selector, 'random_state'))

    def test_query_param_X_cand(self):
        selector = XPAL(clf=self.clf)
        self.assertRaises(ValueError, selector.query, X_cand=[], X=self.X,
                          y=self.y)
        self.assertRaises(ValueError, selector.query,
                          X_cand=np.ones((3, self.X.shape[1] + 1)), X=self.X,
                          y=self.y)
        self.assertRaises(ValueError, selector.query, X_cand=None, X=self.X,
                          y=self.y)
        self.assertRaises(ValueError, selector.query, X_cand=np.nan, X=self.X,
                          y=self.y)
        self.assertRaises(ValueError, selector.query, X_cand=5, X=self.X,
                          y=self.y)

    def test_query_param_X(self):
        selector = XPAL(clf=self.clf)
        self.assertRaises(ValueError, selector.query, X_cand=self.X_cand,
                          X=None, y=self.y)
        self.assertRaises(ValueError, selector.query, X_cand=self.X_cand,
                          X='string', y=self.y)
        self.assertRaises(ValueError, selector.query, X_cand=self.X_cand,
                          X=[], y=self.y)
        self.assertRaises(ValueError, selector.query, X_cand=self.X_cand,
                          X=self.X[0:-1], y=self.y)

    def test_query_param_y(self):
        selector = XPAL(clf=self.clf)
        self.assertRaises(ValueError, selector.query, X_cand=self.X_cand,
                          X=self.X, y=None)
        self.assertRaises(ValueError, selector.query, X_cand=self.X_cand,
                          X=self.X, y='string')
        self.assertRaises(ValueError, selector.query, X_cand=self.X_cand,
                          X=self.X, y=[])
        self.assertRaises(ValueError, selector.query, X_cand=self.X_cand,
                          X=self.X, y=self.y[0:-1])

    def test_query_param_X_eval(self):
        selector = XPAL(clf=self.clf)
        self.assertRaises(ValueError, selector.query, **self.args,
                          X_eval=[])
        self.assertRaises(ValueError, selector.query, **self.args,
                          X_eval=np.ones((3, self.X.shape[1] + 1)))
        self.assertRaises(TypeError, selector.query, **self.args,
                          X_eval=np.nan)
        self.assertRaises(TypeError, selector.query, **self.args,
                          X_eval=5)
        self.assertRaises(ValueError, selector.query, **self.args,
                          X_eval='string')

    def test_query_param_batch_size(self):
        selector = XPAL(clf=self.clf, batch_mode='greedy')
        self.assertRaises(TypeError, selector.query, **self.args,
                          batch_size=1.2)
        self.assertRaises(TypeError, selector.query, **self.args,
                          batch_size='string')
        self.assertRaises(ValueError, selector.query, **self.args,
                          batch_size=0)
        self.assertRaises(ValueError, selector.query, **self.args,
                          batch_size=-10)

        selector = XPAL(clf=self.clf, batch_mode='full')
        self.assertRaises(TypeError, selector.query, **self.args,
                          batch_size=1.2)
        self.assertRaises(TypeError, selector.query, **self.args,
                          batch_size='string')
        self.assertRaises(ValueError, selector.query, **self.args,
                          batch_size=0)
        self.assertRaises(ValueError, selector.query, **self.args,
                          batch_size=-10)

    def test_query_param_sample_weight_cand(self):
        selector = XPAL(clf=self.clf)
        self.assertRaises((TypeError, ValueError), selector.query, **self.args,
                          sample_weight_cand='string',
                          sample_weight=np.ones(len(self.X)))
        self.assertRaises(ValueError, selector.query, **self.args,
                          sample_weight_cand=self.X_cand,
                          sample_weight=np.ones(len(self.X)))
        self.assertRaises(ValueError, selector.query, **self.args,
                          sample_weight_cand=np.ones((len(self.X_cand) - 1)),
                          sample_weight=np.ones(len(self.X)))
        self.assertRaises(ValueError, selector.query, **self.args,
                          sample_weight_cand=np.ones((len(self.X_cand) + 1)),
                          sample_weight=np.ones(len(self.X)))
        self.assertRaises(ValueError, selector.query, **self.args,
                          sample_weight_cand=np.ones((len(self.X_cand) + 1)),
                          sample_weight=None)

    def test_query_param_sample_weight(self):
        selector = XPAL(clf=self.clf)
        self.assertRaises(ValueError, selector.query, **self.args,
                          sample_weight='string',
                          sample_weight_cand=np.ones(len(self.X_cand)))
        self.assertRaises(ValueError, selector.query, **self.args,
                          sample_weight=self.X,
                          sample_weight_cand=np.ones(len(self.X_cand)))
        self.assertRaises(ValueError, selector.query, **self.args,
                          sample_weight=np.ones((len(self.X) - 1)),
                          sample_weight_cand=np.ones(len(self.X_cand)))
        self.assertRaises(ValueError, selector.query, **self.args,
                          sample_weight=np.ones((len(self.X) + 1)),
                          sample_weight_cand=np.ones(len(self.X_cand)))
        self.assertRaises(ValueError, selector.query, **self.args,
                          sample_weight=np.ones((len(self.X) + 1)),
                          sample_weight_cand=None)

    def test_query_param_sample_weight_eval(self):
        selector = XPAL(clf=self.clf)
        self.assertRaises(ValueError, selector.query, **self.args,
                          X_eval=self.X_eval,
                          sample_weight_eval='string',
                          sample_weight=np.ones(len(self.X)),
                          sample_weight_cand=np.ones(len(self.X)))
        self.assertRaises(ValueError, selector.query, **self.args,
                          X_eval=self.X_eval,
                          sample_weight_eval=self.X,
                          sample_weight=np.ones(len(self.X)),
                          sample_weight_cand=np.ones(len(self.X_cand)))
        self.assertRaises(ValueError, selector.query, **self.args,
                          X_eval=self.X_eval,
                          sample_weight_eval=np.empty((len(self.X_eval) - 1)),
                          sample_weight=np.ones(len(self.X)),
                          sample_weight_cand=np.ones(len(self.X_cand)))
        self.assertRaises(ValueError, selector.query, **self.args,
                          X_eval=self.X_eval,
                          sample_weight_eval=np.empty((len(self.X_eval) + 1)),
                          sample_weight=np.ones(len(self.X)),
                          sample_weight_cand=np.ones(len(self.X_cand)))
        # TODO warning/error if sample_weight_eval is given but sample_weight not?
        # self.assertRaises(ValueError, selector.query, **self.args,
        #                  X_eval=self.X_eval,
        #                  sample_weight_eval=np.ones(len(self.X_eval)),
        #                  sample_weight=None,
        #                  sample_weight_cand=None)

    def test_query_param_return_utilities(self):
        selector = XPAL(clf=self.clf)
        self.assertRaises(TypeError, selector.query, **self.args,
                          return_utilities=None)
        self.assertRaises(TypeError, selector.query, **self.args,
                          return_utilities=[])
        self.assertRaises(TypeError, selector.query, **self.args,
                          return_utilities=0)

    def test_general_query(self):
        random_state = np.random.RandomState(1)
        X = random_state.rand(5, 2)
        y_oracle = random_state.randint(2, 4, [5])
        y = np.full(y_oracle.shape, MISSING_LABEL)
        y[2:3] = y_oracle[2:3]

        X_cand = X
        sample_weight_cand = None
        sample_weight = None
        sample_weight_eval = None

        clf = PWC(classes=np.unique(y_oracle))
        prior_cand = 1.e-3
        prior_eval = 1.e-3
        estimator_metric = 'rbf'
        estimator_metric_dict = None

        scorings = ['error', 'macro-accuracy']
        X_evals = [None, X_cand]
        batch_labels_equals = [True, False]
        batch_sizes = [1, 2]
        batch_modes = ['full', 'greedy']
        nonmyopic_max_cands = [1, 2]  # full only 1,
        nonmyopic_independent_probss = [True, False]
        nonmyopic_neighborss = ['same', 'nearest']
        nonmyopic_labels_equals = [True, False]

        params = list(
            product(scorings, X_evals, batch_labels_equals, batch_sizes,
                    batch_modes, [1],
                    [False], ['same'], [True]))
        params += list(
            product(scorings, X_evals, batch_labels_equals, batch_sizes,
                    ['greedy'], [2],
                    nonmyopic_independent_probss, nonmyopic_neighborss,
                    nonmyopic_labels_equals))

        for param in params:
            (scoring, X_eval, batch_labels_equal, batch_size, batch_mode,
             nonmyopic_max_cand, nonmyopic_independent_probs,
             nonmyopic_neighbors, nonmyopic_labels_equal) = param
            with self.subTest(msg="xPAL", params=param):
                selector = XPAL(clf, scoring=scoring, cost_vector=None,
                                cost_matrix=None,
                                custom_perf_func=None,
                                prior_cand=prior_cand, prior_eval=prior_eval,
                                estimator_metric=estimator_metric,
                                estimator_metric_dict=estimator_metric_dict,
                                batch_mode=batch_mode,
                                batch_labels_equal=batch_labels_equal,
                                nonmyopic_max_cand=nonmyopic_max_cand,
                                nonmyopic_neighbors=nonmyopic_neighbors,
                                nonmyopic_labels_equal=nonmyopic_labels_equal,
                                nonmyopic_independent_probs=
                                nonmyopic_independent_probs)

                selector.query(X_cand, X, y, X_eval=X_eval,
                               batch_size=batch_size,
                               sample_weight_cand=sample_weight_cand,
                               sample_weight=sample_weight,
                               sample_weight_eval=sample_weight_eval,
                               return_utilities=True)

            # break

    def test_reduce_candlist_set(self):
        from skactiveml.pool._probal import _reduce_candlist_set
        candidate_sets = [(0,), (1,)]
        reduced_candidate_sets, _ = \
            _reduce_candlist_set(candidate_sets, reduce=True)
        np.testing.assert_equal(type(reduced_candidate_sets), list)
        np.testing.assert_array_equal(candidate_sets, reduced_candidate_sets)

        candidate_sets = [(0,), (1,), (1,)]
        reduced_candidate_sets, _ = \
            _reduce_candlist_set(candidate_sets, reduce=True)
        np.testing.assert_equal(type(reduced_candidate_sets), list)
        np.testing.assert_array_equal(reduced_candidate_sets, [[0], [1]])

        candidate_sets = [(0, 1), (1, 1), (1, 0), (1, 2), (0, 1)]
        reduced_candidate_sets, mapping = \
            _reduce_candlist_set(candidate_sets, reduce=True)
        np.testing.assert_array_equal(reduced_candidate_sets,
                                      [(0, 1), (1, 1), (1, 2)])
        np.testing.assert_array_equal(mapping, [0, 1, 0, 2, 0])

        permutations = list(itertools.permutations(np.arange(10), 3))
        combinations = list(itertools.combinations(np.arange(10), 3))
        reduced_permutations, _ = \
            _reduce_candlist_set(permutations, reduce=True)
        np.testing.assert_equal(len(reduced_permutations), len(combinations))

    def test_calc_sim(self):
        K = lambda X1, X2: pairwise_kernels(X1, X2)
        X = np.array([[1, 2], [5, 8], [8, 4], [5, 4]])
        Y = np.array([[1, 2], [5, 3], [4, 3], [2, 9], [7, 4]])
        idx_X = [0, 2, 3]
        idx_Y = [1, 3]
        kernel = K(X, Y)

        similarity = _calc_sim(K, X, Y)
        self.assertEqual(similarity.shape, (X.shape[0], Y.shape[0]))
        np.testing.assert_array_almost_equal(similarity, kernel)

        similarity = _calc_sim(K, X, Y, idx_X, idx_Y)

        self.assertEqual(similarity.shape, (X.shape[0], Y.shape[0]))
        self.assertEqual(np.count_nonzero(~np.isnan(similarity)),
                         len(idx_X) * len(idx_Y))

        for i in range(len(X)):
            for j in range(len(Y)):
                if i in idx_X and j in idx_Y:
                    np.testing.assert_equal(kernel[i, j], similarity[i, j])
                else:
                    np.testing.assert_equal(np.nan, similarity[i, j])

        # TODO test with 0 instances

    def test_get_nonmyopic_cand_set(self):
        cand_idx = np.arange(5)
        M = 2
        nonmyopic_candidate_sets = _get_nonmyopic_cand_set('same', cand_idx, M)
        correct_array = [[x] for x in cand_idx] + [[x, x] for x in cand_idx]

        x = np.array(sorted(nonmyopic_candidate_sets), dtype=object)
        y = np.array(sorted(correct_array), dtype=object)
        np.testing.assert_array_equal(x, y)

        self.assertRaises(ValueError, _get_nonmyopic_cand_set, 'wrong',
                          cand_idx=[0, 1], similarity=np.eye(2), M=2)

        self.assertRaises(ValueError, _get_nonmyopic_cand_set, 'nearest',
                          cand_idx=[0, 1], M=2)

        neighbors = 'nearest'
        cand_idx = np.arange(5)
        similarity = np.random.random((5, 5))
        similarity = (similarity + similarity.T) / 2
        np.fill_diagonal(similarity, 1)
        M = 3
        nonmyopic_candidate_sets = _get_nonmyopic_cand_set(neighbors, cand_idx,
                                                           M, similarity)
        for i, x in enumerate(cand_idx):
            for m in range(1, M + 1):
                similarity_x = np.argsort(-similarity[i])
                self.assertIn(list(similarity_x[:m]), nonmyopic_candidate_sets)

    def test_dependent_cand_prob(self):
        from skactiveml.pool._probal import _dependent_cand_prob

        X = [[0, 1], [3, 2], [1, 2], [5, 2], [3, 1], [4, 7], [1, 9]]
        sample_weight = [1, 1, 1, 1, 1, 1, 1]
        sim_cand = pairwise_kernels(X, X, metric='rbf')
        prob_est = PWC(metric="precomputed", classes=[0, 1],
                       missing_label=np.nan, class_prior=1,
                       random_state=14)

        # Standard test
        y = [np.nan, 1, np.nan, np.nan, np.nan, 1, 1]
        cand_idx = [3]
        idx_train = [1, 5, 6]
        idx_preselected = [2]
        label_simulations = [([0], [0]), ([0], [1]), ([1], [0]), ([1], [1])]
        prob_preselected = [0.5, 0.5, 0.5, 0.5]
        P = _dependent_cand_prob(cand_idx, idx_train, idx_preselected, X, y,
                                 sample_weight, label_simulations,
                                 prob_preselected, prob_est, sim_cand)
        np.testing.assert_equal(np.argmax(P), 3)

        # No training instances
        y = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
        cand_idx = [0, 1]
        idx_train = []
        idx_preselected = []
        label_simulations = [([], [0, 0]), ([], [1, 1])]
        prob_preselected = [1, 1]
        P = _dependent_cand_prob(cand_idx, idx_train, idx_preselected, X, y,
                                 sample_weight, label_simulations,
                                 prob_preselected, prob_est, sim_cand)
        np.testing.assert_array_equal(P, [0.5, 0.5])

    def test_get_y_sim_list(self):
        from skactiveml.pool._probal import _get_y_sim_list
        label_combinations = _get_y_sim_list([0, 1], 3, labels_equal=False)
        correct = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
                   [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]
        np.testing.assert_array_equal(label_combinations, correct)

        label_combinations = _get_y_sim_list([0, 1, 2], 2, labels_equal=False)
        correct = [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0],
                   [2, 1], [2, 2]]
        np.testing.assert_array_equal(label_combinations, correct)

        label_combinations = _get_y_sim_list([0, 1, 2], 3, labels_equal=True)
        correct = [[0, 0, 0], [1, 1, 1], [2, 2, 2]]
        np.testing.assert_array_equal(label_combinations, correct)

    # def test_params_reduce_candlist_set(self):
    #     # tests for parameter 'candidate_sets':
    #     self.assertRaises(TypeError, _reduce_candlist_set,
    #                       candidate_sets='string', reduce=True)
    #
    #     self.assertRaises(ValueError, _reduce_candlist_set,
    #                       candidate_sets=[(0, 1), (0, 1, 2), (1, 2)],
    #                       reduce=True)
    #
    #     self.assertRaises(ValueError, _reduce_candlist_set,
    #                       candidate_sets=None, reduce=True)
    #
    #     # tests for parameter 'reduce':
    #     candidate_sets = [(0, 1), (0, 2), (1, 2)]
    #     self.assertRaises(ValueError, _reduce_candlist_set,
    #                       candidate_sets=candidate_sets, reduce=None)
    #
    #     self.assertRaises(ValueError, _reduce_candlist_set,
    #                       candidate_sets=candidate_sets, reduce='String')
    #
    #     self.assertRaises(ValueError, _reduce_candlist_set,
    #                       candidate_sets=candidate_sets, reduce=[])
    #
    #     self.assertRaises(ValueError, _reduce_candlist_set,
    #                       candidate_sets=candidate_sets, reduce=0)

    # def test_params_calc_sim(self):
    #     K = lambda X1, X2: pairwise_kernels(X1, X2)
    #     X = [[1, 2], [5, 8], [8, 4], [5, 4]]
    #     Y = [[1, 2], [5, 3], [4, 3], [2, 9], [7, 4]]
    #     idx_X = [0, 2, 3]
    #     idx_Y = [1, 3]
    #
    #     # tests for parameter 'K':
    #     self.assertRaises(ValueError, _calc_sim,
    #                       K=None, X=X, Y=Y, idx_X=idx_X, idx_Y=idx_Y)
    #
    #     self.assertRaises(TypeError, _calc_sim,
    #                       K='String', X=X, Y=Y, idx_X=idx_X, idx_Y=idx_Y)
    #
    #     self.assertRaises(ValueError, _calc_sim,
    #                       K=lambda x: x + x, X=X, Y=Y, idx_X=idx_X,
    #                       idx_Y=idx_Y)
    #
    #     # tests for parameter 'X':
    #     self.assertRaises(ValueError, _calc_sim,
    #                       K=K, X=None, Y=Y, idx_X=idx_X, idx_Y=idx_Y)
    #
    #     self.assertRaises(ValueError, _calc_sim,
    #                       K=K, X='String', Y=Y, idx_X=idx_X, idx_Y=idx_Y)
    #
    #     self.assertRaises(ValueError, _calc_sim,
    #                       K=K, X=[3, 4], Y=Y, idx_X=idx_X, idx_Y=idx_Y)
    #
    #     self.assertRaises(ValueError, _calc_sim,
    #                       K=K, X=[[[3, 4]]], Y=Y, idx_X=idx_X, idx_Y=idx_Y)
    #
    #     self.assertRaises(ValueError, _calc_sim,
    #                       K=K, X=[[1, 2, 3]], Y=Y, idx_X=idx_X, idx_Y=idx_Y)
    #
    #     # tests for parameter 'Y':
    #     self.assertRaises(ValueError, _calc_sim,
    #                       K=K, X=X, Y=None, idx_X=idx_X, idx_Y=idx_Y)
    #
    #     self.assertRaises(ValueError, _calc_sim,
    #                       K=K, X=X, Y='String', idx_X=idx_X, idx_Y=idx_Y)
    #
    #     self.assertRaises(ValueError, _calc_sim,
    #                       K=K, X=X, Y=[3, 4], idx_X=idx_X, idx_Y=idx_Y)
    #
    #     self.assertRaises(ValueError, _calc_sim,
    #                       K=K, X=X, Y=[[[3, 4]]], idx_X=idx_X, idx_Y=idx_Y)
    #
    #     self.assertRaises(ValueError, _calc_sim,
    #                       K=K, X=X, Y=[[1, 2, 3]], idx_X=idx_X, idx_Y=idx_Y)
    #
    #     # tests for parameter 'idx_X':
    #     self.assertRaises(ValueError, _calc_sim,
    #                       K=K, X=X, Y=Y, idx_X=np.nan, idx_Y=idx_Y)
    #
    #     self.assertRaises(ValueError, _calc_sim,
    #                       K=K, X=X, Y=Y, idx_X='String', idx_Y=idx_Y)
    #
    #     self.assertRaises(ValueError, _calc_sim,
    #                       K=K, X=X, Y=Y, idx_X=True, idx_Y=idx_Y)
    #
    #     # tests for parameter 'idx_X':
    #     self.assertRaises(ValueError, _calc_sim,
    #                       K=K, X=X, Y=Y, idx_X=idx_X, idx_Y=np.nan)
    #
    #     self.assertRaises(ValueError, _calc_sim,
    #                       K=K, X=X, Y=Y, idx_X=idx_X, idx_Y='String')
    #
    #     self.assertRaises(ValueError, _calc_sim,
    #                       K=K, X=X, Y=Y, idx_X=idx_X, idx_Y=True)

    # def test_params_get_nonmyopic_cand_set(self):
    #     # tests for parameter 'neighbors':
    #     self.assertRaises(ValueError, _get_nonmyopic_cand_set,
    #                       neighbors='string', cand_idx=np.arange(5), M=2,
    #                       similarity=None)
    #
    #     self.assertRaises(ValueError, _get_nonmyopic_cand_set,
    #                       neighbors=None, cand_idx=np.arange(5), M=2,
    #                       similarity=None)
    #
    #     self.assertRaises(ValueError, _get_nonmyopic_cand_set,
    #                       neighbors=['string'], cand_idx=np.arange(5), M=2,
    #                       similarity=None)
    #
    #     self.assertRaises(ValueError, _get_nonmyopic_cand_set,
    #                       neighbors=0, cand_idx=np.arange(5), M=2,
    #                       similarity=None)
    #
    #     # tests for parameter 'cand_idx':
    #     self.assertRaises(ValueError, _get_nonmyopic_cand_set,
    #                       neighbors='same', cand_idx=[[5], [5]], M=2,
    #                       similarity=None)
    #
    #     self.assertRaises(TypeError, _get_nonmyopic_cand_set,
    #                       neighbors='same', cand_idx='String', M=2,
    #                       similarity=None)
    #
    #     self.assertRaises(ValueError, _get_nonmyopic_cand_set,
    #                       neighbors='same', cand_idx=None, M=2,
    #                       similarity=None)
    #
    #     self.assertRaises(TypeError, _get_nonmyopic_cand_set,
    #                       neighbors='same', cand_idx=True, M=2,
    #                       similarity=None)
    #
    #     # tests for parameter 'M':
    #     self.assertRaises(TypeError, _get_nonmyopic_cand_set,
    #                       neighbors='same', cand_idx=np.arange(5), M=True,
    #                       similarity=None)
    #
    #     self.assertRaises(ValueError, _get_nonmyopic_cand_set,
    #                       neighbors='same', cand_idx=np.arange(5), M=None,
    #                       similarity=None)
    #
    #     self.assertRaises(ValueError, _get_nonmyopic_cand_set,
    #                       neighbors='same', cand_idx=np.arange(5), M=-1,
    #                       similarity=None)
    #
    #     self.assertRaises(TypeError, _get_nonmyopic_cand_set,
    #                       neighbors='same', cand_idx=np.arange(5), M='String',
    #                       similarity=None)
    #
    #     # tests for parameter 'similarity':
    #     self.assertRaises(ValueError, _get_nonmyopic_cand_set,
    #                       neighbors='nearest', cand_idx=np.arange(5), M=2,
    #                       similarity=None)
    #
    #     self.assertRaises(ValueError, _get_nonmyopic_cand_set,
    #                       neighbors='nearest', cand_idx=np.arange(5), M=2,
    #                       similarity='String')
    #
    #     self.assertRaises(ValueError, _get_nonmyopic_cand_set,
    #                       neighbors='nearest', cand_idx=np.arange(5), M=2,
    #                       similarity=5)
    #
    #     self.assertRaises(ValueError, _get_nonmyopic_cand_set,
    #                       neighbors='nearest', cand_idx=np.arange(5), M=2,
    #                       similarity=np.empty((4, 4)))
    #
    #     self.assertRaises(ValueError, _get_nonmyopic_cand_set,
    #                       neighbors='nearest', cand_idx=np.arange(5), M=2,
    #                       similarity=np.empty((5, 5, 5)))

    def test_params_to_int_labels(self):
        # tests for parameter 'est':
        self.assertRaises(TypeError, to_int_labels, est=None,
                          X=self.X, y=self.y)

        self.assertRaises(TypeError, to_int_labels, est='String',
                          X=self.X, y=self.y)

        # tests for parameter 'X':
        self.assertRaises(ValueError, to_int_labels, est=self.clf,
                          X=self.X[0:-1], y=self.y)

        self.assertRaises(ValueError, to_int_labels, est=self.clf,
                          X=None, y=self.y)

        self.assertRaises(ValueError, to_int_labels, est=self.clf,
                          X=np.empty((len(self.X))), y=self.y)

        # tests for parameter 'y':
        self.assertRaises(ValueError, to_int_labels, est=self.clf,
                          X=self.X, y=self.y[0:-1])

        self.assertRaises(TypeError, to_int_labels, est=self.clf,
                          X=self.X, y=None)

        self.assertRaises(TypeError, to_int_labels, est=self.clf,
                          X=self.X, y='String')

    # def test_params_dependent_cand_prob(self):
    #     rs = np.random.RandomState(42)
    #     test_func = _dependent_cand_prob
    #
    #     cand_idx = [9]
    #     idx_train = [5, 6, 7, 8, 9]
    #     idx_preselected = [0]
    #     X = rs.rand(10, 6)
    #     y = rs.randint(0, 2, [10])
    #     sample_weight = np.ones(10)
    #     y_sim_list = [([0], [0]), ([0], [1]), ([1], [0]), ([1], [1])]
    #     prob_y_sim_pre = rs.rand(4)
    #     prob_est = PWC(classes=[0, 1])
    #     sim_cand = rs.rand(10, 10)
    #
    #     params = dict(
    #         cand_idx=cand_idx, idx_train=idx_train,
    #         idx_preselected=idx_preselected,
    #         X=X, y=y,
    #         sample_weight=sample_weight,
    #         y_sim_list=y_sim_list,
    #         prob_y_sim_pre=prob_y_sim_pre,
    #         prob_est=prob_est,
    #         sim_cand=sim_cand
    #     )
    #     test_params = dict(
    #         cand_idx=[([[5], [5]], IndexError), (None, TypeError),
    #                   ('String', TypeError), (True, TypeError)],
    #         idx_train=[([[3]], IndexError), ('String', TypeError),
    #                    (None, TypeError)],
    #         idx_preselected=[(None, TypeError), ('String', TypeError),
    #                          ([[1], [1]], IndexError)],
    #         X=[(X[0:-1], IndexError), (None, TypeError),
    #            ('String', TypeError)],
    #         y=[(y[0:-1], IndexError), (None, TypeError),
    #            ('String', TypeError)],
    #         sample_weight=[(sample_weight[0:-1], IndexError),
    #                        (None, TypeError),
    #                        ('String', TypeError)],
    #         y_sim_list=[(None, TypeError), ([([0])], ValueError),
    #                     ('Sting', ValueError)],
    #         prob_y_sim_pre=[(None, TypeError), ('String', TypeError),
    #                         (np.ones(3), IndexError)],
    #         prob_est=[(None, AttributeError), ('String', AttributeError)],
    #         sim_cand=[(None, TypeError), ('String', TypeError),
    #                   (sim_cand[0:-1], IndexError),
    #                   (sim_cand[0, :], IndexError),
    #                   (sim_cand[:, 0], IndexError)]
    #     )
    #
    #     test_callable(self, test_func, params, test_params)

    # def test_param_get_y_sim_list(self):
    #     # tests for parameter 'classes':
    #     self.assertRaises(TypeError, _get_y_sim_list, classes=None,
    #                       n_instances=11, labels_equal=True)
    #     self.assertRaises(ValueError, _get_y_sim_list, classes='String',
    #                       n_instances=11, labels_equal=True)
    #     self.assertRaises(TypeError, _get_y_sim_list, classes=False,
    #                       n_instances=11, labels_equal=True)
    #
    #     # tests for parameter 'n_instances':
    #     self.assertRaises(ValueError, _get_y_sim_list, classes=[2, 3, 5],
    #                       n_instances=-11, labels_equal=True)
    #     self.assertRaises(TypeError, _get_y_sim_list, classes=[2, 3, 5],
    #                       n_instances='String', labels_equal=True)
    #     self.assertRaises(ValueError, _get_y_sim_list, classes=[2, 3, 5],
    #                       n_instances=None, labels_equal=True)
    #     self.assertRaises(ValueError, _get_y_sim_list, classes=[2, 3, 5],
    #                       n_instances=np.inf, labels_equal=True)
    #     self.assertRaises(ValueError, _get_y_sim_list, classes=[2, 3, 5],
    #                       n_instances=np.nan, labels_equal=True)
    #     self.assertRaises(TypeError, _get_y_sim_list, classes=[2, 3, 5],
    #                       n_instances=[], labels_equal=True)
    #
    #     # tests for parameter 'labels_equal':
    #     self.assertRaises(TypeError, _get_y_sim_list, classes=[2, 3, 5],
    #                       n_instances=11, labels_equal='String')
    #     self.assertRaises(ValueError, _get_y_sim_list, classes=[2, 3, 5],
    #                       n_instances=11, labels_equal=None)
    #     self.assertRaises(TypeError, _get_y_sim_list, classes=[2, 3, 5],
    #                       n_instances=11, labels_equal=42)

    # def test_param_transform_scoring(self):
    #     # tests for parameter 'metric':
    #     self.assertRaises(TypeError, _transform_scoring, metric='String',
    #                       cost_matrix=[[0, 1], [1, 0]], cost_vector=[1, 1],
    #                       perf_func=None, n_classes=2)
    #     self.assertRaises(TypeError, _transform_scoring, metric=None,
    #                       cost_matrix=[[0, 1], [1, 0]], cost_vector=[1, 1],
    #                       perf_func=None, n_classes=2)
    #     self.assertRaises(TypeError, _transform_scoring, metric=5,
    #                       cost_matrix=[[0, 1], [1, 0]], cost_vector=[1, 1],
    #                       perf_func=None, n_classes=2)
    #
    #     # tests for parameter 'cost_matrix':
    #     self.assertRaises(TypeError, _transform_scoring,
    #                       metric='misclassification-loss',
    #                       cost_matrix=None, cost_vector=[1, 1],
    #                       perf_func=None, n_classes=2)
    #     self.assertRaises(TypeError, _transform_scoring,
    #                       metric='misclassification-loss',
    #                       cost_matrix='String', cost_vector=[1, 1],
    #                       perf_func=None, n_classes=2)
    #     self.assertRaises(ValueError, _transform_scoring,
    #                       metric='misclassification-loss',
    #                       cost_matrix=[[0, np.nan], [1, 0]],
    #                       cost_vector=[1, 1],
    #                       perf_func=None, n_classes=2)
    #
    #     # tests for parameter 'cost_vector':
    #     self.assertRaises(ValueError, _transform_scoring, metric='cost-vector',
    #                       cost_matrix=[[0, 1], [1, 0]],
    #                       cost_vector=[np.nan, 1],
    #                       perf_func=None, n_classes=2)
    #     self.assertRaises(TypeError, _transform_scoring, metric='cost-vector',
    #                       cost_matrix=[[0, 1], [1, 0]], cost_vector=None,
    #                       perf_func=None, n_classes=2)
    #     self.assertRaises(TypeError, _transform_scoring, metric='cost-vector',
    #                       cost_matrix=[[0, 1], [1, 0]], cost_vector='String',
    #                       perf_func=None, n_classes=2)
    #
    #     # tests for parameter 'perf_func':
    #     self.assertRaises(TypeError, _transform_scoring, metric='custom',
    #                       cost_matrix=[[0, 1], [1, 0]], cost_vector=[1, 1],
    #                       perf_func=None, n_classes=2)
    #     self.assertRaises(TypeError, _transform_scoring, metric='custom',
    #                       cost_matrix=[[0, 1], [1, 0]], cost_vector=[1, 1],
    #                       perf_func=5, n_classes=2)
    #     self.assertRaises(TypeError, _transform_scoring, metric='custom',
    #                       cost_matrix=[[0, 1], [1, 0]], cost_vector=[1, 1],
    #                       perf_func='String', n_classes=2)
    #
    #     # tests for parameter 'n_classes':
    #     self.assertRaises(ValueError, _transform_scoring,
    #                       metric='misclassification-loss',
    #                       cost_matrix=[[0, 1], [1, 0]], cost_vector=[1, 1],
    #                       perf_func=None, n_classes=3)
    #     self.assertRaises(ValueError, _transform_scoring,
    #                       metric='misclassification-loss',
    #                       cost_matrix=[[0, 1], [1, 0]], cost_vector=[1, 1],
    #                       perf_func=None, n_classes=-3)
    #     self.assertRaises(TypeError, _transform_scoring,
    #                       metric='misclassification-loss',
    #                       cost_matrix=[[0, 1], [1, 0]], cost_vector=[1, 1],
    #                       perf_func=None, n_classes=None)
    #     self.assertRaises(TypeError, _transform_scoring,
    #                       metric='misclassification-loss',
    #                       cost_matrix=[[0, 1], [1, 0]], cost_vector=[1, 1],
    #                       perf_func=None, n_classes='String')

    # def test_params_dperf(self):
    #     rs = np.random.RandomState(42)
    #     test_func = _dperf
    #
    #     params = dict(
    #         probs=rs.rand(10, 2),
    #         pred_old=np.zeros(10, dtype=int),
    #         pred_new=np.ones(10, dtype=int),
    #         sample_weight_eval=np.ones(10),
    #         decomposable=True,
    #         cost_matrix=np.array([[0, 1], [1, 0]]),
    #         perf_func=f1_score_func
    #     )
    #     test_params = dict(
    #         probs=[(np.ones((9, 2)), IndexError),
    #                (None, TypeError),
    #                ('String', TypeError)],
    #         pred_old=[(np.ones(9), IndexError),
    #                   (None, TypeError),
    #                   ('String', TypeError)],
    #         pred_new=[(np.ones(9), IndexError),
    #                   (None, TypeError),
    #                   ('String', TypeError)],
    #         sample_weight_eval=[(np.ones(9), IndexError),
    #                             (None, TypeError),
    #                             ('String', TypeError)],
    #         decomposable=[(5, TypeError),
    #                       (None, TypeError),
    #                       ('String', TypeError)],
    #         cost_matrix=[(None, AttributeError, {'decomposable': True}),
    #                      ('String', AttributeError, {'decomposable': True}),
    #                      (5, AttributeError, {'decomposable': True}),
    #                      (np.array([[1]]), IndexError, {'decomposable': True})],
    #         perf_func=[(None, TypeError, {'decomposable': False}),
    #                    (5, TypeError, {'decomposable': False}),
    #                    ('String', TypeError, {'decomposable': False})]
    #     )
    #     test_callable(self, test_func, params, test_params)

    def test_param_estimate_bandwidth(self):
        # tests for parameter 'n_samples':
        self.assertRaises(ValueError, estimate_bandwidth,
                          n_samples=-1, n_features=1)
        self.assertRaises(TypeError, estimate_bandwidth,
                          n_samples=1.5, n_features=1)
        self.assertRaises(TypeError, estimate_bandwidth,
                          n_samples=np.nan, n_features=1)
        self.assertRaises(TypeError, estimate_bandwidth,
                          n_samples=None, n_features=1)
        self.assertRaises(TypeError, estimate_bandwidth,
                          n_samples='String', n_features=1)

        # tests for parameter 'n_features':
        self.assertRaises(ValueError, estimate_bandwidth,
                          n_samples=1, n_features=-1)
        self.assertRaises(TypeError, estimate_bandwidth,
                          n_samples=1, n_features=1.5)
        self.assertRaises(TypeError, estimate_bandwidth,
                          n_samples=1, n_features=np.nan)
        self.assertRaises(TypeError, estimate_bandwidth,
                          n_samples=1, n_features=None)
        self.assertRaises(TypeError, estimate_bandwidth,
                          n_samples=1, n_features='String')

    def test_param_score_recall(self):
        # tests for parameter 'conf_matrix':
        self.assertRaises(TypeError, score_recall, conf_matrix=1)
        self.assertRaises(IndexError, score_recall, conf_matrix=np.array([1]))
        self.assertRaises(TypeError, score_recall, conf_matrix=np.nan)
        self.assertRaises(TypeError, score_recall, conf_matrix=None)
        self.assertRaises(TypeError, score_recall, conf_matrix='String')

    def test_param_macro_accuracy_func(self):
        # tests for parameter 'conf_matrix':
        self.assertRaises(AttributeError, macro_accuracy_func, conf_matrix=1)
        self.assertRaises(ValueError, macro_accuracy_func, conf_matrix=np.array([1]))
        self.assertRaises(AttributeError, macro_accuracy_func, conf_matrix=np.nan)
        self.assertRaises(AttributeError, macro_accuracy_func, conf_matrix=None)
        self.assertRaises(AttributeError, macro_accuracy_func, conf_matrix='String')

    def test_param_score_accuracy(self):
        # tests for parameter 'conf_matrix':
        self.assertRaises(AttributeError, score_accuracy, conf_matrix=1)
        self.assertRaises(ValueError, score_accuracy, conf_matrix=np.array([1]))
        self.assertRaises(AttributeError, score_accuracy, conf_matrix=np.nan)
        self.assertRaises(AttributeError, score_accuracy, conf_matrix=None)
        self.assertRaises(AttributeError, score_accuracy, conf_matrix='String')

    def test_param_score_precision(self):
        # tests for parameter 'conf_matrix':
        self.assertRaises(TypeError, score_precision, conf_matrix=1)
        self.assertRaises(IndexError, score_precision, conf_matrix=np.array([1]))
        self.assertRaises(TypeError, score_precision, conf_matrix=np.nan)
        self.assertRaises(TypeError, score_precision, conf_matrix=None)
        self.assertRaises(TypeError, score_precision, conf_matrix='String')

    def test_param_f1_score_func(self):
        # tests for parameter 'conf_matrix':
        self.assertRaises(TypeError, f1_score_func, conf_matrix=1)
        self.assertRaises(IndexError, f1_score_func, conf_matrix=np.array([1]))
        self.assertRaises(TypeError, f1_score_func, conf_matrix=np.nan)
        self.assertRaises(TypeError, f1_score_func, conf_matrix=None)
        self.assertRaises(TypeError, f1_score_func, conf_matrix='String')

    def test_param_calculate_optimal_prior(self):
        # tests for parameter 'n_classes':
        self.assertRaises(ValueError, calculate_optimal_prior,
                          n_classes=-2, cost_matrix=None)
        self.assertRaises(TypeError, calculate_optimal_prior,
                          n_classes=None, cost_matrix=None)
        self.assertRaises(TypeError, calculate_optimal_prior,
                          n_classes=np.nan, cost_matrix=None)
        self.assertRaises(TypeError, calculate_optimal_prior,
                          n_classes='String', cost_matrix=None)

        # tests for parameter 'n_classes':
        self.assertRaises(TypeError, calculate_optimal_prior,
                          n_classes=2, cost_matrix=np.nan)
        self.assertRaises(LinAlgError, calculate_optimal_prior,
                          n_classes=2, cost_matrix='String')
        self.assertRaises(LinAlgError, calculate_optimal_prior,
                          n_classes=2, cost_matrix=np.ones((2, 3)))
        self.assertRaises(LinAlgError, calculate_optimal_prior,
                          n_classes=2, cost_matrix=np.ones((3, 2)))
        self.assertRaises(LinAlgError, calculate_optimal_prior,
                          n_classes=2, cost_matrix=np.ones((2, 2, 2)))

    def test_transform_scoring(self):
        from skactiveml.pool._probal import _transform_scoring

        self.assertRaises(ValueError, _transform_scoring, 'wrong-string')

        # error
        np.testing.assert_raises(ValueError, _transform_scoring, 'error')
        _, cost_matrix, _ = _transform_scoring('error', n_classes=3)
        correct = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
        np.testing.assert_array_equal(cost_matrix, correct)

        # cost vector
        np.testing.assert_raises(ValueError, _transform_scoring, 'cost-vector')
        np.testing.assert_raises(ValueError, _transform_scoring, 'cost-vector',
                                 n_classes=3)
        np.testing.assert_raises(ValueError, _transform_scoring, 'cost-vector',
                                 n_classes=3, cost_vector=np.array([0, 1]))
        _, cost_matrix, _ = _transform_scoring('cost-vector', n_classes=3,
                                               cost_vector=np.array([0, 2, 1]))
        correct = [[0, 0, 0], [2, 0, 2], [1, 1, 0]]
        np.testing.assert_array_equal(cost_matrix, correct)

        # misclassification loss
        np.testing.assert_raises(ValueError, _transform_scoring,
                                 'misclassification-loss')
        np.testing.assert_raises(ValueError, _transform_scoring,
                                 'misclassification-loss', n_classes=3)
        np.testing.assert_raises(ValueError, _transform_scoring,
                                 'misclassification-loss', n_classes=3,
                                 cost_matrix=np.array(([0, 1], [1, 0])))
        inp_cost_matrix = np.array([[0, 2, 1], [1, 0, 3], [1, 1, 0]])
        _, cost_matrix, _ = _transform_scoring('misclassification-loss',
                                               n_classes=3,
                                               cost_matrix=inp_cost_matrix)
        np.testing.assert_array_equal(cost_matrix, inp_cost_matrix)

        # mean absolute error
        np.testing.assert_raises(ValueError, _transform_scoring,
                                 'mean-abs-error')
        _, cost_matrix, _ = _transform_scoring('mean-abs-error', n_classes=3)
        for i in range(3):
            for j in range(3):
                np.testing.assert_equal(cost_matrix[i, j], np.abs(i - j))

        # macro accuracy
        _, _, perf_func = _transform_scoring('macro-accuracy')
        self.assertIsNotNone(perf_func)

        # f1-score
        _, _, perf_func = _transform_scoring('f1-score')
        self.assertIsNotNone(perf_func)

        # custom
        self.assertRaises(ValueError, _transform_scoring, 'custom')
        def perf(x): return x
        _, _, perf_func = _transform_scoring('custom', perf_func=perf)
        self.assertIsNotNone(perf_func)

    def test_dperf(self):
        from skactiveml.pool._probal import _dperf

        dperf = _dperf(probs=np.array([[0.2, 0.8], [0.6, 0.4]]),
                       pred_old=np.array([0, 0]),
                       pred_new=np.array([1, 0]),
                       sample_weight_eval=np.array([1, 1]),
                       decomposable=True,
                       cost_matrix=np.array([[0, 1], [1, 0]]))
        self.assertAlmostEqual(dperf, (0.8 - 0.2) / 2)

        def perf_func(matrix): return 1
        dperf = _dperf(probs=np.array([[0.2, 0.8], [0.6, 0.4]]),
                       pred_old=np.array([0, 0]),
                       pred_new=np.array([1, 0]),
                       sample_weight_eval=np.array([1, 1]),
                       decomposable=False,
                       perf_func=perf_func)
        self.assertEqual(dperf, 0)

        probs = np.array([[0.2, 0.8], [0.6, 0.4], [0.7, 0.3]])
        pred_old = np.array([1, 1, 1])
        pred_new = np.array([1, 0, 0])
        n_classes = probs.shape[1]
        conf_mat_old = np.zeros([n_classes, n_classes])
        conf_mat_new = np.zeros([n_classes, n_classes])

        def perf_func(confusion_matrix):
            performance = 0
            for a in range(n_classes):
                for b in range(n_classes):
                    if a != b:
                        performance += confusion_matrix[a, b]
            return performance

        dperf = _dperf(probs=probs,
                       pred_old=pred_old,
                       pred_new=pred_new,
                       sample_weight_eval=np.ones(len(probs)),
                       decomposable=False,
                       perf_func=perf_func)

        for i, probability in enumerate(probs):
            for y in range(n_classes):
                conf_mat_old[y, pred_old[i]] += probability[y]
                conf_mat_new[y, pred_new[i]] += probability[y]
        difference = perf_func(conf_mat_new) - perf_func(conf_mat_old)

        self.assertEqual(dperf, difference)


if __name__ == '__main__':
    unittest.main()
