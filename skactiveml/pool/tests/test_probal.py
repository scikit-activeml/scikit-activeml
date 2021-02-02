import numpy as np
import unittest

from skactiveml.pool import McPAL, XPAL, RandomSampler
from skactiveml.classifier import PWC
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
        self.X = np.array([[1, 2], [5, 8], [8, 4], [5, 4]])
        self.y = np.array([0, 0, 1, 1])
        self.classes = np.array([0, 1])
        self.clf = PWC(classes=self.classes)
        self.kwargs = dict(X_cand=self.X_cand, X=self.X, y=self.y)

    def test_init_param_clf(self):
        # TODO
        selector = XPAL(clf=None)
        self.assertRaises(TypeError, selector.query, **self.kwargs)

        selector = XPAL(clf='string')
        self.assertRaises(TypeError, selector.query, **self.kwargs)

        self.assertTrue(hasattr(selector, 'clf'))

    def test_init_param_scoring(self):
        # TODO
        selector = XPAL(clf=self.clf, scoring=None)
        self.assertRaises(ValueError, selector.query, **self.kwargs)

        selector = XPAL(clf=self.clf, scoring=2)
        self.assertRaises(ValueError, selector.query, **self.kwargs)

        selector = XPAL(clf=self.clf, scoring='String')
        self.assertRaises(ValueError, selector.query, **self.kwargs)

        self.assertTrue(hasattr(selector, 'scoring'))

    def test_init_param_cost_vector(self):
        # TODO
        selector = XPAL(clf=self.clf, scoring='cost-vector',
                        cost_vector='string')
        self.assertRaises(ValueError, selector.query, **self.kwargs)

        selector = XPAL(clf=self.clf, scoring='cost-vector',
                        cost_vector=[3, 2, 1])  # clf.n_classes=2
        self.assertRaises(ValueError, selector.query, **self.kwargs)

        selector = XPAL(clf=self.clf, scoring='cost-vector',
                        cost_vector=None)  # clf.n_classes=2
        self.assertRaises(ValueError, selector.query, **self.kwargs)

        self.assertTrue(hasattr(selector, 'cost_vector'))

    def test_init_param_cost_matrix(self):
        # TODO
        selector = XPAL(clf=self.clf, scoring='misclassification-loss',
                        cost_matrix=None)
        self.assertRaises(ValueError, selector.query, **self.kwargs)

        selector = XPAL(clf=self.clf, scoring='macro-accuracy',
                        cost_matrix=None)
        self.assertRaises(ValueError, selector.query, **self.kwargs)

        selector = XPAL(clf=self.clf, scoring='f1-score',
                        cost_matrix=None)
        self.assertRaises(ValueError, selector.query, **self.kwargs)

        selector = XPAL(clf=self.clf, cost_matrix=np.ones((2, 3)))
        self.assertRaises(ValueError, selector.query, **self.kwargs)

        selector = XPAL(clf=self.clf, cost_matrix='string')
        self.assertRaises(TypeError, selector.query, **self.kwargs)

        selector = XPAL(clf=self.clf, cost_matrix=np.ones((3, 3)))
        self.assertRaises(ValueError, selector.query, **self.kwargs)

        # TODO cost_vector and cost_matrix together?
        selector = XPAL(clf=self.clf, cost_matrix=np.ones((2, 2)),
                        cost_vector=[1, 1])
        self.assertRaises(ValueError, selector.query, **self.kwargs)

        self.assertTrue(hasattr(selector, 'cost_matrix'))

    def test_init_param_custom_perf_func(self):
        # TODO
        selector = XPAL(clf=self.clf, scoring='custom',
                        custom_perf_func='string')
        self.assertRaises(TypeError, selector.query, **self.kwargs)

        selector = XPAL(clf=self.clf, scoring='custom',
                        custom_perf_func=42)
        self.assertRaises(TypeError, selector.query, **self.kwargs)

        selector = XPAL(clf=self.clf, scoring='custom')
        self.assertRaises(TypeError, selector.query, **self.kwargs)

        def func(a):
            return 0
        selector = XPAL(clf=self.clf, scoring='error',
                        custom_perf_func=func)
        self.assertRaises(ValueError, selector.query, **self.kwargs)

        self.assertTrue(hasattr(selector, 'custom_perf_func'))

    def test_init_param_prior_cand(self):
        # TODO
        selector = XPAL(clf=self.clf, prior_cand='string')
        self.assertRaises(np.core._exceptions.UFuncTypeError,
                          selector.query, **self.kwargs)

        selector = XPAL(clf=self.clf, prior_cand=None)
        self.assertRaises(TypeError, selector.query, **self.kwargs)

        selector = XPAL(clf=self.clf, prior_cand=[[1, 2], [1, 2]])
        self.assertRaises(ValueError, selector.query, **self.kwargs)

        selector = XPAL(clf=self.clf, prior_cand=[1, 2, 3])  # clf.n_classes=2
        self.assertRaises(ValueError, selector.query, **self.kwargs)

        self.assertTrue(hasattr(selector, 'prior_cand'))

    def test_init_param_prior_eval(self):
        # TODO
        selector = XPAL(clf=self.clf, prior_eval='string')
        self.assertRaises(np.core._exceptions.UFuncTypeError,
                          selector.query, **self.kwargs)

        selector = XPAL(clf=self.clf, prior_eval=None)
        self.assertRaises(TypeError, selector.query, **self.kwargs)

        selector = XPAL(clf=self.clf, prior_eval=[[1, 2], [1, 2]])
        self.assertRaises(ValueError, selector.query, **self.kwargs)

        selector = XPAL(clf=self.clf, prior_eval=[1, 2, 3])  # clf.n_classes=2
        self.assertRaises(ValueError, selector.query, **self.kwargs)

        self.assertTrue(hasattr(selector, 'prior_eval'))

    def test_init_param_estimator_metric(self):
        # TODO
        selector = XPAL(clf=self.clf, estimator_metric=False)
        self.assertRaises(ValueError, selector.query, **self.kwargs)

        selector = XPAL(clf=self.clf, estimator_metric='String')
        self.assertRaises(ValueError, selector.query, **self.kwargs)

        selector = XPAL(clf=self.clf, estimator_metric=None)
        self.assertRaises(ValueError, selector.query, **self.kwargs)

        selector = XPAL(clf=self.clf, estimator_metric='precomputed')
        self.assertRaises(ValueError, selector.query, **self.kwargs)

        self.assertTrue(hasattr(selector, 'estimator_metric'))

    def test_init_param_estimator_metric_dict(self):
        # TODO
        selector = XPAL(clf=self.clf, estimator_metric_dict='String')
        self.assertRaises(TypeError, selector.query, **self.kwargs)

        selector = XPAL(clf=self.clf, estimator_metric_dict=2)
        self.assertRaises(TypeError, selector.query, **self.kwargs)

        selector = XPAL(clf=self.clf, estimator_metric_dict=True)
        self.assertRaises(TypeError, selector.query, **self.kwargs)

        def func(X1, X2):
            return 0

        selector = XPAL(clf=self.clf, estimator_metric=func,
                        estimator_metric_dict={'arg': 'arg'})
        self.assertRaises(TypeError, selector.query, **self.kwargs)

        self.assertTrue(hasattr(selector, 'estimator_metric_dict'))

    def test_init_param_batch_mode(self):
        # TODO
        selector = XPAL(clf=self.clf, batch_mode='string')
        self.assertRaises(ValueError, selector.query, **self.kwargs)

        selector = XPAL(clf=self.clf, batch_mode=None)
        self.assertRaises(TypeError, selector.query, **self.kwargs)

        selector = XPAL(clf=self.clf, batch_mode=False)
        self.assertRaises(TypeError, selector.query, **self.kwargs)

        self.assertTrue(hasattr(selector, 'batch_mode'))

    def test_init_param_batch_labels_equal(self):
        # TODO
        selector = XPAL(clf=self.clf, batch_labels_equal="string")
        self.assertRaises(TypeError, selector.query, **self.kwargs)

        selector = XPAL(clf=self.clf, batch_labels_equal=None)
        self.assertRaises(TypeError, selector.query, **self.kwargs)

        selector = XPAL(clf=self.clf, batch_labels_equal=[])
        self.assertRaises(TypeError, selector.query, **self.kwargs)

        selector = XPAL(clf=self.clf, batch_labels_equal=2)
        self.assertRaises(TypeError, selector.query, **self.kwargs)

        self.assertTrue(hasattr(selector, 'nonmyopic_labels_equal'))

    def test_init_param_nonmyopic_max_cand(self):
        selector = XPAL(clf=self.clf, nonmyopic_max_cand=2,
                        batch_mode='full')
        self.assertRaises(NotImplementedError, selector.query, **self.kwargs)

        selector = XPAL(clf=self.clf, nonmyopic_max_cand=0,
                        batch_mode='greedy')
        self.assertRaises(ValueError, selector.query, **self.kwargs)

        selector = XPAL(clf=self.clf, nonmyopic_max_cand=1.5,
                        batch_mode='greedy')
        self.assertRaises(TypeError, selector.query, **self.kwargs)

        selector = XPAL(clf=self.clf, nonmyopic_max_cand=-5,
                        batch_mode='greedy')
        self.assertRaises(ValueError, selector.query, **self.kwargs)

        selector = XPAL(clf=self.clf, nonmyopic_max_cand='string',
                        batch_mode='greedy')
        self.assertRaises(TypeError, selector.query, **self.kwargs)

        selector = XPAL(clf=self.clf, nonmyopic_max_cand=None,
                        batch_mode='greedy')
        self.assertRaises(TypeError, selector.query, **self.kwargs)

        self.assertTrue(hasattr(selector, 'nonmyopic_max_cand'))

    def test_init_param_nonmyopic_neighbors(self):
        # nonmyopic_max_cand = 2 (nonmyopic method)
        selector = XPAL(clf=self.clf, nonmyopic_max_cand=2,
                        nonmyopic_neighbors="string", batch_mode='greedy')
        self.assertRaises(ValueError, selector.query, **self.kwargs)

        selector = XPAL(clf=self.clf, nonmyopic_max_cand=2,
                        nonmyopic_neighbors=None, batch_mode='greedy')
        self.assertRaises(TypeError, selector.query, **self.kwargs)

        self.assertTrue(hasattr(selector, 'nonmyopic_labels_equal'))

    def test_init_param_nonmyopic_labels_equal(self):
        # nonmyopic_max_cand = 2 (nonmyopic method)
        selector = XPAL(clf=self.clf, nonmyopic_max_cand=2,
                        nonmyopic_labels_equal=None, batch_mode='greedy')
        self.assertRaises(TypeError, selector.query, **self.kwargs)

        self.assertTrue(hasattr(selector, 'nonmyopic_labels_equal'))

        selector = XPAL(clf=self.clf, nonmyopic_max_cand=2,
                        nonmyopic_labels_equal=[], batch_mode='greedy')
        self.assertRaises(TypeError, selector.query, **self.kwargs)

        selector = XPAL(clf=self.clf, nonmyopic_max_cand=2,
                        nonmyopic_labels_equal=0, batch_mode='greedy')
        self.assertRaises(TypeError, selector.query, **self.kwargs)

        selector = XPAL(clf=self.clf, nonmyopic_max_cand=2,
                        nonmyopic_labels_equal='string', batch_mode='greedy')
        self.assertRaises(TypeError, selector.query, **self.kwargs)

    def test_init_param_nonmyopic_independent_probs(self):
        # nonmyopic_max_cand = 2 (nonmyopic method)
        selector = XPAL(clf=self.clf, nonmyopic_max_cand=2,
                        nonmyopic_independent_probs=0, batch_mode='greedy')
        self.assertRaises(TypeError, selector.query, **self.kwargs)

        self.assertTrue(hasattr(selector, 'nonmyopic_independent_probs'))

        selector = XPAL(clf=self.clf, nonmyopic_max_cand=2,
                        nonmyopic_independent_probs=None, batch_mode='greedy')
        self.assertRaises(TypeError, selector.query, **self.kwargs)

        selector = XPAL(clf=self.clf, nonmyopic_max_cand=2,
                        nonmyopic_independent_probs=[], batch_mode='greedy')
        self.assertRaises(TypeError, selector.query, **self.kwargs)

        selector = XPAL(clf=self.clf, nonmyopic_max_cand=2,
                        nonmyopic_independent_probs='string',
                        batch_mode='greedy')
        self.assertRaises(TypeError, selector.query, **self.kwargs)

    def test_init_param_random_state(self):
        selector = XPAL(clf=self.clf, random_state='string')
        self.assertRaises(ValueError, selector.query, **self.kwargs)

        self.assertTrue(hasattr(selector, 'random_state'))


if __name__ == '__main__':
    unittest.main()
