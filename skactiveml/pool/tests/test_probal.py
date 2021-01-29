import numpy as np
import unittest

from skactiveml.pool import McPAL, XPAL
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
        self.clf = PWC()
        self.kwargs = dict(X_cand=self.X_cand, X=self.X, y=self.y)


    def test_init_param_clf(self):
        # TODO
        pass

    def test_init_param_scoring(self):
        # TODO
        pass

    def test_init_param_cost_vector(self):
        # TODO
        pass

    def test_init_param_cost_matrix(self):
        # TODO
        pass

    def test_init_param_custom_perf_func(self):
        # TODO
        pass

    def test_init_param_prior_cand(self):
        # TODO
        pass

    def test_init_param_prior_eval(self):
        # TODO
        pass

    def test_init_param_estimator_metric(self):
        # TODO
        pass

    def test_init_param_estimator_metric_dict(self):
        # TODO
        pass

    def test_init_param_batch_mode(self):
        # TODO
        pass

    def test_init_param_batch_labels_equal(self):
        # TODO
        pass

    def test_init_param_nonmyopic_look_ahead(self):
        selector = XPAL(clf=self.clf, nonmyopic_look_ahead=0)
        self.assertRaises(ValueError, selector.query, **self.kwargs)

        selector = XPAL(clf=self.clf, nonmyopic_look_ahead=1.5)
        self.assertRaises(TypeError, selector.query, **self.kwargs)

        selector = XPAL(clf=self.clf, nonmyopic_look_ahead=-5)
        self.assertRaises(ValueError, selector.query, **self.kwargs)

        selector = XPAL(clf=self.clf, nonmyopic_look_ahead='string')
        self.assertRaises(TypeError, selector.query, **self.kwargs)

        selector = XPAL(clf=self.clf, nonmyopic_look_ahead=None)
        self.assertRaises(TypeError, selector.query, **self.kwargs)

    def test_init_param_nonmyopic_neighbors(self):
        # TODO
        # nonmyopic_look_ahead = 2 (nonmyopic method)
        selector = XPAL(clf=self.clf, nonmyopic_look_ahead=1,
                        nonmyopic_neighbors="string")
        self.assertRaises(ValueError, selector.query, **self.kwargs)

        selector = XPAL(clf=self.clf, nonmyopic_look_ahead=1,
                        nonmyopic_neighbors=None)
        self.assertRaises(TypeError, selector.query, **self.kwargs)

        self.assertTrue(hasattr(selector, 'nonmyopic_labels_equal'))

    def test_init_param_nonmyopic_labels_equal(self):
        # TODO
        # nonmyopic_look_ahead = 2 (nonmyopic method)
        selector = XPAL(clf=self.clf, nonmyopic_look_ahead=2,
                        nonmyopic_labels_equal=None)
        self.assertRaises(TypeError, selector.query, **self.kwargs)

        self.assertTrue(hasattr(selector, 'nonmyopic_labels_equal'))

        selector = XPAL(clf=self.clf, nonmyopic_look_ahead=2,
                        nonmyopic_labels_equal=[])
        self.assertRaises(TypeError, selector.query, **self.kwargs)

        selector = XPAL(clf=self.clf, nonmyopic_look_ahead=2,
                        nonmyopic_labels_equal=0)
        self.assertRaises(TypeError, selector.query, **self.kwargs)

        selector = XPAL(clf=self.clf, nonmyopic_look_ahead=2,
                        nonmyopic_labels_equal='string')
        self.assertRaises(TypeError, selector.query, **self.kwargs)

    def test_init_param_independent_probs(self):
        # nonmyopic_look_ahead = 2 (nonmyopic method)
        selector = XPAL(clf=self.clf,  nonmyopic_look_ahead=2,
                        independent_probs=0)
        self.assertRaises(TypeError, selector.query, **self.kwargs)

        self.assertTrue(hasattr(selector, 'independent_probs'))

        selector = XPAL(clf=self.clf, nonmyopic_look_ahead=2,
                        independent_probs=None)
        self.assertRaises(TypeError, selector.query, **self.kwargs)

        selector = XPAL(clf=self.clf, nonmyopic_look_ahead=2,
                        independent_probs=[])
        self.assertRaises(TypeError, selector.query, **self.kwargs)

        selector = XPAL(clf=self.clf,  nonmyopic_look_ahead=2,
                        independent_probs='string')
        self.assertRaises(TypeError, selector.query, **self.kwargs)

        # TODO

    def test_init_param_random_state(self):
        selector = XPAL(clf=self.clf, random_state='string')
        self.assertRaises(ValueError, selector.query, **self.kwargs)

        self.assertTrue(hasattr(selector, 'random_state'))


if __name__ == '__main__':
    unittest.main()
