import unittest

import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier

from skactiveml.classifier import ParzenWindowClassifier
from skactiveml.pool import ProbabilisticAL, XProbabilisticAL
from skactiveml.tests.template_query_strategy import \
    TemplateSingleAnnotatorPoolQueryStrategy
from skactiveml.utils import MISSING_LABEL, is_unlabeled, unlabeled_indices


class TestProbabilisticAL(
    TemplateSingleAnnotatorPoolQueryStrategy,
    unittest.TestCase):

    def setUp(self):
        self.classes = [0, 1, 2]
        query_default_params_clf = {
            'X': np.array([[1, 2], [5, 8], [8, 4], [5, 4]]),
            'y': np.array([0, 1, MISSING_LABEL, MISSING_LABEL]),
            'clf': ParzenWindowClassifier(random_state=42,
                                          classes=self.classes),
        }
        super().setUp(qs_class=ProbabilisticAL,
                      init_default_params={},
                      query_default_params_clf=query_default_params_clf)

    # Test init parameters
    def test_init_param_prior(self):
        clf = ParzenWindowClassifier(
            classes=self.classes, missing_label=MISSING_LABEL
        )
        test_cases = [(0, ValueError), (clf, TypeError)]
        self._test_param('init', 'prior', test_cases)

    def test_init_param_m_max(self):
        test_cases = [(-2, ValueError), (1.5, TypeError)]
        self._test_param('init', 'm_max', test_cases)

    # Test query parameters
    def test_query_param_clf(self):
        add_test_cases = [
            (GaussianProcessClassifier(), TypeError),
            (ParzenWindowClassifier(missing_label="missing"), TypeError),
        ]
        super().test_query_param_clf(test_cases=add_test_cases)

    def test_query_param_sample_weight(self):
        X = self.query_default_params_clf['X']
        test_cases = [("string", ValueError), (X, ValueError),
                      (np.empty((len(X) - 1)), ValueError),
                      (np.ones(len(X)), None)]
        super().test_query_param_sample_weight(test_cases)

    def test_query_param_utility_weight(self):
        test_cases = [('string', ValueError), (np.zeros((2, 2)), ValueError),
                      (np.empty(6), ValueError)]
        super().test_query_param_utility_weight(test_cases)

        test_cases = [(np.ones(2), None)]
        self._test_param('query', 'utility_weight', test_cases,
                         replace_query_params={'candidates': [[0, 1], [2, 3]]})

    def test_query(self):
        clf = ParzenWindowClassifier(
            classes=self.classes, missing_label=MISSING_LABEL
        )
        mcpal = ProbabilisticAL()
        self.assertRaises(ValueError, mcpal.query, X=[], y=[], clf=clf)
        self.assertRaises(
            ValueError, mcpal.query, X=[], y=[], clf=clf, candidates=[]
        )
        self.assertRaises(
            ValueError,
            mcpal.query,
            X=np.zeros((6, 2)),
            y=[0, 1, 4, 0, 2, 1],
            clf=clf,
            candidates=[],
        )

        # Test missing labels
        X_cand = [[0], [1], [2], [3]]
        clf = ParzenWindowClassifier(classes=[0, 1])
        mcpal = ProbabilisticAL()
        _, utilities = mcpal.query(
            [[1]],
            [MISSING_LABEL],
            clf,
            candidates=X_cand,
            return_utilities=True,
        )
        self.assertEqual(utilities.shape, (1, len(X_cand)))
        self.assertEqual(len(np.unique(utilities)), 1)

        _, utilities = mcpal.query(
            X=[[0], [1], [2]],
            y=[0, 1, MISSING_LABEL],
            clf=clf,
            candidates=X_cand,
            return_utilities=True,
        )
        self.assertGreater(utilities[0, 2], utilities[0, 1])
        self.assertGreater(utilities[0, 2], utilities[0, 0])

        # Test scenario
        X_cand = [[0], [1], [2], [5]]
        mcpal = ProbabilisticAL()

        best_indices = mcpal.query(X=[[1]], y=[0], clf=clf, candidates=X_cand)
        np.testing.assert_array_equal(best_indices, np.array([3]))

        _, utilities = mcpal.query(
            X=[[1]], y=[0], clf=clf, candidates=X_cand, return_utilities=True
        )
        min_utilities = np.argmin(utilities)
        np.testing.assert_array_equal(min_utilities, np.array([1]))

        best_indices = mcpal.query(
            X=[[0], [2]], y=[0, 1], clf=clf, candidates=[[0], [1], [2]]
        )
        np.testing.assert_array_equal(best_indices, [1])


class TestXProbabilisticAL(
    TemplateSingleAnnotatorPoolQueryStrategy,
    unittest.TestCase):
    def setUp(self):
        # self.X = np.zeros((6, 2))
        # self.candidates = np.zeros((2, 2))
        # self.y = [0, 1, 1, 0, 2, 1]
        self.classes = [0, 1, 2]
        # self.clf = ParzenWindowClassifier(
        #     classes=self.classes, missing_label=MISSING_LABEL
        # )
        # self.kwargs = dict(
        #     X=self.X, y=self.y, candidates=self.candidates, clf=self.clf
        # )
        self.random_state = np.random.RandomState(42)
        query_default_params_clf = {
            'X': np.array([[1, 2], [5, 8], [8, 4], [5, 4]]),
            'y': np.array([0, 1, MISSING_LABEL, MISSING_LABEL]),
            'clf': ParzenWindowClassifier(random_state=42,
                                          classes=self.classes),
        }
        super().setUp(qs_class=XProbabilisticAL,
                      init_default_params={},
                      query_default_params_clf=query_default_params_clf)

    # Test init parameters
    def test_init_param_method(self):
        test_cases = [(1, TypeError), ("string", ValueError),
                      ('inductive', None), ('transductive', None)]
        self._test_param("init", "method", test_cases)

    def test_init_param_cost_matrix(self):
        test_cases = [(np.ones((2, 3)), ValueError), ("string", ValueError),
                      (np.ones((2, 2)), ValueError)]
        self._test_param("init", "cost_matrix", test_cases)

    def test_init_param_candidate_prior(self):
        test_cases = [(-1, ValueError), ("string", TypeError)]
        self._test_param("init", "candidate_prior", test_cases)

    def test_init_param_evaluation_prior(self):
        test_cases = [(-1, ValueError), ("string", TypeError)]
        self._test_param("init", "evaluation_prior", test_cases)

    # Test query parameters
    def test_query_param_clf(self):
        add_test_cases = [
            (GaussianProcessClassifier(), TypeError),
            (ParzenWindowClassifier(missing_label="missing"), TypeError),
        ]
        super().test_query_param_clf(test_cases=add_test_cases)

    def test_query_param_ignore_partial_fit(self):
        test_params = [('string', TypeError), (True, None), (False, None)]
        self._test_param('query', 'ignore_partial_fit', test_params)

    def test_query_param_sample_weight(self):
        X = self.query_default_params_clf['X']
        test_cases = [("string", ValueError), (X, ValueError),
                      (np.empty((len(X) - 1)), ValueError),
                      (np.ones(len(X)), None)]
        super().test_query_param_sample_weight(test_cases)

    def test_query_param_sample_weight_candidates(self):
        raise NotImplementedError('TODO Daniel Kottke')
        query_params = self.query_default_params_clf
        y = self.query_default_params_clf['y']
        candidates = unlabeled_indices(y)

        # sample_wight = None
        test_cases = [(np.ones(candidates), ValueError), (None, None)]
        self._test_param('query', 'sample_weight_candidates', test_cases)

        # sample_wight = 1d array
        query_params['sample_weight'] = np.ones(len(query_params['X']))
        test_cases = [("string", ValueError), (candidates, ValueError),
                      (np.empty((len(candidates) - 1)), ValueError),
                      (np.ones(len(candidates)), None), None]
        self._test_param('query', 'sample_weight_candidates', test_cases)

    def test_query_param_X_eval(self):
        X = self.query_default_params_clf['X']
        test_cases = [("string", ValueError),
                      (1, ValueError), (X, None)]
        self._test_param("query", "X_eval", test_cases)

    def test_query_param_sample_weight_eval(self):
        X_eval = self.random_state.rand(6, 2)
        test_cases = [(np.ones(len(X_eval)), ValueError)]
        self._test_param('query', 'sample_weight_eval', test_cases)

        query_params = self.query_default_params_clf
        query_params['X_eval'] = X_eval
        test_cases = [("string", ValueError), (X_eval, ValueError),
                      (np.empty((len(X_eval) - 1)), ValueError),
                      (np.ones(len(X_eval)), None)]
        self._test_param('query', 'sample_weight_eval', test_cases,
                         replace_query_params=query_params)

    def test_query_param_return_candidate_utilities(self):
        test_params = [('string', TypeError), (True, None), (False, None)]
        self._test_param('query', 'return_candidate_utilities', test_params)

    def test_query(self):
        # TODO
        pass
