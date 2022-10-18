import unittest

import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier

from skactiveml.classifier import ParzenWindowClassifier
from skactiveml.pool import ProbabilisticAL, XPal
from skactiveml.utils import MISSING_LABEL


class TestProbabilisticAL(unittest.TestCase):
    def setUp(self):
        self.X = np.zeros((6, 2))
        self.utility_weight = np.ones(len(self.X)) / len(self.X)
        self.candidates = np.zeros((2, 2))
        self.y = [0, 1, 1, 0, 2, 1]
        self.classes = [0, 1, 2]
        self.C = np.eye(3)
        self.clf = ParzenWindowClassifier(
            classes=self.classes, missing_label=MISSING_LABEL
        )
        self.kwargs = dict(
            X=self.X, y=self.y, candidates=self.candidates, clf=self.clf
        )

    # Test init parameters
    def test_init_param_prior(self):
        pal = ProbabilisticAL(prior=0)
        self.assertTrue(hasattr(pal, "prior"))
        self.assertRaises(ValueError, pal.query, **self.kwargs)

        pal = ProbabilisticAL(self.clf)
        self.assertTrue(hasattr(pal, "prior"))
        self.assertRaises(TypeError, pal.query, **self.kwargs)

    def test_init_param_m_max(self):
        pal = ProbabilisticAL(m_max=-2)
        self.assertTrue(hasattr(pal, "m_max"))
        self.assertRaises(ValueError, pal.query, **self.kwargs)

        pal = ProbabilisticAL(m_max=1.5)
        self.assertTrue(hasattr(pal, "m_max"))
        self.assertRaises(TypeError, pal.query, **self.kwargs)

    def test_query_param_clf(self):
        pal = ProbabilisticAL()
        self.assertRaises(
            TypeError,
            pal.query,
            X=self.X,
            y=self.y,
            clf=GaussianProcessClassifier(),
        )
        self.assertRaises(
            (ValueError, TypeError),
            pal.query,
            X=self.X,
            y=self.y,
            clf=ParzenWindowClassifier(missing_label="missing"),
        )

    def test_query_param_sample_weight(self):
        pal = ProbabilisticAL()
        self.assertRaises(
            ValueError, pal.query, **self.kwargs, sample_weight="string"
        )
        self.assertRaises(
            ValueError, pal.query, **self.kwargs, sample_weight=self.candidates
        )
        self.assertRaises(
            ValueError,
            pal.query,
            **self.kwargs,
            sample_weight=np.empty((len(self.X) - 1))
        )
        self.assertRaises(
            ValueError,
            pal.query,
            **self.kwargs,
            sample_weight=np.empty((len(self.X) + 1))
        )
        self.assertRaises(
            ValueError,
            pal.query,
            **self.kwargs,
            sample_weight=np.ones((len(self.X) + 1))
        )
        self.assertRaises(
            ValueError,
            pal.query,
            X=self.X,
            y=self.y,
            candidates=None,
            clf=self.clf,
            sample_weight=np.ones((len(self.X) + 1)),
        )
        self.assertRaises(
            ValueError,
            pal.query,
            X=self.X,
            y=self.y,
            candidates=[0],
            clf=self.clf,
            sample_weight=np.ones(2),
        )

    def test_query_param_fit_clf(self):
        selector = ProbabilisticAL()
        self.assertRaises(
            TypeError, selector.query, **self.kwargs, fit_clf="string"
        )
        self.assertRaises(
            TypeError, selector.query, **self.kwargs, fit_clf=self.candidates
        )
        self.assertRaises(
            TypeError, selector.query, **self.kwargs, fit_clf=None
        )

    def test_query_param_utility_weight(self):
        pal = ProbabilisticAL()
        self.assertRaises(
            ValueError, pal.query, **self.kwargs, utility_weight="string"
        )
        self.assertRaises(
            ValueError,
            pal.query,
            **self.kwargs,
            utility_weight=self.candidates
        )
        self.assertRaises(
            ValueError,
            pal.query,
            X=self.X,
            y=self.y,
            candidates=None,
            clf=self.clf,
            utility_weight=np.empty(len(self.X)),
        )
        self.assertRaises(
            ValueError,
            pal.query,
            X=self.X,
            y=self.y,
            candidates=None,
            clf=self.clf,
            utility_weight=np.empty((len(self.X) + 1)),
        )
        self.assertRaises(
            ValueError,
            pal.query,
            X=self.X,
            y=self.y,
            candidates=[1],
            clf=self.clf,
            utility_weight=np.ones(1),
        )
        self.assertRaises(
            ValueError,
            pal.query,
            X=self.X,
            y=self.y,
            candidates=self.candidates,
            clf=self.clf,
            utility_weight=np.ones((len(self.candidates) + 1)),
        )

    def test_query(self):
        mcpal = ProbabilisticAL()
        self.assertRaises(ValueError, mcpal.query, X=[], y=[], clf=self.clf)
        self.assertRaises(
            ValueError, mcpal.query, X=[], y=[], clf=self.clf, candidates=[]
        )
        self.assertRaises(
            ValueError,
            mcpal.query,
            X=self.X,
            y=[0, 1, 4, 0, 2, 1],
            clf=self.clf,
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


class TestXPal(unittest.TestCase):
    def setUp(self):
        self.X = np.zeros((6, 2))
        self.candidates = np.zeros((2, 2))
        self.y = [0, 1, 1, 0, 2, 1]
        self.classes = [0, 1, 2]
        self.clf = ParzenWindowClassifier(
            classes=self.classes, missing_label=MISSING_LABEL
        )
        self.kwargs = dict(
            X=self.X, y=self.y, candidates=self.candidates, clf=self.clf
        )

    # Test init parameters
    def test_init_param_method(self):
        qs = XPal()
        self.assertTrue(hasattr(qs, "method"))
        selector = XPal(method="String")
        self.assertRaises(ValueError, selector.query, **self.kwargs)
        selector = XPal(method=1)
        self.assertRaises(TypeError, selector.query, **self.kwargs)

    def test_init_param_candidate_prior(self):
        qs = XPal()
        self.assertTrue(hasattr(qs, "candidate_prior"))
        for candidate_prior in [-1, 0, "string"]:
            qs = XPal(candidate_prior=candidate_prior)
            self.assertRaises((ValueError, TypeError), qs.query, **self.kwargs)

    def test_init_param_evaluation_prior(self):
        qs = XPal()
        self.assertTrue(hasattr(qs, "evaluation_prior"))
        for evaluation_prior in [-1, 0, "string"]:
            qs = XPal(evaluation_prior=evaluation_prior)
            self.assertRaises((ValueError, TypeError), qs.query, **self.kwargs)

    # Test query parameters
    def test_query_param_clf(self):
        qs = XPal()
        self.assertRaises(
            TypeError,
            qs.query,
            X=self.X,
            y=self.y,
            clf=GaussianProcessClassifier(),
        )
        self.assertRaises(
            (ValueError, TypeError),
            qs.query,
            X=self.X,
            y=self.y,
            clf=ParzenWindowClassifier(missing_label="missing"),
        )

    def test_query_param_fit_clf(self):
        qs = XPal()
        self.assertRaises(
            TypeError, qs.query, **self.kwargs, fit_clf="string"
        )
        self.assertRaises(
            TypeError, qs.query, **self.kwargs, fit_clf=self.candidates
        )
        self.assertRaises(
            TypeError, qs.query, **self.kwargs, fit_clf=None
        )

    def test_query_param_ignore_partial_fit(self):
        qs = XPal()
        self.assertRaises(
            TypeError,
            qs.query,
            **self.kwargs,
            ignore_partial_fit="test"
        )

    def test_query_param_sample_weight(self):
        qs = XPal()
        sample_weight_list = [
            "string",
            self.candidates,
            np.empty((len(self.X) - 1)),
            np.empty((len(self.X) + 1)),
        ]
        for sample_weight in sample_weight_list:
            qs = XPal()
            self.assertRaises((ValueError, TypeError), qs.query, **self.kwargs,
                              sample_weight=sample_weight)

        self.assertRaises(
            ValueError,
            qs.query,
            X=self.X,
            y=self.y,
            candidates=None,
            clf=self.clf,
            sample_weight=np.ones((len(self.X) + 1)),
        )
        self.assertRaises(
            ValueError,
            qs.query,
            X=self.X,
            y=self.y,
            candidates=[0],
            clf=self.clf,
            sample_weight=np.ones(2),
        )

    def test_query_param_sample_weight_candidates(self):
        qs = XPal()
        sample_weight_candidates_list = [
            "string",
            self.candidates,
            np.empty((len(self.X) - 1)),
            np.empty((len(self.X) + 1)),
        ]
        for sample_weight_candidates in sample_weight_candidates_list:
            qs = XPal()
            self.assertRaises(
                (ValueError, TypeError), qs.query, **self.kwargs,
                sample_weight_candidates=sample_weight_candidates
            )

        self.assertRaises(
            ValueError,
            qs.query,
            X=self.X,
            y=self.y,
            candidates=None,
            clf=self.clf,
            sample_weight_candidates=np.ones((len(self.X) + 1)),
        )
        self.assertRaises(
            ValueError,
            qs.query,
            X=self.X,
            y=self.y,
            candidates=[0],
            clf=self.clf,
            sample_weight_candidates=np.ones(2),
        )

    def _test_query_param_X(self):
        qs = XPal()
        for X_eval in [None, "str", [], np.ones(5)]:
            self.assertRaises(
                (TypeError, ValueError),
                qs.query,
                **self.kwargs,
                X_eval=X_eval
            )

    def test_query_param_sample_weight_eval(self):
        qs = XPal()
        sample_weight_eval_list = [
            "string",
            self.candidates,
            np.empty((len(self.X) - 1)),
            np.empty((len(self.X) + 1)),
        ]
        for sample_weight_eval in sample_weight_eval_list:
            qs = XPal()
            self.assertRaises(
                (ValueError, TypeError), qs.query, **self.kwargs,
                sample_weight_eval=sample_weight_eval
            )

        self.assertRaises(
            ValueError,
            qs.query,
            X=self.X,
            y=self.y,
            X_eval=None,
            clf=self.clf,
            sample_weight_eval=np.ones((len(self.X) + 1)),
        )
        self.assertRaises(
            ValueError,
            qs.query,
            X=self.X,
            y=self.y,
            X_eval=[[0]],
            clf=self.clf,
            sample_weight_eval=np.ones(2),
        )

    def test_query_param_return_candidate_utilities(self):
        qs = XPal()
        self.assertRaises(
            TypeError,
            qs.query,
            **self.kwargs,
            return_candidate_utilities="test"
        )

    def test_query(self):
        # TODO
        pass
