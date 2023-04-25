import unittest

import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB

from skactiveml.classifier import ParzenWindowClassifier, SklearnClassifier
from skactiveml.pool import ProbabilisticAL
from skactiveml.tests.template_query_strategy import (
    TemplateSingleAnnotatorPoolQueryStrategy,
)
from skactiveml.utils import MISSING_LABEL


class TestProbabilisticAL(
    TemplateSingleAnnotatorPoolQueryStrategy, unittest.TestCase
):
    def setUp(self):
        self.X = np.zeros((6, 2))
        self.utility_weight = np.ones(len(self.X)) / len(self.X)
        self.candidates = np.zeros((2, 2))
        self.y = [0, 1, 1, 0, 2, 1]
        self.classes = [0, 1, 2]
        self.clf = ParzenWindowClassifier(
            classes=self.classes, missing_label=MISSING_LABEL
        )
        self.kwargs = dict(
            X=self.X, y=self.y, candidates=self.candidates, clf=self.clf
        )
        query_default_params_clf = {
            "X": np.array([[1, 2], [5, 8], [8, 4], [5, 4]]),
            "y": np.array([0, 1, MISSING_LABEL, MISSING_LABEL]),
            "clf": ParzenWindowClassifier(
                random_state=42, classes=self.classes
            ),
        }
        super().setUp(
            qs_class=ProbabilisticAL,
            init_default_params={},
            query_default_params_clf=query_default_params_clf,
        )

    # Test init parameters
    def test_init_param_prior(self):
        test_cases = [(0, ValueError), (self.clf, TypeError)]
        self._test_param("init", "prior", test_cases)

    def test_init_param_m_max(self):
        test_cases = [(-2, ValueError), (1.5, TypeError)]
        self._test_param("init", "m_max", test_cases)

    def test_init_param_metric_dict(self):
        pal = ProbabilisticAL(metric="rbf", metric_dict=["gamma"])
        self.assertRaises(TypeError, pal.query, **(self.kwargs))
        pal = ProbabilisticAL(metric="rbf", metric_dict={"test": 0})
        self.assertRaises(TypeError, pal.query, **(self.kwargs))

    def test_init_param_metric(self):
        pal = ProbabilisticAL(metric="string")
        self.assertRaises(ValueError, pal.query, **(self.kwargs))
        pal = ProbabilisticAL(metric=0)
        self.assertRaises(ValueError, pal.query, **(self.kwargs))
        pal = ProbabilisticAL()
        clf = SklearnClassifier(GaussianNB())
        self.assertRaises(
            TypeError,
            pal.query,
            candidates=self.candidates,
            clf=clf,
            X=self.X,
            y=self.y,
        )
        pal = ProbabilisticAL(metric="rbf")
        clf = SklearnClassifier(GaussianNB())
        pal.query(
            candidates=self.candidates,
            clf=clf,
            X=self.X,
            y=self.y,
            fit_clf=True,
        )

    def test_query_param_clf(self):
        add_test_cases = [
            (GaussianProcessClassifier(), TypeError),
            (ParzenWindowClassifier(missing_label="missing"), TypeError),
        ]
        super().test_query_param_clf(test_cases=add_test_cases)

    def test_query_param_sample_weight(self):
        X = self.query_default_params_clf["X"]
        test_cases = [
            ("string", ValueError),
            (X, ValueError),
            (np.empty((len(X) - 1)), ValueError),
            (np.ones(len(X)), None),
        ]
        super().test_query_param_sample_weight(test_cases)

    def test_query_param_utility_weight(self):
        test_cases = [
            ("string", (ValueError, TypeError)),
            (self.candidates, (ValueError, TypeError)),
            (np.empty(len(self.X)), (ValueError, TypeError)),
        ]
        super().test_query_param_utility_weight(test_cases)

        test_cases = [(np.ones(2), None)]
        self._test_param(
            "query",
            "utility_weight",
            test_cases,
            replace_query_params={"candidates": [[0, 1], [2, 3]]},
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
