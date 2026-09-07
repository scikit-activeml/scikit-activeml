from copy import deepcopy
import itertools
from math import exp, factorial, lgamma
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB

from skactiveml.classifier import ParzenWindowClassifier, SklearnClassifier
from skactiveml.pool import ProbabilisticAL, cost_reduction
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

    def test_query_preserves_constructor_parameters(self):
        super().test_query_preserves_constructor_parameters()
        for metric_dict in [None, {}, {"gamma": 0.5}]:
            super().test_query_preserves_constructor_parameters(
                init_param_overrides={
                    "metric": "rbf",
                    "metric_dict": metric_dict,
                }
            )

    def test_query_metric_defaults_and_kernel_change(self):
        X = np.array([[1, 2], [5, 8], [8, 4], [5, 4]])
        y = np.array([0, 1, np.nan, np.nan])
        clf = ParzenWindowClassifier(classes=[0, 1]).fit(X, y)
        query_kwargs = dict(
            X=X,
            y=y,
            clf=clf,
            fit_clf=False,
            return_utilities=True,
        )
        for metric_dict in [None, {}, {"gamma": 0.5}]:
            with self.subTest(metric_dict=metric_dict):
                qs = ProbabilisticAL(
                    metric="rbf", metric_dict=metric_dict, random_state=0
                )
                reference = ProbabilisticAL(
                    metric="rbf",
                    random_state=0,
                    metric_dict=(
                        {"gamma": "mean"}
                        if metric_dict is None
                        else deepcopy(metric_dict)
                    ),
                ).query(**query_kwargs)
                for _ in range(2):
                    indices, utilities = qs.query(**query_kwargs)
                    np.testing.assert_array_equal(indices, reference[0])
                    np.testing.assert_allclose(utilities, reference[1])
                if metric_dict is None:
                    qs.set_params(metric="linear")
                    indices, utilities = qs.query(**query_kwargs)
                    reference = ProbabilisticAL(
                        metric="linear", random_state=0
                    ).query(**query_kwargs)
                    np.testing.assert_array_equal(indices, reference[0])
                    np.testing.assert_allclose(utilities, reference[1])

    def test_fitted_multilabel_classifier_rejected_before_state(self):
        self._test_fitted_multilabel_classifier_rejection()

    def test_query_candidate_mapping_matches_closed_form(self):
        X = np.random.RandomState(0).uniform(size=(8, 2))
        y = np.array([0, 1, 0, np.nan, np.nan, np.nan, np.nan, np.nan])
        clf = ParzenWindowClassifier(classes=[0, 1]).fit(X, y)
        for candidates in [None, np.array([7, 3, 5]), X[[7, 3, 5]]]:
            with self.subTest(candidates=candidates):
                params = dict(
                    X=X,
                    y=y,
                    clf=clf,
                    fit_clf=False,
                    candidates=candidates,
                    batch_size=2,
                    return_utilities=True,
                )
                actual = ProbabilisticAL(random_state=0).query(**params)
                with patch(
                    "skactiveml.pool._probabilistic_al.cost_reduction",
                    side_effect=TestCostReduction._reference,
                ):
                    expected = ProbabilisticAL(random_state=0).query(**params)
                np.testing.assert_array_equal(actual[0], expected[0])
                np.testing.assert_allclose(
                    actual[1], expected[1], rtol=1e-9, atol=1e-11
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


class TestCostReduction(unittest.TestCase):
    @staticmethod
    def _reference(k_vec_list, C=None, m_max=2, prior=1e-3):
        """Evaluate the original closed-form integral without batching."""
        n_classes = len(k_vec_list[0])
        C = 1 - np.eye(n_classes) if C is None else C

        def log_beta(alpha):
            return sum(lgamma(a) for a in alpha) - lgamma(sum(alpha))

        gains = []
        for k in k_vec_list:
            alpha = np.asarray(k) + prior
            risks = []
            for m in range(m_max + 1):
                risk = 0.0
                for counts in itertools.product(
                    range(m + 1), repeat=n_classes
                ):
                    if sum(counts) != m:
                        continue
                    counts = np.asarray(counts)
                    decision = np.argmin((k + counts) @ C)
                    coefficient = factorial(m)
                    for count in counts:
                        coefficient /= factorial(count)
                    for c in range(n_classes):
                        indicator = np.eye(n_classes)[c]
                        risk += (
                            coefficient
                            * C[c, decision]
                            * exp(
                                log_beta(alpha + counts + indicator)
                                - log_beta(alpha)
                            )
                        )
                risks.append(risk)
            gains.append(
                max((risks[0] - risks[m]) / m for m in range(1, m_max + 1))
            )
        return np.asarray(gains)

    def test_closed_form_equivalence(self):
        rng = np.random.RandomState(42)
        for n_classes in [2, 3, 5]:
            k = rng.uniform(0, 5, size=(4, n_classes))
            k[0] = 0
            k[1] = 2
            costs = rng.uniform(size=(n_classes, n_classes))
            for m_max in [1, 2, 3]:
                for prior in [1e-3, 1, 10]:
                    for C in [None, costs]:
                        with self.subTest(
                            n_classes=n_classes,
                            m_max=m_max,
                            prior=prior,
                            default_cost=C is None,
                        ):
                            expected = self._reference(k, C, m_max, prior)
                            actual = cost_reduction(k, C, m_max, prior)
                            np.testing.assert_allclose(
                                actual, expected, rtol=1e-9, atol=1e-11
                            )

    def test_default_cost_and_integer_counts(self):
        counts = [[0, 0, 0], [1, 1, 0], [2, 0, 1]]
        for prior in [1e-30, 1, 10]:
            default = cost_reduction(counts, prior=prior)
            explicit = cost_reduction(counts, C=1 - np.eye(3), prior=prior)
            np.testing.assert_allclose(default, explicit, atol=1e-14)

    def test_underflowing_beta_functions(self):
        for count, prior in [(1000, 1), (0, 1000), (1e6, 1e-3)]:
            # With tied binary counts, one acquired label reduces the risk
            # from 1/2 to alpha / (2 * alpha + 1).
            expected = 1 / (4 * (count + prior) + 2)
            actual = cost_reduction([[count, count]], m_max=1, prior=prior)
            np.testing.assert_allclose(actual, expected, rtol=1e-8)
        np.testing.assert_allclose(
            cost_reduction([[0, 0]], m_max=1, prior=1e-30), [0.5]
        )

    def test_candidate_chunks_and_input_preservation(self):
        counts = np.tile(np.arange(5, dtype=float), (10001, 1))
        before = counts.copy()
        expected = self._reference(counts[:1])
        actual = cost_reduction(counts)
        np.testing.assert_allclose(actual, np.repeat(expected, len(counts)))
        np.testing.assert_array_equal(counts, before)
