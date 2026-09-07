from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np

from skactiveml.base import SkactivemlRegressor
from skactiveml.pool import GreedySamplingX, GreedySamplingTarget
from skactiveml.pool._greedy_sampling import (
    _greedy_sampling,
    _measure_distance,
)
from skactiveml.regressor import NICKernelRegressor, SklearnRegressor
from skactiveml.tests.template_query_strategy import (
    TemplateSingleAnnotatorPoolQueryStrategy,
)
from skactiveml.tests.utils import assert_no_query_state
from skactiveml.utils import MISSING_LABEL, is_labeled, rand_argmax
from sklearn.gaussian_process import GaussianProcessRegressor


class TestGreedySamplingX(
    TemplateSingleAnnotatorPoolQueryStrategy, unittest.TestCase
):
    def setUp(self):
        query_default_params_reg = {
            "X": np.array([[1, 2], [5, 8], [8, 4], [5, 4]]),
            "y": np.array([1.5, -1.2, MISSING_LABEL, MISSING_LABEL]),
        }
        query_default_params_clf = {
            "X": np.array([[1, 2], [5, 8], [8, 4], [5, 4]]),
            "y": np.array([0, 1, MISSING_LABEL, MISSING_LABEL]),
        }
        params_clf_multilabel = {
            "X": np.array([[1, 2], [5, 8], [8, 4], [5, 4]], dtype=float),
            "y": np.array(
                [
                    [0.0, 1.0],
                    [1.0, 0.0],
                    [MISSING_LABEL, MISSING_LABEL],
                    [MISSING_LABEL, MISSING_LABEL],
                ]
            ),
        }
        super().setUp(
            qs_class=GreedySamplingX,
            init_default_params={},
            query_default_params_reg=query_default_params_reg,
            query_default_params_clf=query_default_params_clf,
            query_default_params_clf_multilabel=params_clf_multilabel,
        )

    def test_init_param_metric(self):
        test_cases = [
            (np.nan, TypeError),
            ("illegal", TypeError),
            (1.1, TypeError),
            ("euclidean", None),
        ]
        self._test_param("init", "metric", test_cases)

    def test_init_param_metric_dict(self):
        test_cases = [
            (np.nan, TypeError),
            ("illegal", TypeError),
            ({"test": 2}, TypeError),
            ({"X_norm_squared": np.zeros(3)}, ValueError),
            ({"X_norm_squared": np.zeros((2, 2))}, ValueError),
            ({}, None),
        ]
        n_samples = len(self.query_default_params_reg["X"])
        for query_params in [{}, {"y": np.full(n_samples, MISSING_LABEL)}]:
            self._test_param(
                "init",
                "metric_dict",
                test_cases,
                replace_query_params=query_params,
            )

    def test_query(self):
        X = np.arange(7).reshape(7, 1)
        y = np.append([1], np.full(6, MISSING_LABEL))

        qs = GreedySamplingX()
        utilities = qs.query(X, y, return_utilities=True)[1][0]
        np.testing.assert_array_equal(
            utilities, np.append([MISSING_LABEL], np.arange(1, 7))
        )

    def test_query_matches_dense_reference(self):
        rng = np.random.RandomState(4)
        X = rng.normal(size=(24, 3))
        for n_labeled in [0, 5]:
            y = np.full(len(X), np.nan)
            y[[1, 3, 6, 8, 10][:n_labeled]] = 0
            available = np.flatnonzero(np.isnan(y))
            for candidates in [
                None,
                available[::-2],
                available[:1],
                rng.normal(size=(8, 3)),
            ]:
                for metric, metric_dict in [
                    ("euclidean", {"squared": True}),
                    ("manhattan", {}),
                    ("cosine", {}),
                    ("nan_euclidean", {}),
                ]:
                    with self.subTest(
                        n_labeled=n_labeled,
                        candidates=candidates,
                        metric=metric,
                    ):
                        qs = GreedySamplingX(
                            metric=metric,
                            metric_dict=metric_dict,
                            random_state=42,
                        )
                        kwargs = dict(
                            X=X,
                            y=y,
                            candidates=candidates,
                            batch_size=min(
                                5,
                                (
                                    len(available)
                                    if candidates is None
                                    else len(candidates)
                                ),
                            ),
                            return_utilities=True,
                        )
                        with patch(
                            "skactiveml.pool._greedy_sampling."
                            "_greedy_sampling",
                            _dense_greedy_reference,
                        ):
                            expected = qs.query(**kwargs)
                        actual = qs.query(**kwargs)
                        np.testing.assert_array_equal(actual[0], expected[0])
                        np.testing.assert_allclose(
                            actual[1], expected[1], rtol=1e-12, atol=1e-12
                        )


class TestGreedySamplingTarget(
    TemplateSingleAnnotatorPoolQueryStrategy, unittest.TestCase
):
    def setUp(self):
        query_default_params_reg = {
            "X": np.array([[1, 2], [5, 8], [8, 4], [5, 4]]),
            "y": np.array([1.5, -1.2, MISSING_LABEL, MISSING_LABEL]),
            "reg": NICKernelRegressor(),
        }
        super().setUp(
            qs_class=GreedySamplingTarget,
            init_default_params={},
            query_default_params_reg=query_default_params_reg,
        )

    def test_target_contract(self):
        strategy = GreedySamplingTarget()

        self.assertEqual(strategy.target_type, "auto")
        self.assertEqual(
            strategy._target_capabilities,
            frozenset({("regression", "single-output", "single-annotator")}),
        )

    def test_multi_output_capability_failure_precedes_acquisition_state(self):
        X = np.arange(12, dtype=float).reshape(6, 2)
        y = np.array(
            [
                [0.0, 1.0],
                [1.0, 2.0],
                [MISSING_LABEL, MISSING_LABEL],
                [MISSING_LABEL, MISSING_LABEL],
                [MISSING_LABEL, MISSING_LABEL],
                [MISSING_LABEL, MISSING_LABEL],
            ]
        )
        reg = SklearnRegressor(
            GaussianProcessRegressor(), target_type="multi-output"
        )
        strategy = GreedySamplingTarget()

        with self.assertRaisesRegex(ValueError, "does not support"):
            strategy.query(X, y, reg, fit_reg=False)

        assert_no_query_state(self, strategy)

    def test_query_effective_defaults_match_explicit_parameters(self):
        query_kwargs = deepcopy(self.query_default_params_reg)
        effective_defaults = {"method": "GSi"}
        reference = GreedySamplingTarget(
            random_state=0, **effective_defaults
        ).query(
            **query_kwargs,
            return_utilities=True,
        )
        for explicit in [False, True]:
            with self.subTest(explicit=explicit):
                parameters = deepcopy(effective_defaults) if explicit else {}
                qs = GreedySamplingTarget(random_state=0, **parameters)
                for _ in range(2):
                    indices, utilities = qs.query(
                        **query_kwargs,
                        return_utilities=True,
                    )
                    np.testing.assert_array_equal(indices, reference[0])
                    np.testing.assert_allclose(utilities, reference[1])

    def test_init_param_x_metric(self):
        test_cases = [
            (np.nan, TypeError),
            ("illegal", TypeError),
            (1.1, TypeError),
            ("euclidean", None),
        ]
        self._test_param("init", "x_metric", test_cases)

    def test_init_param_x_metric_dict(self):
        test_cases = [
            (np.nan, TypeError),
            ("illegal", TypeError),
            ({"test": 2}, TypeError),
            ({}, None),
        ]
        self._test_param("init", "x_metric_dict", test_cases)

    def test_init_param_y_metric(self):
        test_cases = [
            (np.nan, TypeError),
            ("illegal", TypeError),
            (1.1, TypeError),
            ("euclidean", None),
        ]
        self._test_param("init", "y_metric", test_cases)

    def test_init_param_y_metric_dict(self):
        test_cases = [
            (np.nan, TypeError),
            ("illegal", TypeError),
            ({"test": 2}, TypeError),
            ({}, None),
        ]
        self._test_param("init", "y_metric_dict", test_cases)

    def test_init_param_method(self):
        test_cases = [
            (np.nan, TypeError),
            ("illegal", TypeError),
            ({"test": 2}, TypeError),
            ("GSy", None),
            ("GSi", None),
        ]
        self._test_param("init", "method", test_cases)

    def test_init_param_n_GSx_samples(self):
        test_cases = [
            (np.nan, TypeError),
            (1.5, TypeError),
            ({"test": 2}, TypeError),
            (0, None),
            (10, None),
        ]
        self._test_param("init", "n_GSx_samples", test_cases)

    def test_query_param_reg(self):
        test_cases = [
            (NICKernelRegressor(), None),
            (GaussianProcessRegressor(), TypeError),
            (SklearnRegressor(GaussianProcessRegressor()), None),
        ]
        super().test_query_param_reg(test_cases=test_cases)

    def test_query(self):
        X = (1 / 2 * np.arange(2 * 7) + 3.7).reshape(7, 2)
        y = [MISSING_LABEL, MISSING_LABEL, MISSING_LABEL, 0, 0, 0, 0]

        class ZeroRegressor(SkactivemlRegressor):
            def fit(self, *args, **kwargs):
                return self

            def predict(self, X):
                return np.zeros(len(X))

        reg = ZeroRegressor()
        for method in ["GSy", "GSi"]:
            qs = GreedySamplingTarget(random_state=42, method=method)
            utilities = qs.query(X, y, reg, return_utilities=True)[1][0]
            np.testing.assert_array_equal(
                utilities, np.where(is_labeled(y), np.nan, 0)
            )

    def test_query_seed_controls_tied_batches(self):
        X = np.zeros((8, 2))
        y = np.full(len(X), np.nan)
        for n_gsx in [0, 2, 4]:
            with self.subTest(n_gsx=n_gsx):
                qs = GreedySamplingTarget(random_state=42, n_GSx_samples=n_gsx)
                global_state = np.random.get_state()
                try:
                    np.random.seed(1)
                    first = qs.query(X, y, NICKernelRegressor(), batch_size=4)
                    np.random.seed(2)
                    second = qs.query(X, y, NICKernelRegressor(), batch_size=4)
                finally:
                    np.random.set_state(global_state)
                np.testing.assert_array_equal(first, second)

    def test_query_matches_dense_reference(self):
        rng = np.random.RandomState(8)
        X = rng.normal(size=(24, 3))
        for n_labeled in [0, 3, 7]:
            y = np.full(len(X), np.nan)
            y[:n_labeled] = rng.normal(size=n_labeled)
            available = np.flatnonzero(np.isnan(y))
            for candidates in [None, available[::-2], rng.normal(size=(8, 3))]:
                for method in ["GSy", "GSi"]:
                    with self.subTest(
                        n_labeled=n_labeled,
                        candidates=candidates,
                        method=method,
                    ):
                        qs = GreedySamplingTarget(
                            method=method,
                            random_state=42,
                            n_GSx_samples=5,
                            x_metric="manhattan",
                            y_metric="euclidean",
                            y_metric_dict={"squared": True},
                        )
                        kwargs = dict(
                            X=X,
                            y=y,
                            candidates=candidates,
                            reg=NICKernelRegressor(),
                            batch_size=6,
                            return_utilities=True,
                        )
                        with patch(
                            "skactiveml.pool._greedy_sampling."
                            "_greedy_sampling",
                            _dense_greedy_reference,
                        ):
                            expected = qs.query(**kwargs)
                        actual = qs.query(**kwargs)
                        np.testing.assert_array_equal(actual[0], expected[0])
                        np.testing.assert_allclose(
                            actual[1], expected[1], rtol=1e-12, atol=1e-12
                        )


class TestGreedySamplingHelper(unittest.TestCase):
    def test_chunked_initialization_with_precomputed_candidate_norms(self):
        rng = np.random.RandomState(42)
        X = rng.normal(size=(270, 3))
        y = rng.normal(size=len(X))
        for shape in [(len(X),), (len(X), 1), (1, len(X))]:
            with self.subTest(shape=shape):
                kwargs = dict(
                    X_cand=X,
                    X=X,
                    y_cand=y,
                    y=y,
                    sample_indices=np.arange(len(X)),
                    selected_indices=np.array([], dtype=int),
                    candidate_indices=np.arange(len(X)),
                    batch_size=3,
                    random_state=42,
                    method="xy",
                    metric_dict_x={
                        "X_norm_squared": np.sum(X**2, axis=1).reshape(shape)
                    },
                    metric_dict_y={"X_norm_squared": (y**2).reshape(shape)},
                )
                actual = _greedy_sampling(**kwargs)
                expected = _dense_greedy_reference(**kwargs)
                np.testing.assert_array_equal(actual[0], expected[0])
                np.testing.assert_allclose(actual[1], expected[1])

    def test_supplied_center_norm_is_retained(self):
        X = np.arange(6, dtype=float).reshape(-1, 1)
        _, utilities = _greedy_sampling(
            X[1:],
            X,
            np.arange(6),
            np.array([0]),
            np.arange(1, 6),
            batch_size=3,
            random_state=42,
            method="x",
            metric_dict_x={"Y_norm_squared": np.array([0.0])},
        )
        np.testing.assert_array_equal(utilities[0], np.arange(1, 6))
        # The supplied center norm is deliberately reused by sklearn even
        # after choosing a different center, as in the original helper.
        np.testing.assert_array_equal(utilities[1, :4], np.zeros(4))
        self.assertTrue(np.isnan(utilities[1, 4]))
        self.assertTrue(np.all(np.nan_to_num(utilities[2]) == 0))

    def test_distances_are_updated_only_between_selections(self):
        X = np.arange(18, dtype=float).reshape(6, 3)
        with patch(
            "skactiveml.pool._greedy_sampling._measure_distance",
            wraps=_measure_distance,
        ) as distance:
            _greedy_sampling(
                X[1:],
                X,
                np.arange(6),
                np.array([0]),
                np.arange(1, 6),
                batch_size=3,
                random_state=42,
                method="x",
            )
        self.assertEqual(distance.call_count, 3)

    def test_chunked_initialization_and_ties(self):
        rng = np.random.RandomState(42)
        X = rng.randint(-3, 4, size=(1050, 3)).astype(float)
        y = rng.randint(-3, 4, size=len(X)).astype(float)
        for n_labeled in [0, 520]:
            candidates = np.arange(n_labeled, len(X))[::-1]
            for method in ["x", "y", "xy"]:
                with self.subTest(n_labeled=n_labeled, method=method):
                    kwargs = dict(
                        X_cand=X[candidates],
                        X=X,
                        y_cand=y[candidates],
                        y=y,
                        sample_indices=np.arange(len(X)),
                        selected_indices=np.arange(n_labeled),
                        candidate_indices=candidates,
                        batch_size=4,
                        random_state=42,
                        method=method,
                        metric_x="manhattan",
                    )
                    actual = _greedy_sampling(**kwargs)
                    expected = _dense_greedy_reference(**kwargs)
                    np.testing.assert_array_equal(actual[0], expected[0])
                    np.testing.assert_allclose(actual[1], expected[1])


def _dense_greedy_reference(
    X_cand,
    X,
    sample_indices,
    selected_indices,
    candidate_indices,
    batch_size,
    y_cand=None,
    y=None,
    random_state=None,
    method=None,
    **kwargs,
):
    """Score against all selected centers afresh at each batch step."""
    centers = np.union1d(selected_indices, candidate_indices)
    if not len(selected_indices):
        centers = np.union1d(centers, sample_indices)
    distances = np.full((len(X_cand), len(X)), np.nan)
    distances[:, centers] = _measure_distance(
        centers, X_cand, y_cand, X, y, method=method, **kwargs
    )
    remaining = np.arange(len(X_cand))
    selected = np.array(selected_indices)
    query_indices = np.empty(batch_size, dtype=int)
    utilities = np.full((batch_size, len(X_cand)), np.nan)
    for i in range(batch_size):
        if len(selected):
            values = np.min(distances[remaining][:, selected], axis=1)
        else:
            values = -np.sum(distances[remaining][:, sample_indices], axis=1)
        utilities[i, remaining] = values
        position = rand_argmax(values, random_state=random_state)[0]
        query_indices[i] = remaining[position]
        selected = np.append(selected, candidate_indices[remaining[position]])
        remaining = np.delete(remaining, position)
    return query_indices, utilities
