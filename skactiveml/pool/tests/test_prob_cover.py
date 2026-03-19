import unittest

import numpy as np
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.datasets import make_blobs
from sklearn.metrics import pairwise_distances, euclidean_distances
from skactiveml.pool import ProbCover
from skactiveml.exceptions import MappingError
from skactiveml.utils import MISSING_LABEL
from skactiveml.tests.template_query_strategy import (
    TemplateSingleAnnotatorPoolQueryStrategy,
)


class TestProbCover(
    TemplateSingleAnnotatorPoolQueryStrategy, unittest.TestCase
):
    def setUp(self):
        query_default_params_clf = {
            "X": np.random.RandomState(0).uniform(
                size=(1000, 10), low=-0.5, high=0.5
            ),
            "y": np.hstack([[0, 1], np.full(998, MISSING_LABEL)]),
        }
        qs_params_clf_multilabel = {
            "X": np.random.RandomState(0).uniform(
                size=(1000, 10), low=-0.5, high=0.5
            ),
            "y": np.vstack(
                [
                    [0.0, 1.0],
                    [1.0, 0.0],
                    *[
                        np.full(2, MISSING_LABEL, dtype=float)
                        for _ in range(998)
                    ],
                ]
            ),
        }
        super().setUp(
            qs_class=ProbCover,
            init_default_params={
                "random_state": 0,
                "deltas": np.linspace(0.2, 2.2, 100),
                "cluster_algo_dict": {"n_init": 10},
            },
            query_default_params_clf=query_default_params_clf,
            query_default_params_clf_multilabel=qs_params_clf_multilabel,
        )

    def test_init_param_n_classes(self, test_cases=None):
        test_cases = [] if test_cases is None else test_cases
        test_cases += [
            (None, None),
            (2, None),
            (10, None),
            ("string", TypeError),
            (1.5, TypeError),
            (0, ValueError),
            (1, ValueError),
        ]
        self._test_param("init", "n_classes", test_cases)

    def test_init_param_deltas(self, test_cases=None):
        test_cases = [] if test_cases is None else test_cases
        test_cases += [
            ([0.5], None),
            ([0.0, 1.0, 2.0], None),
            ([500], None),
            (np.array([0.0, 1.0, 2.0]), None),
            ([-1], ValueError),
            (0.5, ValueError),
            (-1, ValueError),
            ("string", ValueError),
        ]
        self._test_param("init", "deltas", test_cases)

    def test_init_param_alpha(self, test_cases=None):
        test_cases = [] if test_cases is None else test_cases
        test_cases += [
            (0.5, None),
            (0.01, None),
            (0.99, None),
            (1.0, ValueError),
            (0.0, ValueError),
            ("string", TypeError),
        ]
        self._test_param("init", "alpha", test_cases)

    def test_init_param_cluster_algo(self, test_cases=None):
        test_cases = [] if test_cases is None else test_cases
        test_cases += [
            (1, TypeError),
            ("string", TypeError),
            (None, TypeError),
            (ProbCover, TypeError),
            (SpectralClustering, None),
            (KMeans, None),
        ]
        self._test_param(
            "init",
            "cluster_algo",
            test_cases,
            replace_init_params={"random_state": 0},
        )

    def test_init_param_cluster_algo_dict(self, test_cases=None):
        test_cases = [] if test_cases is None else test_cases
        test_cases += [
            (1, TypeError),
            ("string", TypeError),
            (None, None),
            ({}, None),
            ({"n_init": "auto", "random_state": 0}, None),
        ]
        self._test_param("init", "cluster_algo_dict", test_cases)

    def test_init_param_n_cluster_param_name(self, test_cases=None):
        test_cases = [] if test_cases is None else test_cases
        test_cases += [
            (1, TypeError),
            ("string", TypeError),
            (None, TypeError),
            ("n_clusters", None),
        ]
        self._test_param("init", "n_cluster_param_name", test_cases)

    def test_init_param_distance_func(self, test_cases=None):
        test_cases = [] if test_cases is None else test_cases
        test_cases += [
            (pairwise_distances, None),
            (euclidean_distances, None),
            (1, TypeError),
            ("string", TypeError),
        ]
        self._test_param("init", "distance_func", test_cases)

    def test_query_param_update(self, test_cases=None):
        test_cases = [] if test_cases is None else test_cases
        test_cases += [
            (False, None),
            (True, None),
            (1, TypeError),
            ("string", TypeError),
        ]
        self._test_param("query", "update", test_cases)

    def test_query(self):
        X, y_true = make_blobs(n_samples=50, centers=4, random_state=0)
        y_true = y_true.astype(float)
        D = pairwise_distances(X)

        def distance_func(X):
            return D

        for candidates in [None, np.arange(10, 30)]:
            qs = ProbCover(
                deltas=np.linspace(0.5, 2, 100),
                cluster_algo_dict={"random_state": 0},
                random_state=42,
            )
            qs_metric = ProbCover(
                deltas=np.linspace(0.5, 2, 100),
                cluster_algo_dict={"random_state": 0},
                distance_func=distance_func,
                random_state=42,
            )
            qs_cluster = ProbCover(
                deltas=np.linspace(0.5, 2, 100),
                random_state=42,
                cluster_algo_dict={
                    "init": "random",
                    "max_iter": 1,
                    "random_state": 0,
                },
            )

            # With the same random state the initial selection is the same.
            y = np.full(50, MISSING_LABEL)
            self.assertEqual(
                qs.query(X, y, candidates=candidates),
                qs.query(X, y, candidates=candidates),
            )

            # Check non-negativity of `delta_max_`.
            self.assertGreaterEqual(qs.delta_max_, 0.0)

            # All utilities are non-negative integers or np.nan.
            is_unlabeled = np.random.RandomState(0).choice(
                [False, True], size=(len(X),), replace=True
            )
            y = y_true.copy()
            y[is_unlabeled] = np.nan
            query_indices, utilities = qs.query(
                X,
                y,
                update=True,
                candidates=candidates,
                batch_size=2,
                return_utilities=True,
            )
            utilities_copy = utilities.copy()
            is_nan = np.isnan(utilities)
            utilities_copy[is_nan] = 0.0
            is_integer = np.mod(utilities_copy, 1) == 0
            is_non_negative = utilities_copy >= 0
            self.assertTrue(np.logical_and(is_integer, is_non_negative).all())

            # Check functionality of `pairwise_distances_dict`.
            query_indices_metric, utilities_metric = qs_metric.query(
                X,
                y,
                batch_size=2,
                candidates=candidates,
                return_utilities=True,
            )
            np.testing.assert_array_equal(query_indices_metric, query_indices)
            np.testing.assert_array_equal(utilities_metric, utilities)

            # Check consistence of `delta_max_`.
            self.assertEqual(qs.delta_max_, qs_metric.delta_max_)

            # Check functionality of `cluster_algo_dict`.
            _, _ = qs_cluster.query(X, y, candidates=candidates, batch_size=2)
            self.assertTrue(qs_cluster.delta_max_ != qs.delta_max_)

        # Check whether error is raised for `candidates` being not in `X`.
        self.assertRaises(MappingError, qs.query, X, y, candidates=X)

    def test_query_multilabel_n_classes_inference(self):
        X = np.array(
            [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]],
            dtype=float,
        )
        y = np.array(
            [
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [MISSING_LABEL, MISSING_LABEL],
                [MISSING_LABEL, MISSING_LABEL],
                [MISSING_LABEL, MISSING_LABEL],
            ]
        )

        class ClusterRecorder:
            seen_n_clusters = None

            def __init__(self, n_clusters):
                self.n_clusters = n_clusters
                type(self).seen_n_clusters = n_clusters

            def fit_predict(self, X):
                return np.zeros(len(X), dtype=int)

        qs = ProbCover(
            deltas=np.array([0.2, 0.4]),
            cluster_algo=ClusterRecorder,
            cluster_algo_dict={},
            random_state=0,
        )
        qs.query(X, y, update=True)
        self.assertEqual(ClusterRecorder.seen_n_clusters, 3)

    def test_query_with_default_deltas(self):
        X = np.random.RandomState(0).uniform(size=(20, 2), low=-0.5, high=0.5)
        y = np.hstack([[0, 1], np.full(18, MISSING_LABEL)])
        qs = ProbCover(deltas=None, cluster_algo_dict={"random_state": 0})
        query_idx, utilities = qs.query(
            X, y, batch_size=2, return_utilities=True
        )
        self.assertEqual(query_idx.shape, (2,))
        self.assertEqual(utilities.shape, (2, len(X)))
        self.assertGreaterEqual(qs.delta_max_, 0.0)
