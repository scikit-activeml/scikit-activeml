import unittest

import numpy as np
from sklearn.cluster import KMeans, SpectralClustering
from skactiveml.pool._typi_clust import TypiClust
from skactiveml.utils import MISSING_LABEL
from skactiveml.tests.template_query_strategy import (
    TemplateSingleAnnotatorPoolQueryStrategy,
)


class TestTypiClust(
    TemplateSingleAnnotatorPoolQueryStrategy, unittest.TestCase
):
    def setUp(self):
        query_default_params_clf = {
            "X": np.random.RandomState(0).uniform(
                size=(1000, 10), low=-0.5, high=0.5
            ),
            "y": np.hstack([[0, 1], np.full(998, MISSING_LABEL)]),
        }
        params_clf_multilabel = {
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
        query_default_params_reg = {
            "X": np.random.RandomState(0).uniform(
                size=(1000, 10), low=-0.5, high=0.5
            ),
            "y": np.hstack([[1.1, 2.1], np.full(998, MISSING_LABEL)]),
        }
        super().setUp(
            qs_class=TypiClust,
            init_default_params={
                "random_state": 0,
                "cluster_algo_dict": {"n_init": 1},
            },
            query_default_params_clf=query_default_params_clf,
            query_default_params_reg=query_default_params_reg,
            query_default_params_clf_multilabel=params_clf_multilabel,
        )

    def test_degenerate_clusters_produce_unique_eligible_batches(self):
        X = np.zeros((5, 1))
        for target_type in ["single-output", "multi-label"]:
            for labeled in [False, True]:
                for candidates in [None, [1, 3, 4]]:
                    with self.subTest(
                        target_type=target_type,
                        labeled=labeled,
                        candidates=candidates,
                    ):
                        shape = (5, 2) if target_type == "multi-label" else 5
                        y = np.full(shape, np.nan)
                        if labeled:
                            y[0] = 0
                        eligible = (
                            np.arange(int(labeled), 5)
                            if candidates is None
                            else np.array(candidates)
                        )
                        qs = TypiClust(random_state=0, target_type=target_type)
                        batch_size = min(4, len(eligible))
                        indices, utilities = qs.query(
                            X,
                            y,
                            candidates=candidates,
                            batch_size=batch_size,
                            return_utilities=True,
                        )
                        self.assertEqual(len(np.unique(indices)), batch_size)
                        self.assertTrue(np.isin(indices, eligible).all())
                        for i, idx in enumerate(indices):
                            self.assertTrue(np.isfinite(utilities[i, idx]))
                            self.assertTrue(
                                np.isnan(utilities[i, indices[:i]]).all()
                            )
                            self.assertTrue(
                                np.isnan(
                                    utilities[
                                        i, ~np.isin(np.arange(5), eligible)
                                    ]
                                ).all()
                            )

    def test_uncovered_clusters_without_candidates_are_skipped(self):
        X = np.array([[0.0], [0.1], [0.2], [0.3], [10.0], [10.1]])
        indices, utilities = TypiClust(random_state=0).query(
            X,
            np.full(6, np.nan),
            candidates=[4, 5],
            batch_size=2,
            return_utilities=True,
        )
        np.testing.assert_array_equal(np.sort(indices), [4, 5])
        self.assertTrue(np.isfinite(utilities[np.arange(2), indices]).all())

    def test_init_param_cluster_algo(self, test_cases=None):
        test_cases = [] if test_cases is None else test_cases
        test_cases += [
            (1, TypeError),
            ("string", TypeError),
            (None, TypeError),
            (TypiClust, TypeError),
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

    def test_init_param_k(self, test_cases=None):
        test_cases = [] if test_cases is None else test_cases
        test_cases += [
            ("string", TypeError),
            (None, TypeError),
            (10, None),
            (1, None),
            (1.5, TypeError),
            (0, ValueError),
        ]
        self._test_param("init", "k", test_cases)

    def test_query(self):
        # test case 1: with the same random state the init pick up
        # is the same

        random_state = np.random.RandomState(42)

        typi_clust_1 = TypiClust(random_state=42, k=3)

        X = random_state.choice(5, size=(10, 2))
        y = np.full(10, MISSING_LABEL)

        self.assertEqual(typi_clust_1.query(X, y), typi_clust_1.query(X, y))

        # test case 2: all utilities are not negative or np.nan
        y_1 = np.hstack([[0], np.full(9, MISSING_LABEL)])
        _, utilities = typi_clust_1.query(
            X, y_1, batch_size=2, return_utilities=True
        )
        for u in utilities:
            for i in u:
                if not np.isnan(i) and i < 0:
                    self.assertTrue(np.isneginf(i))

        # test case 3: for an uncovered cluster with 2 samples, the utilities
        # with k=1 is for all samples are the same
        X_3 = np.array([[1, 2], [3, 4]])
        y_3 = np.full(2, MISSING_LABEL)
        typi_clust_3 = TypiClust(random_state=42, k=1)
        _, utilities_3 = typi_clust_3.query(
            X_3, y_3, batch_size=1, return_utilities=True
        )
        for u in utilities_3:
            for i in u:
                if not np.isnan(i):
                    self.assertEqual(i, u[0])
                else:
                    self.assertTrue(np.isnan(i))

        # test case 4: for candidates.ndim = 1
        candidates = np.arange(1, 5)
        _, utilities_4 = typi_clust_1.query(
            X, y_1, batch_size=1, candidates=candidates, return_utilities=True
        )
        for u in utilities_4:
            for i in u:
                if not np.isnan(i) and i < 0:
                    self.assertTrue(np.isneginf(i))
        self.assertEqual(10, utilities_4.shape[1])
        self.assertEqual(1, utilities_4.shape[0])

        # test case 5: duplicate samples
        typi_clust_1 = TypiClust(random_state=0, k=3)
        X_dup = X[[1, 2, 0, 0, 0, 0, 0, 0, 0, 0]]
        y_dup = [0, 1] + [np.nan] * (len(X_dup) - 2)
        typi_clust_1.query(X_dup, y_dup, batch_size=5, return_utilities=True)
