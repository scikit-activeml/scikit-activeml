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
            "X": np.linspace(0, 1, 20).reshape(10, 2),
            "y": np.hstack([[0, 1], np.full(8, MISSING_LABEL)]),
        }
        query_default_params_reg = {
            "X": np.linspace(0, 1, 20).reshape(10, 2),
            "y": np.hstack([[1.1, 2.1], np.full(8, MISSING_LABEL)]),
        }
        cluster_dict = {"random_state": 0, "n_init": 1}
        super().setUp(
            qs_class=TypiClust,
            init_default_params={"cluster_algo_dict": cluster_dict},
            query_default_params_clf=query_default_params_clf,
            query_default_params_reg=query_default_params_reg,
        )

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
