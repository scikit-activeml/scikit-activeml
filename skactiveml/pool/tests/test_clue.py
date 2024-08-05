import unittest

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.datasets import make_blobs
from sklearn.cluster import SpectralClustering, KMeans, MiniBatchKMeans
from skactiveml.pool import Clue
from skactiveml.classifier import ParzenWindowClassifier, SklearnClassifier
from skactiveml.utils import MISSING_LABEL
from skactiveml.tests.template_query_strategy import (
    TemplateSingleAnnotatorPoolQueryStrategy,
)
from skactiveml.tests.utils import (
    ParzenWindowClassifierEmbedding,
    ParzenWindowClassifierTuple,
)


class TestClue(TemplateSingleAnnotatorPoolQueryStrategy, unittest.TestCase):
    def setUp(self):
        X = np.linspace(0, 1, 20).reshape(10, 2)
        y = np.hstack([[0, 1], np.full(8, MISSING_LABEL)])
        self.classes = [0, 1]
        self.query_default_params_clf = {
            "X": X,
            "y": y,
            "clf": ParzenWindowClassifier(
                classes=self.classes, random_state=42
            ),
        }
        self.query_default_params_clf_embedding = {
            "X": X,
            "y": y,
            "clf": ParzenWindowClassifierEmbedding(
                classes=self.classes,
            ),
        }
        cluster_dict = {"random_state": 0, "n_init": 1}
        super().setUp(
            qs_class=Clue,
            init_default_params={"cluster_algo_dict": cluster_dict},
            query_default_params_clf=self.query_default_params_clf,
        )

    def test_init_param_cluster_algo(self, test_cases=None):
        test_cases = [] if test_cases is None else test_cases
        test_cases += [
            (1, TypeError),
            ("string", TypeError),
            (None, TypeError),
            (SpectralClustering, AttributeError),
            (Clue, TypeError),
            (MiniBatchKMeans, None),
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

    def test_init_param_method(self, test_cases=None):
        test_cases = [] if test_cases is None else test_cases
        test_cases += [
            (1, ValueError),
            ("string", ValueError),
            ("entropy", None),
            ("margin_sampling", None),
            ("least_confident", None),
        ]
        self._test_param("init", "method", test_cases)

    def test_init_param_clf_embedding_flag_name(self, test_cases=None):
        test_cases = [] if test_cases is None else test_cases
        test_cases += [
            (1, TypeError),
            (None, None),
            (False, TypeError),
            (True, TypeError),
            ("return_embeddings", None),
        ]
        self._test_param(
            "init",
            "clf_embedding_flag_name",
            replace_query_params=self.query_default_params_clf_embedding,
            test_cases=test_cases,
        )

    def test_query_param_clf(self, test_cases=None):
        test_cases = [] if test_cases is None else test_cases
        test_cases += [
            (SVC(), TypeError),
            (SklearnClassifier(SVC()), AttributeError),
            (SklearnClassifier(SVC(probability=True)), None),
            (
                SklearnClassifier(LogisticRegression(), classes=self.classes),
                None,
            ),
        ]
        super().test_query_param_clf(test_cases=test_cases)

    def test_query(self):
        X, y_true = make_blobs(n_samples=100, centers=4, random_state=0)
        y_true = y_true.astype(float)
        clf_orig = ParzenWindowClassifier(classes=np.unique(y_true))
        clf_emb = ParzenWindowClassifierEmbedding(classes=np.unique(y_true))
        clf_tuple = ParzenWindowClassifierTuple(classes=np.unique(y_true))
        for candidates in [None, np.arange(10, 30)]:
            prev_clf_utilities = None
            for clf in [clf_orig, clf_emb, clf_tuple]:
                qs_plus = Clue(
                    cluster_algo_dict={"init": "k-means++", "random_state": 0},
                    random_state=42,
                )
                qs_random = Clue(
                    cluster_algo_dict={"init": "random", "random_state": 0},
                    random_state=42,
                )

                # Without labeled samples, the initial selection is the same,
                # even for different clustering initializations.
                y = np.full(len(y_true), MISSING_LABEL)
                query_indices_plus, utilities_plus = qs_plus.query(
                    X, y, clf=clf, candidates=candidates, return_utilities=True
                )
                query_indices_random, utilities_random = qs_random.query(
                    X, y, clf=clf, candidates=candidates, return_utilities=True
                )
                np.testing.assert_array_equal(
                    query_indices_plus, query_indices_random
                )
                np.testing.assert_almost_equal(
                    utilities_plus, utilities_random
                )
                if candidates is not None:
                    self.assertTrue(np.isnan(utilities_plus[0, :10]).all())
                    self.assertTrue(np.isnan(utilities_plus[0, 30:]).all())
                    self.assertTrue((utilities_plus[30:50] <= 0).all())
                else:
                    self.assertTrue((utilities_plus <= 0).all())

                # All utilities are non-positive values or np.nan.
                is_unlabeled = np.random.RandomState(0).choice(
                    [False, True], size=(len(X),), replace=True
                )
                y = y_true.copy()
                y[is_unlabeled] = np.nan
                prev_utilities = None
                for qs in [qs_plus, qs_random]:
                    query_indices, utilities = qs.query(
                        X,
                        y,
                        clf=clf,
                        candidates=candidates,
                        batch_size=2,
                        return_utilities=True,
                    )
                    utilities_copy = utilities.copy()
                    is_nan = np.isnan(utilities)
                    utilities_copy[is_nan] = 0.0
                    is_non_positive = utilities_copy <= 0
                    self.assertTrue(is_non_positive.all())
                    if prev_utilities is not None:
                        # Check that different clustering initializations
                        # lead to different utilities.
                        self.assertTrue(
                            np.nansum(utilities) != np.nansum(prev_utilities)
                        )
                    else:
                        # Check that the utilities are consistent across
                        # classifiers using different syntax to return
                        # embeddings.
                        if prev_clf_utilities is not None:
                            np.testing.assert_array_equal(
                                prev_clf_utilities, utilities
                            )
                        prev_clf_utilities = utilities

                    prev_utilities = utilities
