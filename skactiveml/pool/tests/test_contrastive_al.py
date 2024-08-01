import unittest

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.datasets import make_blobs
from skactiveml.pool import ContrastiveAL
from skactiveml.classifier import ParzenWindowClassifier, SklearnClassifier
from skactiveml.utils import MISSING_LABEL
from skactiveml.tests.template_query_strategy import (
    TemplateSingleAnnotatorPoolQueryStrategy,
)
from skactiveml.tests.utils import (
    ParzenWindowClassifierEmbedding,
    ParzenWindowClassifierTuple,
)


class TestContrastiveAL(
    TemplateSingleAnnotatorPoolQueryStrategy, unittest.TestCase
):
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
        super().setUp(
            qs_class=ContrastiveAL,
            init_default_params={"nearest_neighbors_dict": {"n_neighbors": 2}},
            query_default_params_clf=self.query_default_params_clf,
        )

    def test_init_param_nearest_neighbors_dict(self, test_cases=None):
        test_cases = [] if test_cases is None else test_cases
        test_cases += [
            (1, TypeError),
            ("string", TypeError),
            (None, None),
            ({}, None),
            ({"n_neighbors": 2}, None),
        ]
        self._test_param("init", "nearest_neighbors_dict", test_cases)

    def test_init_param_eps(self):
        test_cases = [
            (0, ValueError),
            (1e-3, None),
            (0.1, None),
            ("1", TypeError),
            (1, ValueError),
        ]
        self._test_param("init", "eps", test_cases)

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
        for candidates in [None, np.arange(10, 30), X[10:30]]:
            prev_clf_utilities = None
            for clf in [clf_orig, clf_emb, clf_tuple]:
                qs_1 = ContrastiveAL(
                    nearest_neighbors_dict={"n_neighbors": 1},
                    random_state=42,
                )
                qs_5 = ContrastiveAL(
                    nearest_neighbors_dict={"n_neighbors": 5},
                    random_state=42,
                )

                # Without labeled samples, the initial selection is the same,
                # even for different nearest neighbor numbers.
                y = np.full(len(y_true), MISSING_LABEL)
                query_indices_1, utilities_1 = qs_1.query(
                    X, y, clf=clf, candidates=candidates, return_utilities=True
                )
                query_indices_5, utilities_5 = qs_5.query(
                    X, y, clf=clf, candidates=candidates, return_utilities=True
                )
                np.testing.assert_array_equal(query_indices_1, query_indices_5)
                np.testing.assert_array_equal(utilities_1, utilities_5)
                expected_utilities = np.zeros_like(utilities_1)
                if candidates is not None and candidates.ndim == 1:
                    expected_utilities[:, :10] = np.nan
                    expected_utilities[:, 30:] = np.nan
                np.testing.assert_array_equal(utilities_1, expected_utilities)

                # All utilities are non-negative values or np.nan.
                is_unlabeled = np.random.RandomState(0).choice(
                    [False, True], size=(len(X),), replace=True
                )
                y = y_true.copy()
                y[is_unlabeled] = np.nan
                prev_utilities = None
                for qs in [qs_1, qs_5]:
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
                    is_non_negative = utilities_copy >= 0
                    self.assertTrue(is_non_negative.all())
                    if prev_utilities is not None:
                        # Check that different numbers of nearest neighbors
                        # lead to different utilities.
                        self.assertTrue(
                            np.nansum(utilities) != np.nansum(prev_utilities)
                        )
                    else:
                        # Check that the utilties are consistent across
                        # classifiers using different syntax to return
                        # embeddings.
                        if prev_clf_utilities is not None:
                            np.testing.assert_array_equal(
                                prev_clf_utilities, utilities
                            )
                        prev_clf_utilities = utilities

                    prev_utilities = utilities
