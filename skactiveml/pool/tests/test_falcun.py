import unittest

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.datasets import make_blobs
from skactiveml.pool import Falcun
from skactiveml.classifier import ParzenWindowClassifier, SklearnClassifier
from skactiveml.utils import MISSING_LABEL
from skactiveml.tests.template_query_strategy import (
    TemplateSingleAnnotatorPoolQueryStrategy,
)


class TestFalcun(TemplateSingleAnnotatorPoolQueryStrategy, unittest.TestCase):
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
            "clf": ParzenWindowClassifier(
                classes=self.classes,
            ),
        }
        super().setUp(
            qs_class=Falcun,
            init_default_params={"gamma": 1},
            query_default_params_clf=self.query_default_params_clf,
        )

    def test_init_param_gamma(self):
        test_cases = [
            (0, None),
            (1e-3, None),
            (0.1, None),
            ("1", TypeError),
            (-1, ValueError),
            (-0.5, ValueError),
        ]
        self._test_param("init", "gamma", test_cases)

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
        clf = ParzenWindowClassifier(classes=np.unique(y_true))
        cand_iter = zip([None, np.arange(10, 30), X[10:30]], [len(X), 20, 20])
        for candidates, n_candidates in cand_iter:
            qs_1 = Falcun(random_state=42)
            qs_5 = Falcun(random_state=42, gamma=5)

            # Without labeled samples, the initial selection is the same,
            # even for different `gamma` values.
            y = np.full(len(y_true), MISSING_LABEL)
            query_indices_1, utilities_1 = qs_1.query(
                X, y, clf=clf, candidates=candidates, return_utilities=True
            )
            query_indices_5, utilities_5 = qs_5.query(
                X, y, clf=clf, candidates=candidates, return_utilities=True
            )
            np.testing.assert_array_equal(query_indices_1, query_indices_5)
            np.testing.assert_array_equal(utilities_1, utilities_5)
            expected_utilities = np.ones_like(utilities_1)
            expected_utilities /= n_candidates
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
                np.testing.assert_allclose(
                    np.nansum(utilities, axis=1), [1, 1]
                )
                if prev_utilities is not None:
                    self.assertTrue((utilities_copy != prev_utilities).sum())
                    # Check that different `gamma` values lead to the same
                    # selection regarding the batch's first sample.
                    max_idx = utilities_copy[0].argmax()
                    prev_max_idx = prev_utilities[0].argmax()
                    self.assertTrue(max_idx, prev_max_idx)
                    # Check that different `gamma` values lead to different
                    # utilities.
                    self.assertGreater(
                        np.abs(utilities_copy[0] - prev_utilities[0]).sum(),
                        0.1,
                    )
                else:
                    prev_utilities = utilities_copy
