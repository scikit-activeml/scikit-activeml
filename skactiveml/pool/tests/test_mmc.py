import unittest
import warnings
from unittest.mock import patch

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import GaussianNB

from skactiveml.classifier import ParzenWindowClassifier, SklearnClassifier
from skactiveml.pool import MaxLossReductionMaxConfidence
from skactiveml.pool.tests._multilabel_target_semantics import (
    MultilabelOnlyTargetSemanticsMixin,
)
from skactiveml.tests.template_query_strategy import (
    TemplateSingleAnnotatorPoolQueryStrategy,
)
from skactiveml.utils import MISSING_LABEL, unlabeled_indices


class TestMaxLossReductionMaxConfidence(
    MultilabelOnlyTargetSemanticsMixin,
    TemplateSingleAnnotatorPoolQueryStrategy,
    unittest.TestCase,
):
    def setUp(self):
        self.X = np.linspace(0, 1, 16).reshape(8, 2)
        self.y = np.array(
            [
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [np.nan, np.nan],
                [np.nan, np.nan],
                [np.nan, np.nan],
                [np.nan, np.nan],
                [np.nan, np.nan],
            ]
        )
        self.unld_idx = unlabeled_indices(
            self.y, missing_label=MISSING_LABEL, target_type="multi-label"
        )
        self.clf = SklearnClassifier(
            estimator=MultiOutputClassifier(GaussianNB()),
            classes=[[0, 1], [0, 1]],
            missing_label=MISSING_LABEL,
            proba_format="array",
            random_state=0,
        )
        self.discriminator = ParzenWindowClassifier(
            classes=[0, 1, 2], missing_label=-1, random_state=0
        )
        self.qs = MaxLossReductionMaxConfidence(random_state=0)

        self.strategy_class = MaxLossReductionMaxConfidence
        TemplateSingleAnnotatorPoolQueryStrategy.setUp(
            self,
            qs_class=MaxLossReductionMaxConfidence,
            init_default_params={"random_state": 0},
            query_default_params_clf_multilabel={
                "X": self.X,
                "y": self.y,
                "discriminator": self.discriminator,
                "clf": self.clf,
            },
        )

    def _query_strategy(self, strategy, y, clf, **kwargs):
        return strategy.query(
            self.X,
            y,
            discriminator=self.discriminator,
            clf=clf,
            **kwargs,
        )

    def test_query_param_clf(self):
        super().test_query_param_clf(test_cases=[])

    def test_query_param_discriminator(self):
        self._test_param(
            "query",
            "discriminator",
            [("invalid", TypeError), (self.discriminator, None)],
        )

    def test_query(self):
        query_idx1, utilities1 = self.qs.query(
            self.X,
            self.y,
            discriminator=self.discriminator,
            clf=self.clf,
            return_utilities=True,
        )
        query_idx2, utilities2 = self.qs.query(
            self.X,
            self.y,
            discriminator=self.discriminator,
            clf=self.clf,
            candidates=self.unld_idx,
            return_utilities=True,
        )
        query_idx3, utilities3 = self.qs.query(
            self.X,
            self.y,
            discriminator=self.discriminator,
            clf=self.clf,
            candidates=self.X[self.unld_idx],
            return_utilities=True,
        )

        np.testing.assert_array_equal(query_idx1, query_idx2)
        np.testing.assert_allclose(utilities1, utilities2, equal_nan=True)
        np.testing.assert_allclose(
            utilities1[0][self.unld_idx], utilities3[0], equal_nan=True
        )
        self.assertEqual(query_idx3.shape, (1,))
        self.assertEqual(utilities3.shape, (1, len(self.unld_idx)))

    def test_query_multilabel_list_probas(self):
        # Regression test: `MultiOutputClassifier` natively returns a list of
        # binary probability matrices, which used to reach the boolean label
        # mask unchanged for `proba_format="list"`.
        clf = SklearnClassifier(
            estimator=MultiOutputClassifier(GaussianNB()),
            classes=[[0, 1], [0, 1]],
            missing_label=MISSING_LABEL,
            proba_format="list",
            random_state=0,
        )

        for candidates in [None, self.unld_idx, self.X[self.unld_idx]]:
            with self.subTest(candidates=candidates):
                query_idx, utilities = self.qs.query(
                    self.X,
                    self.y,
                    discriminator=self.discriminator,
                    clf=clf,
                    candidates=candidates,
                    return_utilities=True,
                )
                exp_idx, exp_utilities = self.qs.query(
                    self.X,
                    self.y,
                    discriminator=self.discriminator,
                    clf=self.clf,
                    candidates=candidates,
                    return_utilities=True,
                )

                np.testing.assert_array_equal(query_idx, exp_idx)
                np.testing.assert_allclose(
                    utilities, exp_utilities, equal_nan=True
                )

    def test_query_array_native_estimator_matches_list_format(self):
        # `OneVsRestClassifier` natively returns an array, so the wrapper has
        # to translate it into the list format and MMC must consume both.
        results = []
        for proba_format in ["array", "list"]:
            clf = SklearnClassifier(
                estimator=OneVsRestClassifier(LogisticRegression()),
                classes=[[0, 1], [0, 1]],
                missing_label=MISSING_LABEL,
                proba_format=proba_format,
                random_state=0,
            )
            results.append(
                MaxLossReductionMaxConfidence(random_state=0).query(
                    self.X,
                    self.y,
                    discriminator=self.discriminator,
                    clf=clf,
                    return_utilities=True,
                )
            )

        self.assertIn(results[0][0][0], self.unld_idx)
        np.testing.assert_array_equal(results[0][0], results[1][0])
        np.testing.assert_allclose(
            results[0][1], results[1][1], equal_nan=True
        )

    def test_query_rejects_malformed_multilabel_probas(self):
        cases = [
            (
                "output count",
                [np.full((len(self.X), 2), 0.5) for _ in range(3)],
                "contains 3 outputs, expected 2",
            ),
            (
                "sample count",
                [np.full((len(self.X) + 1, 2), 0.5) for _ in range(2)],
                r"`probas\[0\]` has 9 samples, expected 8",
            ),
            (
                "binary width",
                [np.full((len(self.X), 3), 1 / 3) for _ in range(2)],
                r"`probas\[0\]` must have shape `\(n_samples, 2\)`",
            ),
            (
                "per-output dimension",
                np.full(len(self.X), 0.5),
                r"`probas` must have shape `\(n_samples, n_outputs\)`",
            ),
            (
                "array output count",
                np.full((len(self.X), 3), 0.5),
                "has 3 outputs, expected 2",
            ),
        ]

        for msg, probas, error_regex in cases:
            with self.subTest(msg=msg):
                with patch.object(
                    SklearnClassifier, "predict_proba", return_value=probas
                ):
                    with self.assertRaisesRegex(ValueError, error_regex):
                        self.qs.query(
                            self.X,
                            self.y,
                            discriminator=self.discriminator,
                            clf=self.clf,
                        )

    def test_query_batch_variation(self):
        query_idx, utilities = self.qs.query(
            self.X,
            self.y,
            discriminator=self.discriminator,
            clf=self.clf,
            batch_size=3,
            return_utilities=True,
        )
        self.assertEqual(query_idx.shape, (3,))
        self.assertEqual(utilities.shape, (3, len(self.X)))

        self.assertWarns(
            Warning,
            self.qs.query,
            self.X,
            self.y,
            discriminator=self.discriminator,
            clf=self.clf,
            batch_size=len(self.unld_idx) + 1,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            query_idx = self.qs.query(
                self.X,
                self.y,
                discriminator=self.discriminator,
                clf=self.clf,
                batch_size=len(self.unld_idx) + 1,
            )
            self.assertEqual(len(query_idx), len(self.unld_idx))
