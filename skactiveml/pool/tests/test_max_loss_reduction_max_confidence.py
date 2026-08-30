import unittest
import warnings
from unittest.mock import patch

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import GaussianNB

from skactiveml.classifier import ParzenWindowClassifier, SklearnClassifier
from skactiveml.pool import (
    MaxLossReductionMaxConfidence,
    max_loss_reduction_max_confidence,
)
from skactiveml.tests.template_query_strategy import (
    TemplateMultilabelOnlySingleAnnotatorPoolQueryStrategy,
)
from skactiveml.utils import MISSING_LABEL, unlabeled_indices


class RecordingDiscriminator(ParzenWindowClassifier):
    """Discriminator recording the label cardinalities it is fitted on."""

    last_y = None

    def fit(self, X, y, sample_weight=None):
        RecordingDiscriminator.last_y = np.array(y, copy=True)
        return super().fit(X, y, sample_weight=sample_weight)


class TestMaxLossReductionMaxConfidence(
    TemplateMultilabelOnlySingleAnnotatorPoolQueryStrategy,
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

        TemplateMultilabelOnlySingleAnnotatorPoolQueryStrategy.setUp(
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

    def test_query_param_clf(self):
        super().test_query_param_clf(test_cases=[])

    def test_query_param_discriminator(self):
        self._test_param(
            "query",
            "discriminator",
            [("invalid", TypeError), (self.discriminator, None)],
        )

    def test_query_rejects_multilabel_discriminator(self):
        discriminator = ParzenWindowClassifier(
            classes=[[0, 1], [0, 1]],
            missing_label=-1,
            random_state=0,
            target_type="multi-label",
        )

        with self.assertRaisesRegex(
            ValueError,
            "`discriminator` must support single-output classification",
        ):
            self.qs.query(
                self.X,
                self.y,
                discriminator=discriminator,
                clf=self.clf,
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

    def test_query_labeled_candidates(self):
        # A given `candidates` is authoritative, i.e., labeled samples remain
        # candidates, e.g., to relabel them or to recompute their utilities.
        lbld_idx = np.setdiff1d(np.arange(len(self.X)), self.unld_idx)
        query_idx, utilities = self.qs.query(
            self.X,
            self.y,
            discriminator=self.discriminator,
            clf=self.clf,
            candidates=np.arange(len(self.X)),
            batch_size=4,
            return_utilities=True,
        )
        self.assertFalse(np.isnan(utilities[0]).any())
        self.assertTrue(np.isin(query_idx, lbld_idx).any())

        # Restricting `candidates` to labeled samples selects among those.
        query_idx_lbld = self.qs.query(
            self.X,
            self.y,
            discriminator=self.discriminator,
            clf=self.clf,
            candidates=lbld_idx,
            batch_size=2,
        )
        self.assertTrue(np.isin(query_idx_lbld, lbld_idx).all())

        # In contrast, `candidates=None` considers unlabeled samples only.
        query_idx_none = self.qs.query(
            self.X,
            self.y,
            discriminator=self.discriminator,
            clf=self.clf,
            batch_size=3,
        )
        self.assertTrue(np.isin(query_idx_none, self.unld_idx).all())

    def test_query_delegates_to_public_utility(self):
        recorded = {}
        acquisition_scores = 1.0 + np.arange(len(self.unld_idx), dtype=float)

        def _record(probas, n_positive_labels):
            recorded["probas"] = probas
            recorded["n_positive_labels"] = n_positive_labels
            return acquisition_scores

        with patch(
            "skactiveml.pool._max_loss_reduction_max_confidence."
            "max_loss_reduction_max_confidence",
            side_effect=_record,
        ) as utility_mock:
            query_indices, utilities = self.qs.query(
                self.X,
                self.y,
                discriminator=self.discriminator,
                clf=self.clf,
                return_utilities=True,
            )

        utility_mock.assert_called_once()
        # The query passes canonical positive-class probabilities of the
        # candidates and the discriminator's label-cardinality predictions.
        probas = recorded["probas"]
        self.assertIsInstance(probas, np.ndarray)
        self.assertEqual(probas.shape, (len(self.unld_idx), 2))
        self.assertTrue(((probas >= 0) & (probas <= 1)).all())
        self.assertEqual(
            np.shape(recorded["n_positive_labels"]), (len(self.unld_idx),)
        )
        self.assertTrue(
            np.isin(recorded["n_positive_labels"], [0, 1, 2]).all()
        )
        # The returned acquisition scores drive utilities and selection.
        np.testing.assert_allclose(
            utilities[0, self.unld_idx], acquisition_scores
        )
        self.assertEqual(
            query_indices[0], self.unld_idx[np.argmax(acquisition_scores)]
        )

    def test_query_encodes_custom_class_vocabularies(self):
        missing_label = -1
        y = np.array(
            [
                [0, 0],
                [0, 5],
                [2, 5],
                *[[missing_label, missing_label] for _ in range(5)],
            ]
        )
        clf = SklearnClassifier(
            estimator=MultiOutputClassifier(GaussianNB()),
            classes=[[0, 2], [0, 5]],
            missing_label=missing_label,
            proba_format="array",
            random_state=0,
        )
        RecordingDiscriminator.last_y = None

        query_indices, utilities = MaxLossReductionMaxConfidence(
            missing_label=missing_label, random_state=0
        ).query(
            self.X,
            y,
            discriminator=RecordingDiscriminator(
                classes=[0, 1, 2], missing_label=missing_label, random_state=0
            ),
            clf=clf,
            return_utilities=True,
        )

        # The discriminator learns encoded label cardinalities, i.e., 0, 1,
        # and 2 instead of the raw class value sums 0, 5, and 7.
        np.testing.assert_array_equal(RecordingDiscriminator.last_y, [0, 1, 2])
        self.assertTrue(np.isfinite(utilities[0, 3:]).all())
        self.assertIn(query_indices[0], range(3, len(self.X)))

    def test_query_handles_zero_positive_class_probabilities(self):
        y = self.y.copy()
        y[:3] = 0
        clf = ParzenWindowClassifier(
            classes=[[0, 1], [0, 1]],
            missing_label=MISSING_LABEL,
            random_state=0,
            target_type="multi-label",
        )

        query_indices, utilities = self.qs.query(
            self.X,
            y,
            discriminator=self.discriminator,
            clf=clf,
            return_utilities=True,
        )

        self.assertEqual(query_indices.shape, (1,))
        self.assertTrue(np.isfinite(utilities[0, self.unld_idx]).all())

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


class TestMaxLossReductionMaxConfidenceFunction(unittest.TestCase):
    def test_max_loss_reduction_max_confidence(self):
        # The `n_positive_labels` most probable labels of a candidate are
        # predicted positive. The loss reduction then sums `1 - p` over these
        # labels and `p` over the labels predicted negative.
        probas = np.array([[0.9, 0.2, 0.6], [0.1, 0.4, 0.3]])
        n_positive_labels = np.array([2, 0])

        utilities = max_loss_reduction_max_confidence(
            probas, n_positive_labels
        )

        np.testing.assert_allclose(utilities, [0.7, 0.8])

    def test_extreme_label_cardinalities(self):
        # Predicting no positive label sums all probabilities, whereas
        # predicting every label positive sums their complements.
        probas = np.array([[0.9, 0.2, 0.6], [0.9, 0.2, 0.6]])

        utilities = max_loss_reduction_max_confidence(probas, [0, 3])

        np.testing.assert_allclose(utilities, [1.7, 1.3])

    def test_without_candidates(self):
        utilities = max_loss_reduction_max_confidence(
            np.empty((0, 3)), np.empty(0, dtype=int)
        )

        self.assertEqual(utilities.shape, (0,))

    def test_binary_probability_matrices(self):
        # The documented list of per-output binary probability matrices is
        # canonicalized to the same positive-class probabilities.
        probas = np.array([[0.9, 0.2, 0.6], [0.1, 0.4, 0.3]])
        probas_list = [
            np.column_stack([1 - probas[:, j], probas[:, j]])
            for j in range(probas.shape[1])
        ]

        utilities = max_loss_reduction_max_confidence(probas_list, [2, 0])

        np.testing.assert_allclose(utilities, [0.7, 0.8])

    def test_param_probas(self):
        cases = [
            (
                "one-dimensional probabilities",
                np.array([0.9, 0.2]),
                r"`probas` must have shape `\(n_samples, n_outputs\)`",
            ),
            (
                "mismatching candidate counts",
                np.array([[0.9, 0.2, 0.6]]),
                r"`probas` has 1 samples, expected 2",
            ),
            (
                "mismatching candidate counts per output",
                [np.full((2, 2), 0.5), np.full((3, 2), 0.5)],
                r"`probas\[1\]` has 3 samples, expected 2",
            ),
            (
                "probabilities outside the unit interval",
                np.array([[0.9, 1.2, 0.6], [0.1, 0.4, 0.3]]),
                r"`probas` must contain probabilities within `\[0, 1\]`",
            ),
            (
                "missing probability",
                np.array([[0.9, np.nan, 0.6], [0.1, 0.4, 0.3]]),
                r"`probas` must contain probabilities within `\[0, 1\]`",
            ),
        ]

        for msg, probas, error_regex in cases:
            with self.subTest(msg=msg):
                with self.assertRaisesRegex(ValueError, error_regex):
                    max_loss_reduction_max_confidence(probas, [1, 1])

    def test_param_n_positive_labels(self):
        probas = np.array([[0.9, 0.2, 0.6], [0.1, 0.4, 0.3]])
        cases = [
            (
                "two-dimensional label counts",
                np.array([[1], [1]]),
                r"`n_positive_labels` must have shape `\(n_candidates,\)`",
            ),
            (
                "too many positive labels",
                [1, 4],
                r"`n_positive_labels` must contain integers within `\[0, 3\]`",
            ),
            (
                "negative label count",
                [1, -1],
                r"`n_positive_labels` must contain integers within `\[0, 3\]`",
            ),
            (
                "fractional label count",
                [1, 1.5],
                r"`n_positive_labels` must contain integers within `\[0, 3\]`",
            ),
        ]

        for msg, n_positive_labels, error_regex in cases:
            with self.subTest(msg=msg):
                with self.assertRaisesRegex(ValueError, error_regex):
                    max_loss_reduction_max_confidence(
                        probas, n_positive_labels
                    )
