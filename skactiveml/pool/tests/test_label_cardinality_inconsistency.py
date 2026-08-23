import unittest
import warnings
from unittest.mock import patch

import numpy as np
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import GaussianNB

from skactiveml.base import SkactivemlClassifier
from skactiveml.classifier import SklearnClassifier
from skactiveml.pool import (
    LabelCardinalityInconsistency,
    label_cardinality_inconsistency,
)
from skactiveml.tests.template_query_strategy import (
    TemplateMultilabelOnlySingleAnnotatorPoolQueryStrategy,
)
from skactiveml.utils import ExtLabelEncoder, MISSING_LABEL, unlabeled_indices


class DummyMultilabelClassifier(SkactivemlClassifier):
    last_sample_weight = None

    def __init__(
        self, prediction=None, missing_label=MISSING_LABEL, classes=None
    ):
        classes = [[0, 1], [0, 1]] if classes is None else classes
        super().__init__(classes=classes, missing_label=missing_label)
        self.prediction = [1, 0] if prediction is None else prediction
        self.missing_label = missing_label
        self.classes = classes
        self.target_type = "multi-label"

    @property
    def _target_capabilities(self):
        return frozenset(
            {("classification", "multi-label", "single-annotator")}
        )

    def fit(self, X, y, sample_weight=None):
        DummyMultilabelClassifier.last_sample_weight = (
            None
            if sample_weight is None
            else np.array(sample_weight, copy=True)
        )
        target_spec = self._resolve_target_spec(y)
        self._validate_data(
            X=X,
            y=y,
            sample_weight=sample_weight,
            target_spec=target_spec,
        )
        self.target_spec_ = target_spec
        return self

    def predict_proba(self, X, **kwargs):
        raise NotImplementedError

    def predict(self, X, **kwargs):
        return np.tile(self.prediction, (len(X), 1))


class TestLabelCardinalityInconsistency(
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
        self.qs = LabelCardinalityInconsistency(random_state=0)

        TemplateMultilabelOnlySingleAnnotatorPoolQueryStrategy.setUp(
            self,
            qs_class=LabelCardinalityInconsistency,
            init_default_params={"random_state": 0},
            query_default_params_clf_multilabel={
                "X": self.X,
                "y": self.y,
                "clf": self.clf,
            },
        )

    def test_query_param_clf(self):
        super().test_query_param_clf(test_cases=[])

    def test_query(self):
        query_idx1, utilities1 = self.qs.query(
            self.X, self.y, clf=self.clf, return_utilities=True
        )
        query_idx2, utilities2 = self.qs.query(
            self.X,
            self.y,
            clf=self.clf,
            candidates=self.unld_idx,
            return_utilities=True,
        )
        query_idx3, utilities3 = self.qs.query(
            self.X,
            self.y,
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
            clf=self.clf,
            candidates=np.arange(len(self.X)),
            batch_size=4,
            return_utilities=True,
        )
        self.assertFalse(np.isnan(utilities[0]).any())
        self.assertTrue(np.isin(query_idx, lbld_idx).any())

        # Restricting `candidates` to labeled samples selects among those.
        query_idx_lbld = self.qs.query(
            self.X, self.y, clf=self.clf, candidates=lbld_idx, batch_size=2
        )
        self.assertTrue(np.isin(query_idx_lbld, lbld_idx).all())

        # In contrast, `candidates=None` considers unlabeled samples only.
        query_idx_none = self.qs.query(
            self.X, self.y, clf=self.clf, batch_size=3
        )
        self.assertTrue(np.isin(query_idx_none, self.unld_idx).all())

    def test_query_batch_variation(self):
        query_idx, utilities = self.qs.query(
            self.X,
            self.y,
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
            clf=self.clf,
            batch_size=len(self.unld_idx) + 1,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            query_idx = self.qs.query(
                self.X,
                self.y,
                clf=self.clf,
                batch_size=len(self.unld_idx) + 1,
            )
            self.assertEqual(len(query_idx), len(self.unld_idx))

    def test_query_with_sample_weight(self):
        DummyMultilabelClassifier.last_sample_weight = None
        sample_weight = np.ones_like(self.y)
        self.qs.query(
            self.X,
            self.y,
            clf=DummyMultilabelClassifier(),
            sample_weight=sample_weight,
        )
        np.testing.assert_array_equal(
            DummyMultilabelClassifier.last_sample_weight, sample_weight
        )

        sample_weight = np.ones(len(self.y))
        self.qs.query(
            self.X,
            self.y,
            clf=DummyMultilabelClassifier(),
            sample_weight=sample_weight,
        )
        np.testing.assert_array_equal(
            DummyMultilabelClassifier.last_sample_weight, sample_weight
        )

    def test_label_cardinality_ignores_unlabeled_rows(self):
        missing_label = -1
        y = np.where(np.isnan(self.y), missing_label, self.y)
        strategy = LabelCardinalityInconsistency(
            missing_label=missing_label, random_state=0
        )
        clf = DummyMultilabelClassifier(missing_label=missing_label)

        _, utilities = strategy.query(
            self.X, y, clf=clf, return_utilities=True
        )

        np.testing.assert_allclose(utilities[0, self.unld_idx], 0)

    def test_query_delegates_to_public_utility(self):
        recorded = {}
        acquisition_scores = 1.0 + np.arange(len(self.unld_idx), dtype=float)

        def _record(y_pred, y_labeled):
            recorded["y_pred"] = y_pred
            recorded["y_labeled"] = y_labeled
            return acquisition_scores

        with patch(
            "skactiveml.pool._label_cardinality_inconsistency."
            "label_cardinality_inconsistency",
            side_effect=_record,
        ) as utility_mock:
            query_indices, utilities = self.qs.query(
                self.X, self.y, clf=self.clf, return_utilities=True
            )

        utility_mock.assert_called_once()
        # The query passes encoded predictions and encoded observed labels.
        self.assertEqual(np.shape(recorded["y_pred"]), (len(self.unld_idx), 2))
        self.assertTrue(np.isin(recorded["y_pred"], [0, 1]).all())
        np.testing.assert_array_equal(
            recorded["y_labeled"], [[0, 0], [0, 1], [1, 1]]
        )
        # The returned acquisition scores drive utilities and selection.
        np.testing.assert_allclose(
            utilities[0, self.unld_idx], acquisition_scores
        )
        self.assertEqual(
            query_indices[0], self.unld_idx[np.argmax(acquisition_scores)]
        )

    def test_label_cardinality_uses_positive_class_vocabularies(self):
        missing_label = -1
        y = np.array(
            [
                [0, 0],
                [0, 5],
                [2, 5],
                *[[missing_label, missing_label] for _ in range(5)],
            ]
        )
        classes = [[0, 2], [0, 5]]
        strategy = LabelCardinalityInconsistency(
            missing_label=missing_label, random_state=0
        )
        clf = DummyMultilabelClassifier(
            prediction=[0, 5],
            missing_label=missing_label,
            classes=classes,
        )

        _, utilities = strategy.query(
            self.X, y, clf=clf, return_utilities=True
        )

        np.testing.assert_allclose(utilities[0, 3:], 0)


class TestLabelCardinalityInconsistencyFunction(unittest.TestCase):
    def test_label_cardinality_inconsistency(self):
        # Worked example: the labeled pool contains 0, 1, and 2 positive
        # labels, i.e., a mean label cardinality of one. The candidates
        # deviate from it by 1, 1, and 0 positive labels.
        y_labeled = np.array([[0, 0], [0, 1], [1, 1]])
        y_pred = np.array([[1, 1], [0, 0], [1, 0]])

        utilities = label_cardinality_inconsistency(y_pred, y_labeled)

        np.testing.assert_allclose(utilities, [1.0, 1.0, 0.0])

    def test_custom_class_vocabularies(self):
        # Encoded targets carry no trace of the raw class values, so a custom
        # vocabulary is scored exactly like a `{0, 1}` vocabulary.
        encoder = ExtLabelEncoder(
            classes=[[0, 2], [0, 5]],
            missing_label=-1,
            target_type="multi-label",
        ).fit(np.array([[0, 0], [2, 5]]))
        y_labeled = encoder.transform(np.array([[0, 0], [2, 5]]))
        y_pred = encoder.transform(np.array([[2, 0], [2, 5]]))

        utilities = label_cardinality_inconsistency(y_pred, y_labeled)

        # Labeled cardinalities are 0 and 2, i.e., a mean of one, while the
        # candidates have predicted cardinalities of one and two.
        np.testing.assert_allclose(utilities, [0.0, 1.0])

    def test_without_labeled_samples(self):
        # An empty labeled pool has a label cardinality of zero, so each
        # candidate's utility is its own number of predicted positive labels.
        utilities = label_cardinality_inconsistency(
            np.array([[1, 1], [0, 0], [0, 1]]), np.empty((0, 2), dtype=int)
        )

        np.testing.assert_allclose(utilities, [2.0, 0.0, 1.0])

    def test_without_candidates(self):
        utilities = label_cardinality_inconsistency(
            np.empty((0, 2), dtype=int), np.array([[0, 1]])
        )

        self.assertEqual(utilities.shape, (0,))

    def test_param_y_pred(self):
        y_labeled = np.array([[0, 0], [0, 1], [1, 1]])
        cases = [
            (
                "one-dimensional prediction",
                np.array([1, 0]),
                r"`y_pred` must have shape `\(n_samples, n_outputs\)`",
            ),
            (
                "unencoded prediction",
                np.array([[2, 0]]),
                r"`y_pred` must contain encoded labels",
            ),
            (
                "raw class vocabulary",
                np.array([["yes", "no"]]),
                r"`y_pred` must contain encoded labels",
            ),
        ]

        for msg, y_pred, error_regex in cases:
            with self.subTest(msg=msg):
                with self.assertRaisesRegex(ValueError, error_regex):
                    label_cardinality_inconsistency(y_pred, y_labeled)

    def test_param_y_labeled(self):
        y_pred = np.array([[1, 1], [0, 0]])
        cases = [
            (
                "one-dimensional labels",
                np.array([1, 0]),
                r"`y_labeled` must have shape `\(n_samples, n_outputs\)`",
            ),
            (
                "mismatching output counts",
                np.array([[0, 0, 1]]),
                r"`y_labeled` has 3 outputs, expected 2",
            ),
            (
                "unlabeled rows",
                np.array([[-1, -1]]),
                r"`y_labeled` must contain encoded labels",
            ),
            (
                "raw class vocabulary",
                np.array([["yes", "no"]]),
                r"`y_labeled` must contain encoded labels",
            ),
        ]

        for msg, y_labeled, error_regex in cases:
            with self.subTest(msg=msg):
                with self.assertRaisesRegex(ValueError, error_regex):
                    label_cardinality_inconsistency(y_pred, y_labeled)
