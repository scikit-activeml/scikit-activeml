import unittest
import warnings

import numpy as np
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import GaussianNB

from skactiveml.base import SkactivemlClassifier
from skactiveml.classifier import SklearnClassifier
from skactiveml.pool import LabelCardinalityInconsistency
from skactiveml.pool.tests._multilabel_target_semantics import (
    MultilabelOnlyTargetSemanticsMixin,
)
from skactiveml.tests.template_query_strategy import (
    TemplateSingleAnnotatorPoolQueryStrategy,
)
from skactiveml.utils import MISSING_LABEL, unlabeled_indices


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
        self.qs = LabelCardinalityInconsistency(random_state=0)

        self.strategy_class = LabelCardinalityInconsistency
        TemplateSingleAnnotatorPoolQueryStrategy.setUp(
            self,
            qs_class=LabelCardinalityInconsistency,
            init_default_params={"random_state": 0},
            query_default_params_clf_multilabel={
                "X": self.X,
                "y": self.y,
                "clf": self.clf,
            },
        )

    def _query_strategy(self, strategy, y, clf, **kwargs):
        return strategy.query(self.X, y, clf=clf, **kwargs)

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
