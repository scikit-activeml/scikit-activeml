import unittest
import warnings

import numpy as np
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import GaussianNB

from skactiveml.base import SkactivemlClassifier
from skactiveml.classifier import SklearnClassifier
from skactiveml.pool import LabelCardinalityInconsistency
from skactiveml.utils import MISSING_LABEL, unlabeled_indices


class DummyMultilabelClassifier(SkactivemlClassifier):
    last_sample_weight = None

    def __init__(self, prediction=None):
        super().__init__(classes=[[0, 1], [0, 1]], missing_label=MISSING_LABEL)
        self.prediction = [1, 0] if prediction is None else prediction

    def fit(self, X, y, sample_weight=None):
        DummyMultilabelClassifier.last_sample_weight = (
            None
            if sample_weight is None
            else np.array(sample_weight, copy=True)
        )
        self._validate_data(
            X=X,
            y=y,
            sample_weight=sample_weight,
            y_ensure_1d=False,
            multioutput_ensure_multilabel=True,
        )
        return self

    def predict_proba(self, X, **kwargs):
        raise NotImplementedError

    def predict(self, X, **kwargs):
        return np.tile(self.prediction, (len(X), 1))


class TestLabelCardinalityInconsistency(unittest.TestCase):
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
            self.y, missing_label=MISSING_LABEL, is_multioutput=True
        )
        self.clf = SklearnClassifier(
            estimator=MultiOutputClassifier(GaussianNB()),
            classes=[[0, 1], [0, 1]],
            missing_label=MISSING_LABEL,
            proba_format="array",
            random_state=0,
        )
        self.qs = LabelCardinalityInconsistency(random_state=0)

    def test_query_requires_multilabel_y(self):
        y = np.array([0.0, 1.0, 0.0, np.nan, np.nan, np.nan, np.nan, np.nan])
        self.assertRaises(ValueError, self.qs.query, self.X, y, clf=self.clf)

    def test_query_candidate_variation(self):
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
