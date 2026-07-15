import unittest
import warnings

import numpy as np
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
