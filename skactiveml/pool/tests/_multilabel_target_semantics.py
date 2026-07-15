import numpy as np
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import GaussianNB

from skactiveml.classifier import ParzenWindowClassifier, SklearnClassifier
from skactiveml.classifier.multiannotator import AnnotatorEnsembleClassifier


class MultilabelOnlyTargetSemanticsMixin:
    """Shared target-contract tests for multi-label-only strategies."""

    def _assert_no_acquisition_state(self, strategy):
        self.assertFalse(hasattr(strategy, "n_features_in_"))
        self.assertFalse(hasattr(strategy, "missing_label_"))
        self.assertFalse(hasattr(strategy, "random_state_"))

    def test_query_requires_multilabel_y(self):
        y = np.array([0.0, 1.0, 0.0, np.nan, np.nan, np.nan, np.nan, np.nan])
        clf = SklearnClassifier(estimator=GaussianNB())
        strategy = self.strategy_class()

        with self.assertRaisesRegex(
            ValueError,
            rf"{type(strategy).__name__} does not support target capability",
        ):
            self._query_strategy(strategy, y, clf)

        self._assert_no_acquisition_state(strategy)

    def test_target_contract(self):
        strategy = self.strategy_class()

        self.assertEqual(strategy.target_type, "auto")
        self.assertEqual(
            strategy._target_capabilities,
            frozenset({("classification", "multi-label", "single-annotator")}),
        )
        self.assertNotIn(
            ("classification", "multi-label", "multi-annotator"),
            strategy._target_capabilities,
        )

    def test_query_rejects_fitted_target_spec_conflict_before_state(self):
        clf = self.clf.fit(self.X, self.y)
        strategy = self.strategy_class(target_type="single-output")

        with self.assertRaisesRegex(ValueError, "conflicts"):
            self._query_strategy(strategy, self.y, clf, fit_clf=False)

        self._assert_no_acquisition_state(strategy)

    def test_query_reuses_fitted_target_spec_without_class_evidence(self):
        clf = SklearnClassifier(
            estimator=MultiOutputClassifier(GaussianNB()),
            classes=None,
            target_type="multi-label",
            proba_format="array",
            random_state=0,
        ).fit(self.X, self.y)
        established_spec = clf.target_spec_
        y_query = np.array(
            [
                [0.0, 1.0],
                [0.0, 1.0],
                *[[np.nan, np.nan] for _ in range(6)],
            ]
        )

        query_idx, utilities = self._query_strategy(
            self.qs,
            y_query,
            clf,
            fit_clf=False,
            return_utilities=True,
        )

        self.assertEqual(established_spec.classes, ((0.0, 1.0),) * 2)
        self.assertIs(clf.target_spec_, established_spec)
        self.assertIn(query_idx[0], range(2, len(self.X)))
        self.assertTrue(np.isnan(utilities[0, :2]).all())
        self.assertFalse(hasattr(self.qs, "target_spec_"))

    def test_query_resolves_explicit_multilabel_without_classes(self):
        clf = SklearnClassifier(
            estimator=MultiOutputClassifier(GaussianNB()),
            classes=None,
            target_type="multi-label",
            proba_format="array",
            random_state=0,
        )

        query_idx = self._query_strategy(self.strategy_class(), self.y, clf)

        self.assertIn(query_idx[0], self.unld_idx)
        self.assertEqual(clf.target_type, "multi-label")
        self.assertFalse(hasattr(clf, "target_spec_"))

    def test_query_supports_custom_binary_vocabularies(self):
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
            target_type="multi-label",
            proba_format="array",
            random_state=0,
        )
        strategy = self.strategy_class(
            missing_label=missing_label, random_state=0
        )

        query_idx, utilities = self._query_strategy(
            strategy, y, clf, return_utilities=True
        )

        self.assertIn(query_idx[0], range(3, len(y)))
        self.assertTrue(np.isfinite(utilities[0, 3:]).all())

    def test_query_rejects_partially_observed_rows_before_state(self):
        clf = self.clf.fit(self.X, self.y)
        y = self.y.copy()
        y[1, 0] = np.nan
        strategy = self.strategy_class()

        with self.assertRaisesRegex(ValueError, "no mixing"):
            self._query_strategy(strategy, y, clf, fit_clf=False)

        self._assert_no_acquisition_state(strategy)

    def test_query_rejects_other_target_capabilities_before_state(self):
        multioutput_y = np.array(
            [
                [0.0, 0.0],
                [1.0, 1.0],
                [2.0, 0.0],
                *[[np.nan, np.nan] for _ in range(5)],
            ]
        )
        multioutput_clf = SklearnClassifier(
            estimator=MultiOutputClassifier(GaussianNB()),
            classes=[[0, 1, 2], [0, 1]],
            target_type="multi-output",
        )
        multiannotator_clf = AnnotatorEnsembleClassifier(
            estimators=[
                ("pwc-0", ParzenWindowClassifier(classes=[0, 1])),
                ("pwc-1", ParzenWindowClassifier(classes=[0, 1])),
            ],
            classes=[0, 1],
        ).fit(self.X, self.y)

        cases = [
            (multioutput_y, multioutput_clf, "SklearnClassifier"),
            (self.y, multiannotator_clf, self.strategy_class.__name__),
        ]
        for y, clf, component in cases:
            with self.subTest(component=component):
                strategy = self.strategy_class()
                with self.assertRaisesRegex(
                    ValueError,
                    rf"{component} does not support target capability",
                ):
                    self._query_strategy(strategy, y, clf, fit_clf=False)
                self._assert_no_acquisition_state(strategy)
