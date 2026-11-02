import unittest
from copy import deepcopy
from itertools import product
from unittest.mock import PropertyMock, patch

import numpy as np
from sklearn import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import DataConversionWarning, NotFittedError

from skactiveml.base import SkactivemlClassifier
from skactiveml.classifier import (
    ParzenWindowClassifier,
    SklearnClassifier,
    SlidingWindowClassifier,
)
from skactiveml.classifier.multiannotator import (
    AnnotatorEnsembleClassifier,
    AnnotatorLogisticRegression,
)
from skactiveml.pool import (
    Badge,
    BatchBALD,
    DropQuery,
    Falcun,
    GreedyBALD,
    ParallelUtilityEstimationWrapper,
    ProbabilisticAL,
    QueryByCommittee,
    SubSamplingWrapper,
    UncertaintySampling,
)
from skactiveml.pool.multiannotator import SingleAnnotatorWrapper
from skactiveml.tests.utils import assert_state_unchanged
from skactiveml.utils import majority_vote


class _PredictionClassifier(SkactivemlClassifier):
    """Expose identical predictions for either training annotation type."""

    def __init__(self, annotation_type="single-annotator", classes=(0, 1, 2)):
        super().__init__(classes=classes, missing_label=-1, random_state=0)
        self.annotation_type = annotation_type
        self._annotation_type = annotation_type

    @property
    def _target_capabilities(self):
        return frozenset(
            {("classification", "single-output", self.annotation_type)}
        )

    def fit(self, X, y, sample_weight=None):
        self._validate_data(X, y, sample_weight)
        return self

    def predict_proba(self, X, return_embeddings=False):
        X = np.asarray(X)
        scores = 1 + np.square(X[:, :1] - np.arange(len(self.classes_)))
        probas = scores / scores.sum(axis=1, keepdims=True)
        if return_embeddings:
            return probas, np.cos(X)
        return probas


class TestFittedMultiannotatorTargets(unittest.TestCase):
    def setUp(self):
        self.X = np.arange(16, dtype=float).reshape(8, 2) / 4
        self.y = np.array([0, 1, 2, -1, -1, -1, -1, -1])
        self.y_matrix = np.column_stack([self.y, np.full(len(self.y), -1)])
        self.single_clf = _PredictionClassifier().fit(self.X, self.y)
        self.multi_clf = _PredictionClassifier("multi-annotator").fit(
            self.X, self.y_matrix
        )

    def test_prediction_equivalence_and_fitted_state(self):
        strategies = [
            UncertaintySampling(method="margin_sampling"),
            Badge(),
            Badge(clf_embedding_flag_name="return_embeddings"),
            Falcun(),
        ]
        candidate_options = [None, np.arange(len(self.X)), self.X[2:]]
        labels = [self.y, np.full_like(self.y, -1)]
        original_state = deepcopy(self.multi_clf)
        original_spec = self.multi_clf.target_spec_

        with patch.object(
            _PredictionClassifier, "fit", side_effect=AssertionError("refit")
        ):
            for strategy, candidates, y in product(
                strategies, candidate_options, labels
            ):
                with self.subTest(
                    strategy=strategy, candidates=candidates, y=y
                ):
                    strategy = clone(strategy).set_params(
                        missing_label=-1, random_state=0
                    )
                    results = [
                        clone(strategy).query(
                            self.X,
                            y,
                            clf=clf,
                            fit_clf=False,
                            candidates=candidates,
                            batch_size=2,
                            return_utilities=True,
                        )
                        for clf in (self.single_clf, self.multi_clf)
                    ]
                    np.testing.assert_array_equal(results[0][0], results[1][0])
                    np.testing.assert_allclose(results[0][1], results[1][1])

        self.assertIs(self.multi_clf.target_spec_, original_spec)
        assert_state_unchanged(self, self.multi_clf, original_state)

    def test_column_vector_remains_supported(self):
        strategy = UncertaintySampling(missing_label=-1, random_state=0)
        expected = strategy.query(
            self.X, self.y, self.multi_clf, fit_clf=False
        )
        with self.assertWarns(DataConversionWarning):
            actual = strategy.query(
                self.X, self.y[:, None], self.multi_clf, fit_clf=False
            )
        np.testing.assert_array_equal(actual, expected)

    def test_hard_voting_queries_preserve_fitted_classifier_state(self):
        X = np.random.RandomState(0).normal(size=(20, 2))
        X[:2] = 0
        y = np.full(len(X), -1)
        y[:2] = [0, 1]
        clf = AnnotatorEnsembleClassifier(
            estimators=[
                (
                    str(i),
                    ParzenWindowClassifier(missing_label=-1, random_state=i),
                )
                for i in range(2)
            ],
            classes=[0, 1],
            missing_label=-1,
            random_state=0,
        ).fit(X, np.column_stack([y, y]))
        original_state = deepcopy(clf)
        for strategy_cls in (UncertaintySampling, DropQuery):
            with self.subTest(strategy=strategy_cls.__name__):
                results = [
                    strategy_cls(missing_label=-1, random_state=0).query(
                        X,
                        y,
                        clf=clf,
                        fit_clf=False,
                        batch_size=3,
                        return_utilities=True,
                    )
                    for _ in range(3)
                ]
                for actual in results[1:]:
                    np.testing.assert_array_equal(actual[0], results[0][0])
                    np.testing.assert_allclose(actual[1], results[0][1])
                assert_state_unchanged(self, clf, original_state)

    def test_wrapper_prediction_equivalence(self):
        strategy = UncertaintySampling(missing_label=-1, random_state=0)
        subsampling = SubSamplingWrapper(
            strategy, max_candidates=4, missing_label=-1, random_state=0
        )
        wrappers = [
            subsampling,
            clone(subsampling).set_params(exclude_non_subsample=True),
            ParallelUtilityEstimationWrapper(
                strategy,
                n_jobs=2,
                parallel_dict={"prefer": "threads"},
                missing_label=-1,
                random_state=0,
            ),
            SingleAnnotatorWrapper(
                Badge(missing_label=-1, random_state=0),
                missing_label=-1,
                random_state=0,
            ),
            SingleAnnotatorWrapper(
                subsampling, missing_label=-1, random_state=0
            ),
        ]
        with patch.object(
            _PredictionClassifier, "fit", side_effect=AssertionError("refit")
        ):
            for wrapper in wrappers:
                with self.subTest(wrapper=wrapper):
                    is_multiannotator = isinstance(
                        wrapper, SingleAnnotatorWrapper
                    )
                    y = self.y_matrix if is_multiannotator else self.y
                    results = [
                        clone(wrapper).query(
                            self.X,
                            y,
                            clf=clf,
                            fit_clf=False,
                            candidates=np.arange(len(self.X)),
                            return_utilities=True,
                        )
                        for clf in (self.single_clf, self.multi_clf)
                    ]
                    np.testing.assert_array_equal(results[0][0], results[1][0])
                    np.testing.assert_allclose(results[0][1], results[1][1])
                    if is_multiannotator:
                        self.assertTrue(
                            np.all(self.y_matrix[tuple(results[1][0].T)] == -1)
                        )

    def test_rejections(self):
        strategy = UncertaintySampling(missing_label=-1, random_state=0)
        invalid_y = self.y.copy()
        invalid_y[0] = 3
        cases = [
            (self.y_matrix, {}, "one-dimensional or a column vector"),
            (invalid_y, {}, "outside `classes`"),
            (self.y, {"missing_label": np.nan}, "must be equal"),
        ]
        for y, init_params, error in cases:
            with self.subTest(error=error):
                qs = clone(strategy).set_params(**init_params)
                with self.assertRaisesRegex(ValueError, error):
                    qs.query(self.X, y, clf=self.multi_clf, fit_clf=False)

    def test_refitting_multiannotator_classifier_still_rejected(self):
        strategy = UncertaintySampling(missing_label=-1)
        wrappers = [
            strategy,
            SubSamplingWrapper(strategy, max_candidates=4, missing_label=-1),
            SingleAnnotatorWrapper(strategy, missing_label=-1),
        ]
        for wrapper, clf in product(
            wrappers, (self.multi_clf, clone(self.multi_clf))
        ):
            with self.subTest(wrapper=wrapper, clf=clf):
                y = (
                    self.y_matrix
                    if isinstance(wrapper, SingleAnnotatorWrapper)
                    else self.y
                )
                with self.assertRaisesRegex(ValueError, "does not support"):
                    clone(wrapper).query(self.X, y, clf=clf, fit_clf=True)

    def test_estimator_capability_uses_training_specification(self):
        with patch.object(
            _PredictionClassifier,
            "_target_capabilities",
            new_callable=PropertyMock,
            return_value=frozenset(
                {("classification", "single-output", "single-annotator")}
            ),
        ):
            strategy = UncertaintySampling(missing_label=-1)
            with self.assertRaisesRegex(
                ValueError, "_PredictionClassifier does not support"
            ):
                strategy.query(self.X, self.y, self.multi_clf, fit_clf=False)

    def test_committee_refitting_rejected_before_any_fit(self):
        annotator_ensemble = AnnotatorEnsembleClassifier(
            estimators=[("pwc", ParzenWindowClassifier(missing_label=-1))],
            classes=[0, 1, 2],
            missing_label=-1,
            random_state=0,
        ).fit(self.X, self.y[:, None])
        annotator_clf = AnnotatorLogisticRegression(
            classes=[0, 1, 2], missing_label=-1, random_state=0
        ).fit(self.X, self.y_matrix)
        ensembles = {
            "annotator ensemble": annotator_ensemble,
            "multiannotator members": [annotator_clf, annotator_clf],
            "multiannotator first": [self.multi_clf, self.single_clf],
            "multiannotator last": [self.single_clf, self.multi_clf],
            "tuple members": (self.single_clf, self.multi_clf),
        }
        strategies = [QueryByCommittee, BatchBALD, GreedyBALD]
        for strategy_cls, fitted, column_y, (name, ensemble) in product(
            strategies, (False, True), (False, True), ensembles.items()
        ):
            with self.subTest(
                strategy=strategy_cls.__name__,
                fitted=fitted,
                column_y=column_y,
                ensemble=name,
            ):
                ensemble = deepcopy(ensemble) if fitted else clone(ensemble)
                original_state = deepcopy(ensemble)
                strategy = strategy_cls(missing_label=-1, random_state=0)
                with (
                    patch.object(
                        _PredictionClassifier,
                        "fit",
                        side_effect=AssertionError("refit"),
                    ),
                    patch.object(
                        AnnotatorEnsembleClassifier,
                        "fit",
                        side_effect=AssertionError("refit"),
                    ),
                    patch.object(
                        AnnotatorLogisticRegression,
                        "fit",
                        side_effect=AssertionError("refit"),
                    ),
                    self.assertRaisesRegex(ValueError, "does not support"),
                ):
                    strategy.query(
                        self.X,
                        self.y[:, None] if column_y else self.y,
                        ensemble,
                        fit_ensemble=True,
                    )
                if isinstance(ensemble, (list, tuple)):
                    for member, original in zip(ensemble, original_state):
                        assert_state_unchanged(self, member, original)
                else:
                    assert_state_unchanged(self, ensemble, original_state)

    def test_committee_refitting_preserves_nested_class_declarations(self):
        y = np.full_like(self.y, -1)
        members = [
            SlidingWindowClassifier(
                ParzenWindowClassifier(
                    classes=[0, 1, 2], missing_label=-1, random_state=i
                ),
                missing_label=-1,
                random_state=i,
            )
            for i in range(2)
        ]
        forest = SlidingWindowClassifier(
            SklearnClassifier(
                RandomForestClassifier(n_estimators=2, random_state=0),
                classes=[0, 1, 2],
                missing_label=-1,
                random_state=0,
            ),
            missing_label=-1,
            random_state=0,
        )
        for strategy_cls, ensemble in product(
            (QueryByCommittee, BatchBALD, GreedyBALD), (members, forest)
        ):
            with self.subTest(
                strategy=strategy_cls.__name__, ensemble=ensemble
            ):
                fitted_ensemble = (
                    [clone(member).fit(self.X, y) for member in ensemble]
                    if isinstance(ensemble, list)
                    else clone(ensemble).fit(self.X, y)
                )
                results = [
                    strategy_cls(missing_label=-1, random_state=0).query(
                        self.X,
                        y,
                        supplied_ensemble,
                        fit_ensemble=fit_ensemble,
                        return_utilities=True,
                    )
                    for supplied_ensemble, fit_ensemble in (
                        (ensemble, True),
                        (fitted_ensemble, False),
                    )
                ]
                np.testing.assert_array_equal(results[0][0], results[1][0])
                np.testing.assert_allclose(results[0][1], results[1][1])

    def test_mixed_fitted_authorities_share_prediction_target(self):
        strategy = SubSamplingWrapper(
            QueryByCommittee(missing_label=-1, random_state=0),
            max_candidates=4,
            missing_label=-1,
            random_state=0,
        )
        expected = clone(strategy).query(
            self.X,
            self.y,
            ensemble=[self.single_clf, self.single_clf],
            fit_ensemble=False,
            return_utilities=True,
        )
        for ensemble in (
            [self.single_clf, self.multi_clf],
            [self.multi_clf, self.single_clf],
        ):
            with self.subTest(ensemble=ensemble):
                actual = clone(strategy).query(
                    self.X,
                    self.y,
                    ensemble=ensemble,
                    fit_ensemble=False,
                    return_utilities=True,
                )
                np.testing.assert_array_equal(actual[0], expected[0])
                np.testing.assert_allclose(actual[1], expected[1])

    def test_annotator_ensemble_members_form_committee(self):
        y_matrix = np.column_stack([self.y, np.roll(self.y, 1)])
        pwc = ParzenWindowClassifier(classes=[0, 1, 2], missing_label=-1)
        clf = AnnotatorEnsembleClassifier(
            estimators=[("pwc_0", clone(pwc)), ("pwc_1", clone(pwc))],
            classes=[0, 1, 2],
            missing_label=-1,
        ).fit(self.X, y_matrix)
        y = majority_vote(
            y_matrix, classes=[0, 1, 2], missing_label=-1, random_state=0
        )
        members = [est for _, est in clf.estimators_]
        strategies = [
            QueryByCommittee(missing_label=-1, random_state=0),
            BatchBALD(missing_label=-1, random_state=0),
            GreedyBALD(missing_label=-1, random_state=0),
        ]
        for strategy, pair_type in product(strategies, (tuple, list)):
            with self.subTest(strategy=strategy, pair_type=pair_type):
                named_ensemble = (
                    clone(clf)
                    .set_params(
                        estimators=[pair_type(pair) for pair in clf.estimators]
                    )
                    .fit(self.X, y_matrix)
                )
                results = [
                    clone(strategy).query(
                        self.X,
                        y,
                        ensemble=ensemble,
                        fit_ensemble=False,
                        return_utilities=True,
                    )
                    for ensemble in (members, named_ensemble)
                ]
                np.testing.assert_array_equal(results[0][0], results[1][0])
                np.testing.assert_allclose(results[0][1], results[1][1])

    def test_query_preserves_fitted_committee_members(self):
        X = np.arange(40, dtype=float).reshape(-1, 2)
        y = np.full(len(X), -1)
        clf = AnnotatorEnsembleClassifier(
            estimators=[
                (
                    f"pwc_{i}",
                    ParzenWindowClassifier(
                        classes=[0, 1], missing_label=-1, random_state=i
                    ),
                )
                for i in range(2)
            ],
            classes=[0, 1],
            missing_label=-1,
        ).fit(X, np.column_stack([y, y]))
        original_state = deepcopy(clf)
        strategies = [
            QueryByCommittee(method=method, missing_label=-1, random_state=0)
            for method in ("KL_divergence", "vote_entropy", "variation_ratios")
        ] + [
            BatchBALD(missing_label=-1, random_state=0),
            GreedyBALD(missing_label=-1, random_state=0),
        ]
        with patch.object(
            ParzenWindowClassifier, "fit", side_effect=AssertionError("refit")
        ):
            for strategy in strategies:
                with self.subTest(strategy=strategy):
                    clf = deepcopy(original_state)
                    original_spec = clf.target_spec_
                    members = [est for _, est in clf.estimators_]
                    expected = clone(strategy).query(
                        X,
                        y,
                        ensemble=members,
                        fit_ensemble=False,
                        batch_size=2,
                        return_utilities=True,
                    )
                    for _ in range(2):
                        actual = strategy.query(
                            X,
                            y,
                            ensemble=clf,
                            fit_ensemble=False,
                            batch_size=2,
                            return_utilities=True,
                        )
                        np.testing.assert_array_equal(actual[0], expected[0])
                        np.testing.assert_allclose(actual[1], expected[1])
                    self.assertIs(clf.target_spec_, original_spec)
                    assert_state_unchanged(self, clf, original_state)

    def test_empty_trained_committee_is_rejected(self):
        pwc = ParzenWindowClassifier(classes=[0, 1, 2], missing_label=-1)
        base = AnnotatorEnsembleClassifier(
            estimators=[("pwc_0", clone(pwc)), ("pwc_1", clone(pwc))],
            classes=[0, 1, 2],
            missing_label=-1,
        )
        strategies = [QueryByCommittee(), BatchBALD(), GreedyBALD()]
        for X, y in (([], []), (np.empty((0, 2)), np.empty((0, 2)))):
            clf = clone(base).fit(X, y)
            np.testing.assert_allclose(clf.predict_proba(self.X), 1 / 3)
            original_state = deepcopy(clf)
            for strategy in strategies:
                with self.subTest(strategy=strategy, X=X):
                    strategy = clone(strategy).set_params(missing_label=-1)
                    with self.assertRaisesRegex(
                        NotFittedError, "member.*not fitted"
                    ):
                        strategy.query(self.X, self.y, clf, fit_ensemble=False)
                    assert_state_unchanged(self, clf, original_state)


class TestWrappedClassVocabulary(unittest.TestCase):
    def test_query_refitting_respects_wrapped_class_vocabulary(self):
        X = np.arange(16, dtype=float).reshape(8, 2) / 4
        y = np.array([0, 1, -1, -1, -1, -1, -1, -1])
        unlabeled = np.full_like(y, -1)
        clf = SlidingWindowClassifier(
            ParzenWindowClassifier(
                classes=[0, 1, 2], missing_label=-1, random_state=0
            ),
            missing_label=-1,
            random_state=0,
        )
        strategy = UncertaintySampling(missing_label=-1, random_state=0)
        wrappers = [
            strategy,
            ProbabilisticAL(metric="rbf", missing_label=-1, random_state=0),
            SubSamplingWrapper(
                strategy, max_candidates=4, missing_label=-1, random_state=0
            ),
            SingleAnnotatorWrapper(strategy, missing_label=-1, random_state=0),
        ]
        for wrapper, labels in product(wrappers, (y, unlabeled)):
            with self.subTest(wrapper=wrapper, labels=labels):
                query_y = (
                    labels[:, None]
                    if isinstance(wrapper, SingleAnnotatorWrapper)
                    else labels
                )
                original_state = deepcopy(clf)
                results = [
                    clone(wrapper).query(
                        X,
                        query_y,
                        clf=supplied_clf,
                        fit_clf=fit_clf,
                        return_utilities=True,
                    )
                    for supplied_clf, fit_clf in (
                        (clone(clf).fit(X, labels), False),
                        (clf, True),
                    )
                ]
                np.testing.assert_array_equal(results[0][0], results[1][0])
                np.testing.assert_allclose(results[0][1], results[1][1])
                assert_state_unchanged(self, clf, original_state)
