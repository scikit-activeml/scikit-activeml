import unittest
import warnings
from itertools import product
from unittest.mock import patch

import numpy as np
from sklearn.utils._testing import assert_allclose
from sklearn.utils.validation import check_array

from skactiveml.base import (
    QueryStrategy,
    SingleAnnotatorPoolQueryStrategy,
    MultiAnnotatorPoolQueryStrategy,
    SkactivemlClassifier,
    ClassFrequencyEstimator,
    BudgetManager,
    SingleAnnotatorStreamQueryStrategy,
    SkactivemlRegressor,
    ProbabilisticRegressor,
)
from skactiveml.exceptions import MappingError
from skactiveml.pool import RandomSampling
from skactiveml.utils import MISSING_LABEL, is_unlabeled, match_signature
from skactiveml.tests.utils import assert_predicts_class_dtype

successful_skorch_torch_import = False
try:
    import torch
    from skactiveml.base import SkorchMixin

    successful_skorch_torch_import = True
except ImportError:  # pragma: no cover
    pass


class DummySkactivemlClassifier(SkactivemlClassifier):
    @property
    def _target_capabilities(self):
        return frozenset(
            {
                ("classification", "single-output", "single-annotator"),
                ("classification", "multi-label", "single-annotator"),
            }
        )

    def __init__(
        self,
        classes=None,
        missing_label=MISSING_LABEL,
        cost_matrix=None,
        random_state=None,
        probas=None,
        target_type="auto",
    ):
        super().__init__(
            classes=classes,
            missing_label=missing_label,
            cost_matrix=cost_matrix,
            random_state=random_state,
            target_type=target_type,
        )
        self.probas = probas

    def fit(self, X, y, sample_weight=None):
        self._initialize_label_state(y)
        self.validated_ = self._validate_data(
            X=X,
            y=y,
            sample_weight=sample_weight,
        )
        return self

    def partial_fit(self, X, y, sample_weight=None):
        target_spec = self._resolve_target_spec_for_fit(
            y,
            is_incremental=hasattr(self, "target_spec_"),
        )
        self.validated_ = self._validate_data(
            X=X,
            y=y,
            sample_weight=sample_weight,
            target_spec=target_spec,
        )
        return self

    def predict_proba(self, X, **kwargs):
        if self.probas is None:
            raise NotImplementedError
        return self.probas


class DummyMultiAnnotatorClassifier(DummySkactivemlClassifier):
    _annotation_type = "multi-annotator"

    @property
    def _target_capabilities(self):
        return frozenset(
            {("classification", "single-output", "multi-annotator")}
        )


class DummySingleAnnotatorPoolQueryStrategy(SingleAnnotatorPoolQueryStrategy):
    def query(
        self,
        X,
        y,
        candidates=None,
        batch_size=1,
        return_utilities=False,
    ):
        return self._validate_data(
            X=X,
            y=y,
            candidates=candidates,
            batch_size=batch_size,
            return_utilities=return_utilities,
        )


class DummyRegressionPoolQueryStrategy(DummySingleAnnotatorPoolQueryStrategy):
    @property
    def _target_capabilities(self):
        return frozenset({("regression", "single-output", "single-annotator")})


class DummyClassFrequencyEstimator(ClassFrequencyEstimator):
    def __init__(
        self,
        freq=None,
        class_prior=0,
        classes=None,
        missing_label=MISSING_LABEL,
        target_type="auto",
    ):
        super().__init__(
            classes=classes,
            missing_label=missing_label,
            class_prior=class_prior,
            target_type=target_type,
        )
        self.freq = freq

    def fit(self, X, y, sample_weight=None):
        self.validated_ = self._validate_data(
            X=X,
            y=y,
            sample_weight=sample_weight,
        )
        return self

    def predict_freq(self, X):
        if self.freq is None:
            raise NotImplementedError
        return self.freq


class DummyMultilabelClassFrequencyEstimator(DummyClassFrequencyEstimator):
    @property
    def _target_capabilities(self):
        return super()._target_capabilities | frozenset(
            {("classification", "multi-label", "single-annotator")}
        )


class DummyDistribution:
    def __init__(self, mean, std, entropy):
        self._mean = np.asarray(mean)
        self._std = np.asarray(std)
        self._entropy = np.asarray(entropy)

    def mean(self):
        return self._mean

    def std(self):
        return self._std

    def entropy(self):
        return self._entropy

    def rvs(self, size=None, random_state=None):
        if random_state is None:
            random_state = np.random.RandomState(0)
        elif isinstance(random_state, (int, np.integer)):
            random_state = np.random.RandomState(random_state)
        return random_state.normal(loc=self._mean, scale=self._std, size=size)


class DummyProbabilisticRegressor(ProbabilisticRegressor):
    def __init__(self, mean=None, std=None, entropy=None, missing_label=-1):
        super().__init__(missing_label=missing_label)
        self.mean_ = (
            np.asarray([0.0, 1.0]) if mean is None else np.asarray(mean)
        )
        self.std_ = np.asarray([1.0, 2.0]) if std is None else np.asarray(std)
        self.entropy_ = (
            np.asarray([0.5, 1.5]) if entropy is None else np.asarray(entropy)
        )

    def fit(self, X, y, sample_weight=None):
        return self

    def predict_target_distribution(self, X):
        n = len(X)
        return DummyDistribution(
            mean=np.resize(self.mean_, n),
            std=np.resize(self.std_, n),
            entropy=np.resize(self.entropy_, n),
        )


class DummyMultiAnnotatorPoolQueryStrategy(MultiAnnotatorPoolQueryStrategy):
    def query(
        self,
        X,
        y,
        candidates=None,
        annotators=None,
        batch_size=1,
        return_utilities=False,
    ):
        return self._validate_data(
            X=X,
            y=y,
            candidates=candidates,
            annotators=annotators,
            batch_size=batch_size,
            return_utilities=return_utilities,
        )


class ExhaustedCandidatePoolGuardTest(unittest.TestCase):
    def test_guard_does_not_rewrap_an_inherited_query(self):
        class ParentStrategy(DummySingleAnnotatorPoolQueryStrategy):
            pass

        class ChildStrategy(ParentStrategy):
            pass

        self.assertIs(ChildStrategy.query, ParentStrategy.query)

        with self.assertWarnsRegex(UserWarning, "exhausted"):
            query_indices = ChildStrategy().query(
                X=np.arange(4).reshape(2, 2),
                y=np.array([0, 1]),
            )

        self.assertEqual(query_indices.shape, (0,))

    def test_guard_covers_a_query_published_through_a_descriptor(self):
        # `match_signature` publishes `query` as a descriptor, which stays in
        # place while the function it binds carries the guard.
        class DescriptorPublishingStrategy(SingleAnnotatorPoolQueryStrategy):
            @match_signature("query_strategy", "query")
            def query(
                self,
                X,
                y,
                candidates=None,
                batch_size=1,
                return_utilities=False,
            ):
                self._validate_data(
                    X, y, candidates, batch_size, return_utilities
                )
                raise AssertionError("The guard did not abort the query.")

            def __init__(self, query_strategy):
                super().__init__()
                self.query_strategy = query_strategy

        strategy = DescriptorPublishingStrategy(RandomSampling())
        with self.assertWarnsRegex(UserWarning, "exhausted"):
            query_indices = strategy.query(
                X=np.arange(4).reshape(2, 2),
                y=np.array([0, 1]),
            )

        self.assertEqual(query_indices.shape, (0,))

    def test_guard_rejects_an_uncovered_query_publication(self):
        with self.assertRaisesRegex(TypeError, "publishes `query` as"):

            class PropertyPublishingStrategy(SingleAnnotatorPoolQueryStrategy):
                query = property(lambda self: None)

    def test_guard_preserves_an_abstract_query(self):
        # The guard wraps `query` while the class is created, i.e., before the
        # abstract methods are collected, so it must not hide abstractness.
        for base in (
            SingleAnnotatorPoolQueryStrategy,
            MultiAnnotatorPoolQueryStrategy,
        ):
            with self.subTest(base=base.__name__):
                self.assertEqual(
                    base.__abstractmethods__, frozenset({"query"})
                )


class QueryStrategyTest(unittest.TestCase):
    @patch.multiple(QueryStrategy, __abstractmethods__=set())
    def setUp(self):
        self.qs = QueryStrategy()

    def test_query(self):
        self.assertRaises(NotImplementedError, self.qs.query, candidates=None)


class SingleAnnotPoolBasedQueryStrategyTest(unittest.TestCase):
    @patch.multiple(
        SingleAnnotatorPoolQueryStrategy, __abstractmethods__=set()
    )
    def setUp(self):
        self.qs = SingleAnnotatorPoolQueryStrategy()

    def test_query(self):
        self.assertRaises(
            NotImplementedError, self.qs.query, X=None, y=None, candidates=None
        )

    def test__transform_candidates(self):
        self.qs.missing_label_ = MISSING_LABEL
        self.assertRaises(
            MappingError,
            self.qs._transform_candidates,
            np.array([[3]]),
            np.array([[2]]),
            np.array([0]),
            True,
        )

        self.assertRaises(
            ValueError,
            self.qs._transform_candidates,
            np.array([0]),
            np.array([[2]]),
            np.array([0]),
            True,
            allow_only_unlabeled=True,
        )

        X = np.array([[2], [3]])
        X_cand, mapping = self.qs._transform_candidates(
            candidates=np.array([0]),
            X=X,
            y=np.array([0, 1]),
        )
        np.testing.assert_array_equal(X_cand, X[mapping])

        y_ml = np.array([[0, 1], [np.nan, np.nan], [1, 0]])
        X_ml = np.arange(6).reshape(3, 2)
        X_cand, mapping = self.qs._transform_candidates(
            candidates=None,
            X=X_ml,
            y=y_ml,
            target_type="multi-label",
        )
        np.testing.assert_array_equal(mapping, np.array([1]))
        np.testing.assert_array_equal(X_cand, X_ml[[1]])

        X_cand, mapping = self.qs._transform_candidates(
            candidates=np.array([[9, 9]]),
            X=X_ml,
            y=y_ml,
        )
        self.assertIsNone(mapping)
        np.testing.assert_array_equal(X_cand, np.array([[9, 9]]))

    def test__validate_data_multilabel(self):
        X = np.arange(6).reshape(3, 2)
        y = np.array([[0, 1], [np.nan, np.nan], [np.nan, np.nan]])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            X_v, y_v, candidates_v, batch_size_v, return_utilities_v = (
                self.qs._validate_data(
                    X=X,
                    y=y,
                    candidates=None,
                    batch_size=3,
                    return_utilities=True,
                    target_type="multi-label",
                )
            )
        self.assertEqual(len(w), 1)
        self.assertEqual(batch_size_v, 2)
        self.assertTrue(return_utilities_v)
        np.testing.assert_array_equal(X_v, X)
        np.testing.assert_array_equal(y_v, y)
        self.assertIsNone(candidates_v)

        X_v, y_v, candidates_v, batch_size_v, return_utilities_v = (
            self.qs._validate_data(
                X=X,
                y=np.array([0.0, np.nan, 1.0]),
                candidates=np.array([1, 2]),
                batch_size=1,
                return_utilities=False,
                target_type="single-output",
            )
        )
        self.assertEqual(y_v.ndim, 1)
        np.testing.assert_array_equal(candidates_v, np.array([1, 2]))
        self.assertEqual(batch_size_v, 1)
        self.assertFalse(return_utilities_v)

    def test__validate_data_checks_multilabel_inputs_once(self):
        X = np.arange(6).reshape(3, 2)
        y = np.array([[0, 1], [np.nan, np.nan], [np.nan, np.nan]])

        with patch("skactiveml.base.check_array", wraps=check_array) as check:
            self.qs._validate_data(
                X=X,
                y=y,
                candidates=None,
                batch_size=1,
                return_utilities=False,
                target_type="multi-label",
            )

        self.assertEqual(check.call_count, 2)

    def test_public_query_resolves_target_capabilities(self):
        X = np.arange(8).reshape(4, 2)
        y_single_output = np.array([0, 1, -1, -1])
        y_multi_label = np.array([[0, 1], [1, 0], [-1, -1], [-1, -1]])

        single_output_indices = RandomSampling(
            missing_label=-1,
            random_state=0,
            target_type="single-output",
        ).query(X, y_single_output, batch_size=1)
        multi_label_indices = RandomSampling(
            missing_label=-1,
            random_state=0,
            target_type="multi-label",
        ).query(X, y_multi_label, batch_size=1)

        self.assertEqual(single_output_indices.shape, (1,))
        self.assertEqual(multi_label_indices.shape, (1,))

    def test_public_query_uses_default_single_output_capability(self):
        strategy = DummySingleAnnotatorPoolQueryStrategy(
            missing_label=-1, random_state=0
        )

        validated = strategy.query(
            X=np.arange(8).reshape(4, 2),
            y=np.array([0, 1, -1, -1]),
            batch_size=1,
        )

        self.assertEqual(validated[3], 1)

        strategy.classes = [0, 1]
        validated_with_classes = strategy.query(
            X=np.arange(8).reshape(4, 2),
            y=np.array([0, 1, -1, -1]),
            batch_size=1,
        )
        self.assertEqual(validated_with_classes[3], 1)

    def test_public_query_resolves_regression_capability(self):
        validated = DummyRegressionPoolQueryStrategy(
            missing_label=-1, random_state=0
        ).query(
            X=np.arange(8).reshape(4, 2),
            y=np.array([0.0, 1.0, -1.0, -1.0]),
            batch_size=1,
        )

        self.assertEqual(validated[3], 1)


class MultiAnnotatorPoolQueryStrategyTest(unittest.TestCase):
    @patch.multiple(MultiAnnotatorPoolQueryStrategy, __abstractmethods__=set())
    def setUp(self):
        self.qs = MultiAnnotatorPoolQueryStrategy()
        self.qs.missing_label_ = MISSING_LABEL

    def test_query(self):
        self.assertRaises(
            NotImplementedError,
            self.qs.query,
            X=np.array([[1, 2]]),
            y=np.array(
                [
                    [
                        1,
                    ]
                ]
            ),
        )

    def test_public_query_allows_initially_unobserved_targets(self):
        strategy = DummyMultiAnnotatorPoolQueryStrategy(
            missing_label=-1, random_state=0
        )
        validated = strategy.query(
            X=np.zeros((2, 1)),
            y=np.full((2, 2), -1),
            batch_size=1,
        )

        self.assertEqual(validated[4], 1)
        with self.assertRaises(ValueError):
            strategy.query(
                X=np.zeros((2, 1)),
                y=np.zeros((2, 2, 1)),
                batch_size=1,
            )

    def test__validate_data(self):
        self.assertRaises(
            ValueError,
            self.qs._validate_data,
            candidates=np.array([[1, 2], [0, 1]]),
            annotators=np.array([[False, True], [True, True]]).reshape(
                2, 2, 1
            ),
            X=np.array([[1, 2], [0, 1]]),
            y=np.array([[1, MISSING_LABEL], [2, 3]]),
            batch_size=2,
            return_utilities=False,
        )

        X = np.array([[1, 2], [0, 1]])
        y = np.array([[1, MISSING_LABEL], [2, 3]])
        candidates_values = [
            None,
            np.array([0, 1]),
            np.array([[3, 4], [0, 1]]),
        ]
        annotators_values = [
            None,
            np.array([0]),
            np.array([[False, True], [True, True]]),
        ]

        batch_size_initial = 4
        batch_sizes_expected = [[1, 2, 3], [4, 2, 3], [4, 2, 3]]

        for (i, candidates), (j, annotators) in product(
            enumerate(candidates_values), enumerate(annotators_values)
        ):
            X, y, candidates, annotators, batch_size, return_utilities = (
                self.qs._validate_data(
                    candidates=candidates,
                    annotators=annotators,
                    X=X,
                    y=y,
                    batch_size=batch_size_initial,
                    return_utilities=False,
                )
            )
            self.assertEqual(batch_sizes_expected[i][j], batch_size)

    def test__transform_cand_annot(self):

        self.assertRaises(
            ValueError,
            self.qs._transform_cand_annot,
            candidates=np.array([[0, 2]]),
            annotators=None,
            X=np.array([[1, 2]]),
            y=np.array(
                [
                    [
                        1,
                    ]
                ]
            ),
            enforce_mapping=True,
        )

        X = np.array([[1, 2], [0, 1]])
        y = np.array([[1, MISSING_LABEL], [2, 3]])
        candidates_values = [
            None,
            np.array([0, 1]),
            np.array([[3, 4], [0, 1]]),
        ]
        annotators_values = [
            None,
            np.array([0]),
            np.array([[False, True], [True, True]]),
        ]

        for (i, candidates), (j, annotators) in product(
            enumerate(candidates_values), enumerate(annotators_values)
        ):
            X_cand, mapping, A_cand = self.qs._transform_cand_annot(
                candidates=candidates,
                annotators=annotators,
                X=X,
                y=y,
            )
            self.assertEqual(len(A_cand), len(X_cand))

            if i == 0 and j == 0:
                np.testing.assert_array_equal(A_cand, is_unlabeled(y)[mapping])
                np.testing.assert_array_equal(X[mapping], X_cand)
                np.testing.assert_array_equal(
                    mapping, np.nonzero(np.any(is_unlabeled(y), axis=1))[0]
                )
            if i == 0 and j == 1:
                expected_A_cand = np.full((len(X_cand), len(y.T)), False)
                expected_A_cand[:, annotators] = True
                np.testing.assert_array_equal(A_cand, expected_A_cand)
                np.testing.assert_array_equal(X[mapping], X_cand)
            if i == 0 and j == 2:
                np.testing.assert_array_equal(annotators[mapping], A_cand)
                np.testing.assert_array_equal(X[mapping], X_cand)
                np.testing.assert_array_equal(
                    mapping, np.nonzero(np.any(A_cand, axis=1))[0]
                )
            if i == 1 and j == 0:
                np.testing.assert_array_equal(X[mapping], X_cand)
                np.testing.assert_array_equal(
                    A_cand, np.full((len(X_cand), len(y.T)), True)
                )
                np.testing.assert_array_equal(mapping, candidates)
            if i == 1 and j == 1:
                expected_A_cand = np.full((len(X_cand), len(y.T)), False)
                expected_A_cand[:, annotators] = True
                np.testing.assert_array_equal(A_cand, expected_A_cand)
                np.testing.assert_array_equal(X[mapping], X_cand)
                np.testing.assert_array_equal(mapping, candidates)
            if i == 1 and j == 2:
                np.testing.assert_array_equal(annotators[mapping], A_cand)
                np.testing.assert_array_equal(X[mapping], X_cand)
                np.testing.assert_array_equal(
                    mapping, candidates[np.any(A_cand, axis=1)]
                )
            if i == 2 and j == 0:
                self.assertEqual(mapping, None)
                np.testing.assert_array_equal(X_cand, candidates)
                np.testing.assert_array_equal(
                    A_cand, np.full((len(X_cand), len(y.T)), True)
                )
            if i == 2 and j == 1:
                self.assertEqual(mapping, None)
                np.testing.assert_array_equal(X_cand, candidates)
                expected_A_cand = np.full((len(X_cand), len(y.T)), False)
                expected_A_cand[:, annotators] = True
                np.testing.assert_array_equal(A_cand, expected_A_cand)
            if i == 2 and j == 2:
                self.assertEqual(mapping, None)
                np.testing.assert_array_equal(X_cand, candidates)
                np.testing.assert_array_equal(A_cand, annotators)

        re_val = self.qs._transform_cand_annot(
            candidates=np.arange(2),
            annotators=np.arange(2),
            X=np.array([[1, 2], [0, 1]]),
            y=np.array([[1, MISSING_LABEL], [2, 3]]),
        )
        X_cand, mapping, A_cand = re_val
        np.testing.assert_array_equal(X_cand, np.array([[1, 2], [0, 1]]))

        re_val = self.qs._transform_cand_annot(
            candidates=None,
            annotators=np.array([[False, True], [True, True]]),
            X=np.array([[1, 2], [0, 1]]),
            y=np.array([[1, MISSING_LABEL], [2, 3]]),
        )
        X_cand, mapping, A_cand = re_val
        np.testing.assert_array_equal(
            A_cand, np.array([[False, True], [True, True]])
        )

    def test_consistency_validate_and_transform(self):
        X = np.array([[1, 2], [0, 1]])
        y = np.array([[1, MISSING_LABEL], [2, 3]])
        batch_size_initial = y.shape[0] * y.shape[1]
        candidates_values = [
            None,
            np.array([0, 1]),
            np.array([[3, 4], [0, 1]]),
        ]
        annotators_values = [
            None,
            np.array([0]),
            np.array([[False, True], [True, True]]),
        ]

        for (i, candidates), (j, annotators) in product(
            enumerate(candidates_values), enumerate(annotators_values)
        ):
            X, y, candidates, annotators, batch_size, return_utilities = (
                self.qs._validate_data(
                    candidates=candidates,
                    annotators=annotators,
                    X=X,
                    y=y,
                    batch_size=batch_size_initial,
                    return_utilities=False,
                )
            )

            X_cand, mapping, A_cand = self.qs._transform_cand_annot(
                candidates=candidates,
                annotators=annotators,
                X=X,
                y=y,
            )

            self.assertEqual(np.sum(A_cand).item(), batch_size)


class SkactivemlClassifierTest(unittest.TestCase):
    @patch.multiple(SkactivemlClassifier, __abstractmethods__=set())
    def setUp(self):
        self.clf = SkactivemlClassifier(classes=[0, 1], missing_label=-1)

    def test_fit(self):
        self.assertRaises(NotImplementedError, self.clf.fit, X=None, y=None)

    def test_predict_proba(self):
        self.assertRaises(NotImplementedError, self.clf.predict_proba, X=None)

    def test__validate_data(self):
        X = np.ones((10, 2))
        y = np.random.rand(10)
        # TODO: Wait for scikit-learn update.
        # self.assertRaises(ValueError, self.clf._validate_data, X=X, y=y)
        y = np.full(10, fill_value=-1)
        self.clf.classes = None
        self.assertRaises(ValueError, self.clf._validate_data, X=X, y=y)

    def test__validate_data_multilabel(self):
        X = np.arange(8).reshape(4, 2)
        y = np.array([[0, 1], [-1, -1], [1, 0], [0, 0]])
        clf = DummySkactivemlClassifier(
            classes=[[0, 1], [0, 1]],
            missing_label=-1,
            target_type="multi-label",
        )

        clf.fit(
            X,
            y,
            sample_weight=np.ones(len(y), dtype=float),
        )
        X_valid, y_valid, sample_weight_valid = clf.validated_

        np.testing.assert_array_equal(X_valid, X)
        np.testing.assert_array_equal(y_valid, y)
        np.testing.assert_array_equal(
            sample_weight_valid, np.ones(len(y), dtype=float)
        )
        self.assertEqual(clf.target_spec_.target_type, "multi-label")
        self.assertIsNone(clf.cost_matrix_)

    def test_public_fit_handles_empty_data_and_incremental_target_specs(self):
        empty_clf = DummySkactivemlClassifier(classes=[0, 1], missing_label=-1)
        empty_clf.fit(np.empty((0, 2)), np.empty(0, dtype=int))
        self.assertEqual(empty_clf.classes_.tolist(), [0, 1])

        clf = DummySkactivemlClassifier(missing_label=-1)
        X = np.arange(4).reshape(2, 2)
        clf.fit(X, np.array([0, 1]))
        clf.partial_fit(X, np.array([0, 1]))
        established_spec = clf.target_spec_
        clf.partial_fit(np.empty((0, 2)), np.empty(0, dtype=int))
        self.assertIs(clf.target_spec_, established_spec)
        np.testing.assert_array_equal(clf.classes_, [0, 1])
        clf.classes = [0, 1, 2]
        with self.assertRaisesRegex(ValueError, "cannot change"):
            clf.partial_fit(X, np.array([0, 1, 2]))

        multi_annotator_clf = DummyMultiAnnotatorClassifier(
            classes=[0, 1], missing_label=-1
        )
        multi_annotator_clf.fit(np.empty((0, 2)), np.empty(0, dtype=int))
        self.assertEqual(
            multi_annotator_clf.target_spec_.annotation_type, "multi-annotator"
        )

    def test_partial_fit_rejects_changed_multilabel_dtype_kind(self):
        X = np.arange(4).reshape(2, 2)
        y = np.array([[0, 2], [1, 3]])
        clf = DummySkactivemlClassifier(
            classes=((0, 1), (2, 3)),
            missing_label=-1,
            target_type="multi-label",
        )
        clf.partial_fit(X, y)
        established_spec = clf.target_spec_

        clf.classes = ((0.0, 1.0), (2.0, 3.0))
        with self.assertRaisesRegex(ValueError, "cannot change"):
            clf.partial_fit(X, y.astype(float))

        self.assertIs(clf.target_spec_, established_spec)

    def test_public_fit_validates_weights_and_orders_cost_matrix(self):
        X = np.arange(4).reshape(2, 2)
        y = np.array([0, 1])
        clf = DummySkactivemlClassifier(classes=[0, 1], missing_label=-1)

        with self.assertRaises(ValueError):
            clf.fit(X, y, sample_weight=np.ones(1))
        with self.assertRaises(ValueError):
            clf.fit(X, y, sample_weight=np.ones((2, 1)))

        clf = DummySkactivemlClassifier(
            classes=[1, 0],
            cost_matrix=[[0, 2], [3, 0]],
            missing_label=-1,
        )
        clf.fit(X, y)
        np.testing.assert_array_equal(
            clf.cost_matrix_, np.array([[0, 3], [2, 0]])
        )

    def test_public_predict_preserves_extra_probability_outputs(self):
        clf = DummySkactivemlClassifier(
            classes=[0, 1],
            missing_label=-1,
            probas=(np.array([[0.9, 0.1], [0.1, 0.9]]), "extra"),
            random_state=0,
        )
        X = np.arange(4).reshape(2, 2)
        clf.fit(X, np.array([0, 1]))

        prediction = clf.predict(X)

        self.assertEqual(prediction[1], "extra")
        np.testing.assert_array_equal(prediction[0], [0, 1])

    def test__validate_data_multilabel_invalid_rows(self):
        X = np.arange(6).reshape(3, 2)
        y = np.array([[0, 1], [-1, 1], [1, 0]])
        clf = DummySkactivemlClassifier(
            classes=[[0, 1], [0, 1]],
            missing_label=-1,
            target_type="multi-label",
        )

        self.assertRaises(
            ValueError,
            clf.fit,
            X,
            y,
        )

    def test__validate_data_multilabel_sample_weight(self):
        X = np.arange(8).reshape(4, 2)
        y = np.array([[0, 1], [-1, -1], [1, 0], [0, 0]])
        clf = DummySkactivemlClassifier(
            classes=[[0, 1], [0, 1]],
            missing_label=-1,
            target_type="multi-label",
        )

        clf.fit(
            X,
            y,
            sample_weight=np.ones_like(y, dtype=float),
        )
        np.testing.assert_array_equal(
            clf.validated_[2], np.ones_like(y, dtype=float)
        )
        self.assertRaises(
            ValueError,
            clf.fit,
            X,
            y,
            sample_weight="invalid",
        )
        self.assertRaises(
            ValueError,
            clf.fit,
            X,
            y,
            sample_weight=np.ones((len(y) - 1, y.shape[1])),
        )

    def test__validate_data_multilabel_binary_enforcement(self):
        X = np.arange(6).reshape(3, 2)
        y = np.array([[0, 0], [1, 1], [2, 0]])
        clf = DummySkactivemlClassifier(
            classes=[[0, 1, 2], [0, 1]],
            missing_label=-1,
            target_type="multi-label",
        )

        self.assertRaises(
            ValueError,
            clf.fit,
            X,
            y,
        )

    def test_predict_multilabel(self):
        X = np.arange(4).reshape(2, 2)
        y = np.array([[1, 1], [1, 0]])
        clf = DummySkactivemlClassifier(
            classes=[[0, 1], [0, 1]],
            missing_label=-1,
            probas=np.array([[0.9, 0.9], [0.8, 0.2]]),
            target_type="multi-label",
        )

        clf.fit(X, y)
        np.testing.assert_array_equal(clf.predict(X), y)

    def test_predict_dtype_matches_class_dtype(self):
        # `missing_label=np.nan` widens the label encoder to `float64`, so
        # the decoded labels must be narrowed back to the class dtype.
        X = np.arange(4).reshape(2, 2)
        y = np.array([0, np.nan])
        clf = DummySkactivemlClassifier(
            classes=[0, 1],
            missing_label=np.nan,
            probas=np.array([[0.9, 0.1], [0.2, 0.8]]),
        )

        clf.fit(X, y)
        y_pred = clf.predict(X)

        assert_predicts_class_dtype(self, y_pred, clf.classes_)
        # The decoded labels are usable where class labels are expected.
        np.testing.assert_array_equal(np.array(["a", "b"])[y_pred], ["a", "b"])

    def test_predict_dtype_matches_class_dtype_multilabel(self):
        X = np.arange(4).reshape(2, 2)
        y = np.array([[1, 1], [np.nan, np.nan]])
        clf = DummySkactivemlClassifier(
            classes=[[0, 1], [0, 1]],
            missing_label=np.nan,
            probas=np.array([[0.9, 0.9], [0.8, 0.2]]),
            target_type="multi-label",
        )

        clf.fit(X, y)
        y_pred = clf.predict(X)

        assert_predicts_class_dtype(self, y_pred, clf.classes_)

    def test_predict_dtype_preserves_extra_outputs(self):
        # Only the class labels are narrowed; extra outputs are untouched.
        X = np.arange(4).reshape(2, 2)
        y = np.array([0, np.nan])
        extra = np.array([[0.5], [0.5]], dtype=np.float32)
        clf = DummySkactivemlClassifier(
            classes=[0, 1],
            missing_label=np.nan,
            probas=(np.array([[0.9, 0.1], [0.2, 0.8]]), extra),
        )

        clf.fit(X, y)
        y_pred, extra_pred = clf.predict(X)

        assert_predicts_class_dtype(self, y_pred, clf.classes_)
        self.assertEqual(extra_pred.dtype, np.float32)

    def test_score_multilabel(self):
        X = np.arange(4).reshape(2, 2)
        y = np.array([[1, 1], [1, 1]])
        clf = DummySkactivemlClassifier(
            classes=[[0, 1], [0, 1]],
            missing_label=-1,
            probas=np.array([[0.9, 0.9], [0.8, 0.2]]),
            target_type="multi-label",
        )

        clf.fit(X, y)
        self.assertEqual(clf.score(X, y), 0.5)

    def test_multi_output_is_rejected_before_fitted_state(self):
        X = np.arange(4).reshape(2, 2)
        y = np.array([[2, 1], [1, 0]])
        clf = DummySkactivemlClassifier(
            classes=[[0, 1, 2], [0, 1]],
            missing_label=-1,
            target_type="multi-output",
        )
        with self.assertRaisesRegex(ValueError, "Supported capabilities"):
            clf.fit(X, y)
        self.assertFalse(hasattr(clf, "target_spec_"))


class ClassFrequencyEstimatorTest(unittest.TestCase):
    @patch.multiple(ClassFrequencyEstimator, __abstractmethods__=set())
    def setUp(self):
        self.clf = ClassFrequencyEstimator()

    def test_predict_freq(self):
        self.assertRaises(NotImplementedError, self.clf.predict_freq, X=None)

    def test_base_does_not_advertise_multilabel_capability(self):
        self.assertNotIn(
            ("classification", "multi-label", "single-annotator"),
            self.clf._target_capabilities,
        )

    def test_public_fit_validates_class_prior(self):
        clf = DummyClassFrequencyEstimator(class_prior=1.0)
        clf.fit(np.zeros((2, 1)), np.array([0, 1]))
        np.testing.assert_array_equal(clf.class_prior_, [1.0, 1.0])

    def test_multilabel_fit_validates_class_prior(self):
        fit_params = {
            "X": np.zeros((2, 1)),
            "y": np.array([["no", "on"], ["yes", "off"]]),
        }
        classes = [["no", "yes"], ["off", "on"]]

        clf = DummyMultilabelClassFrequencyEstimator(
            class_prior=1.5,
            classes=classes,
            missing_label=None,
            target_type="multi-label",
        ).fit(**fit_params)
        np.testing.assert_array_equal(clf.class_prior_, np.full((2, 2), 1.5))

        clf = DummyMultilabelClassFrequencyEstimator(
            class_prior=[[1, 2], [3, 4]],
            classes=classes,
            missing_label=None,
            target_type="multi-label",
        ).fit(**fit_params)
        np.testing.assert_array_equal(clf.class_prior_, [[1, 2], [3, 4]])

        for invalid_prior in ([1, 2], [1, 2, 3], [[1, 2]], [[1, -1], [2, 3]]):
            with self.subTest(invalid_prior=invalid_prior):
                clf = DummyMultilabelClassFrequencyEstimator(
                    class_prior=invalid_prior,
                    classes=classes,
                    missing_label=None,
                    target_type="multi-label",
                )
                with self.assertRaisesRegex(ValueError, "n_outputs, 2"):
                    clf.fit(**fit_params)

    def test_predict_proba(self):
        clf = DummyClassFrequencyEstimator(
            freq=np.array([[0.0, 0.0], [1.0, 3.0]]),
            class_prior=0,
        )
        clf.classes_ = np.array([0, 1])
        clf.class_prior_ = np.array([0.0, 0.0])
        P = clf.predict_proba(np.zeros((2, 1)))
        np.testing.assert_array_equal(P[0], np.array([0.5, 0.5]))
        np.testing.assert_array_equal(P[1], np.array([0.25, 0.75]))

    def test_multilabel_predict_proba_returns_positive_classes(self):
        clf = DummyMultilabelClassFrequencyEstimator(
            freq=np.array(
                [
                    [[0.0, 0.0], [1.0, 3.0]],
                    [[3.0, 1.0], [0.0, 0.0]],
                ]
            ),
            class_prior=0,
            classes=[["no", "yes"], ["off", "on"]],
            missing_label=None,
            target_type="multi-label",
        ).fit(
            np.zeros((2, 1)),
            np.array([["no", "on"], ["yes", "off"]]),
        )

        P = clf.predict_proba(np.zeros((2, 1)))

        np.testing.assert_array_equal(P, [[0.5, 0.75], [0.25, 0.5]])

    def test_sample_proba(self):
        clf = DummyClassFrequencyEstimator(
            freq=np.array([[1.0, 2.0], [3.0, 4.0]]),
            class_prior=1.0,
        )
        clf.classes_ = np.array([0, 1])
        clf.class_prior_ = np.array([1.0, 1.0])
        P = clf.sample_proba(np.zeros((2, 1)), n_samples=3, random_state=0)
        self.assertEqual(P.shape, (3, 2, 2))

        clf_zero = DummyClassFrequencyEstimator(
            freq=np.zeros((2, 2)),
            class_prior=0.0,
        )
        clf_zero.classes_ = np.array([0, 1])
        clf_zero.class_prior_ = np.array([0.0, 0.0])
        self.assertRaises(
            ValueError,
            clf_zero.sample_proba,
            np.zeros((2, 1)),
        )

    def test_multilabel_sample_proba_returns_full_binary_vectors(self):
        clf = DummyMultilabelClassFrequencyEstimator(
            freq=np.array(
                [
                    [[1.0, 2.0], [3.0, 4.0]],
                    [[5.0, 6.0], [7.0, 8.0]],
                ]
            ),
            class_prior=1,
            classes=[["no", "yes"], ["off", "on"]],
            missing_label=None,
            target_type="multi-label",
        ).fit(
            np.zeros((2, 1)),
            np.array([["no", "on"], ["yes", "off"]]),
        )

        P = clf.sample_proba(np.zeros((2, 1)), n_samples=3, random_state=0)

        self.assertEqual(P.shape, (3, 2, 2, 2))
        assert_allclose(P.sum(axis=-1), 1)

        clf.freq = np.zeros((2, 2, 2))
        clf.class_prior_ = np.zeros((2, 2))
        with self.assertRaisesRegex(ValueError, "class_prior > 0"):
            clf.sample_proba(np.zeros((2, 1)), random_state=0)


class TestBudgetManager(unittest.TestCase):
    @patch.multiple(BudgetManager, __abstractmethods__=set())
    def setUp(self):
        self.bm = BudgetManager()

    def test_query_by_utility(self):
        self.assertRaises(
            NotImplementedError, self.bm.query_by_utility, utilities=None
        )

    def test_update(self):
        self.assertRaises(
            NotImplementedError,
            self.bm.update,
            candidates=None,
            queried_indices=None,
        )

    def test_validate_budget_and_data(self):
        self.bm.budget = None
        self.bm._validate_budget()
        self.assertEqual(self.bm.budget_, 0.1)

        self.bm.budget = 0.2
        self.bm._validate_budget()
        self.assertEqual(self.bm.budget_, 0.2)

        utilities = np.array([0.1, 0.2])
        np.testing.assert_array_equal(
            self.bm._validate_data(utilities), utilities
        )
        self.assertRaises(TypeError, self.bm._validate_data, [0.1, 0.2])


class SingleAnnotatorStreamQueryStrategyTest(unittest.TestCase):
    @patch.multiple(
        SingleAnnotatorStreamQueryStrategy, __abstractmethods__=set()
    )
    def setUp(self):
        self.qs = SingleAnnotatorStreamQueryStrategy(budget=None)

    def test_query(self):
        self.assertRaises(NotImplementedError, self.qs.query, candidates=None)

    def test_update(self):
        self.assertRaises(
            NotImplementedError,
            self.qs.update,
            candidates=None,
            queried_indices=None,
        )

    def test_validate_helpers(self):
        self.qs._validate_random_state()
        self.assertTrue(hasattr(self.qs, "random_state_"))
        self.qs._validate_budget()
        self.assertEqual(self.qs.budget_, 0.1)
        self.qs.budget = 0.2
        self.qs._validate_budget()
        self.assertEqual(self.qs.budget_, 0.2)

        candidates, return_utilities = self.qs._validate_data(
            candidates=np.zeros((2, 1)),
            return_utilities=True,
        )
        self.assertEqual(candidates.shape, (2, 1))
        self.assertTrue(return_utilities)


class SkactivemlRegressorTest(unittest.TestCase):
    @patch.multiple(SkactivemlRegressor, __abstractmethods__=set())
    def setUp(self):
        self.reg = SkactivemlRegressor(missing_label=-1)

    def test_fit(self):
        self.assertRaises(NotImplementedError, self.reg.fit, X=None, y=None)

    def test_predict(self):
        self.assertRaises(NotImplementedError, self.reg.predict, X=None)

    def test_validate_data(self):
        X = np.arange(5 * 2).reshape(5, 2)
        y = 1 / 2 * np.arange(5)
        self.assertRaises(
            ValueError,
            self.reg._validate_data,
            X=X,
            y=y,
            sample_weight=np.arange(1, 5),
        )

        X_valid, y_valid, sample_weight_valid = self.reg._validate_data(
            X=np.arange(4).reshape(2, 2),
            y=np.array([0.0, 1.0]),
            sample_weight=np.array([1.0, 1.0]),
        )
        np.testing.assert_array_equal(X_valid, np.arange(4).reshape(2, 2))
        np.testing.assert_array_equal(y_valid, np.array([0.0, 1.0]))
        np.testing.assert_array_equal(
            sample_weight_valid, np.array([1.0, 1.0])
        )

        X_empty, y_empty, sample_weight_empty = self.reg._validate_data(
            X=np.array([]),
            y=np.array([]),
        )
        np.testing.assert_array_equal(X_empty, np.array([]))
        np.testing.assert_array_equal(y_empty, np.array([]))
        self.assertIsNone(sample_weight_empty)


class TargetDistributionEstimatorTest(unittest.TestCase):
    @patch.multiple(ProbabilisticRegressor, __abstractmethods__=set())
    def setUp(self):
        self.reg = ProbabilisticRegressor(missing_label=-1)

    def test_predict_target_distribution(self):
        self.assertRaises(
            NotImplementedError, self.reg.predict_target_distribution, X=None
        )

    def test_predict_and_sample_y(self):
        reg = DummyProbabilisticRegressor()
        X = np.zeros((2, 1))
        mu = reg.predict(X)
        np.testing.assert_array_equal(mu, np.array([0.0, 1.0]))

        mu_std = reg.predict(X, return_std=True)
        self.assertEqual(len(mu_std), 2)
        np.testing.assert_array_equal(mu_std[0], np.array([0.0, 1.0]))
        np.testing.assert_array_equal(mu_std[1], np.array([1.0, 2.0]))

        mu_std_entropy = reg.predict(X, return_std=True, return_entropy=True)
        self.assertEqual(len(mu_std_entropy), 3)
        np.testing.assert_array_equal(mu_std_entropy[2], np.array([0.5, 1.5]))

        y_samples = reg.sample_y(X, n_samples=3, random_state=0)
        self.assertEqual(y_samples.shape, (2, 3))


if successful_skorch_torch_import:

    class NeuralNetDummy:
        def forward(self, X):
            return X

    class TestSkorchMixin(unittest.TestCase):
        @patch.multiple(SkorchMixin, __abstractmethods__=set())
        def setUp(self):
            self.sk = SkorchMixin()
            self.sk.neural_net_ = NeuralNetDummy()

        def test__net_parts(self):
            self.assertRaises(
                NotImplementedError, self.sk._net_parts, X=None, y=None
            )

        def test__validate_data_kwargs(self):
            self.assertRaises(
                NotImplementedError, self.sk._validate_data_kwargs
            )

        def test__validate_data(self):
            self.assertRaises(
                NotImplementedError, self.sk._validate_data, X=None, y=None
            )

        def test__return_training_data(self):
            self.assertRaises(
                NotImplementedError,
                self.sk._return_training_data,
                X=None,
                y=None,
            )

        def test_public_initialize_builds_a_neural_net(self):
            self.sk._net_parts = lambda X=None, y=None: (
                torch.nn.Linear,
                torch.nn.MSELoss,
                {
                    "module__in_features": 1,
                    "module__out_features": 1,
                },
            )
            self.sk._validate_data_kwargs = lambda: {}
            self.sk._validate_data = lambda X, y, **kwargs: (
                np.asarray(X),
                np.asarray(y),
                None,
            )
            self.sk._return_training_data = lambda X, y: (X, y)

            initialized = self.sk.initialize()
            initialized_with_data = self.sk.initialize(
                X=np.zeros((1, 1)),
                y=np.zeros(1),
                enforce_check_X_y=True,
            )

            self.assertIs(initialized, self.sk)
            self.assertIs(initialized_with_data[0], self.sk)
            self.assertEqual(initialized_with_data[1].shape, (1, 1))

            self.sk._net_parts = lambda X=None, y=None: (
                torch.nn.Linear,
                torch.nn.MSELoss,
                {"module": torch.nn.Linear},
            )
            with self.assertRaisesRegex(ValueError, "module"):
                self.sk.initialize()

        def test_public_fit_initializes_and_reuses_a_neural_net(self):
            class PublicSkorchEstimator(SkorchMixin):
                def _net_parts(self, X=None, y=None):
                    return (
                        torch.nn.Linear,
                        torch.nn.MSELoss,
                        {
                            "module__in_features": 1,
                            "module__out_features": 1,
                            "max_epochs": 1,
                            "train_split": None,
                            "verbose": 0,
                        },
                    )

                def _validate_data_kwargs(self):
                    return {}

                def _validate_data(self, X, y, **kwargs):
                    return (
                        np.asarray(X, dtype=np.float32),
                        np.asarray(y, dtype=np.float32),
                        None,
                    )

                def _return_training_data(self, X, y):
                    return X, y

                def fit(self, X, y):
                    return self._fit("fit", X, y)

            estimator = PublicSkorchEstimator()
            X = np.array([[0.0], [1.0]])
            y = np.array([[0.0], [1.0]])

            estimator.fit(X, y)
            estimator.target_spec_ = "established"
            estimator.neural_net_.warm_start = True
            estimator.fit(X, y)

            self.assertIsNotNone(estimator.neural_net_)

        def test___forward_with_named_outputs(self):
            # Single-output tests.
            X = torch.ones((5, 10))
            self.assertRaises(
                TypeError,
                self.sk._forward_with_named_outputs,
                X=X,
                forward_outputs=None,
            )
            self.assertRaises(
                ValueError,
                self.sk._forward_with_named_outputs,
                X=X,
                forward_outputs={},
            )
            self.assertRaises(
                TypeError,
                self.sk._forward_with_named_outputs,
                X=X,
                forward_outputs={"samples": (0,)},
            )
            self.assertRaises(
                ValueError,
                self.sk._forward_with_named_outputs,
                X=X,
                forward_outputs={"samples": (1, None)},
            )
            self.assertRaises(
                ValueError,
                self.sk._forward_with_named_outputs,
                X=X,
                forward_outputs={"samples": (-1, None)},
            )
            self.assertRaises(
                TypeError,
                self.sk._forward_with_named_outputs,
                X=X,
                forward_outputs={"samples": (0, "no callable or None")},
            )
            fw_out = self.sk._forward_with_named_outputs(
                X, forward_outputs={"samples": (0, None)}
            )
            self.assertIsInstance(fw_out, np.ndarray)
            np.testing.assert_array_equal(np.ones_like(X), fw_out)
            fw_out = self.sk._forward_with_named_outputs(
                X, forward_outputs={"samples": (0, lambda x: x + 2)}
            )
            self.assertIsInstance(fw_out, np.ndarray)
            np.testing.assert_array_equal(np.ones_like(X) + 2, fw_out)

            # Tuple-output tests.
            X_tuple = (X, X, X)
            for i in range(3):
                fw_out = self.sk._forward_with_named_outputs(
                    X_tuple, forward_outputs={"samples": (i, lambda x: x + 2)}
                )
                self.assertIsInstance(fw_out, np.ndarray)
                np.testing.assert_array_equal(np.ones_like(X) + 2, fw_out)
            forward_outputs = {
                "out_0": (0, None),
                "out_1": (1, lambda x: x + 1),
                "out_2": (2, lambda x: x + 2),
            }
            fw_out = self.sk._forward_with_named_outputs(
                X_tuple,
                forward_outputs=forward_outputs,
            )
            self.assertIsInstance(fw_out, np.ndarray)
            np.testing.assert_array_equal(np.ones_like(X), fw_out)
            fw_out = self.sk._forward_with_named_outputs(
                X_tuple,
                forward_outputs=forward_outputs,
                extra_outputs=["out_2", "out_1"],
            )
            self.assertEqual(len(fw_out), 3)
            np.testing.assert_array_equal(np.ones_like(X), fw_out[0])
            np.testing.assert_array_equal(np.ones_like(X) + 2, fw_out[1])
            np.testing.assert_array_equal(np.ones_like(X) + 1, fw_out[2])

        def test__normalize_extra_outputs(self):
            norm = self.sk._normalize_extra_outputs

            allowed = ["a", "b", "c"]

            # extra_outputs is None -> returns []
            self.assertEqual(norm(None, allowed_names=allowed), [])

            # extra_outputs is str -> single name list
            self.assertEqual(
                norm("a", allowed_names=allowed, primary_name=None),
                ["a"],
            )

            # extra_outputs is sequence of str (list)
            self.assertEqual(
                norm(["a", "c"], allowed_names=allowed, primary_name="b"),
                ["a", "c"],
            )

            # extra_outputs is sequence of str (tuple)
            self.assertEqual(
                norm(("a",), allowed_names=allowed, primary_name=None),
                ["a"],
            )

            # extra_outputs of invalid type ->
            # TypeError (non-str / non-sequence)
            with self.assertRaises(TypeError):
                norm(123, allowed_names=allowed)

            # extra_outputs sequence with non-str element -> TypeError
            with self.assertRaises(TypeError):
                norm(["a", 1], allowed_names=allowed)

            # duplicate names -> ValueError
            with self.assertRaises(ValueError):
                norm(["a", "a"], allowed_names=allowed)

            # unknown names -> ValueError
            with self.assertRaises(ValueError):
                norm(["a", "z"], allowed_names=allowed)

            # primary_name included in extras -> ValueError
            with self.assertRaises(ValueError):
                norm(["a", "b"], allowed_names=allowed, primary_name="b")
