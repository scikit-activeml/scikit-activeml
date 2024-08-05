import unittest
from itertools import product
from unittest.mock import patch

import numpy as np

from skactiveml.base import (
    QueryStrategy,
    SingleAnnotatorPoolQueryStrategy,
    MultiAnnotatorPoolQueryStrategy,
    SkactivemlClassifier,
    ClassFrequencyEstimator,
    AnnotatorModelMixin,
    BudgetManager,
    SingleAnnotatorStreamQueryStrategy,
    SkactivemlRegressor,
    ProbabilisticRegressor,
)
from skactiveml.exceptions import MappingError
from skactiveml.utils import MISSING_LABEL, is_unlabeled


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


class ClassFrequencyEstimatorTest(unittest.TestCase):
    @patch.multiple(ClassFrequencyEstimator, __abstractmethods__=set())
    def setUp(self):
        self.clf = ClassFrequencyEstimator()

    def test_predict_freq(self):
        self.assertRaises(NotImplementedError, self.clf.predict_freq, X=None)


class AnnotatorModelMixinTest(unittest.TestCase):
    @patch.multiple(AnnotatorModelMixin, __abstractmethods__=set())
    def setUp(self):
        self.clf = AnnotatorModelMixin()

    def test_predict_annotator_perf(self):
        self.assertRaises(
            NotImplementedError, self.clf.predict_annotator_perf, X=None
        )


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


class TargetDistributionEstimatorTest(unittest.TestCase):
    @patch.multiple(ProbabilisticRegressor, __abstractmethods__=set())
    def setUp(self):
        self.reg = ProbabilisticRegressor(missing_label=-1)

    def test_predict_target_distribution(self):
        self.assertRaises(
            NotImplementedError, self.reg.predict_target_distribution, X=None
        )
