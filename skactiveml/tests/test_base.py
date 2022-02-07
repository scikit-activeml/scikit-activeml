import unittest
from unittest.mock import patch

import numpy as np

from skactiveml.base import (
    QueryStrategy,
    SingleAnnotPoolBasedQueryStrategy,
    MultiAnnotPoolBasedQueryStrategy,
    SkactivemlClassifier,
    ClassFrequencyEstimator,
    AnnotModelMixin,
    BudgetManager,
    SingleAnnotStreamBasedQueryStrategy,
)
from skactiveml.utils import MISSING_LABEL


class QueryStrategyTest(unittest.TestCase):

    @patch.multiple(QueryStrategy, __abstractmethods__=set())
    def setUp(self):
        self.qs = QueryStrategy()

    def test_fit(self):
        self.assertRaises(NotImplementedError, self.qs.query, X_cand=None)


class SingleAnnotPoolBasedQueryStrategyTest(unittest.TestCase):

    @patch.multiple(SingleAnnotPoolBasedQueryStrategy,
                    __abstractmethods__=set())
    def setUp(self):
        self.qs = SingleAnnotPoolBasedQueryStrategy()

    def test_fit(self):
        self.assertRaises(NotImplementedError, self.qs.query, X=None, y=None,
                          X_cand=None)

    def test__transform_candidates(self):
        for args in [(np.array([[2]]), np.array([[1]]), np.array([0]), True)]:
            with self.assertRaises(ValueError, msg=f'args: {args}'):
                self.qs._transform_candidates(*args)

        self.qs._transform_candidates(candidates=np.array([[2]]),
                                      X=np.array([[2]]),
                                      y=np.array([0]),
                                      enforce_mapping=True)

        X_cand, mapping = self.qs._transform_candidates(
            candidates=np.array([[1]]),
            X=np.array([[2]]),
            y=np.array([0]),
            enforce_mapping=False
        )
        np.testing.assert_array_equal(X_cand, np.array([[1]]))


class MultiAnnotPoolBasedQueryStrategyTest(unittest.TestCase):

    @patch.multiple(MultiAnnotPoolBasedQueryStrategy,
                    __abstractmethods__=set())
    def setUp(self):
        self.qs = MultiAnnotPoolBasedQueryStrategy()
        self.qs.missing_label_ = MISSING_LABEL

    def test_fit(self):
        self.assertRaises(NotImplementedError, self.qs.query,
                          X=np.array([[1, 2]]), y=np.array([[1, ]]))

    def test_transform_cand_annot(self):
        self.assertRaises(ValueError, self.qs._transform_cand_annot,
                          candidates=np.array([[0, 2]]), annotators=None,
                          X=np.array([[1, 2]]), y=np.array([[1, ]]),
                          enforce_mapping=True)
        re_val = self.qs._transform_cand_annot(candidates=np.arange(2),
                                               annotators=np.arange(2),
                                               X=np.array([[1, 2], [0, 1]]),
                                               y=np.array([[1, MISSING_LABEL],
                                                           [2, 3]]))
        X_cand, mapping, A_cand = re_val
        np.testing.assert_array_equal(X_cand, np.array([[1, 2]]))

        re_val = self.qs._transform_cand_annot(candidates=None,
                                               annotators=np.array(
                                                   [[False, True], [True, True]]
                                               ),
                                               X=np.array([[1, 2], [0, 1]]),
                                               y=np.array([[1, MISSING_LABEL],
                                                           [2, 3]]))
        X_cand, mapping, A_cand = re_val
        np.testing.assert_array_equal(A_cand, np.array([[False, True],
                                                        [True, True]]))

        re_val = self.qs._validate_data(candidates=None,
                                        annotators=[1],
                                        X=np.array([[1, 2], [0, 1]]),
                                        y=np.array([[1, MISSING_LABEL],
                                                    [2, 3]]),
                                        batch_size=2,
                                        return_utilities=False)

        X, y, candidates, annotators, batch_size, return_utilities = re_val
        self.assertEqual(1, batch_size)

        re_val = self.qs._validate_data(candidates=[1],
                                        annotators=[1],
                                        X=np.array([[1, 2], [0, 1]]),
                                        y=np.array([[1, MISSING_LABEL],
                                                    [2, 3]]),
                                        batch_size=2,
                                        return_utilities=False)

        X, y, candidates, annotators, batch_size, return_utilities = re_val
        self.assertEqual(0, batch_size)


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
        self.assertRaises(ValueError, self.clf._validate_data, X=X, y=y)
        y = np.full(10, fill_value=-1)
        self.clf.classes = None
        self.assertRaises(ValueError, self.clf._validate_data, X=X, y=y)


class ClassFrequencyEstimatorTest(unittest.TestCase):

    @patch.multiple(ClassFrequencyEstimator, __abstractmethods__=set())
    def setUp(self):
        self.clf = ClassFrequencyEstimator()

    def test_predict_freq(self):
        self.assertRaises(NotImplementedError, self.clf.predict_freq, X=None)


class AnnotModelMixinTest(unittest.TestCase):

    @patch.multiple(AnnotModelMixin, __abstractmethods__=set())
    def setUp(self):
        self.clf = AnnotModelMixin()

    def test_predict_annot_proba(self):
        self.assertRaises(NotImplementedError, self.clf.predict_annot_perf,
                          X=None)


class BudgetManagerTest(unittest.TestCase):
    @patch.multiple(BudgetManager, __abstractmethods__=set())
    def setUp(self):
        self.bm = BudgetManager()

    def test_fit(self):
        self.assertRaises(NotImplementedError, self.bm.query_by_utility,
                          utilities=None)

    def test_update(self):
        self.assertRaises(
            NotImplementedError,
            self.bm.update,
            X_cand=None,
            queried_indices=None,
        )


class SingleAnnotStreamBasedQueryStrategyTest(unittest.TestCase):
    @patch.multiple(
        SingleAnnotStreamBasedQueryStrategy, __abstractmethods__=set()
    )
    def setUp(self):
        self.qs = SingleAnnotStreamBasedQueryStrategy(budget=None)

    def test_fit(self):
        self.assertRaises(NotImplementedError, self.qs.query, X_cand=None)

    def test_update(self):
        self.assertRaises(NotImplementedError, self.qs.update, X_cand=None,
                          queried_indices=None)


