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
        self.assertRaises(NotImplementedError, self.qs.query, X_cand=None)


class MultiAnnotPoolBasedQueryStrategyTest(unittest.TestCase):

    @patch.multiple(MultiAnnotPoolBasedQueryStrategy,
                    __abstractmethods__=set())
    def setUp(self):
        self.qs = MultiAnnotPoolBasedQueryStrategy()

    def test_fit(self):
        self.assertRaises(NotImplementedError, self.qs.query, X_cand=None)


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
        self.qs = SingleAnnotStreamBasedQueryStrategy(budget_manager=None)

    def test_fit(self):
        self.assertRaises(NotImplementedError, self.qs.query, X_cand=None)

    def test_update(self):
        self.assertRaises(NotImplementedError, self.qs.update, X_cand=None,
                          queried_indices=None)

    def test_get_default_budget_manager(self):
        self.assertRaises(NotImplementedError,
                          self.qs.get_default_budget_manager)
