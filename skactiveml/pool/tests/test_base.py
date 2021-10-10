import unittest
from unittest.mock import patch

from skactiveml.base import QueryStrategy, SingleAnnotPoolBasedQueryStrategy, \
    MultiAnnotPoolBasedQueryStrategy


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
