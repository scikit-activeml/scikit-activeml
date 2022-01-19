import unittest

import numpy as np
from sklearn.datasets import make_classification

from skactiveml.classifier import PWC
from skactiveml.stream import (
    FixedUncertainty,
    VariableUncertainty,
    Split,
    RandomVariableUncertainty,
)


class TemplateTestUncertainty:
    def setUp(self):
        # initialise valid data to test uncertainty parameters
        rand = np.random.RandomState(0)
        stream_length = 1000
        train_init_size = 10
        X, y = make_classification(
            n_samples=stream_length + train_init_size,
            random_state=rand.randint(2 ** 31 - 1),
            shuffle=True,
        )

        self.X = X[:train_init_size, :]
        self.X_cand = X[train_init_size:, :]
        self.y = y[:train_init_size]
        self.clf = PWC()
        self.kwargs = dict(
            X_cand=self.X_cand, clf=self.clf, X=self.X, y=self.y
        )

    def test_init_param_budget(self):
        # budget must be defined as a float greater than 0
        query_strategy = self.get_query_strategy()(budget=[])
        self.assertRaises(TypeError, query_strategy.query, **(self.kwargs))
        query_strategy = self.get_query_strategy()(budget="string")
        self.assertRaises(TypeError, query_strategy.query, **(self.kwargs))
        query_strategy = self.get_query_strategy()(budget=-1)
        self.assertRaises(TypeError, query_strategy.query, **(self.kwargs))

    def test_init_param_budget_manager(self):
        # budget_manager must be defined as an object of an budget manager
        # class
        query_strategy = self.get_query_strategy()(budget_manager=[])
        self.assertRaises(TypeError, query_strategy.query, **(self.kwargs))

    def test_init_param_random_state(self):
        query_strategy = self.get_query_strategy()(random_state="string",)
        self.assertRaises(ValueError, query_strategy.query, **(self.kwargs))

    def test_query_param_X_cand(self):
        # X_cand must be defined as a two dimensinal array
        query_strategy = self.get_query_strategy()()
        self.assertRaises(
            ValueError,
            query_strategy.query,
            X_cand=1,
            clf=self.clf,
            X=self.X,
            y=self.y,
        )
        self.assertRaises(
            ValueError,
            query_strategy.query,
            X_cand=None,
            clf=self.clf,
            X=self.X,
            y=self.y,
        )
        self.assertRaises(
            ValueError,
            query_strategy.query,
            X_cand=np.ones(5),
            clf=self.clf,
            X=self.X,
            y=self.y,
        )

    def test_query_param_clf(self):
        # clf must be defined as a classifier
        query_strategy = self.get_query_strategy()()
        self.assertRaises(
            TypeError,
            query_strategy.query,
            X_cand=self.X_cand,
            clf="string",
            X=self.X,
            y=self.y,
        )
        query_strategy = self.get_query_strategy()()
        self.assertRaises(
            TypeError,
            query_strategy.query,
            X_cand=self.X_cand,
            clf=1,
            X=self.X,
            y=self.y,
        )

    def test_query_param_X(self):
        # X must be defined as a two dimensinal array and must be equal in
        # length to y
        query_strategy = self.get_query_strategy()()
        self.assertRaises(
            ValueError,
            query_strategy.query,
            X_cand=self.X_cand,
            clf=self.clf,
            X=1,
            y=self.y,
            fit_clf=True
        )
        self.assertRaises(
            ValueError,
            query_strategy.query,
            X_cand=self.X_cand,
            clf=self.clf,
            X=None,
            y=self.y,
            fit_clf=True
        )
        self.assertRaises(
            ValueError,
            query_strategy.query,
            X_cand=self.X_cand,
            clf=self.clf,
            X=np.ones(5),
            y=self.y,
            fit_clf=True
        )
        self.assertRaises(
            ValueError,
            query_strategy.query,
            X_cand=self.X_cand,
            clf=self.clf,
            X=self.X[1:],
            y=self.y,
            fit_clf=True
        )

    def test_query_param_y(self):
        # y must be defined as a one Dimensional array and must be equal in
        # length to X
        query_strategy = self.get_query_strategy()()
        self.assertRaises(
            TypeError,
            query_strategy.query,
            X_cand=self.X_cand,
            clf=self.clf,
            X=self.X,
            y=1,
            fit_clf=True
        )
        self.assertRaises(
            TypeError,
            query_strategy.query,
            X_cand=self.X_cand,
            clf=self.clf,
            X=self.X,
            y=None,
            fit_clf=True
        )
        self.assertRaises(
            ValueError,
            query_strategy.query,
            X_cand=self.X_cand,
            clf=self.clf,
            X=self.X,
            y=self.y[1:],
            fit_clf=True
        )

    def test_query_param_sample_weight(self):
        # sample weight needs to be a list that can be convertet to float
        # equal in size of y
        query_strategy = self.get_query_strategy()()
        self.assertRaises(
            TypeError,
            query_strategy.query,
            X_cand=self.X_cand,
            clf=self.clf,
            X=self.X,
            y=self.y[1:],
            sample_weight="string",
            fit_clf=True
        )
        self.assertRaises(
            ValueError,
            query_strategy.query,
            X_cand=self.X_cand,
            clf=self.clf,
            X=self.X,
            y=self.y[1:],
            sample_weight=["string", "numbers", "test"],
            fit_clf=True
        )
        self.assertRaises(
            ValueError,
            query_strategy.query,
            X_cand=self.X_cand,
            clf=self.clf,
            X=self.X,
            y=self.y[1:],
            sample_weight=[1],
            fit_clf=True
        )

    def test_query_param_fit_clf(self):
        # fit_clf needs to be a boolean
        query_strategy = self.get_query_strategy()()
        self.assertRaises(
            TypeError,
            query_strategy.query,
            X_cand=self.X_cand,
            clf=self.clf,
            X=self.X,
            y=self.y,
            fit_clf="string",
        )
        self.assertRaises(
            TypeError,
            query_strategy.query,
            X_cand=self.X_cand,
            clf=self.clf,
            X=self.X,
            y=self.y,
            fit_clf=1,
        )

    def test_query_param_return_utilities(self):
        # return_utilities needs to be a boolean
        query_strategy = self.get_query_strategy()()
        self.assertRaises(
            TypeError,
            query_strategy.query,
            X_cand=self.X_cand,
            clf=self.clf,
            X=self.X,
            y=self.y[1:],
            return_utilities="string",
        )
        self.assertRaises(
            TypeError,
            query_strategy.query,
            X_cand=self.X_cand,
            clf=self.clf,
            X=self.X,
            y=self.y[1:],
            return_utilities=1,
        )


class TestSplit(TemplateTestUncertainty, unittest.TestCase):
    def get_query_strategy(self):
        return Split


class TestFixedUncertainty(TemplateTestUncertainty, unittest.TestCase):
    def get_query_strategy(self):
        return FixedUncertainty


class TestVariableUncertainty(TemplateTestUncertainty, unittest.TestCase):
    def get_query_strategy(self):
        return VariableUncertainty


class TestRandomVariableUncertainty(
    TemplateTestUncertainty, unittest.TestCase
):
    def get_query_strategy(self):
        return RandomVariableUncertainty
