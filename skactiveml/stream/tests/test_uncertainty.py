import unittest
import numpy as np

from sklearn.datasets import make_classification
from sklearn.utils import check_random_state

from .._uncertainty import FixedUncertainty, VariableUncertainty, Split

class TestUncertainty (unittest.TestCase):
    
    def setup():
        rand = np.random.RandomState(0)
        stream_length = 1000
        train_init_size = 10
        training_size = 100
        X, y = make_classification(
            n_samples=stream_length+train_init_size,
            random_state=rand.randint(2**32-1), shuffle=True)
        
        self.X = X[:train_init_size, :]
        self.X_cand = X[train_init_size:, :]
        self.y = y[:train_init_size]
        self.clf = PWC()
        self.kwargs = dict(X_cand=self.X_cand, X=self.X, y=self.y)
    
    
    def test_fixed_uncertainty(self):
        # init param test
        self._test_init_param_clf(FixedUncertainty)
        self._test_init_param_budget_manager(FixedUncertainty)
        
        # quary param test
        self._test_query_param_X_cand(FixedUncertainty)
        self._test_query_param_X(FixedUncertainty)
        self._test_query_param_y(FixedUncertainty)
        
    def test_var_uncertainty(self):
        # init param test
        self._test_init_param_clf(FixedUncertainty)
        self._test_init_param_budget_manager(FixedUncertainty)
        self._test_init_param_theta(FixedUncertainty)
        self._test_init_param_s(FixedUncertainty)
        
        # quary param test
        self._test_query_param_X_cand(FixedUncertainty)
        self._test_query_param_X(FixedUncertainty)
        self._test_query_param_y(FixedUncertainty)
        
    def test_split(self):
        # init param test
        self._test_init_param_clf(Split)
        self._test_init_param_budget_manager(Split)
        self._test_init_param_theta(Split)
        self._test_init_param_s(Split)
        query_strategy = Split(clf=self.clf, v="string")
        self.assertRaises(TypeError, query_strategy.query, **self.kwargs)
        query_strategy = Split(clf=self.clf, v=1.1)
        self.assertRaises(ValueError, query_strategy.query, **self.kwargs)
        
        # quary param test
        self._test_query_param_X_cand(FixedUncertainty)
        self._test_query_param_X(FixedUncertainty)
        self._test_query_param_y(FixedUncertainty)
        
        
    def _test_init_param_clf(self, query_strategy_name):
        query_strategy = query_strategy_name(clf='string')
        self.assertRaises(TypeError, query_strategy.query, **self.kwargs)
        query_strategy = query_strategy_name(clf=1)
        self.assertRaises(TypeError, query_strategy.query, **self.kwargs)
        query_strategy = query_strategy_name()
        self.assertRaises(TypeError, query_strategy.query, **self.kwargs)
        
    def _test_init_param_budget_manager(self, query_strategy_name):
        query_strategy = query_strategy_name(clf=self.clf, 
                                             budget_manager="string")
        self.assertRaises(TypeError, query_strategy.query, **self.kwargs)
        query_strategy = query_strategy_name(clf=self.clf, 
                                             budget_manager=None)
        self.assertRaises(TypeError, query_strategy.query, **self.kwargs)
        
    def _test_init_param_theta(self, query_strategy_name):
        query_strategy = query_strategy_name(clf=self.clf, theta="string")
        self.assertRaises(TypeError, query_strategy.query, **self.kwargs)
        query_strategy = query_strategy_name(clf=self.clf, theta=1.1)
        self.assertRaises(ValueError, query_strategy.query, **self.kwargs)
    
    def _test_init_param_s(self, query_strategy_name):
        query_strategy = query_strategy_name(clf=self.clf, s="string")
        self.assertRaises(TypeError, query_strategy.query, **self.kwargs)
        query_strategy = query_strategy_name(clf=self.clf, s=1.1)
        self.assertRaises(ValueError, query_strategy.query, **self.kwargs)
        
    def _test_query_param_X_cand (self, query_strategy_name):
        query_strategy = query_strategy_name(self.clf)
        self.assertRaises(ValueError, query_strategy.query, X_cand=None,
                          X=self.X,
                          y=self.y)
        self.assertRaises(ValueError, query_strategy.query, X_cand=np.ones(5),
                          X=self.X,
                          y=self.y)
        
    def _test_query_param_X (self, query_strategy_name):
        query_strategy = query_strategy_name(self.clf)
        self.assertRaises(ValueError, query_strategy.query, X_cand=self.X_cand,
                          X=None,
                          y=self.y)
        self.assertRaises(ValueError, query_strategy.query, X_cand=self.X_cand,
                          X=np.ones(5),
                          y=self.y)
        
    def _test_query_param_y (self, query_strategy_name):
        query_strategy = query_strategy_name(self.clf)
        self.assertRaises(ValueError, query_strategy.query, X_cand=self.X_cand,
                          X=self.X,
                          y=None)
        self.assertRaises(ValueError, query_strategy.query, X_cand=self.X_cand,
                          X=self.X,
                          y=np.zeros((len(self.y), 2)))
        