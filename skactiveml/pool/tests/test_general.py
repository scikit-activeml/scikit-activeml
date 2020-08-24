import numpy as np
import unittest

from sklearn.utils.validation import NotFittedError, check_is_fitted
from sklearn.datasets import make_blobs
# from sklearn.preprocessing import StandardScaler
from ...classifier import PWC
from ...semi_supervised import IgnUnlabeledWrapper
from ...utils import is_labeled, is_unlabeled, MISSING_LABEL

from ... import pool


def initialize_class(class_obj, **kwargs):
    parameters = class_obj.__init__.__code__.co_varnames
    kwargs = dict(filter(lambda e: e[0] in parameters, kwargs.items()))
    return class_obj(**kwargs)

class TestGeneral(unittest.TestCase):

    def setUp(self):
        self.MISSING_LABEL = MISSING_LABEL
        self.X, self.y_true = make_blobs(n_samples=10, n_features=2, centers=2, cluster_std=1, random_state=1)
        self.clf = IgnUnlabeledWrapper(PWC(n_classes=2), unlabeled_class=MISSING_LABEL)
        self.budget = 5

        self.query_strategies = {}
        for qs_name in pool.__all__:
            self.query_strategies[qs_name] = getattr(pool, qs_name)


    def test_query_strategies(self):
        for qs_name in self.query_strategies:
            print(qs_name)
            y = np.full(self.y_true.shape, self.MISSING_LABEL)

            qs = initialize_class(self.query_strategies[qs_name],
                          clf=self.clf, perf_est=self.clf, classes = np.unique(self.y_true), random_state=1)

            for b in range(self.budget):
                unlabeled = np.where(is_unlabeled(y))[0]
                self.clf.fit(self.X, y)
                unlabeled_id = qs.query(self.X[unlabeled], X=self.X, y=y, X_eval=self.X, weights=np.ones_like(self.y_true))
                sample_id = unlabeled[unlabeled_id]
                y[sample_id] = self.y_true[sample_id]

