import unittest

import numpy as np
from sklearn.datasets import make_blobs

from skactiveml import pool
from skactiveml.classifier import PWC, CMM
from skactiveml.utils import is_unlabeled, MISSING_LABEL, initialize_class_with_kwargs


class TestGeneral(unittest.TestCase):

    def setUp(self):
        self.MISSING_LABEL = MISSING_LABEL
        self.X, self.y_true = make_blobs(n_samples=10, n_features=2, centers=2, cluster_std=1, random_state=1)
        self.budget = 5

        self.query_strategies = {}
        for qs_name in pool.__all__:
            self.query_strategies[qs_name] = getattr(pool, qs_name)
        print(self.query_strategies.keys())

    def test_query_strategies(self):
        for qs_name in self.query_strategies:
            if qs_name == "FourDS":
                clf = CMM(classes=np.unique(self.y_true), missing_label=MISSING_LABEL)
            else:
                clf = PWC(classes=np.unique(self.y_true), missing_label=MISSING_LABEL)

            with self.subTest(msg="Basic Testing of query strategy", qs_name=qs_name):
                y = np.full(self.y_true.shape, self.MISSING_LABEL)
                y[0:5] = self.y_true[0:5]

                qs = initialize_class_with_kwargs(self.query_strategies[qs_name],
                                                  clf=clf, perf_est=clf, classes=np.unique(self.y_true), random_state=1)

                for b in range(self.budget):
                    unlabeled = np.where(is_unlabeled(y))[0]
                    clf.fit(self.X, y)
                    unlabeled_id = qs.query(self.X[unlabeled], X=self.X, y=y, X_eval=self.X,
                                            sample_weight=np.ones(len(unlabeled)))
                    sample_id = unlabeled[unlabeled_id]
                    y[sample_id] = self.y_true[sample_id]
