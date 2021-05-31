import unittest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

from skactiveml.classifier import PWC, SklearnClassifier
from skactiveml.pool import UncertaintySampling
from skactiveml.utils import is_unlabeled, MISSING_LABEL
from skactiveml.visualization import _feature_space
from sklearn.utils.multiclass import type_of_target
from sklearn.linear_model import LogisticRegression







class TestFeatureSpace(unittest.TestCase):

    def setUp(self):

        self.X = make_classification(n_features=2, n_redundant=0, random_state=0)[0]
        self.y_oracle = make_classification(n_features=2, n_redundant=0, random_state=0)[1]
        self.y = np.full(shape=self.y_oracle.shape, fill_value=[0, 1])


    def test_input_validation(self):

        self.assertEqual(len(self.X), len(self.y))
        self.assertEqual(len(self.y), len(self.y_oracle))


    def test_on_data_set(self):

        X, y_true = make_classification(n_features=2, n_redundant=0, random_state=0)
        clf = SklearnClassifier(LogisticRegression(), classes=np.unique(y_true))
        qs = UncertaintySampling(clf, method='entropy', random_state=42)
        y = np.full(shape=y_true.shape, fill_value=MISSING_LABEL)
        unlbld_idx = np.where(is_unlabeled(y))[0]
        X_cand = X[unlbld_idx]
        query_idx = unlbld_idx[qs.query(X_cand=X_cand, X=X, y=y, batch_size=1)]
        y[query_idx] = y_true[query_idx]
        clf.fit(X, y)
        y_pred = clf.predict(X)
        self.assertTrue(accuracy_score(y_true, y_pred) > 0.5)



if __name__ == '__main__':
    unittest.main()
