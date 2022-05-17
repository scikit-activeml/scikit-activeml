import unittest

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import StandardScaler

from skactiveml.classifier import MixtureModelClassifier, SklearnClassifier
from skactiveml.pool import FourDs


class TestFourDs(unittest.TestCase):
    def setUp(self):
        self.random_state = 1
        self.X, self.y = load_breast_cancer(return_X_y=True)
        self.y_unlblb = np.full_like(self.y, -1)
        self.X = StandardScaler().fit_transform(self.X)
        mixture_model = BayesianGaussianMixture(n_components=2)
        mixture_model.fit(self.X)
        self.clf = MixtureModelClassifier(
            mixture_model=mixture_model,
            classes=np.unique(self.y),
            missing_label=-1,
        )

    def test_init_param_lmbda(self):
        al4ds = FourDs(lmbda=True, missing_label=-1)
        self.assertRaises(
            TypeError, al4ds.query, X=self.X, y=self.y_unlblb, clf=self.clf
        )
        al4ds = FourDs(lmbda=1.1, missing_label=-1)
        self.assertRaises(
            ValueError, al4ds.query, X=self.X, y=self.y_unlblb, clf=self.clf
        )

    def test_query_param_clf(self):
        al4ds = FourDs(missing_label=-1)
        for clf in [None, SklearnClassifier(GaussianProcessClassifier())]:
            self.assertRaises(
                TypeError, al4ds.query, X=self.X, y=self.y_unlblb, clf=clf
            )
        al4ds = FourDs(missing_label=0)
        self.assertRaises(
            ValueError, al4ds.query, X=self.X, y=self.y_unlblb, clf=self.clf
        )

    def test_query_param_fit_clf(self):
        al4ds = FourDs(missing_label=-1)
        for fit_clf in ["string", None]:
            self.assertRaises(
                TypeError,
                al4ds.query,
                X=self.X,
                y=self.y_unlblb,
                clf=self.clf,
                fit_clf="string",
            )

    def test_query_param_sample_weight(self):
        al4ds = FourDs(missing_label=-1)
        for sample_weight in [np.ones(1), "test"]:
            self.assertRaises(
                ValueError,
                al4ds.query,
                X=self.X,
                y=self.y_unlblb,
                clf=self.clf,
                sample_weight=sample_weight,
            )

    def test_query(self):
        al4ds = FourDs(missing_label=-1, random_state=self.random_state)
        clf = self.clf.fit(self.X, self.y_unlblb)
        query_indices = al4ds.query(
            X=self.X, y=self.y_unlblb, clf=clf, fit_clf=False
        )
        self.assertEqual(1, len(query_indices))
        query_indices, utilities = al4ds.query(
            X=self.X,
            y=self.y_unlblb,
            clf=clf,
            fit_clf=False,
            return_utilities=True,
        )
        self.assertEqual(1, len(query_indices))
        self.assertEqual((1, len(self.y)), utilities.shape)
        self.assertEqual(0, np.sum(utilities < 0))
        query_indices, utilities = al4ds.query(
            X=self.X,
            y=self.y_unlblb,
            clf=clf,
            batch_size=3,
            fit_clf=False,
            return_utilities=True,
        )
        self.assertEqual(3, len(query_indices))
        self.assertEqual((3, len(self.y)), utilities.shape)
        self.assertEqual(3, np.sum(np.isnan(utilities)))
        query_indices, utilities = al4ds.query(
            X=self.X,
            y=self.y_unlblb,
            clf=clf,
            fit_clf=False,
            batch_size=len(self.X) + 1,
            return_utilities=True,
        )
        self.assertEqual(len(self.y), len(query_indices))
        self.assertEqual((len(self.y), len(self.y)), utilities.shape)
        self.assertEqual(
            np.sum(np.arange(0, len(self.y))), np.sum(np.isnan(utilities))
        )
