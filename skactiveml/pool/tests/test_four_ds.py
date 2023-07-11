import unittest

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import StandardScaler

from skactiveml.classifier import (
    MixtureModelClassifier,
    ParzenWindowClassifier,
    SklearnClassifier,
)
from skactiveml.pool import FourDs
from skactiveml.utils import MISSING_LABEL, is_unlabeled
from skactiveml.tests.template_query_strategy import (
    TemplateSingleAnnotatorPoolQueryStrategy,
)


class TestFourDs(TemplateSingleAnnotatorPoolQueryStrategy, unittest.TestCase):
    def setUp(self):
        X, y = load_breast_cancer(return_X_y=True)
        X = StandardScaler().fit_transform(X)
        y = y.astype(float)
        y[:50] = MISSING_LABEL
        y[350:] = MISSING_LABEL
        mixture_model = BayesianGaussianMixture(n_components=2, random_state=0)
        mixture_model.fit(X)
        clf = MixtureModelClassifier(
            mixture_model=mixture_model,
            classes=[0, 1],
            missing_label=MISSING_LABEL,
        )
        query_default_params_clf = {
            "X": X,
            "y": y,
            "clf": clf,
            "fit_clf": True,
        }
        super().setUp(
            qs_class=FourDs,
            init_default_params={},
            query_default_params_clf=query_default_params_clf,
        )

    def test_init_param_lmbda(self):
        test_cases = [
            (np.nan, ValueError),
            ("state", TypeError),
            (1.1, ValueError),
            (-0.1, ValueError),
            (0.5, None),
        ]
        self._test_param("init", "lmbda", test_cases)

    def test_query_param_clf(self):
        test_cases = [
            (ParzenWindowClassifier(), TypeError),
            (SklearnClassifier(estimator=ParzenWindowClassifier()), TypeError),
            (MixtureModelClassifier(), None),
        ]
        self._test_param("query", "clf", test_cases)

    def test_query(self):
        init_params = self.init_default_params.copy()
        query_params = self.query_default_params_clf.copy()
        is_unlbld = is_unlabeled(
            query_params["y"], missing_label=MISSING_LABEL
        )
        al4ds = FourDs(**init_params)
        query_indices, utilities = al4ds.query(
            **query_params,
            return_utilities=True,
        )
        self.assertEqual(0, np.sum(utilities[:, is_unlbld] < 0))
        self.assertEqual(0, np.sum(utilities[:, is_unlbld] > 1))
        init_params["lmbda"] = 0
        query_params["y"] = np.full_like(query_params["y"], np.nan)
        query_params["batch_size"] = 10
        al4ds = FourDs(**init_params)
        query_indices, utilities = al4ds.query(
            **query_params,
            return_utilities=True,
        )
        self.assertEqual(0, np.sum(utilities < 0))
        self.assertEqual(0, np.sum(utilities > 1))
