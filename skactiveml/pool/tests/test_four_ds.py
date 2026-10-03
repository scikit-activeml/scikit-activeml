import unittest

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
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

    def test_query_single_class(self):
        X = np.arange(6, dtype=float).reshape(-1, 1)
        y = np.array(
            [
                0.0,
                MISSING_LABEL,
                MISSING_LABEL,
                MISSING_LABEL,
                MISSING_LABEL,
                MISSING_LABEL,
            ]
        )
        mixture_model = BayesianGaussianMixture(
            n_components=1, random_state=0
        ).fit(X)
        clf = MixtureModelClassifier(
            mixture_model=mixture_model,
            random_state=0,
        )

        query_indices, utilities = FourDs(random_state=0).query(
            X, y, clf, return_utilities=True
        )

        self.assertIn(query_indices[0], range(1, len(X)))
        self.assertTrue(np.isfinite(utilities[0, 1:]).all())

    def test_query_equal_density(self):
        for X in (
            np.ones((4, 1)),
            np.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]]),
        ):
            y = np.full(len(X), np.nan)
            clf = MixtureModelClassifier(
                mixture_model=GaussianMixture(n_components=1, random_state=0),
                classes=[0, 1],
            ).fit(X, y)
            for lmbda in (None, 0, 0.2, 1):
                for candidates in (None, [3, 1, 2], X[[3, 1, 2]]):
                    with self.subTest(X=X, lmbda=lmbda, candidates=candidates):
                        batch_size = (
                            len(X) if candidates is None else len(candidates)
                        )
                        qs = FourDs(lmbda=lmbda, random_state=0)
                        with np.errstate(divide="raise", invalid="raise"):
                            indices, utilities = qs.query(
                                X,
                                y,
                                clf,
                                fit_clf=False,
                                candidates=candidates,
                                batch_size=batch_size,
                                return_utilities=True,
                            )
                        self.assertEqual(indices.shape, (batch_size,))
                        self.assertEqual(len(np.unique(indices)), batch_size)
                        eligible = np.ones(utilities.shape[1], dtype=bool)
                        if isinstance(candidates, list):
                            eligible[:] = False
                            eligible[candidates] = True
                        for index, row in zip(indices, utilities):
                            self.assertTrue(eligible[index])
                            self.assertTrue(np.isfinite(row[eligible]).all())
                            self.assertTrue(np.isnan(row[~eligible]).all())
                            self.assertTrue((row[eligible] >= 0).all())
                            self.assertTrue((row[eligible] <= 1).all())
                            eligible[index] = False

    def test_query_batch_prefix(self):
        X = np.array([[-4.0], [-3.0], [0.0], [0.3], [2.0], [5.0]])
        for y in (
            np.full(len(X), np.nan),
            np.array([0, 1, np.nan, np.nan, np.nan, np.nan]),
        ):
            clf = MixtureModelClassifier(
                mixture_model=GaussianMixture(n_components=2, random_state=0),
                classes=[0, 1],
            ).fit(X, y)
            for lmbda in (0, 0.2, 1):
                params = dict(
                    X=X, y=y, clf=clf, fit_clf=False, return_utilities=True
                )
                expected_indices, expected_utilities = FourDs(
                    lmbda=lmbda, random_state=0
                ).query(**params, batch_size=4)
                for batch_size in (1, 2, 3):
                    with self.subTest(y=y, lmbda=lmbda, batch_size=batch_size):
                        indices, utilities = FourDs(
                            lmbda=lmbda, random_state=0
                        ).query(**params, batch_size=batch_size)
                        np.testing.assert_array_equal(
                            indices, expected_indices[:batch_size]
                        )
                        np.testing.assert_allclose(
                            utilities, expected_utilities[:batch_size]
                        )

    def test_query_distribution_with_selected_samples(self):
        X = np.array(
            [[-12.0], [-11.0], [-10.0], [-9.0], [-8.0], [-7.0], [9.0], [11.0]]
        )
        for labeled_indices in ([], [6, 7]):
            with self.subTest(labeled_indices=labeled_indices):
                y = np.full(len(X), np.nan)
                if labeled_indices:
                    y[labeled_indices] = [0, 1]
                clf = MixtureModelClassifier(
                    mixture_model=GaussianMixture(
                        n_components=2, random_state=0
                    ),
                    classes=[0, 1],
                ).fit(X, y)
                responsibilities = clf.mixture_model_.predict_proba(X)
                weights = clf.mixture_model_.weights_
                if labeled_indices:
                    difference = np.abs(
                        weights
                        - responsibilities[labeled_indices].mean(axis=0)
                    ).sum()
                    self.assertGreaterEqual(difference, 1)
                indices, utilities = FourDs(lmbda=0, random_state=0).query(
                    X,
                    y,
                    clf,
                    fit_clf=False,
                    batch_size=4,
                    return_utilities=True,
                )
                selected = []
                for index, row in zip(indices, utilities):
                    expected = np.full(len(X), np.nan)
                    for candidate in range(len(X)):
                        if (
                            candidate in labeled_indices
                            or candidate in selected
                        ):
                            continue
                        prospective = labeled_indices + selected + [candidate]
                        mean = responsibilities[prospective].mean(axis=0)
                        expected[candidate] = np.minimum(weights, mean).sum()
                    np.testing.assert_allclose(row, expected)
                    selected.append(index)
