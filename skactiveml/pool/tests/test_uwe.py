import unittest

import numpy as np
from sklearn.base import clone
from sklearn.datasets import make_blobs
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from skactiveml.classifier import ParzenWindowClassifier, SklearnClassifier
from skactiveml.pool import UWE, CoreSet, uncertainty_scores
from skactiveml.regressor import NadarayaWatsonRegressor
from skactiveml.utils import MISSING_LABEL
from skactiveml.tests.template_query_strategy import (
    TemplateSingleAnnotatorPoolQueryStrategy,
)
from skactiveml.tests.utils import (
    ParzenWindowClassifierEmbedding,
    ParzenWindowClassifierEmbeddingUncertainty,
    ParzenWindowClassifierTriplet,
    ParzenWindowClassifierTuple,
)


class NadarayaWatsonRegressorUncertainty(NadarayaWatsonRegressor):
    """Regressor returning its predictive standard deviations as
    uncertainties."""

    def predict(self, X):
        return super().predict(X, return_std=True, return_entropy=False)


class ParzenWindowClassifierCustomOutput(ParzenWindowClassifier):
    """Classifier returning the given outputs in addition to its class
    probabilities, where a callable output is evaluated on `X`."""

    def __init__(self, outputs=(), classes=None, random_state=None):
        super().__init__(classes=classes, random_state=random_state)
        self.outputs = outputs

    def predict_proba(self, X):
        probas = super().predict_proba(X)
        outputs = [o(X) if callable(o) else o for o in self.outputs]
        return (probas, *outputs)


def _reference_uwe_kcg(
    X_cand, X_lbld, uncertainties_cand, uncertainties_lbld, batch_size, tau=1
):
    """Literal implementation of Algorithm 1 in [1] based on explicit
    pairwise distances, used to check the vectorized implementation."""
    if not np.any(uncertainties_cand) and not np.any(uncertainties_lbld):
        uncertainties_cand = np.ones_like(uncertainties_cand)
        uncertainties_lbld = np.ones_like(uncertainties_lbld)
    uwe_cand = (uncertainties_cand**tau)[:, None] * X_cand
    uwe_lbld = (uncertainties_lbld**tau)[:, None] * X_lbld
    centers = list(uwe_lbld)
    selected = []
    for _ in range(batch_size):
        if len(centers) == 0:
            dist = [np.linalg.norm(e) for e in uwe_cand]
        else:
            dist = [
                min(np.linalg.norm(e - c) for c in centers) for e in uwe_cand
            ]
        dist = np.asarray(dist, dtype=float)
        dist[selected] = -np.inf
        idx = int(np.argmax(dist))
        selected.append(idx)
        centers.append(uwe_cand[idx])
    return np.array(selected)


class TestUWE(TemplateSingleAnnotatorPoolQueryStrategy, unittest.TestCase):
    def setUp(self):
        self.classes = [0, 1]
        X = np.linspace(0, 1, 20).reshape(10, 2)
        y = np.hstack([[0, 1], np.full(8, MISSING_LABEL)])
        self.query_default_params_clf = {
            "X": X,
            "y": y,
            "estimator": ParzenWindowClassifier(
                classes=self.classes, random_state=42
            ),
        }
        self.query_default_params_clf_embedding = {
            "X": X,
            "y": y,
            "estimator": ParzenWindowClassifierEmbeddingUncertainty(
                classes=self.classes,
            ),
        }
        self.query_default_params_reg = {
            "X": X,
            "y": y,
            "estimator": NadarayaWatsonRegressorUncertainty(),
        }
        super().setUp(
            qs_class=UWE,
            init_default_params={},
            query_default_params_clf=self.query_default_params_clf,
            query_default_params_reg=self.query_default_params_reg,
        )

    def test_target_contract(self):
        strategy = UWE()
        self.assertEqual(strategy.target_type, "auto")
        self.assertEqual(
            strategy._target_capabilities,
            frozenset(
                {
                    (
                        "classification",
                        "single-output",
                        "single-annotator",
                    ),
                    ("regression", "single-output", "single-annotator"),
                }
            ),
        )

    def test_fitted_multilabel_classifier_rejected_before_state(self):
        self._test_fitted_multilabel_classifier_rejection(
            estimator_param="estimator", fit_param="fit_estimator"
        )

    def test_init_param_predict_dict(self, test_cases=None):
        test_cases = [] if test_cases is None else test_cases
        test_cases += [
            (1, TypeError),
            (None, None),
            (False, TypeError),
            (True, TypeError),
            ({"return_embeddings": True}, None),
            ({"return_uncertainties": "1d"}, None),
            ({"return_embeddings": True, "return_uncertainties": False}, None),
            ({"return_embeddings": True, "return_uncertainties": "1d"}, None),
            (
                {"return_embeddings": True, "return_uncertainties": "2d"},
                ValueError,
            ),
            ({"test": True}, TypeError),
        ]
        self._test_param(
            "init",
            "predict_dict",
            replace_query_params=self.query_default_params_clf_embedding,
            test_cases=test_cases,
        )

    def test_init_param_method(self, test_cases=None):
        test_cases = [] if test_cases is None else test_cases
        test_cases += [
            (1, ValueError),
            (None, ValueError),
            ("string", ValueError),
            ("entropy", None),
            ("margin_sampling", None),
            ("least_confident", None),
        ]
        self._test_param("init", "method", test_cases)

    def test_init_param_tau(self, test_cases=None):
        test_cases = [] if test_cases is None else test_cases
        test_cases += [
            ("1", TypeError),
            (None, TypeError),
            (True, TypeError),
            ([1], TypeError),
            (-1, ValueError),
            (-0.1, ValueError),
            (np.nan, ValueError),
            (np.inf, ValueError),
            (0, None),
            (0.1, None),
            (1, None),
            (2.5, None),
        ]
        self._test_param("init", "tau", test_cases)

    def test_query_param_estimator(self, test_cases=None):
        test_cases = [] if test_cases is None else test_cases
        test_cases += [
            (SVC(), TypeError),
            (SklearnClassifier(SVC()), AttributeError),
            (SklearnClassifier(SVC(probability=True)), None),
            (
                SklearnClassifier(LogisticRegression(), classes=self.classes),
                None,
            ),
            # Regressors must provide uncertainties.
            (NadarayaWatsonRegressor(), ValueError),
        ]
        self._test_param("query", "estimator", test_cases)

    def test_query_param_fit_estimator(self, test_cases=None):
        test_cases = [] if test_cases is None else test_cases
        test_cases += [
            (1, TypeError),
            (0, TypeError),
            (None, TypeError),
            (True, None),
            (False, NotFittedError),
        ]
        self._test_param("query", "fit_estimator", test_cases)

    def test_query(self):
        X, y_true = make_blobs(n_samples=60, centers=4, random_state=0)
        y_true = y_true.astype(float)
        classes = np.unique(y_true)
        clf = ParzenWindowClassifier(classes=classes)
        X_new = np.random.RandomState(1).uniform(-5, 10, size=(15, 2))
        for n_labeled in [0, 1, 8]:
            y = np.full(len(X), MISSING_LABEL)
            y[:n_labeled] = y_true[:n_labeled]
            clf_fit = clone(clf).fit(X, y)
            lbld = np.arange(n_labeled)
            for method in ["entropy", "least_confident", "margin_sampling"]:
                for tau in [0.1, 1, 2]:
                    for candidates in [
                        None,
                        np.arange(20, 50),
                        X_new,
                    ]:
                        with self.subTest(
                            n_labeled=n_labeled,
                            method=method,
                            tau=tau,
                            candidates=(
                                None
                                if candidates is None
                                else candidates.shape
                            ),
                        ):
                            if candidates is None:
                                cand_idx = np.arange(n_labeled, len(X))
                                X_cand = X[cand_idx]
                            elif candidates.ndim == 1:
                                cand_idx = candidates
                                X_cand = X[cand_idx]
                            else:
                                cand_idx = np.arange(len(candidates))
                                X_cand = candidates

                            u_cand = uncertainty_scores(
                                clf_fit.predict_proba(X_cand), method=method
                            )
                            u_lbld = (
                                uncertainty_scores(
                                    clf_fit.predict_proba(X[lbld]),
                                    method=method,
                                )
                                if n_labeled > 0
                                else np.empty(0)
                            )
                            expected = cand_idx[
                                _reference_uwe_kcg(
                                    X_cand, X[lbld], u_cand, u_lbld, 5, tau
                                )
                            ]
                            qs = UWE(method=method, tau=tau, random_state=0)
                            query_indices, utilities = qs.query(
                                X,
                                y,
                                estimator=clf,
                                candidates=candidates,
                                batch_size=5,
                                return_utilities=True,
                            )
                            np.testing.assert_array_equal(
                                query_indices, expected
                            )
                            self.assertEqual(query_indices.dtype.kind, "i")
                            n_util = len(X) if X_cand is not X_new else 15
                            self.assertEqual(utilities.shape, (5, n_util))

                            # Utilities are masked outside the candidates and
                            # for already selected samples, and the selected
                            # sample maximizes the utilities of its round.
                            is_cand = np.zeros(n_util, dtype=bool)
                            is_cand[cand_idx] = True
                            self.assertTrue(
                                np.isnan(utilities[:, ~is_cand]).all()
                            )
                            for i, idx in enumerate(query_indices):
                                self.assertTrue(
                                    np.isnan(
                                        utilities[i, query_indices[:i]]
                                    ).all()
                                )
                                self.assertEqual(
                                    utilities[i, idx],
                                    np.nanmax(utilities[i]),
                                )
                                self.assertTrue(
                                    (utilities[i, is_cand] >= 0)[
                                        ~np.isnan(utilities[i, is_cand])
                                    ].all()
                                )
                            # The covering radius is non-increasing. Without
                            # labeled samples, the first row holds the norms.
                            radius = np.nanmax(utilities, axis=1)
                            if n_labeled == 0:
                                radius = radius[1:]
                            self.assertTrue((np.diff(radius) <= 1e-12).all())

    def test_query_classifier_output_syntaxes(self):
        # Embeddings and uncertainties returned by the classifier in any
        # supported syntax lead to the same selection as the fallback.
        X, y_true = make_blobs(n_samples=50, centers=3, random_state=0)
        y = np.full(len(X), MISSING_LABEL)
        y[:6] = y_true[:6]
        classes = np.unique(y_true)
        expected = None
        for clf, predict_dict in [
            (ParzenWindowClassifier(classes=classes), None),
            (
                ParzenWindowClassifierEmbedding(classes=classes),
                {"return_embeddings": True},
            ),
            (ParzenWindowClassifierTuple(classes=classes), None),
            (
                ParzenWindowClassifierEmbeddingUncertainty(classes=classes),
                {"return_embeddings": True, "return_uncertainties": "1d"},
            ),
            (
                ParzenWindowClassifierEmbeddingUncertainty(classes=classes),
                {"return_uncertainties": "1d"},
            ),
            (ParzenWindowClassifierTriplet(classes=classes), None),
        ]:
            with self.subTest(clf=type(clf).__name__, kwargs=predict_dict):
                query_indices, utilities = UWE(
                    predict_dict=predict_dict, random_state=0
                ).query(
                    X,
                    y,
                    estimator=clf,
                    batch_size=4,
                    return_utilities=True,
                )
                if expected is None:
                    expected = (query_indices, utilities)
                else:
                    np.testing.assert_array_equal(query_indices, expected[0])
                    np.testing.assert_allclose(utilities, expected[1])

    def test_query_custom_embeddings_and_uncertainties(self):
        # The embeddings are taken from the estimator and the uncertainties
        # weight them, i.e., `uwe(x) = u(x)**tau * f(x)`.
        X = np.arange(12, dtype=float).reshape(6, 2)
        y = np.array([0, 1] + [MISSING_LABEL] * 4)
        emb = np.array([[0, 0], [10, 0], [1, 0], [2, 0], [3, 0], [4, 0.0]])
        unc = np.array([1, 1, 4, 1, 1, 1.75])

        def outputs_for(X_pred):
            idx = (X_pred[:, 0] // 2).astype(int)
            return emb[idx], unc[idx]

        clf = ParzenWindowClassifierCustomOutput(
            outputs=(
                lambda X_pred: outputs_for(X_pred)[0],
                lambda X_pred: outputs_for(X_pred)[1],
            ),
            classes=[0, 1],
        )
        # UWEs: [0, 0], [10, 0], [4, 0], [2, 0], [3, 0], [7, 0].
        query_indices, utilities = UWE(random_state=0).query(
            X, y, estimator=clf, batch_size=3, return_utilities=True
        )
        np.testing.assert_array_equal(query_indices, [2, 5, 3])
        np.testing.assert_allclose(
            utilities,
            [
                [np.nan, np.nan, 4, 2, 3, 3],
                [np.nan, np.nan, np.nan, 2, 1, 3],
                [np.nan, np.nan, np.nan, 2, 1, np.nan],
            ],
        )

        # `tau=0` ignores the uncertainties.
        query_indices = UWE(tau=0, random_state=0).query(
            X, y, estimator=clf, batch_size=2
        )
        np.testing.assert_array_equal(query_indices, [5, 3])

    def test_query_tau_zero_equals_core_set(self):
        X, y_true = make_blobs(n_samples=80, centers=4, random_state=2)
        y = np.full(len(X), MISSING_LABEL)
        y[:10] = y_true[:10]
        clf = ParzenWindowClassifier(classes=np.unique(y_true))
        for candidates in [None, np.arange(30, 70)]:
            with self.subTest(candidates=candidates):
                query_indices_uwe = UWE(tau=0, random_state=0).query(
                    X, y, estimator=clf, candidates=candidates, batch_size=6
                )
                query_indices_core_set = CoreSet(random_state=0).query(
                    X, y, candidates=candidates, batch_size=6
                )
                np.testing.assert_array_equal(
                    query_indices_uwe, query_indices_core_set
                )

    def test_query_without_labeled_samples(self):
        # Without labeled samples, the first pick has the largest embedding
        # norm, and the origin is no center for the subsequent rounds.
        X = np.array([[0.0, 1.0], [0.0, 2.0], [0.0, 3.0], [0.0, -0.5]])
        y = np.full(len(X), MISSING_LABEL)
        clf = ParzenWindowClassifierCustomOutput(
            outputs=(lambda X_pred: np.ones(len(X_pred)),), classes=[0, 1]
        )
        query_indices, utilities = UWE(random_state=0).query(
            X, y, estimator=clf, batch_size=3, return_utilities=True
        )
        np.testing.assert_array_equal(query_indices, [2, 3, 0])
        np.testing.assert_allclose(utilities[0], [1, 2, 3, 0.5])
        np.testing.assert_allclose(utilities[1], [2, 1, np.nan, 3.5])
        np.testing.assert_allclose(utilities[2], [1.5, 1, np.nan, np.nan])

    def test_query_zero_uncertainties(self):
        # If all uncertainties are zero, UWE falls back to the
        # K-Center-Greedy algorithm on the sample representations.
        X, y_true = make_blobs(n_samples=40, centers=3, random_state=0)
        y = np.full(len(X), MISSING_LABEL)
        y[:5] = y_true[:5]
        clf = ParzenWindowClassifierCustomOutput(
            outputs=(lambda X_pred: np.zeros(len(X_pred)),),
            classes=np.unique(y_true),
        )
        expected = CoreSet(random_state=0).query(X, y, batch_size=4)
        query_indices = UWE(random_state=0).query(
            X, y, estimator=clf, batch_size=4
        )
        np.testing.assert_array_equal(query_indices, expected)

        # The same fallback applies if all uncertainties are undefined, e.g.,
        # for a regressor fitted without labeled samples.
        clf = ParzenWindowClassifierCustomOutput(
            outputs=(lambda X_pred: np.full(len(X_pred), np.nan),),
            classes=np.unique(y_true),
        )
        query_indices = UWE(random_state=0).query(
            X, y, estimator=clf, batch_size=4
        )
        np.testing.assert_array_equal(query_indices, expected)

        # Zero uncertainties of some samples are kept, i.e., their
        # embeddings collapse onto the origin.
        unc = np.zeros(len(X))
        unc[[7, 8]] = 1.0
        clf = ParzenWindowClassifierCustomOutput(
            outputs=(lambda X_pred: unc[: len(X_pred)],),
            classes=np.unique(y_true),
        )
        X_pool = X[:20]
        y_pool = np.full(20, MISSING_LABEL)
        query_indices, utilities = UWE(random_state=0).query(
            X_pool, y_pool, estimator=clf, batch_size=1, return_utilities=True
        )
        self.assertIn(query_indices[0], [7, 8])
        self.assertTrue((utilities[0, unc[:20] == 0] == 0).all())

    def test_query_invalid_estimator_outputs(self):
        X, y_true = make_blobs(n_samples=20, centers=2, random_state=0)
        y = np.full(len(X), MISSING_LABEL)
        y[:4] = y_true[:4]
        classes = np.unique(y_true)
        for name, outputs in [
            ("negative", (lambda X_pred: -np.ones(len(X_pred)),)),
            (
                "partially nan",
                (
                    lambda X_pred: np.hstack(
                        [np.nan, np.ones(len(X_pred) - 1)]
                    ),
                ),
            ),
            ("inf", (lambda X_pred: np.full(len(X_pred), np.inf),)),
            ("wrong length", (lambda X_pred: np.ones(len(X_pred) + 1),)),
            ("wrong embeddings", (lambda X_pred: np.ones((2, 3)),)),
            ("3d output", (lambda X_pred: np.ones((len(X_pred), 1, 1)),)),
            (
                "two uncertainties",
                (
                    lambda X_pred: np.ones(len(X_pred)),
                    lambda X_pred: np.ones(len(X_pred)),
                ),
            ),
        ]:
            with self.subTest(name=name):
                clf = ParzenWindowClassifierCustomOutput(
                    outputs=outputs, classes=classes
                )
                self.assertRaises(ValueError, UWE().query, X, y, estimator=clf)

    def test_query_regression(self):
        X, _ = make_blobs(n_samples=40, centers=3, random_state=0)
        y = np.full(len(X), MISSING_LABEL)
        y[:5] = np.arange(5, dtype=float)
        reg = NadarayaWatsonRegressorUncertainty()
        query_indices, utilities = UWE(random_state=0).query(
            X, y, estimator=reg, batch_size=3, return_utilities=True
        )
        reg_fit = NadarayaWatsonRegressor().fit(X, y)
        _, std = reg_fit.predict(X, return_std=True)
        expected = 5 + _reference_uwe_kcg(
            X[5:], X[:5], std[5:], std[:5], batch_size=3
        )
        np.testing.assert_array_equal(query_indices, expected)
        self.assertTrue(np.isnan(utilities[:, :5]).all())

        # A regressor without uncertainties is rejected.
        self.assertRaises(
            ValueError,
            UWE().query,
            X,
            y,
            estimator=NadarayaWatsonRegressor(),
        )

    def test_query_labeled_candidates(self):
        # An explicit labeled candidate remains eligible, but its distance to
        # the labeled samples is zero.
        X, y_true = make_blobs(n_samples=30, centers=3, random_state=0)
        y = np.full(len(X), MISSING_LABEL)
        y[:5] = y_true[:5]
        clf = ParzenWindowClassifier(classes=np.unique(y_true))
        query_indices, utilities = UWE(random_state=0).query(
            X,
            y,
            estimator=clf,
            candidates=[4],
            batch_size=1,
            return_utilities=True,
        )
        np.testing.assert_array_equal(query_indices, [4])
        self.assertEqual(utilities[0, 4], 0)
        self.assertTrue(np.isnan(np.delete(utilities[0], 4)).all())

        query_indices, utilities = UWE(random_state=0).query(
            X,
            y,
            estimator=clf,
            candidates=[3, 4, 10, 11],
            batch_size=4,
            return_utilities=True,
        )
        self.assertEqual(set(query_indices[:2]), {10, 11})
        self.assertEqual(set(query_indices[2:]), {3, 4})
        np.testing.assert_array_equal(utilities[0, [3, 4]], 0)

    def test_query_ties_are_broken_by_random_state(self):
        # Four candidates with identical distances to the labeled sample.
        X = np.array([[0.0, 0.0], [1.0, 0], [-1, 0], [0, 1], [0, -1]])
        y = np.array([0] + [MISSING_LABEL] * 4)
        clf = ParzenWindowClassifierCustomOutput(
            outputs=(lambda X_pred: np.ones(len(X_pred)),), classes=[0, 1]
        )
        selections = {
            int(UWE(random_state=seed).query(X, y, estimator=clf)[0])
            for seed in range(20)
        }
        self.assertTrue(selections.issubset({1, 2, 3, 4}))
        self.assertGreater(len(selections), 1)
        for seed in range(3):
            np.testing.assert_array_equal(
                UWE(random_state=seed).query(X, y, estimator=clf),
                UWE(random_state=seed).query(X, y, estimator=clf),
            )

    def test_query_distinct_indices_per_batch(self):
        # Duplicates have zero distance to each other once one of them is
        # selected, but each sample is selected at most once.
        X = np.repeat([[0.0, 0.0], [5.0, 5.0]], 4, axis=0)
        y = np.full(len(X), MISSING_LABEL)
        clf = ParzenWindowClassifierCustomOutput(
            outputs=(lambda X_pred: np.ones(len(X_pred)),), classes=[0, 1]
        )
        query_indices = UWE(random_state=0).query(
            X, y, estimator=clf, batch_size=len(X)
        )
        self.assertEqual(len(np.unique(query_indices)), len(X))
