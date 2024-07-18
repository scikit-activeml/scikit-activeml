import unittest
from copy import deepcopy

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from skactiveml.classifier import SklearnClassifier, ParzenWindowClassifier
from skactiveml.pool import EpistemicUncertaintySampling
from skactiveml.pool._epistemic_uncertainty_sampling import (
    _interpolate,
    _pwc_ml_1,
    _pwc_ml_0,
    _epistemic_uncertainty_pwc,
    _loglike_logreg,
    _pi_h,
    _epistemic_uncertainty_logreg,
    _theta,
)
from skactiveml.tests.template_query_strategy import (
    TemplateSingleAnnotatorPoolQueryStrategy,
)
from skactiveml.utils import MISSING_LABEL


class TestEpistemicUncertaintySampling(
    TemplateSingleAnnotatorPoolQueryStrategy, unittest.TestCase
):
    def setUp(self):
        self.random_state = 1
        self.candidates = np.array([[8, 1], [9, 1], [5, 1]])
        self.X = np.array([[1, 2], [5, 8], [8, 4], [5, 4]])
        self.y = np.array([0, 0, 1, 1])
        self.y_MISSING_LABEL = np.array(
            [MISSING_LABEL, MISSING_LABEL, MISSING_LABEL, MISSING_LABEL]
        )
        self.classes = [0, 1]
        self.clf = ParzenWindowClassifier(
            classes=self.classes, random_state=self.random_state
        )
        self.kwargs = dict(candidates=self.candidates, X=self.X, y=self.y)
        self.kwargs_MISSING_LABEL = dict(
            candidates=self.candidates, X=self.X, y=self.y_MISSING_LABEL
        )

        query_default_params_clf = {
            "X": np.array([[1, 2], [5, 8], [8, 4], [5, 4]]),
            "y": np.array([0, 1, MISSING_LABEL, MISSING_LABEL]),
            "clf": ParzenWindowClassifier(
                random_state=42, classes=self.classes
            ),
        }
        super().setUp(
            qs_class=EpistemicUncertaintySampling,
            init_default_params={},
            query_default_params_clf=query_default_params_clf,
        )

    def test_init_param_precompute(self, test_cases=None):
        test_cases = [] if test_cases is None else test_cases
        test_cases += [(None, TypeError), ([], TypeError), (0, TypeError)]
        self._test_param("init", "precompute", test_cases)

    def test_query_param_clf(self):
        add_test_cases = [
            (LogisticRegression(), TypeError),
            (
                SklearnClassifier(
                    DecisionTreeClassifier(), classes=self.classes
                ),
                TypeError,
            ),
            (
                SklearnClassifier(LogisticRegression(), classes=self.classes),
                None,
            ),
            (ParzenWindowClassifier(), None),
        ]
        super().test_query_param_clf(test_cases=add_test_cases)

    def test_query_param_sample_weight(self, test_cases=None):
        test_cases = [] if test_cases is None else test_cases
        X = self.query_default_params_clf["X"]
        test_cases += [
            ("string", ValueError),
            (X, ValueError),
            (np.empty((len(X) - 1)), ValueError),
        ]
        super().test_query_param_sample_weight(test_cases)

    # tests for epistemic ParzenWindowClassifier
    def test_interpolate(self):
        interpolated = _interpolate(
            np.array([[0, 0], [1, 1]]), np.array([[0.5, 0.5]])
        )
        np.testing.assert_array_equal(interpolated, np.array([0.5]))

    def test_pwc_ml_1(self):
        self.assertEqual(1.0, -_pwc_ml_1(None, 0.0, 0.0))
        self.assertEqual(0.0, -_pwc_ml_1(1, 0.5, 0.8))
        self.assertAlmostEqual(0.36, -_pwc_ml_1(0.9, 1, 1))
        self.assertEqual(-0.8, -_pwc_ml_1(0.1, 1, 1))

    def test_pwc_ml_0(self):
        self.assertEqual(1.0, -_pwc_ml_0(None, 0.0, 0.0))
        self.assertEqual(-1.0, -_pwc_ml_0(1, 0.5, 0.8))
        self.assertAlmostEqual(0.36, -_pwc_ml_0(0.1, 1, 1))
        self.assertEqual(-0.8, -_pwc_ml_0(0.9, 1, 1))

    def test_epistemic_uncertainty_pwc(self):
        freq = np.empty((121, 2))
        for n in range(11):
            for p in range(11):
                freq[n * 11 + p] = n, p

        indices = [39, 27, 18, 68, 20]
        expected = np.array(
            [
                0.23132135217407046,
                0.22057583593855598,
                0.056099946963575974,
                0.16316360415548017,
                0.021220951860586187,
            ]
        )

        utilities, arr = _epistemic_uncertainty_pwc(freq, None)
        self.assertEqual(utilities.shape, (121,))
        np.testing.assert_allclose(expected, utilities[indices])
        _epistemic_uncertainty_pwc(np.array([[2.5, 1.5]]), None)

        val_utilities = utilities
        precompute_array = np.full((1, 1), np.nan)

        utilities, precompute_array = _epistemic_uncertainty_pwc(
            freq, precompute_array
        )
        np.testing.assert_array_equal(val_utilities, utilities)
        np.testing.assert_array_equal(
            val_utilities, precompute_array[:11, :11].flatten()
        )

        class Dummy_PWC(ParzenWindowClassifier):
            def predict_freq(self, X):
                return freq

        qs = EpistemicUncertaintySampling(precompute=True)
        query_params = deepcopy(self.query_default_params_clf)
        query_params["return_utilities"] = True
        query_params["candidates"] = np.zeros_like(freq)
        query_params["clf"] = Dummy_PWC(classes=self.classes)
        _, utilities = qs.query(**query_params)
        np.testing.assert_array_equal(val_utilities, utilities[0])

        qs = EpistemicUncertaintySampling()
        query_params["clf"] = ParzenWindowClassifier(classes=[0, 1, 2])
        self.assertRaises(ValueError, qs.query, **query_params)

    # tests for epistemic logistic regression
    def test_loglike_logreg(self):
        w = np.array([0, 0])
        X = np.array([[0]])
        y = np.array([0])
        self.assertEqual(0, _loglike_logreg(None, X=[], y=[]))
        self.assertEqual(2.0, np.exp(_loglike_logreg(w=w, X=X, y=y)))

    def test_pi_h(self):
        w = np.array([0, 0])
        X = np.array([[3]])
        y = np.array([1])
        expected = np.exp(-_loglike_logreg(w, X, y))

        # w has to be np.zeros for the follow tests
        self.assertEqual(expected, _pi_h(w, 1, X, y))
        self.assertEqual(2 ** -len(X), _pi_h(w, 1, X, y))

    def test_theta(self):
        def func1(x):
            return x[0] * x[0] - x[1] * x[1]

        def func2(x):
            return -x[0] * x[0] + x[1] * x[1]

        alpha = 0.5
        x0 = np.array([1, 0])
        A = np.array([0, 1])
        self.assertTrue((_theta(func1, alpha, x0, A) == 0).all())
        self.assertTrue(np.isnan(_theta(func2, alpha, x0, A)).all())

    def test_epistemic_uncertainty_logreg(self):
        clf = SklearnClassifier(
            LogisticRegression(),
            classes=[0, 1, 2],
            random_state=42,
        )
        self.assertRaises(
            ValueError,
            _epistemic_uncertainty_logreg,
            X_cand=self.candidates,
            X=self.X,
            y=self.y,
            clf=clf,
        )

        clf = SklearnClassifier(
            DecisionTreeClassifier(),
            classes=[0, 1],
            random_state=42,
        )
        self.assertRaises(
            TypeError,
            _epistemic_uncertainty_logreg,
            X_cand=self.candidates,
            X=self.X,
            y=self.y,
            clf=clf,
        )

        self.assertRaises(
            TypeError,
            _epistemic_uncertainty_logreg,
            X_cand=self.candidates,
            X=self.X,
            y=self.y,
            clf=self.clf,
        )

        probas = np.array([[0.5, 0.5]])
        X = np.array([[0]])
        X_cand = np.array([[3]])
        y = np.array([0])
        # utils_expected = np.array()
        clf = SklearnClassifier(LogisticRegression(), classes=[0, 1])
        clf.fit(X, y)
        utils = _epistemic_uncertainty_logreg(X_cand, X, y, clf, probas)
        np.testing.assert_array_equal([0], utils)

    def test_query(self):
        query_params = deepcopy(self.query_default_params_clf)
        query_params["return_utilities"] = True

        # query - ParzenWindowClassifier
        clf = ParzenWindowClassifier(
            classes=self.classes, random_state=self.random_state
        )
        qs = EpistemicUncertaintySampling()
        qs.query(**self.kwargs_MISSING_LABEL, clf=clf)

        best_indices, utilities = qs.query(
            **self.kwargs, clf=clf, return_utilities=True
        )
        self.assertEqual(utilities.shape, (1, len(self.candidates)))
        self.assertEqual(best_indices.shape, (1,))

        # query - logistic regression
        clf = SklearnClassifier(
            LogisticRegression(),
            classes=self.classes,
            random_state=42,
        )

        qs = EpistemicUncertaintySampling()
        qs.query(**self.kwargs_MISSING_LABEL, clf=clf)

        best_indices, utilities = qs.query(
            **self.kwargs, clf=clf, return_utilities=True
        )
        self.assertEqual(utilities.shape, (1, len(self.candidates)))
        self.assertEqual(best_indices.shape, (1,))

        best_indices_s, utilities_s = qs.query(
            **self.kwargs,
            clf=clf,
            return_utilities=True,
            sample_weight=[0.5, 1, 1, 1]
        )
        comp = utilities_s == utilities
        self.assertTrue(not comp.all())
