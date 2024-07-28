import unittest

import numpy as np
from sklearn.utils.validation import NotFittedError
from sklearn.datasets import make_blobs
from skactiveml.tests.template_estimator import TemplateClassFrequencyEstimator

from skactiveml.classifier import ParzenWindowClassifier


class TestParzenWindowClassifier(
    TemplateClassFrequencyEstimator, unittest.TestCase
):
    def setUp(self):
        estimator_class = ParzenWindowClassifier
        init_default_params = {
            "missing_label": "nan",
        }
        fit_default_params = {
            "X": np.zeros((3, 1)),
            "y": ["tokyo", "nan", "paris"],
        }
        predict_default_params = {"X": [[1]]}
        super().setUp(
            estimator_class=estimator_class,
            init_default_params=init_default_params,
            fit_default_params=fit_default_params,
            predict_default_params=predict_default_params,
        )
        self.y_nan = ["nan", "nan", "nan"]
        self.w = [2, np.nan, 1]

    def test_init_param_metric(self):
        test_cases = []
        test_cases += [
            ("rbf", None),
            (lambda x, y: ((x - y) ** 2).sum(), None),
            (None, ValueError),
            ([], ValueError),
        ]
        self._test_param("init", "metric", test_cases)

    def test_init_param_metric_dict(self):
        test_cases = []
        test_cases += [
            ({"gamma": "mean"}, None),
            ("gamma", TypeError),
            ([], TypeError),
        ]
        self._test_param("init", "metric_dict", test_cases)

    def test_init_param_n_neighbors(self):
        test_cases = []
        test_cases += [
            (1, None),
            (0, ValueError),
            (-1, ValueError),
            (1.5, TypeError),
        ]
        self._test_param("init", "n_neighbors", test_cases)

    def test_fit(self):
        pwc = ParzenWindowClassifier(
            classes=["tokyo", "paris", "new york"], missing_label="nan"
        )
        pwc.fit(X=self.fit_default_params["X"], y=self.y_nan)
        self.assertIsNone(pwc.cost_matrix)
        np.testing.assert_array_equal(1 - np.eye(3), pwc.cost_matrix_)
        np.testing.assert_array_equal(np.zeros((3, 3)), pwc.V_)
        pwc.fit(X=self.fit_default_params["X"], y=self.fit_default_params["y"])
        self.assertIsNone(pwc.cost_matrix)
        np.testing.assert_array_equal(1 - np.eye(3), pwc.cost_matrix_)
        np.testing.assert_array_equal(
            [[0, 0, 1], [0, 0, 0], [0, 1, 0]], pwc.V_
        )
        pwc.fit(
            X=self.fit_default_params["X"],
            y=self.fit_default_params["y"],
            sample_weight=self.w,
        )
        np.testing.assert_array_equal(
            [[0, 0, 2], [0, 0, 0], [0, 1, 0]], pwc.V_
        )

    def test_predict_freq(self):
        pwc = ParzenWindowClassifier(
            classes=["tokyo", "paris", "new york"],
            missing_label="nan",
            n_neighbors=10,
            metric="rbf",
            metric_dict={"gamma": 2},
        )
        self.assertRaises(
            NotFittedError, pwc.predict_freq, X=self.fit_default_params["X"]
        )
        pwc.fit(X=self.fit_default_params["X"], y=self.y_nan)
        F = pwc.predict_freq(X=self.fit_default_params["X"])
        np.testing.assert_array_equal(
            np.zeros((len(self.fit_default_params["X"]), 3)), F
        )
        pwc.fit(
            X=self.fit_default_params["X"],
            y=self.fit_default_params["y"],
            sample_weight=self.w,
        )
        F = pwc.predict_freq(X=[self.fit_default_params["X"][0]])
        np.testing.assert_array_equal([[0, 1, 2]], F)
        pwc = ParzenWindowClassifier(
            classes=["tokyo", "paris", "new york"],
            missing_label="nan",
            n_neighbors=1,
        )
        pwc.fit(
            X=self.fit_default_params["X"],
            y=self.fit_default_params["y"],
            sample_weight=self.w,
        )
        F = pwc.predict_freq(X=[self.fit_default_params["X"][0]])
        np.testing.assert_array_equal([[0, 1, 0]], F)
        pwc = ParzenWindowClassifier(
            classes=["tokyo", "paris", "new york"],
            missing_label="nan",
            n_neighbors=1,
            metric="precomputed",
        )
        pwc.fit(
            X=self.fit_default_params["X"],
            y=self.fit_default_params["y"],
            sample_weight=self.w,
        )
        self.assertRaises(ValueError, pwc.predict_freq, X=[[1, 0]])
        self.assertRaises(ValueError, pwc.predict_freq, X=[[1], [0]])
        F = pwc.predict_freq(X=[[1, 0, 0]])
        np.testing.assert_array_equal([[0, 0, 2]], F)

        def rbf_kernel(x, y, gamma):
            return np.exp(-gamma * np.sum((x - y) ** 2))

        pwc = ParzenWindowClassifier(
            classes=["tokyo", "paris"],
            missing_label="nan",
            random_state=0,
            metric=rbf_kernel,
            metric_dict={"gamma": 2},
        )
        F_call = pwc.fit(
            X=self.fit_default_params["X"], y=self.fit_default_params["y"]
        ).predict_freq(np.ones_like(self.fit_default_params["X"]))
        pwc = ParzenWindowClassifier(
            classes=["tokyo", "paris"],
            missing_label="nan",
            metric="rbf",
            metric_dict={"gamma": 2},
            random_state=0,
        )
        F_rbf = pwc.fit(
            X=self.fit_default_params["X"], y=self.fit_default_params["y"]
        ).predict_freq(np.ones_like(self.fit_default_params["X"]))
        np.testing.assert_array_equal(F_call, F_rbf)

    def test_predict_proba(self):
        pwc = ParzenWindowClassifier(
            classes=["tokyo", "paris"], missing_label="nan"
        )
        self.assertRaises(
            NotFittedError, pwc.predict_proba, X=self.fit_default_params["X"]
        )
        pwc.fit(X=self.fit_default_params["X"], y=self.y_nan)
        P = pwc.predict_proba(X=self.fit_default_params["X"])
        np.testing.assert_array_equal(
            np.ones((len(self.fit_default_params["X"]), 2)) * 0.5, P
        )
        pwc.fit(
            X=self.fit_default_params["X"],
            y=self.fit_default_params["y"],
            sample_weight=self.w,
        )
        P = pwc.predict_proba(X=[self.fit_default_params["X"][0]])
        np.testing.assert_array_equal([[1 / 3, 2 / 3]], P)
        pwc = ParzenWindowClassifier(
            classes=["tokyo", "paris", "new york"],
            missing_label="nan",
            n_neighbors=1,
            metric="precomputed",
            class_prior=1,
        )
        pwc.fit(
            X=self.fit_default_params["X"],
            y=self.fit_default_params["y"],
            sample_weight=self.w,
        )
        P = pwc.predict_proba(X=[[1, 0, 0]])
        np.testing.assert_array_equal([[1 / 5, 1 / 5, 3 / 5]], P)
        pwc = ParzenWindowClassifier(
            classes=["tokyo", "paris", "new york"],
            missing_label="nan",
            n_neighbors=1,
            metric="precomputed",
            class_prior=[0, 0, 1],
        )
        pwc.fit(
            X=self.fit_default_params["X"],
            y=self.fit_default_params["y"],
            sample_weight=self.w,
        )
        P = pwc.predict_proba(X=[[1, 0, 0]])
        np.testing.assert_array_equal([[0, 0, 1]], P)

    def test_sample(self):
        # Setup test cases.
        X, y_full = make_blobs(n_samples=200, centers=4, random_state=0)
        classes = np.unique(y_full)
        pwc = ParzenWindowClassifier(
            classes=classes, class_prior=1, missing_label=-1
        )
        y_missing = np.full_like(y_full, fill_value=-1)
        y_partial_missing = y_full.copy()
        y_partial_missing[30:50] = -1
        y_class_0_missing = y_full.copy()
        y_class_0_missing[y_full == 0] = -1

        for y in [y_missing, y_partial_missing, y_class_0_missing, y_full]:
            pwc.fit(X, y)

            for n_samples in [1, 10]:
                # Check shape of probabilities.
                P_sampled = pwc.sample(X, n_samples=n_samples)
                shape_Expected = [n_samples, len(X), len(classes)]
                np.testing.assert_array_equal(P_sampled.shape, shape_Expected)

                # Check normalization of probabilities.
                P_sums = P_sampled.sum(axis=-1)
                P_sums_expected = np.ones_like(P_sums)
                np.testing.assert_allclose(P_sums, P_sums_expected)

        # Check value error if `alphas` as input to dirichlet are zero.
        pwc = ParzenWindowClassifier(
            classes=np.unique(y_full), class_prior=0, missing_label=-1
        )
        pwc.fit(X, y_missing)
        self.assertRaises(ValueError, pwc.sample, X=X, n_samples=10)

        pwc.fit(X, y_class_0_missing)
        self.assertRaises(ValueError, pwc.sample, X=X, n_samples=10)

    def test_predict(self):
        pwc = ParzenWindowClassifier(
            classes=["tokyo", "paris"], missing_label="nan", random_state=0
        )
        self.assertRaises(
            NotFittedError, pwc.predict, X=self.fit_default_params["X"]
        )
        pwc.fit(X=self.fit_default_params["X"], y=self.y_nan)
        y = pwc.predict(self.fit_default_params["X"])
        np.testing.assert_array_equal(["tokyo", "paris", "tokyo"], y)
        pwc = ParzenWindowClassifier(
            classes=["tokyo", "paris"], missing_label="nan", random_state=1
        )
        pwc.fit(X=self.fit_default_params["X"], y=self.y_nan)
        y = pwc.predict(self.fit_default_params["X"])
        np.testing.assert_array_equal(["tokyo", "tokyo", "paris"], y)
        pwc.fit(
            X=self.fit_default_params["X"],
            y=self.fit_default_params["y"],
            sample_weight=self.w,
        )
        y = pwc.predict(self.fit_default_params["X"])
        np.testing.assert_array_equal(["tokyo", "tokyo", "tokyo"], y)
        pwc = ParzenWindowClassifier(
            classes=["tokyo", "paris", "new york"],
            missing_label="nan",
            cost_matrix=[[0, 1, 4], [10, 0, 5], [2, 2, 0]],
        )
        pwc.fit(X=self.fit_default_params["X"], y=self.y_nan)
        y = pwc.predict(self.fit_default_params["X"])
        np.testing.assert_array_equal(["paris", "paris", "paris"], y)
        pwc = ParzenWindowClassifier(
            classes=["tokyo", "paris"],
            missing_label="nan",
            cost_matrix=[[0, 1], [10, 0]],
        )
        pwc.fit(
            X=self.fit_default_params["X"],
            y=self.fit_default_params["y"],
            sample_weight=self.w,
        )
        y = pwc.predict(self.fit_default_params["X"])
        np.testing.assert_array_equal(["paris", "paris", "paris"], y)

    def test__calculate_mean_gamma(self):
        # test without missing labels
        n_features = 1
        variance = [0.66666667]
        gamma = 6.907755278982137
        gamma2 = ParzenWindowClassifier._calculate_mean_gamma(
            3, variance, n_features
        )
        self.assertAlmostEqual(gamma, gamma2)
        # test with 0 variance
        variance = [0]
        gamma = 1 / n_features
        gamma2 = ParzenWindowClassifier._calculate_mean_gamma(
            3, variance, n_features
        )
        self.assertAlmostEqual(gamma, gamma2)
        # test with 0 variance
        variance = [0]
        n_features2 = 2
        gamma = 1 / n_features2
        gamma2 = ParzenWindowClassifier._calculate_mean_gamma(
            3, variance, n_features2
        )
        self.assertAlmostEqual(gamma, gamma2)
        # test with 1 missing label
        variance = [0.66666667]
        gamma = 5.050851362881613
        gamma2 = ParzenWindowClassifier._calculate_mean_gamma(
            2, variance, n_features
        )
        self.assertAlmostEqual(gamma, gamma2)
        # test mutli dimensional X
        variance = [0.5, 1.1875]
        gamma = 2.7289897398447946
        gamma2 = ParzenWindowClassifier._calculate_mean_gamma(
            3, variance, n_features
        )
        self.assertAlmostEqual(gamma, gamma2)
        # test if increasing N increases gamma
        N = np.arange(2, 100)
        variance = 1
        gamma = [
            ParzenWindowClassifier._calculate_mean_gamma(
                n, variance, n_features
            )
            for n in N
        ]
        self.assertTrue(np.all(np.diff(gamma) > 0))
