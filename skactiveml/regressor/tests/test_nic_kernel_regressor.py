import unittest
from itertools import product

import numpy as np
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.metrics.pairwise import pairwise_kernels

from skactiveml.regressor._nic_kernel_regressor import (
    NICKernelRegressor,
    NadarayaWatsonRegressor,
)
from skactiveml.utils import MISSING_LABEL
from scipy.stats import norm

from skactiveml.tests.template_estimator import TemplateProbabilisticRegressor


class TemplateTestNICKernelEstimator(TemplateProbabilisticRegressor):
    def setUp(
        self,
        estimator_class,
        init_default_params,
        fit_default_params=None,
        predict_default_params=None,
    ):
        super().setUp(
            estimator_class,
            init_default_params,
            fit_default_params,
            predict_default_params,
        )

    def test_init_param_metric(self):
        test_cases = []
        test_cases += [
            ("rbf", None),
            ("invalid", TypeError),
            (None, TypeError),
            ([], TypeError),
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

    def test_init_param_mu_0(self):
        if hasattr(self.estimator_class, "mu_0"):
            test_cases = []
            test_cases += [(0, None), (0.2, None), ("Test", TypeError)]
            self._test_param("init", "mu_0", test_cases)

    def test_init_param_kappa_0(self):
        if hasattr(self.estimator_class, "kappa_0"):
            test_cases = []
            test_cases += [
                (0.1, None),
                (1, None),
                (-1.0, None),
                ("Test", TypeError),
            ]
            self._test_param("init", "kappa_0", test_cases)

    def test_init_param_sigma_sq_0(self):
        if hasattr(self.estimator_class, "sigma_sq_0"):
            test_cases = []
            test_cases += [
                (1.0, None),
                (1, None),
                (-1.0, None),
                ("Test", TypeError),
            ]
            self._test_param("init", "sigma_sq_0", test_cases)

    def test_init_param_nu_0(self):
        if hasattr(self.estimator_class, "nu_0"):
            test_cases = []
            test_cases += [
                (2.5, None),
                (1, None),
                (-1.0, None),
                ("Test", TypeError),
            ]
            self._test_param("init", "nu_0", test_cases)

    def test_predict(self):
        reg = self.estimator_class(**self.start_parameter)
        X = np.array([[0, 0], [1, 1], [2, 2]])
        y_missing = np.full(3, MISSING_LABEL)
        reg.fit(X, y_missing)
        y_return, y_std, y_entropy = reg.predict(
            [[0, 0]], return_std=True, return_entropy=True
        )
        np.testing.assert_array_equal(y_return, [0])
        if self.estimator_class_string == "NadarayaWatsonRegressor":
            np.testing.assert_array_equal(y_std, [1])
            self.assertTrue(np.isfinite(y_entropy).all())

        start_params = self.start_parameter
        if self.estimator_class_string == "NICKernelRegressor":
            start_params["kappa_0"] = 0
            start_params["nu_0"] = 2
        reg = self.estimator_class(**start_params)

        X = np.zeros((3, 1))
        y = np.arange(3)

        for i in range(3):
            w = np.array([0, 0, 0])
            w[i] = 1
            reg.fit(X, y, sample_weight=w)
            y_return = reg.predict([[0]])[0]
            self.assertEqual(y_return, y[i])

        X = np.zeros((500, 1))
        y = norm.rvs(loc=1.24, scale=0.0245, size=500, random_state=0)
        if self.estimator_class_string == "NICKernelRegressor":
            reg = self.estimator_class(kappa_0=0, nu_0=0)
            reg.fit(X, y)
            mu, sigma = reg.predict([[0]], return_std=True)
            np.testing.assert_almost_equal(mu, 1.24, decimal=3)
            np.testing.assert_almost_equal(sigma, 0.0245, decimal=3)

    def test_predict_target_distribution(self):
        reg = self.estimator_class(**self.start_parameter).fit(
            **self.fit_default_params
        )
        X = self.predict_default_params["X"]

        y_pred = reg.predict_target_distribution(X).logpdf(0)

        self.assertEqual(y_pred.shape, (len(X),))

    def test_numeric_dtypes_match_float64_predictions(self):
        X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        X_test = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 1.0]])
        y = np.array([0.5, np.nan, 2.5, 4.0])
        for metric in ("rbf", "precomputed"):
            X_train = X if metric == "rbf" else X @ X.T + 1
            X_query = X_test if metric == "rbf" else X_test @ X.T + 1
            for weights in (None, np.ones(4), np.array([1.0, 0.0, 2.0, 3.0])):
                reference = self.estimator_class(metric=metric).fit(
                    X_train, y, sample_weight=weights
                )
                expected = reference.predict(
                    X_query, return_std=True, return_entropy=True
                )
                expected_dist = reference.predict_target_distribution(X_query)
                for dtype, target_dtype in product(
                    (np.int64, np.float32, np.float64),
                    (np.float32, np.float64),
                ):
                    with self.subTest(
                        metric=metric,
                        dtype=dtype,
                        target_dtype=target_dtype,
                        sample_weight=weights,
                    ):
                        typed_weights = (
                            None if weights is None else weights.astype(dtype)
                        )
                        reg = self.estimator_class(metric=metric).fit(
                            X_train.astype(dtype),
                            y.astype(target_dtype),
                            sample_weight=typed_weights,
                        )
                        actual = reg.predict(
                            X_query.astype(dtype),
                            return_std=True,
                            return_entropy=True,
                        )
                        np.testing.assert_allclose(
                            actual, expected, rtol=1e-6, atol=1e-8
                        )
                        dist = reg.predict_target_distribution(
                            X_query.astype(dtype)
                        )
                        np.testing.assert_allclose(
                            dist.stats(moments="mv"),
                            expected_dist.stats(moments="mv"),
                            rtol=1e-6,
                            atol=1e-8,
                        )
                        for method in ("pdf", "logpdf"):
                            np.testing.assert_allclose(
                                getattr(dist, method)([0.0, 1.0, 2.0]),
                                getattr(expected_dist, method)(
                                    [0.0, 1.0, 2.0]
                                ),
                                rtol=1e-6,
                                atol=1e-8,
                            )

    def test_unlabeled_sample_weights_preserve_fallback(self):
        datasets = [
            ("rbf", np.array([[0.0], [1.0], [2.0]]), [[0.5], [1.5]]),
            ("rbf", np.empty((0, 1)), [[0.5], [1.5]]),
            ("precomputed", np.eye(3), np.ones((2, 3))),
        ]
        for missing_label, (metric, X, X_test) in product(
            (np.nan, -1, None, "missing"), datasets
        ):
            y = np.full(len(X), missing_label)
            reg = self.estimator_class(
                metric=metric, missing_label=missing_label
            ).fit(X, y)
            expected = reg.predict(
                X_test, return_std=True, return_entropy=True
            )
            for weight in (0.0, 1.0):
                with self.subTest(
                    metric=metric,
                    missing_label=missing_label,
                    n_samples=len(X),
                    weight=weight,
                ):
                    reg.fit(X, y, sample_weight=np.full(len(X), weight))
                    actual = reg.predict(
                        X_test, return_std=True, return_entropy=True
                    )
                    np.testing.assert_allclose(actual, expected)
                    self.assertTrue(np.isfinite(actual).all())

    def test_fit_rejects_zero_weights_on_labeled_samples(self):
        X = np.array([[0.0], [1.0], [2.0]])
        for y, weights in (
            ([0.0, 1.0, 2.0], [0.0, 0.0, 0.0]),
            ([0.0, np.nan, 2.0], [0.0, 1.0, 0.0]),
        ):
            with self.subTest(y=y, sample_weight=weights):
                with self.assertRaisesRegex(
                    ValueError, "must not be all zero"
                ):
                    self.estimator_class().fit(X, y, sample_weight=weights)

    def test_callable_kernel_matches_builtin_and_precomputed(self):
        def kernel(x, y, gamma):
            return np.exp(-gamma * np.sum((x - y) ** 2))

        X = np.array([[-1.0, 0.0], [0.0, 0.5], [1.0, 1.5], [2.0, 1.0]])
        X_test = np.array([[-0.5, 0.25], [0.5, 1.0], [1.5, 1.25]])
        y = np.array([0.0, np.nan, 3.0, 2.0])
        gamma = 0.7
        K_train = pairwise_kernels(X, metric="rbf", gamma=gamma)
        K_test = pairwise_kernels(X_test, X, metric="rbf", gamma=gamma)
        for weights in (None, np.array([0.5, 1.0, 2.0, 3.0])):
            reference = self.estimator_class(
                metric="rbf", metric_dict={"gamma": gamma}
            ).fit(X, y, sample_weight=weights)
            expected = reference.predict(
                X_test, return_std=True, return_entropy=True
            )
            for metric, metric_dict, X_train, X_query in (
                (kernel, {"gamma": gamma}, X, X_test),
                ("precomputed", None, K_train, K_test),
            ):
                with self.subTest(metric=metric, sample_weight=weights):
                    reg = self.estimator_class(
                        metric=metric, metric_dict=metric_dict
                    ).fit(X_train, y, sample_weight=weights)
                    actual = reg.predict(
                        X_query, return_std=True, return_entropy=True
                    )
                    np.testing.assert_allclose(actual, expected)

    def test_precomputed_matches_feature_kernel_on_fit_and_refit(self):
        X = np.array([[-1.0, 0.0], [0.0, 0.5], [1.0, 1.5], [2.0, 1.0]])
        X_test = np.array([[-0.5, 0.25], [0.5, 1.0], [1.5, 1.25]])
        gamma = 0.7
        training_sets = [
            (X, np.array([0.0, 1.0, 3.0, 2.0])),
            (X[:3], np.array([0.0, np.nan, 3.0])),
            (X, np.full(4, np.nan)),
        ]

        precomputed = self.estimator_class(metric="precomputed")
        for X_train, y in training_sets:
            K_train = pairwise_kernels(X_train, metric="rbf", gamma=gamma)
            precomputed.fit(K_train, y)
            self.assertEqual(precomputed.n_features_in_, len(X_train))

            feature = self.estimator_class(
                metric="rbf", metric_dict={"gamma": gamma}
            ).fit(X_train, y)
            for X_test_batch in (X_test[:1], X_test):
                K_test = pairwise_kernels(
                    X_test_batch, X_train, metric="rbf", gamma=gamma
                )
                expected = feature.predict(
                    X_test_batch, return_std=True, return_entropy=True
                )
                actual = precomputed.predict(
                    K_test, return_std=True, return_entropy=True
                )
                for actual_part, expected_part in zip(actual, expected):
                    np.testing.assert_allclose(actual_part, expected_part)

    def test_precomputed_fit_requires_a_square_training_kernel(self):
        with self.assertRaisesRegex(
            ValueError, "n_train_samples, n_train_samples"
        ):
            self.estimator_class(metric="precomputed").fit(
                np.ones((3, 2)), [0.0, 1.0, 2.0]
            )

    def test_precomputed_prediction_requires_all_training_columns(self):
        X = np.array([[-1.0, 0.0], [0.0, 0.5], [1.0, 1.5], [2.0, 1.0]])
        X_test = np.array([[-0.5, 0.25], [0.5, 1.0], [1.5, 1.25]])
        y = np.array([0.0, np.nan, 3.0, np.nan])
        gamma = 0.7
        K_train = pairwise_kernels(X, metric="rbf", gamma=gamma)
        reg = self.estimator_class(metric="precomputed").fit(K_train, y)
        K_test = pairwise_kernels(X_test, X, metric="rbf", gamma=gamma)

        with self.assertRaisesRegex(
            ValueError, "n_test_samples, n_train_samples"
        ):
            reg.predict(K_test[:, [0, 2]])


class TestNICKernelEstimator(
    TemplateTestNICKernelEstimator, unittest.TestCase
):
    def setUp(self):
        estimator_class = NICKernelRegressor
        self.estimator_class_string = "NICKernelRegressor"
        init_default_params = {
            "missing_label": MISSING_LABEL,
        }
        self.random_state = 0
        fit_default_params = {
            "X": np.zeros((3, 1)),
            "y": [0.5, 0.6, MISSING_LABEL],
        }
        predict_default_params = {"X": [[1]]}
        self.start_parameter = {
            "kappa_0": 1,
            "nu_0": 2,
            "mu_0": 0,
            "sigma_sq_0": 1,
            "metric": "rbf",
            "metric_dict": {"gamma": 10.0},
            "missing_label": MISSING_LABEL,
            "random_state": self.random_state,
        }
        super().setUp(
            estimator_class=estimator_class,
            init_default_params=init_default_params,
            fit_default_params=fit_default_params,
            predict_default_params=predict_default_params,
        )
        self.X = np.array([[0, 1], [1, 0], [2, 3]])
        self.y = np.array([1, 2, 3])
        self.X_cand = np.array([[2, 1], [3, 5]])

    def test_fit(self):
        reg = NICKernelRegressor(**self.start_parameter)
        X = 5
        y = 7
        self.assertRaises(TypeError, reg.fit, X, y)

        w = np.zeros_like(self.y)
        self.assertRaises(ValueError, reg.fit, self.X, self.y, w)

    def test_fit_preserves_default_metric_dict_parameter(self):
        reg = NICKernelRegressor()

        reg.fit([[0.0], [1.0]], [0.0, 1.0])

        self.assertIsNone(reg.metric_dict)
        self.assertIsNone(reg.get_params()["metric_dict"])
        self.assertIsNone(clone(reg).metric_dict)
        self.assertEqual(reg.metric_dict_, {})

    def test_prediction_uses_captured_metric_dict(self):
        metric_dict = {"gamma": 0.5}
        reg = NICKernelRegressor(metric_dict=metric_dict).fit(
            [[0.0], [1.0]], [0.0, 1.0]
        )
        expected = reg.predict([[0.25]])

        metric_dict["gamma"] = 50.0

        self.assertIs(reg.metric_dict, metric_dict)
        self.assertIsNot(reg.metric_dict_, metric_dict)
        self.assertEqual(reg.metric_dict_, {"gamma": 0.5})
        np.testing.assert_allclose(reg.predict([[0.25]]), expected)

    def test_missing_label(self):
        self.missing_label = -1
        start_params = self.start_parameter.copy()
        start_params["missing_label"] = -1
        reg_other_missing_label = NICKernelRegressor(**start_params)
        reg_usual_missing_label = NICKernelRegressor(**self.start_parameter)
        X = np.array([[0, 1], [0, 1], [1, 0], [0, 0]])
        y_other_missing_label = np.array([0.1, 0.2, -1, -1])
        y_usual_missing_label = np.array(
            [0.1, 0.2, MISSING_LABEL, MISSING_LABEL]
        )
        reg_other_missing_label.fit(X, y_other_missing_label)
        reg_usual_missing_label.fit(X, y_usual_missing_label)
        y_return_other = reg_other_missing_label.predict([[0, 0]])[0]
        y_return_usual = reg_usual_missing_label.predict([[0, 0]])[0]

        self.assertEqual(y_return_other, y_return_usual)

    def test_random_state(self):
        reg = NICKernelRegressor(**self.start_parameter)

        X_test = norm.rvs(size=(10, 2), random_state=self.random_state)
        X = norm.rvs(size=(8, 2), random_state=self.random_state)
        y = norm.rvs(size=8, random_state=self.random_state)

        reg.fit(X, y)
        prediction_1 = reg.predict(X_test)
        reg.fit(X, y)
        prediction_2 = reg.predict(X_test)

        np.testing.assert_almost_equal(prediction_1, prediction_2)

    def test_predict_rejects_an_improper_prior_without_labels(self):
        reg = NICKernelRegressor(kappa_0=0).fit([[0.0]], [np.nan])

        with self.assertRaisesRegex(ValueError, "no evidence"):
            reg.predict([[0.0]])
        with self.assertRaisesRegex(ValueError, "no evidence"):
            reg.predict_target_distribution([[0.0]])

    def test_predict_rejects_a_negative_kernel_mass(self):
        reg = NICKernelRegressor(metric="linear").fit(
            [[-2.0], [1.0]], [0.0, 5.0]
        )

        with self.assertRaisesRegex(ValueError, "kernel mass"):
            reg.predict([[1.0]])
        with self.assertRaisesRegex(ValueError, "kernel mass"):
            reg.predict_target_distribution([[1.0]])

    def test_predict_rejects_a_negative_scatter(self):
        reg = NICKernelRegressor(metric="linear").fit(
            [[-1.0], [3.0]], [10.0, 0.0]
        )

        with self.assertRaisesRegex(ValueError, "scatter"):
            reg.predict([[1.0]])

    def test_predict_rejects_negative_sample_weights(self):
        reg = NICKernelRegressor().fit(
            [[0.0], [1.0]], [0.0, 1.0], sample_weight=[-1.0, 2.0]
        )

        with self.assertRaisesRegex(ValueError, "sample_weight"):
            reg.predict([[0.0]])

    def test_predict_accepts_non_negative_kernels(self):
        for metric in ("rbf", "laplacian"):
            with self.subTest(metric=metric):
                reg = NICKernelRegressor(metric=metric).fit(
                    [[-2.0], [1.0]], [0.0, 5.0]
                )

                mean, std = reg.predict([[1.0]], return_std=True)

                self.assertTrue(np.all(np.isfinite(mean)))
                self.assertTrue(np.all(std > 0))

    def test_zero_kernel_evidence_preserves_prior(self):
        params = dict(mu_0=2.0, kappa_0=3.0, nu_0=5.0, sigma_sq_0=4.0)
        reg = NICKernelRegressor(**params).fit(
            [[0.0], [1.0]], [0.0, 1.0], sample_weight=[2.0, 1.0]
        )
        prior = NICKernelRegressor(**params).fit(
            [[0.0], [1.0]], [np.nan, np.nan]
        )
        nearby = reg.predict([[0.5]], return_std=True)
        with np.errstate(divide="raise", invalid="raise"):
            actual = reg.predict([[100.0], [0.5]], return_std=True)
        expected_prior = prior.predict([[100.0]], return_std=True)
        for actual_part, prior_part, nearby_part in zip(
            actual, expected_prior, nearby
        ):
            np.testing.assert_allclose(actual_part[:1], prior_part)
            np.testing.assert_allclose(actual_part[1:], nearby_part)
            self.assertTrue(np.isfinite(actual_part).all())
        np.testing.assert_allclose(actual[0][:1], [params["mu_0"]])
        self.assertNotAlmostEqual(actual[0][1], params["mu_0"])


class TestNadarayaWatsonRegressor(
    TemplateTestNICKernelEstimator, unittest.TestCase
):
    def setUp(self):
        estimator_class = NadarayaWatsonRegressor
        self.estimator_class_string = "NadarayaWatsonRegressor"
        self.X = np.array([[0, 1], [1, 0], [2, 3]])
        self.random_state = 0
        init_default_params = {
            "metric": "rbf",
            "metric_dict": {"gamma": 10.0},
            "missing_label": MISSING_LABEL,
            "random_state": self.random_state,
        }
        self.start_parameter = init_default_params
        fit_default_params = {"X": np.zeros((3, 1)), "y": [0.5, 0.6, np.nan]}
        predict_default_params = {"X": [[1]]}
        super().setUp(
            estimator_class=estimator_class,
            init_default_params=init_default_params,
            fit_default_params=fit_default_params,
            predict_default_params=predict_default_params,
        )

    def test_fit(self):
        reg = NadarayaWatsonRegressor(**self.init_default_params)
        self.assertRaises(NotFittedError, reg.predict, self.X)
        X = np.zeros((3, 1))
        y = np.arange(3)
        reg.fit(X, y)
        y_pred = reg.predict([[0]])[0]
        self.assertEqual(y_pred, np.average(y))

    def test_predict_reports_samples_without_posterior_evidence(self):
        reg = NadarayaWatsonRegressor(metric_dict={"gamma": 500.0}).fit(
            [[0.0], [1.0]], [0.0, 1.0]
        )

        with self.assertRaisesRegex(ValueError, "no evidence"):
            reg.predict([[50.0]])
        with self.assertRaisesRegex(ValueError, "no evidence"):
            reg.predict_target_distribution([[50.0]])

        mean, std = reg.predict([[0.1]], return_std=True)
        self.assertTrue(np.all(np.isfinite(mean)))
        self.assertTrue(np.all(np.isfinite(std)))
