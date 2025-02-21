import unittest
import numpy as np

from copy import deepcopy

from sklearn import clone
from sklearn.exceptions import NotFittedError
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression, ARDRegression, SGDRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVC
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted
from sklearn.datasets import make_regression

from skactiveml.base import SkactivemlRegressor
from skactiveml.regressor import (
    SklearnRegressor,
    SklearnNormalRegressor,
)
from skactiveml.utils import MISSING_LABEL
from skactiveml.tests.template_estimator import (
    TemplateSkactivemlRegressor,
    TemplateProbabilisticRegressor,
)


class TestWrapper(TemplateSkactivemlRegressor, unittest.TestCase):
    def setUp(self):
        estimator_class = SklearnRegressor
        estimator = SGDRegressor()
        init_default_params = {"estimator": estimator, "missing_label": np.nan}
        fit_default_params = {"X": np.zeros((3, 1)), "y": [0.5, 0.6, np.nan]}
        predict_default_params = {"X": [[1]]}
        super().setUp(
            estimator_class=estimator_class,
            init_default_params=init_default_params,
            fit_default_params=fit_default_params,
            predict_default_params=predict_default_params,
        )
        self.random_state = 0
        self.X = np.array([[0, 1], [1, 0], [2, 3]])
        self.y = np.array([1, 1, 1])

        self.X_cand = np.array([[2, 1], [3, 5]])

    def test_init_param_estimator(self):
        test_cases = []
        test_cases += [
            (GaussianProcessRegressor(), None),
            (SVC(), TypeError),
            ("Test", AttributeError),
        ]
        self._test_param("init", "estimator", test_cases)

    def test_fit_predict(self):
        estimator = LinearRegression()
        reg = SklearnRegressor(estimator=estimator)
        y = np.full(3, MISSING_LABEL)
        reg.fit(self.X, y)
        self.assertRaises(NotFittedError, check_is_fitted, reg.estimator_)
        y = np.zeros(3)
        reg.fit(self.X, y)
        check_is_fitted(reg.estimator_)

        reg_1 = SklearnRegressor(
            estimator=MLPRegressor(
                random_state=self.random_state, max_iter=1000
            ),
            random_state=self.random_state,
        )

        X = np.array([[0], [1], [2], [3], [4]])
        y = np.array([3, 4, 1, 2, 1])

        reg_2 = clone(reg_1)
        sample_weight = np.arange(1, len(y) + 1)
        self.assertRaises(
            TypeError, reg_1.fit, X, y, sample_weight=sample_weight
        )
        reg_2.fit(X, y)

        reg_1 = SklearnRegressor(estimator=LinearRegression())
        reg_2 = clone(reg_1)
        reg_1.fit(X, y, sample_weight=sample_weight)
        reg_2.fit(X, y)
        self.assertTrue(np.any(reg_1.predict(X) != reg_2.predict(X)))

    def test_fit(self):
        class DummyRegressor(SkactivemlRegressor):
            def predict(self, X):
                raise NotFittedError()

            def fit(self, X, y, sample_weight=None):
                raise ValueError()

        reg = SklearnRegressor(DummyRegressor())

        X = np.arange(3 * 2).reshape(3, 2)
        y = np.append(np.full(2, MISSING_LABEL), [1.7])

        self.assertWarns(Warning, reg.fit, X=X, y=y)
        self.assertWarns(Warning, reg.predict, X=X)

    def test_predict(self):
        reg = SklearnRegressor(
            estimator=ARDRegression(),
            random_state=self.random_state,
        )

        X = np.arange(3 * 2).reshape(3, 2)
        y = np.full(3, MISSING_LABEL)

        reg.fit(X, y)
        y_pred = reg.predict(X)
        np.testing.assert_array_equal(np.zeros(3), y_pred)
        _, std_pred = reg.predict(X, return_std=True)
        np.testing.assert_array_equal(np.ones(3), std_pred)
        self.assertRaises(ValueError, reg.predict, X=[])

    def test_getattr(self):
        reg = SklearnRegressor(
            estimator=LinearRegression(),
            random_state=self.random_state,
        )
        self.assertTrue(hasattr(reg, "positive"))
        reg.fit(self.X, self.y)
        self.assertTrue(hasattr(reg, "coef_"))

    def test_sample_y(self):
        gpr = GaussianProcessRegressor(random_state=0)
        reg = SklearnRegressor(estimator=gpr)
        X = np.arange(4 * 2).reshape(4, 2)
        y = np.arange(4) - 1
        X_sample = 1 / 2 * np.arange(3 * 2).reshape(3, 2) + 1
        reg.fit(X, y)
        y_sample = reg.sample_y(X_sample, 5)
        y_sample_exp = gpr.fit(X, y).sample_y(X_sample, 5)
        np.testing.assert_array_equal(y_sample, y_sample_exp)

        lin_reg = LinearRegression()
        lin_reg.sample_y = lambda X, n_samples=1: np.vstack(
            [lin_reg.predict(X) for _ in range(n_samples)]
        )
        reg = SklearnRegressor(lin_reg)

        X = np.array([[0], [1], [2], [3], [4]])
        y = np.array([3, 4, 1, 2, 1])

        # Test without labels.
        reg.fit(X=[], y=[])
        y_sample = reg.sample_y(X, 10)
        np.testing.assert_array_equal(y_sample.shape, [5, 10])
        reg.fit(X=X, y=np.full_like(y, MISSING_LABEL))
        y_sample = reg.sample_y(X, 10)
        np.testing.assert_array_equal(y_sample.shape, [5, 10])

        # Test with labels.
        reg.fit(X=X, y=y)
        y_sample_exp = lin_reg.fit(X=X, y=y).sample_y(X, 10)
        y_sample = reg.sample_y(X, 10)
        np.testing.assert_array_equal(y_sample, y_sample_exp)
        self.assertRaises(ValueError, reg.sample_y, X=[])

    def test_sample(self):
        lin_reg = LinearRegression()
        lin_reg.sample = lambda X, n_samples=1: np.vstack(
            [lin_reg.predict(X) for _ in range(n_samples)]
        )
        reg = SklearnRegressor(lin_reg)

        X = np.array([[0], [1], [2], [3], [4]])
        y = np.array([3, 4, 1, 2, 1])

        # Test without labels.
        reg.fit(X=[], y=[])
        y_sample = reg.sample(X, 10)
        np.testing.assert_array_equal(y_sample.shape, [5, 10])
        reg.fit(X=X, y=np.full_like(y, MISSING_LABEL))
        y_sample = reg.sample(X, 10)
        np.testing.assert_array_equal(y_sample.shape, [5, 10])

        # Test with labels.
        reg.fit(X=X, y=y)
        y_sample_exp = lin_reg.fit(X=X, y=y).sample(X, 10)
        y_sample = reg.sample(X, 10)
        np.testing.assert_array_equal(y_sample, y_sample_exp)
        self.assertRaises(ValueError, reg.sample, X=[])

    def test_partial_fit(self):
        reg_1 = SklearnRegressor(
            SGDRegressor(random_state=self.random_state),
            random_state=self.random_state,
        )
        reg_2 = SklearnRegressor(
            SGDRegressor(random_state=self.random_state),
            random_state=self.random_state,
        )

        X = np.array([[0], [1], [2], [3], [4]])
        y = np.array([3, 4, 1, 2, 1])

        reg_1.partial_fit(X, y)
        reg_2.fit(X, y)
        self.assertTrue(
            np.any(np.not_equal(reg_1.predict(X), reg_2.predict(X)))
        )

    def test_pipeline(self):
        X = np.linspace(-3, 3, 100)
        y_true = X**2
        X = X.reshape(-1, 1)
        pipline = Pipeline(
            (
                ("scaler", PolynomialFeatures(degree=2)),
                ("lr", LinearRegression()),
            )
        )
        reg = SklearnRegressor(pipline, missing_label=np.nan, random_state=0)
        reg = reg.fit(X, y_true)
        check_is_fitted(reg)
        self.assertRaises(NotFittedError, check_is_fitted, pipline)
        self.assertGreaterEqual(reg.score(X, y_true), 0.9)
        y_missing = np.full_like(y_true, np.nan)
        reg.fit(X, y_missing)
        check_is_fitted(reg)
        y_pred = reg.predict(X)
        np.testing.assert_array_equal(np.zeros_like(y_pred), y_pred)


class TestSklearnProbabilisticRegressor(
    TemplateProbabilisticRegressor, unittest.TestCase
):
    def setUp(self):
        estimator_class = SklearnNormalRegressor
        estimator = GaussianProcessRegressor()
        init_default_params = {"estimator": estimator, "missing_label": np.nan}
        fit_default_params = {"X": np.zeros((3, 1)), "y": [0.5, 0.6, np.nan]}
        predict_default_params = {"X": [[1]]}
        super().setUp(
            estimator_class=estimator_class,
            init_default_params=init_default_params,
            fit_default_params=fit_default_params,
            predict_default_params=predict_default_params,
        )
        self.random_state = 0
        self.X = np.array([[0, 1], [1, 0], [2, 3]])
        self.y = np.array([1, 2, 3])
        self.X_cand = np.array([[2, 1], [3, 5]])

    def test_init_param_estimator(self):
        test_cases = []
        test_cases += [
            (GaussianProcessRegressor(), None),
            (SVC(), TypeError),
            ("Test", AttributeError),
        ]
        self._test_param("init", "estimator", test_cases)

    def test_fit_param_sample_weight(self, test_cases=None):
        replace_init_params = {"estimator": SGDRegressor()}
        super().test_fit_param_sample_weight(
            test_cases,
            replace_init_params=replace_init_params,
        )

    def test_partial_fit_param_X(self, test_cases=None):
        replace_init_params = {"estimator": SGDRegressor()}
        super().test_partial_fit_param_X(
            test_cases,
            replace_init_params=replace_init_params,
        )

    def test_partial_fit_param_y(self, test_cases=None):
        replace_init_params = {"estimator": SGDRegressor()}
        super().test_partial_fit_param_y(
            test_cases, replace_init_params=replace_init_params
        )

    def test_partial_fit_param_sample_weight(self, test_cases=None):
        replace_init_params = {"estimator": SGDRegressor()}
        super().test_partial_fit_param_sample_weight(
            test_cases,
            replace_init_params=replace_init_params,
        )

    def test_predict_target_distribution(self):
        reg = SklearnNormalRegressor(estimator=GaussianProcessRegressor())
        reg.fit(self.X, self.y)

        y_pred = reg.predict_target_distribution(self.X_cand).logpdf(0)
        self.assertEqual(y_pred.shape, (len(self.X_cand),))

        reg = SklearnNormalRegressor(estimator=LinearRegression())
        reg.fit(self.X, self.y)
        self.assertRaises(
            ValueError, reg.predict_target_distribution, self.X_cand
        )

    def test_fit(self):
        class DummyRegressor(SkactivemlRegressor):
            def predict(self, X, return_std=None, return_entropy=None):
                raise NotFittedError()

            def fit(self, X, y, sample_weight=None):
                raise ValueError()

        reg = SklearnNormalRegressor(DummyRegressor())

        X = np.arange(3 * 2).reshape(3, 2)
        y = np.append(np.full(2, MISSING_LABEL), [1.7])

        self.assertWarns(Warning, reg.fit, X=X, y=y)
        self.assertWarns(Warning, reg.predict, X=X)

    def test_predict(self):
        reg = SklearnNormalRegressor(
            estimator=ARDRegression(),
            random_state=self.random_state,
        )

        X = np.arange(3 * 2).reshape(3, 2)
        y = np.full(3, MISSING_LABEL)

        reg.fit(X, y)
        y_pred = reg.predict(X)
        np.testing.assert_array_equal(np.zeros(3), y_pred)
        _, std_pred = reg.predict(X, return_std=True)
        np.testing.assert_array_equal(np.ones(3), std_pred)

    def test_partial_fit(self):
        X_all, y_all = make_regression(n_samples=300, random_state=0)
        X_fit, y_fit = X_all[:200], y_all[:200]
        X_new, y_new = X_all[200:], y_all[200:]

        class GaussianProcessRegressorDummy(GaussianProcessRegressor):
            def partial_fit(self, X, y):
                return self.fit(X, y)

        reg = SklearnNormalRegressor(
            estimator=GaussianProcessRegressorDummy(),
            random_state=self.random_state,
        ).fit(X_fit, y_fit)
        y_pred = reg.predict(X_new)
        reg.partial_fit(X_new, y_new)
        y_pred_new = reg.predict(X_new)
        self.assertTrue(np.abs(y_pred_new - y_pred).sum() != 0)

    def test_pretrained_estimator(self):
        random_state = np.random.RandomState(0)
        X_full, y_full = make_regression(150, random_state=0)
        X_train = X_full[:100]
        y_train = y_full[:100]
        X_test = X_full[100:]
        missing_label = np.nan

        sgd_regressor_instance = SGDRegressor(
            loss="huber",
            random_state=0,
        )
        gp_regressor_instance = GaussianProcessRegressor(random_state=0)
        lr_regressor_instance = LinearRegression()
        # TODO: Is there a scikit-learn regressor that supports .sample(..)?
        # GaussianProcessRegressor does not seem to throw a NotFittedError
        cases = [
            (sgd_regressor_instance, NotFittedError),
            (gp_regressor_instance, None),
            (lr_regressor_instance, NotFittedError),
        ]

        for estimator, fit_exception in cases:
            # check that non-pretrained regressors fail without fitting
            reg_no_pretrain = SklearnRegressor(
                estimator=deepcopy(estimator),
                missing_label=missing_label,
                random_state=0,
            )
            if fit_exception is not None:
                self.assertRaises(
                    fit_exception, reg_no_pretrain.predict, X_test
                )

            for use_partial_fit in [False, True]:
                # pretrain regressor and test consistency of results after
                # wrapping
                pretrained_estimator = deepcopy(estimator)
                pretrained_estimator.fit(X_train, y_train)

                has_sample = hasattr(pretrained_estimator, "sample")
                has_sample_y = hasattr(pretrained_estimator, "sample_y")
                has_partial_fit = hasattr(pretrained_estimator, "partial_fit")

                reg = SklearnRegressor(
                    estimator=deepcopy(pretrained_estimator),
                    missing_label=missing_label,
                    random_state=0,
                )

                if use_partial_fit and has_partial_fit:
                    # update classifier and check results for consistency
                    # afterwards
                    y_train_random = random_state.permutation(y_train)

                    pretrained_estimator.partial_fit(X_train, y_train_random)
                    reg.partial_fit(X_train, y_train_random)

                if has_sample:
                    sample_orig_0 = pretrained_estimator.sample(X_test)
                    sample_wrapped_0 = reg.sample_y(X_test)
                    np.testing.assert_array_equal(
                        sample_orig_0, sample_wrapped_0
                    )

                if has_sample_y:
                    sample_y_orig_0 = pretrained_estimator.sample_y(X_test)
                    sample_y_wrapped_0 = reg.sample_y(X_test)
                    np.testing.assert_array_equal(
                        sample_y_orig_0, sample_y_wrapped_0
                    )

                pred_orig_0 = pretrained_estimator.predict(X_test)
                pred_wrapped_0 = reg.predict(X_test)
                np.testing.assert_array_equal(pred_orig_0, pred_wrapped_0)
