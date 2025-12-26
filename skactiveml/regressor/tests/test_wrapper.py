import random
import unittest
import numpy as np

from copy import deepcopy

from sklearn import clone
from sklearn.exceptions import NotFittedError
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression, ARDRegression, SGDRegressor
from sklearn.svm import SVC
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
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

successful_skorch_torch_import = False
try:
    import torch
    from torch import nn
    from skactiveml.regressor import SkorchRegressor
    from skorch.utils import to_numpy
    from torch.nn import HuberLoss

    successful_skorch_torch_import = True
except ImportError:
    pass  # pragma: no cover


class TestSklearnRegressor(TemplateSkactivemlRegressor, unittest.TestCase):
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
        test_cases = [
            (GaussianProcessRegressor(), None),
            (SVC(), TypeError),
            ("Test", AttributeError),
        ]
        self._test_param("init", "estimator", test_cases)

    def test_init_param_include_unlabeled_samples(self):
        test_cases = [
            (True, None),
            (False, None),
            ("String", TypeError),
        ]
        self._test_param("init", "include_unlabeled_samples", test_cases)

    def test_fit_predict(self):
        estimator = LinearRegression()
        reg = SklearnRegressor(estimator=estimator)
        y = np.full(3, MISSING_LABEL)
        reg.fit(self.X, y)
        self.assertRaises(NotFittedError, check_is_fitted, reg.estimator_)
        y = np.zeros(3)
        reg.fit(self.X, y)
        check_is_fitted(reg.estimator_)

        reg_1 = SklearnRegressor(estimator=LinearRegression())
        X = np.array([[0], [1], [2], [3], [4]])
        y = np.array([3, 4, 1, 2, 1])
        sample_weight = np.arange(1, len(y) + 1)

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

        # Test semi-supervised learning for regression.
        X = np.arange(10).astype(np.float64).reshape(-1, 1)
        y = np.arange(10).astype(np.float64)
        y_partial = np.full_like(y, fill_value=-100)
        y_partial[:5] = y[:5]
        reg = SklearnRegressor(
            GaussianProcessRegressor(random_state=0),
            random_state=0,
            include_unlabeled_samples=False,
            missing_label=-100,
        )
        y_pred_sl = reg.fit(X, y_partial).predict(X)
        reg = SklearnRegressor(
            GaussianProcessRegressor(random_state=0),
            random_state=0,
            include_unlabeled_samples=True,
            missing_label=-100,
        )
        y_pred_ssl = reg.fit(X, y_partial).predict(X)
        self.assertTrue(y_pred_ssl.mean() < y_pred_sl.mean())

    def test_partial_fit(self):

        reg_sklearn = SGDRegressor()
        reg = SklearnRegressor(reg_sklearn)
        self.assertRaises(NotFittedError, reg.predict, X=self.X)
        reg.partial_fit(X=self.X, y=self.y)
        check_is_fitted(reg)
        reg_no_partial_fit = SklearnRegressor(GaussianProcessRegressor())
        self.assertFalse(hasattr(reg_no_partial_fit, "partial_fit"))

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

        self.assertGreaterEqual(reg.score(X, y_true), 0.9)
        y_missing = np.full_like(y_true, np.nan)
        reg.fit(X, y_missing)
        check_is_fitted(reg)
        y_pred = reg.predict(X)
        np.testing.assert_array_equal(np.zeros_like(y_pred), y_pred)


class TestSklearnNormalRegressor(
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

        class GaussianProcessRegressorDummy(GaussianProcessRegressor):
            def partial_fit(self, X, y, sample_weight=None):
                return self.fit(X, y, sample_weight=sample_weight)

        self.prob_reg_partial_fit = GaussianProcessRegressorDummy()

    def test_init_param_estimator(self):
        test_cases = []
        test_cases += [
            (GaussianProcessRegressor(), None),
            (SVC(), ValueError),
            (LinearRegression(), ValueError),
            ("Test", AttributeError),
        ]
        self._test_param("init", "estimator", test_cases)

    def test_fit_param_sample_weight(self, test_cases=None):
        replace_init_params = {
            "estimator": Pipeline(
                (("s", StandardScaler()), ("r", SGDRegressor()))
            )
        }
        super().test_fit_param_sample_weight(
            test_cases,
            replace_init_params=replace_init_params,
        )

    def test_partial_fit_param_X(self, test_cases=None):
        replace_init_params = {"estimator": self.prob_reg_partial_fit}
        super().test_partial_fit_param_X(
            test_cases,
            replace_init_params=replace_init_params,
        )

    def test_partial_fit_param_y(self, test_cases=None):
        replace_init_params = {"estimator": self.prob_reg_partial_fit}
        super().test_partial_fit_param_y(
            test_cases, replace_init_params=replace_init_params
        )

    def test_partial_fit_param_sample_weight(self, test_cases=None):
        replace_init_params = {"estimator": self.prob_reg_partial_fit}
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
        self.assertRaises(ValueError, reg.fit, self.X, self.y)

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

        reg = SklearnNormalRegressor(
            estimator=self.prob_reg_partial_fit,
            random_state=self.random_state,
        ).fit(X_fit, y_fit)
        y_pred = reg.predict(X_new)
        reg.partial_fit(X_new, y_new)
        y_pred_new = reg.predict(X_new)
        self.assertTrue(np.abs(y_pred_new - y_pred).sum() != 0)
        reg = SklearnNormalRegressor(
            estimator=SGDRegressor(),
        )
        self.assertRaises(ValueError, reg.partial_fit, X_new, y_new)

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

    def test_pipeline(self):
        X = np.linspace(-3, 3, 100)
        y_true = X**2
        X = X.reshape(-1, 1)
        pipeline = Pipeline(
            (
                ("scaler", StandardScaler()),
                ("lr", LinearRegression()),
            )
        )

        reg = SklearnNormalRegressor(
            pipeline, missing_label=np.nan, random_state=0
        )
        reg.fit(X, y_true)

        self.assertRaises(ValueError, reg.predict, X)
        pipline = Pipeline(
            (
                ("scaler", StandardScaler()),
                ("lr", GaussianProcessRegressor()),
            )
        )
        reg = SklearnRegressor(pipline, missing_label=np.nan, random_state=0)
        reg = reg.fit(X, y_true)
        check_is_fitted(reg)
        reg.predict(X)


if successful_skorch_torch_import:

    class TestSkorchRegressor(TemplateSkactivemlRegressor, unittest.TestCase):
        def setUp(self):
            # Set global seeds.
            torch.manual_seed(0)
            np.random.seed(0)
            random.seed(0)
            self.X, self.y_true = make_regression(
                n_samples=200, n_features=10, random_state=0
            )
            self.X = self.X.astype(np.float32)
            self.y = np.copy(self.y_true).astype(np.float32)
            self.y[:100] = MISSING_LABEL
            self.y_ulbld = np.full_like(self.y, fill_value=MISSING_LABEL)

            estimator_class = SkorchRegressor
            self.neural_net_param_dict = {
                "train_split": None,
                "verbose": False,
                "optimizer": torch.optim.RAdam,
                "device": "cpu",
                "lr": 0.01,
                "max_epochs": 100,
                "batch_size": 32,
            }
            init_default_params = {
                "module": TestNeuralNet,
                "criterion": nn.HuberLoss,
                "missing_label": MISSING_LABEL,
                "random_state": 1,
                "neural_net_param_dict": self.neural_net_param_dict,
            }
            fit_default_params = {
                "X": self.X,
                "y": self.y,
            }
            predict_default_params = {"X": self.X}
            super().setUp(
                estimator_class=estimator_class,
                init_default_params=init_default_params,
                fit_default_params=fit_default_params,
                predict_default_params=predict_default_params,
            )

        def test_init_param_module(self, test_cases=None):
            reg = SkorchRegressor(module="Test")
            self.assertEqual(reg.module, "Test")

            test_cases = [] if test_cases is None else test_cases
            test_cases += [
                ("Test", TypeError),
                (None, TypeError),
                ([("nn.Module", TestNeuralNet)], TypeError),
            ]
            self._test_param("init", "module", test_cases)

        def test_init_param_criterion(self, test_cases=None):
            test_cases = [] if test_cases is None else test_cases
            test_cases += [
                ("Test", TypeError),
                (None, TypeError),
                (nn.MSELoss, None),
                (nn.HuberLoss, None),
                (nn.MSELoss(), None),
                (nn.HuberLoss(), None),
            ]
            self._test_param("init", "criterion", test_cases)

        def test_init_param_include_unlabeled_samples(self, test_cases=None):
            test_cases = [] if test_cases is None else test_cases
            test_cases += [
                (True, None),
                (False, None),
                ("String", TypeError),
            ]
            self._test_param("init", "include_unlabeled_samples", test_cases)

            test_cases = [(True, None)]

            class IgnoreIndexHuberLoss(HuberLoss):
                def forward(self, input, target):
                    is_lbld = ~torch.isnan(target)
                    input = input[is_lbld]
                    target = target[is_lbld]
                    return super().forward(input, target)

            self._test_param(
                "init",
                "include_unlabeled_samples",
                test_cases,
                replace_init_params={"criterion": IgnoreIndexHuberLoss},
            )

        def test_fit(self):
            # Check standard fitting cases.
            reg = SkorchRegressor(**self.init_default_params)
            self.assertRaises(NotFittedError, check_is_fitted, reg)
            reg.fit(self.X, self.y)
            check_is_fitted(reg)

            # Check fitting without `warm_restart`.
            init_default_params1 = self.init_default_params.copy()
            init_default_params1["neural_net_param_dict"]["warm_start"] = False
            reg = SkorchRegressor(**init_default_params1)
            reg.fit(self.X, self.y_ulbld)
            init_weights = to_numpy(
                deepcopy(reg.neural_net_.module_.input_to_hidden.weight)
            )
            reg.fit(self.X, self.y_ulbld)
            new_weights = to_numpy(
                deepcopy(reg.neural_net_.module_.input_to_hidden.weight)
            )
            self.assertRaises(
                AssertionError,
                np.testing.assert_array_equal,
                init_weights,
                new_weights,
            )

            # Check fitting with `warm_restart`.
            init_default_params2 = self.init_default_params.copy()
            init_default_params2["neural_net_param_dict"]["warm_start"] = True
            reg = SkorchRegressor(**init_default_params2)
            self.assertRaises(NotFittedError, check_is_fitted, reg)
            reg.fit(self.X, self.y_ulbld)
            check_is_fitted(reg)
            init_weights = to_numpy(
                deepcopy(reg.neural_net_.module_.input_to_hidden.weight)
            )
            reg.fit(self.X, self.y_ulbld)
            new_weights = to_numpy(
                deepcopy(reg.neural_net_.module_.input_to_hidden.weight)
            )
            np.testing.assert_array_equal(init_weights, new_weights)
            reg.fit(self.X, self.y)
            new_weights = to_numpy(
                deepcopy(reg.neural_net_.module_.input_to_hidden.weight)
            )
            self.assertRaises(
                AssertionError,
                np.testing.assert_array_equal,
                init_weights,
                new_weights,
            )

            # Setup for initialized Pytorch module as input.
            init_default_params3 = self.init_default_params.copy()
            reg_module = TestNeuralNet()
            init_weights = to_numpy(
                deepcopy(reg_module.input_to_hidden.weight)
            )
            init_default_params3["module"] = reg_module
            reg = SkorchRegressor(**init_default_params3)

            # Fitting with only unlabeled data must preserve weights.
            reg.fit(self.X, self.y_ulbld)
            new_weights = to_numpy(deepcopy(reg_module.input_to_hidden.weight))
            np.testing.assert_array_equal(init_weights, new_weights)

            # Fitting with partially label data must change weights.
            reg.fit(self.X, self.y)
            new_weights = to_numpy(
                deepcopy(reg.neural_net_.module_.input_to_hidden.weight)
            )
            self.assertRaises(
                AssertionError,
                np.testing.assert_array_equal,
                init_weights,
                new_weights,
            )

        def test_partial_fit(self):
            reg = SkorchRegressor(**self.init_default_params)
            self.assertRaises(NotFittedError, check_is_fitted, reg)
            reg.partial_fit(self.X, self.y)
            check_is_fitted(reg)

            init_default_params2 = self.init_default_params.copy()
            reg = SkorchRegressor(**init_default_params2)
            self.assertRaises(NotFittedError, check_is_fitted, reg)
            reg.partial_fit(self.X, self.y_ulbld)
            reg.partial_fit(self.X, self.y)
            check_is_fitted(reg)

            y_pred_0 = reg.predict(self.X)
            reg.partial_fit(self.X, self.y_ulbld)
            y_pred_1 = reg.predict(self.X)
            np.testing.assert_almost_equal(y_pred_0, y_pred_1)

        def test_predict(self):
            init_default_params = self.init_default_params.copy()
            init_default_params["forward_outputs"] = {
                "output": (0, torch.ravel),
                "exp-output": (0, torch.exp),
                "emb": (1, None),
            }
            reg = SkorchRegressor(**init_default_params)
            y_pred, X_embed, y_pred_exp = reg.predict(
                self.X, extra_outputs=["emb", "exp-output"]
            )
            self.assertEqual(len(y_pred), len(self.X))
            self.assertTrue(X_embed.shape[1], 2)
            np.testing.assert_almost_equal(np.exp(y_pred), y_pred_exp.ravel())
            init_default_params = self.init_default_params.copy()
            reg = SkorchRegressor(**init_default_params)
            y_pred_0 = reg.predict(self.X)
            reg.partial_fit(self.X, self.y_ulbld)
            y_pred_1 = reg.predict(self.X)
            np.testing.assert_almost_equal(y_pred_0, y_pred_1)
            reg.fit(self.X, self.y)
            self.assertGreaterEqual(reg.score(self.X, self.y_true), 0.9)

        def test_init_param_sample_dtype(self):
            test_cases = [
                (None, None),
                (np.float32, None),
                (np.int32, RuntimeError),
            ]
            self._test_param("init", "sample_dtype", test_cases)

        def test_init_param_neural_net_param_dict(self):
            default_dict = self.init_default_params["neural_net_param_dict"]
            test_cases = [
                (None, None),
                (default_dict, None),
                (default_dict, None),
                (np.int32, TypeError),
                ("a", TypeError),
                ({"abcdefg": 0}, ValueError),
                ({"predict_nonlinearity": nn.Identity()}, ValueError),
                ({"module": TestNeuralNet}, ValueError),
                ({"criterion": nn.MSELoss}, ValueError),
            ]
            self._test_param("init", "neural_net_param_dict", test_cases)

        def test_init_param_forward_outputs(self):
            test_cases = [
                (None, None),
                ({"output": (0, None)}, None),
                ({"output": (0, None), "emb": (1, None)}, None),
                (
                    {
                        "output": (0, None),
                        "exp-output": (0, torch.exp),
                        "emb": (1, None),
                    },
                    None,
                ),
                ({"output": (0,)}, TypeError),
                ({"output": (-1, None)}, ValueError),
                ({"output": ("str", None)}, TypeError),
                ({"output": (2, None)}, ValueError),
            ]
            self._test_param("init", "forward_outputs", test_cases)

            test_cases = [
                (None, None),
                ({"output": (0, torch.exp)}, None),
            ]
            self._test_param(
                "init",
                "forward_outputs",
                test_cases,
                replace_init_params={"criterion": nn.MSELoss},
            )

        def test_init_param_criterion_output_keys(self):
            test_cases = [
                (None, None),
                ("output", None),
                (["output"], None),
                ("test", ValueError),
                (["test"], ValueError),
                (False, TypeError),
            ]
            self._test_param("init", "criterion_output_keys", test_cases)

            replace_init_params = {
                "forward_outputs": {
                    "output": (0, None),
                    "exp-output": (0, torch.exp),
                    "emb": (1, None),
                },
                "criterion": nn.MSELoss,
            }
            test_cases += [
                ("output", None),
                (["output"], None),
                ("emb", None),
                (["emb"], None),
                (["output", "exp-output"], ValueError),
                (["exp-output", "emb"], AttributeError),
            ]
            self._test_param(
                "init",
                "criterion_output_keys",
                test_cases,
                replace_init_params=replace_init_params,
            )
            nn_rep = self.init_default_params["neural_net_param_dict"].copy()
            nn_rep["module__return_embeddings"] = False
            test_cases = [
                (None, None),
                ("output", None),
                (["output"], None),
                ("test", ValueError),
                (["test"], ValueError),
                (False, TypeError),
            ]
            self._test_param(
                "init",
                "criterion_output_keys",
                test_cases,
                replace_init_params={"neural_net_param_dict": nn_rep},
            )

        def test_predict_param_extra_outputs(self):
            test_cases = [
                (None, None),
                ([], None),
                ("outputs", ValueError),
                (["outputs"], ValueError),
                ("emb", ValueError),
                (["emb"], ValueError),
                ("exp-outputs", ValueError),
                (["exp-outputs", "emb"], ValueError),
                (["emb", "exp-outputs"], ValueError),
                (False, TypeError),
            ]
            self._test_param(
                "predict",
                "extra_outputs",
                test_cases,
                extras_params={"X": self.X},
            )
            test_cases = [
                (None, None),
                ([], None),
                ("outputs", ValueError),
                (["outputs"], ValueError),
                ("emb", None),
                (["emb"], None),
                ("exp-outputs", None),
                (["exp-outputs", "emb"], None),
                (["emb", "exp-outputs"], None),
                (False, TypeError),
            ]
            replace_init_params = {
                "forward_outputs": {
                    "outputs": (0, None),
                    "exp-outputs": (0, torch.exp),
                    "emb": (1, None),
                }
            }
            self._test_param(
                "predict",
                "extra_outputs",
                test_cases,
                extras_params={"X": self.X},
                replace_init_params=replace_init_params,
            )

    class TestNeuralNet(nn.Module):
        def __init__(self, return_embeddings=True):
            super().__init__()
            self.return_embeddings = return_embeddings
            self.input_to_hidden = nn.Linear(
                in_features=10,
                out_features=128,
                bias=True,
                dtype=torch.float32,
            )
            self.hidden_to_output = nn.Linear(
                in_features=128, out_features=1, bias=True, dtype=torch.float32
            )

        def forward(self, X):
            hidden = self.input_to_hidden(X)
            hidden = torch.relu(hidden)
            output_values = self.hidden_to_output(hidden)
            if self.return_embeddings:
                return output_values, hidden
            else:
                return output_values

    class TestSkorchProbabilisticRegressor(TestSkorchRegressor):
        pass
