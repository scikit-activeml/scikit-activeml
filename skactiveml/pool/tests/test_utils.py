import itertools
import unittest
from itertools import product

import numpy as np
from scipy.stats import norm
from sklearn.exceptions import NotFittedError
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import pairwise_kernels
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import make_blobs

from skactiveml.classifier import ParzenWindowClassifier, SklearnClassifier
from skactiveml.pool.utils import _cross_entropy
from skactiveml.pool.utils import (
    IndexClassifierWrapper,
    _conditional_expect,
    _reshape_scipy_dist,
    _update_X_y,
    _update_reg,
)
from skactiveml.pool._expected_model_change_maximization import (
    _bootstrap_estimators,
)
from skactiveml.regressor import (
    NICKernelRegressor,
    SklearnRegressor,
    SklearnNormalRegressor,
)
from skactiveml.utils import (
    MISSING_LABEL,
    is_unlabeled,
    unlabeled_indices,
    labeled_indices,
)


class TestIndexClassifierWrapper(unittest.TestCase):
    def setUp(self):
        self.X = np.linspace(0, 1, 4).reshape(-1, 1)
        self.y = np.array([0, 1, MISSING_LABEL, MISSING_LABEL])
        self.y2 = np.array([0.0, 1.0, 0.0, 1.0])
        self.y3 = np.array([0, MISSING_LABEL, MISSING_LABEL, MISSING_LABEL])
        self.clf = ParzenWindowClassifier(classes=[0, 1])
        self.kwargs = dict(X=self.X, y=self.y, clf=self.clf)
        self.iclf = lambda **kw: IndexClassifierWrapper(
            self.clf, self.X, self.y, **kw
        )

    def test_init_param_clf(self):
        self.assertTrue(hasattr(self.iclf(), "clf"))
        self.assertRaises(
            TypeError, IndexClassifierWrapper, clf="str", X=self.X, y=self.y
        )
        clf = self.clf.fit(self.X, self.y)
        iclf = IndexClassifierWrapper(clf=clf, X=self.X, y=self.y)
        np.testing.assert_array_equal(clf.X_, iclf.clf_.X_)

    def test_dtype_error(self):
        X, y = make_blobs(
            n_samples=2000, centers=[(-2, 2), (2, -2)], n_features=2
        )
        clf = ParzenWindowClassifier(
            classes=np.unique(y), missing_label=MISSING_LABEL
        )
        y_known = np.full(len(y), MISSING_LABEL)
        self.assertRaises(
            TypeError,
            IndexClassifierWrapper,
            clf=clf,
            X=X,
            y=y,
            missing_label=MISSING_LABEL,
        )
        y = y.astype(float)
        id_clf = IndexClassifierWrapper(clf, X, y, missing_label=MISSING_LABEL)
        id_clf.fit(np.arange(len(X)), y_known, set_base_clf=True)

    def test_init_param_X(self):
        self.assertTrue(hasattr(self.iclf(), "X"))
        self.assertRaises(
            (ValueError, TypeError),
            IndexClassifierWrapper,
            clf=self.clf,
            X="str",
            y=self.y,
        )

    def test_init_param_y(self):
        self.assertTrue(hasattr(self.iclf(), "y"))
        self.assertRaises(
            (ValueError, TypeError),
            IndexClassifierWrapper,
            clf=self.clf,
            X=self.X,
            y="str",
        )
        self.assertRaises(
            ValueError, IndexClassifierWrapper, clf=self.clf, X=self.X, y=[0]
        )

    def test_init_param_sample_weight(self):
        self.assertTrue(hasattr(self.iclf(), "sample_weight"))
        self.assertEqual(self.iclf().sample_weight, None)
        self.assertRaises(
            (ValueError, TypeError),
            IndexClassifierWrapper,
            clf=self.clf,
            X=self.X,
            y=self.y,
            sample_weight="s",
        )
        self.assertRaises(
            (ValueError, TypeError),
            IndexClassifierWrapper,
            clf=self.clf,
            X=self.X,
            y=self.y,
            sample_weight=[0],
        )

    def test_init_param_fit_base_clf(self):
        self.assertRaises(TypeError, self.iclf, set_base_clf="string")
        self.assertRaises(NotFittedError, self.iclf, set_base_clf=True)

        clf = self.clf.fit(self.X, self.y)
        iclf = IndexClassifierWrapper(
            clf=clf, X=self.X, y=self.y, set_base_clf=True
        )
        np.testing.assert_array_equal(clf.X_, iclf.base_clf_.X_)

    def test_init_param_ignore_partial_fit(self):
        self.assertTrue(hasattr(self.iclf(), "ignore_partial_fit"))
        self.assertEqual(self.iclf().ignore_partial_fit, False)

        self.assertRaises(TypeError, self.iclf, ignore_partial_fit="string")

    def test_init_param_enforce_unique_samples(self):
        self.assertTrue(hasattr(self.iclf(), "enforce_unique_samples"))
        self.assertEqual(self.iclf().enforce_unique_samples, False)

        self.assertRaises(
            TypeError, self.iclf, enforce_unique_samples="string"
        )

        self.assertWarns(
            Warning,
            IndexClassifierWrapper,
            clf=SklearnClassifier(GaussianNB()),
            X=self.X,
            y=self.y,
            enforce_unique_samples=True,
        )

    def test_init_param_use_speed_up(self):
        self.assertTrue(hasattr(self.iclf(), "use_speed_up"))
        self.assertEqual(self.iclf().use_speed_up, False)

        self.assertRaises(TypeError, self.iclf, use_speed_up="string")

        clf = ParzenWindowClassifier().fit(self.X, self.y)
        iclf = IndexClassifierWrapper(
            clf, self.X, self.y, use_speed_up=True, ignore_partial_fit=False
        )
        self.assertWarns(Warning, iclf.predict, [0])
        self.assertWarns(Warning, iclf.predict_proba, [0])
        self.assertWarns(Warning, iclf.predict_freq, [0])

    def test_init_param_missing_label(self):
        self.assertTrue(hasattr(self.iclf(), "missing_label"))
        self.assertTrue(
            is_unlabeled(
                [self.iclf().missing_label], missing_label=MISSING_LABEL
            )
        )

        self.assertRaises(TypeError, self.iclf, missing_label="string")

    def test_precompute_param_idx_fit(self):
        iclf = self.iclf()
        self.assertRaises((ValueError, TypeError), iclf.precompute, "str", [0])
        self.assertRaises((ValueError, TypeError), iclf.precompute, [10], [0])

    def test_precompute_param_idx_pred(self):
        iclf = self.iclf()
        self.assertRaises((ValueError, TypeError), iclf.precompute, [0], "str")
        self.assertRaises((ValueError, TypeError), iclf.precompute, [0], [10])

    def test_precompute_param_fit_params(self):
        iclf = self.iclf(use_speed_up=True)
        self.assertRaises(
            (ValueError, TypeError), iclf.precompute, [0], [0], fit_params=2
        )
        self.assertRaises(
            (ValueError, TypeError),
            iclf.precompute,
            [0],
            [0],
            fit_params="wrong_str",
        )

    def test_precompute_param_pred_params(self):
        iclf = self.iclf(use_speed_up=True)
        self.assertRaises(
            (ValueError, TypeError), iclf.precompute, [0], [0], pred_params=2
        )
        self.assertRaises(
            (ValueError, TypeError),
            iclf.precompute,
            [0],
            [0],
            pred_params="wrong_str",
        )

    def test_precompute(self):
        all_idx = np.arange(len(self.X))
        params = [
            ("all", all_idx),
            ("labeled", labeled_indices(self.y)),
            ("unlabeled", unlabeled_indices(self.y)),
        ]
        for (fit_str, fit_idx), (pred_str, pred_idx) in list(
            product(params, params)
        ):
            with self.subTest(msg="Sub", fit_str=fit_str, pred_str=pred_str):
                iclf = self.iclf(use_speed_up=True)
                iclf.precompute(
                    all_idx, all_idx, fit_params=fit_str, pred_params=pred_str
                )
                K = np.full([len(all_idx), len(all_idx)], np.nan)
                K[np.ix_(fit_idx, pred_idx)] = pairwise_kernels(
                    self.X[fit_idx], self.X[pred_idx], metric="rbf"
                )

                np.testing.assert_array_equal(K, iclf.pwc_K_)

    def test_fit_param_idx(self):
        iclf = self.iclf()
        self.assertRaises((ValueError, TypeError), iclf.fit, 0)
        self.assertRaises((ValueError, TypeError), iclf.fit, "wrong_str")
        self.assertRaises((ValueError, TypeError), iclf.fit, [10])

    def test_fit_param_y(self):
        iclf = self.iclf()
        self.assertRaises((ValueError, TypeError), iclf.fit, [0], y="str")
        self.assertRaises((ValueError, TypeError), iclf.fit, [0], y=[0, 0])

    def test_fit_param_sample_weight(self):
        iclf = self.iclf()
        self.assertRaises(
            (ValueError, TypeError), iclf.fit, [0], sample_weight="str"
        )
        self.assertRaises(
            (ValueError, TypeError), iclf.fit, [0], sample_weight=[0, 0]
        )

    def test_fit_param_set_base_clf(self):
        iclf = self.iclf()
        self.assertRaises(TypeError, iclf.fit, [0], set_base_clf="string")

    def test_partial_fit_param_idx(self):
        iclf = self.iclf().fit([0])
        self.assertRaises((ValueError, TypeError), iclf.partial_fit, 0)
        self.assertRaises((ValueError, TypeError), iclf.partial_fit, "str")
        self.assertRaises((ValueError, TypeError), iclf.partial_fit, [10])

    def test_partial_fit_param_y(self):
        iclf = self.iclf().fit([0])
        self.assertRaises(
            (ValueError, TypeError), iclf.partial_fit, [0], y="str"
        )
        self.assertRaises(
            (ValueError, TypeError), iclf.partial_fit, [0], y=[0, 0]
        )

    def test_partial_fit_param_sample_weight(self):
        iclf = self.iclf().fit([0])
        self.assertRaises(
            (ValueError, TypeError), iclf.partial_fit, [0], sample_weight="str"
        )
        self.assertRaises(
            (ValueError, TypeError),
            iclf.partial_fit,
            [0],
            sample_weight=[0, 0],
        )

    def test_partial_fit_param_use_base_clf(self):
        iclf = self.iclf().fit([0])
        self.assertRaises(
            TypeError, iclf.partial_fit, [0], use_base_clf="string"
        )

    def test_partial_fit_param_set_base_clf(self):
        iclf = self.iclf().fit([0])
        self.assertRaises(
            TypeError, iclf.partial_fit, [0], set_base_clf="string"
        )

    def test_predict_param_idx(self):
        iclf = self.iclf().fit([0])
        self.assertRaises((ValueError, TypeError, IndexError), iclf.predict, 0)
        self.assertRaises(
            (ValueError, TypeError, IndexError), iclf.predict, "str"
        )
        self.assertRaises(
            (ValueError, TypeError, IndexError), iclf.predict, [10]
        )

        iclf = self.iclf(use_speed_up=True).fit([0])
        self.assertRaises(ValueError, iclf.predict, [1])

    def test_predict_proba_param_idx(self):
        iclf = self.iclf().fit([0])
        self.assertRaises(
            (ValueError, TypeError, IndexError), iclf.predict_proba, 0
        )
        self.assertRaises(
            (ValueError, TypeError, IndexError), iclf.predict_proba, "str"
        )
        self.assertRaises(
            (ValueError, TypeError, IndexError), iclf.predict_proba, [10]
        )

        iclf = self.iclf(use_speed_up=True).fit([0])
        self.assertRaises(ValueError, iclf.predict_proba, [1])

    def test_predict_freq_param_idx(self):
        iclf = self.iclf().fit([0])
        self.assertRaises(
            (ValueError, TypeError, IndexError), iclf.predict_freq, 0
        )
        self.assertRaises(
            (ValueError, TypeError, IndexError), iclf.predict_freq, "str"
        )
        self.assertRaises(
            (ValueError, TypeError, IndexError), iclf.predict_freq, [10]
        )

        iclf = self.iclf(use_speed_up=True).fit([0])
        self.assertRaises(ValueError, iclf.predict_freq, [1])

    def test_getattr(self):
        clf = ParzenWindowClassifier(classes=[0, 1])
        iclf = IndexClassifierWrapper(clf, self.X, self.y)
        self.assertEqual(iclf.clf.classes, iclf.classes)
        iclf.fit([0, 1])
        self.assertEqual(iclf.clf_.classes, iclf.classes)

    def test__concat_sw(self):
        iclf = IndexClassifierWrapper(
            clf=ParzenWindowClassifier(), X=self.X, y=self.y
        )
        self.assertRaises(ValueError, iclf._concat_sw, [1], None)

    def test_fit(self):
        iclf = IndexClassifierWrapper(
            clf=ParzenWindowClassifier(), X=self.X, y=self.y
        )
        # self.assertWarns(Warning, iclf.fit, [2, 3])
        self.assertRaises(ValueError, iclf.fit, [2, 3])

        base_clfs = [
            lambda: ParzenWindowClassifier(classes=[0, 1]),
            lambda: SklearnClassifier(GaussianNB(), classes=[0, 1]),
        ]
        speed_ups = [True, False]
        sample_weights = [None, np.linspace(0.2, 1, 4)]
        preds = ["predict", "predict_proba", "predict_freq"]

        params = list(
            product(base_clfs[:1], speed_ups, sample_weights, preds)
        ) + list(product(base_clfs[1:], speed_ups, sample_weights, preds[:2]))

        for BaseClf, speed_up, sample_weight, pred in params:
            with self.subTest(
                msg="Test fit via init",
                BaseClf=BaseClf,
                speed_up=speed_up,
                sample_weight=sample_weight,
                pred=pred,
            ):
                for i in range(1, 4):
                    clf = BaseClf()
                    sw_ = None if sample_weight is None else sample_weight[:i]
                    clf.fit(self.X[:i], self.y2[:i], sample_weight=sw_)

                    iclf = IndexClassifierWrapper(
                        BaseClf(),
                        self.X,
                        self.y2,
                        sample_weight=sample_weight,
                        use_speed_up=speed_up,
                    )

                    if speed_up:
                        iclf.precompute(np.arange(i), np.arange(4))

                    iclf.fit(np.arange(i))
                    np.testing.assert_allclose(
                        getattr(iclf, pred)(np.arange(4)),
                        getattr(clf, pred)(self.X),
                    )

            with self.subTest(
                msg="Test direct fit",
                BaseClf=BaseClf,
                speed_up=speed_up,
                sample_weight=sample_weight,
                pred=pred,
            ):
                for i in range(1, 4):
                    clf = BaseClf()
                    sw_ = None if sample_weight is None else sample_weight[:i]
                    clf.fit(self.X[:i], self.y2[:i], sample_weight=sw_)

                    iclf = IndexClassifierWrapper(
                        BaseClf(),
                        self.X,
                        np.full(4, np.nan),
                        use_speed_up=speed_up,
                    )

                    if speed_up:
                        iclf.precompute(np.arange(i), np.arange(4))

                    iclf.fit(np.arange(i), y=self.y2[:i], sample_weight=sw_)
                    np.testing.assert_allclose(
                        getattr(iclf, pred)(np.arange(4)),
                        getattr(clf, pred)(self.X),
                    )

    def test_partial_fit(self):
        iclf = self.iclf()
        self.assertRaises(
            NotFittedError, iclf.partial_fit, [0], use_base_clf=False
        )
        self.assertRaises(
            NotFittedError, iclf.partial_fit, [0], use_base_clf=True
        )
        iclf = IndexClassifierWrapper(
            clf=ParzenWindowClassifier().fit(self.X, self.y),
            X=self.X,
            y=self.y,
            set_base_clf=True,
        )
        self.assertRaises(
            NotFittedError, iclf.partial_fit, [0], use_base_clf=False
        )
        self.assertRaises(
            NotFittedError, iclf.partial_fit, [0], use_base_clf=True
        )

        iclf = IndexClassifierWrapper(
            clf=ParzenWindowClassifier(), X=self.X, y=self.y
        )
        iclf.fit([0, 1])
        self.assertWarns(Warning, iclf.partial_fit, [2, 3])

        base_clfs = [
            lambda: ParzenWindowClassifier(classes=[0, 1]),
            lambda: SklearnClassifier(GaussianNB(), classes=[0, 1]),
        ]
        speed_ups = [True, False]
        sample_weights = [None, np.linspace(0.2, 1, 4)]
        preds = ["predict", "predict_proba"]
        enforce_uniques = [True, False]

        params = list(
            product(
                base_clfs[:1],
                speed_ups,
                sample_weights,
                preds,
                enforce_uniques,
            )
        )

        for BaseClf, speed_up, sample_weight, pred, enforce_unique in params:
            with self.subTest(
                msg="ParzenWindowClassifier use base data",
                BaseClf=str(BaseClf()),
                speed_up=speed_up,
                sample_weight=sample_weight,
                pred=pred,
                enforce_unique=enforce_unique,
            ):
                iclf = IndexClassifierWrapper(
                    BaseClf(),
                    self.X,
                    self.y2,
                    sample_weight=sample_weight,
                    use_speed_up=speed_up,
                    enforce_unique_samples=enforce_unique,
                )
                if speed_up:
                    iclf.precompute(np.arange(4), np.arange(4))
                init_idx = [0]
                iclf.fit(init_idx)
                all_idx = list(init_idx)
                for add_idx in [[1], [2, 3], [3]]:
                    all_idx = np.concatenate([all_idx, add_idx], axis=0)
                    if enforce_unique:
                        all_idx = np.unique(all_idx)
                    clf = BaseClf()
                    sw_ = (
                        None
                        if sample_weight is None
                        else sample_weight[all_idx]
                    )
                    clf.fit(
                        self.X[all_idx], self.y2[all_idx], sample_weight=sw_
                    )

                    iclf.partial_fit(add_idx)
                    np.testing.assert_allclose(
                        getattr(iclf, pred)(np.arange(4)),
                        getattr(clf, pred)(self.X),
                    )

        for BaseClf, speed_up, sample_weight, pred, enforce_unique in params:
            with self.subTest(
                msg="ParzenWindowClassifier use fit data",
                BaseClf=str(BaseClf()),
                speed_up=speed_up,
                sample_weight=sample_weight,
                pred=pred,
                enforce_unique=enforce_unique,
            ):
                iclf = IndexClassifierWrapper(
                    BaseClf(),
                    self.X,
                    self.y3,
                    sample_weight=sample_weight,
                    use_speed_up=speed_up,
                    enforce_unique_samples=enforce_unique,
                )
                if speed_up:
                    iclf.precompute(np.arange(4), np.arange(4))
                init_idx = [0]
                iclf.fit(init_idx)
                all_idx = list(init_idx)
                for add_idx in [[1], [2, 3], [3]]:
                    all_idx = np.concatenate([all_idx, add_idx], axis=0)
                    if enforce_unique:
                        all_idx = np.unique(all_idx)
                    clf = BaseClf()
                    sw_ = (
                        None
                        if sample_weight is None
                        else sample_weight[all_idx]
                    )
                    clf.fit(
                        self.X[all_idx], self.y2[all_idx], sample_weight=sw_
                    )

                    sw_add_ = (
                        None
                        if sample_weight is None
                        else sample_weight[add_idx]
                    )
                    iclf.partial_fit(
                        add_idx, y=self.y2[add_idx], sample_weight=sw_add_
                    )
                    np.testing.assert_allclose(
                        getattr(iclf, pred)(np.arange(4)),
                        getattr(clf, pred)(self.X),
                    )

        for BaseClf, speed_up, sample_weight, pred, enforce_unique in params:
            with self.subTest(
                msg="ParzenWindowClassifier use fit data with base clf",
                BaseClf=str(BaseClf()),
                speed_up=speed_up,
                sample_weight=sample_weight,
                pred=pred,
                enforce_unique=enforce_unique,
            ):
                iclf = IndexClassifierWrapper(
                    BaseClf(),
                    self.X,
                    self.y3,
                    sample_weight=sample_weight,
                    use_speed_up=speed_up,
                    enforce_unique_samples=enforce_unique,
                )
                if speed_up:
                    iclf.precompute(np.arange(4), np.arange(4))
                init_idx = [0]
                iclf.fit(init_idx, set_base_clf=True)
                all_idx = list(init_idx)
                for add_idx in [[1], [2, 3], [3]]:
                    all_idx = np.concatenate([all_idx, add_idx], axis=0)
                    if enforce_unique:
                        all_idx = np.unique(all_idx)
                    clf = BaseClf()
                    sw_ = (
                        None
                        if sample_weight is None
                        else sample_weight[all_idx]
                    )
                    clf.fit(
                        self.X[all_idx], self.y2[all_idx], sample_weight=sw_
                    )

                    sw_add_ = (
                        None
                        if sample_weight is None
                        else sample_weight[add_idx]
                    )
                    iclf.partial_fit(
                        add_idx,
                        y=self.y2[add_idx],
                        sample_weight=sw_add_,
                        use_base_clf=True,
                        set_base_clf=True,
                    )
                    np.testing.assert_allclose(
                        getattr(iclf, pred)(np.arange(4)),
                        getattr(clf, pred)(self.X),
                    )

        params = list(
            product(base_clfs[1:], speed_ups, sample_weights, preds[:2])
        )
        for BaseClf, speed_up, sample_weight, pred in params:
            with self.subTest(
                msg="NB use fit data",
                BaseClf=str(BaseClf()),
                speed_up=speed_up,
                sample_weight=sample_weight,
                pred=pred,
            ):
                iclf = IndexClassifierWrapper(
                    BaseClf(),
                    self.X,
                    self.y,
                    sample_weight=sample_weight,
                    use_speed_up=speed_up,
                    ignore_partial_fit=False,
                )
                clf = BaseClf()
                if speed_up:
                    iclf.precompute(np.arange(4), np.arange(4))
                init_idx = [0, 1]
                iclf.fit(init_idx, set_base_clf=True)
                sw_ = (
                    None if sample_weight is None else sample_weight[init_idx]
                )
                clf.fit(self.X[init_idx], self.y2[init_idx], sample_weight=sw_)

                for add_idx in [[2], [3]]:
                    sw_add_ = (
                        None
                        if sample_weight is None
                        else sample_weight[add_idx]
                    )
                    iclf.partial_fit(
                        add_idx,
                        y=self.y2[add_idx],
                        sample_weight=sw_add_,
                        use_base_clf=True,
                        set_base_clf=True,
                    )
                    clf.partial_fit(
                        self.X[add_idx],
                        y=self.y2[add_idx],
                        sample_weight=sw_add_,
                    )
                    np.testing.assert_allclose(
                        getattr(iclf, pred)(np.arange(4)),
                        getattr(clf, pred)(self.X),
                    )


class TestApproximation(unittest.TestCase):
    def setUp(self):
        self.random_state = 0

    def test_conditional_expectation_params(self):
        dummy_func_1 = np.zeros_like

        def dummy_func_2(x, y):
            return np.zeros_like(y)

        X = np.arange(4 * 2).reshape(4, 2)
        y = np.arange(4, dtype=float)
        reg = NICKernelRegressor().fit(X, y)

        illegal_argument_dict = {
            "X": ["illegal", np.arange(3)],
            "func": ["illegal", dummy_func_2],
            "reg": ["illegal", SklearnRegressor(LinearRegression())],
            "method": ["illegal", 7, dict],
            "quantile_method": ["illegal", 7, dict],
            "n_integration_samples": ["illegal", 0, dict],
            "quad_dict": ["illegal", 7, dict],
            "random_state": ["illegal", dict],
            "include_x": ["illegal", dict, 7],
            "include_idx": ["illegal", dict, 7],
            "vector_func": ["illegal", dict, 7],
        }

        for parameter in illegal_argument_dict:
            for illegal_argument in illegal_argument_dict[parameter]:
                param_dict = dict(
                    X=X, func=dummy_func_1, reg=reg, method="quantile"
                )
                param_dict[parameter] = illegal_argument
                self.assertRaises(
                    (TypeError, ValueError), _conditional_expect, **param_dict
                )

    def test_conditional_expectation(self):
        reg = SklearnNormalRegressor(estimator=GaussianProcessRegressor())
        X_train = np.array([[0, 2, 3], [1, 3, 4], [2, 4, 5], [3, 6, 7]])
        y_train = np.array([-1, 2, 1, 4])
        reg.fit(X_train, y_train)

        parameters_1 = [
            {"method": "assume_linear"},
            {"method": "monte_carlo", "n_integration_samples": 2},
            {"method": "monte_carlo", "n_integration_samples": 4},
            {"method": None, "quantile_method": "trapezoid"},
            {"method": "quantile", "quantile_method": "simpson"},
            {"method": "quantile", "quantile_method": "romberg"},
            {"method": "quantile", "quantile_method": "trapezoid"},
            {"method": "quantile", "quantile_method": "average"},
            {"method": "quantile", "quantile_method": "quadrature"},
            {"method": "dynamic_quad"},
            {"method": "gauss_hermite"},
        ]

        parameters_2 = [
            {"vector_func": True},
            {"vector_func": False},
            {"vector_func": "both"},
        ]

        X = np.arange(2 * 3).reshape((2, 3))

        for parameter_1, parameter_2 in itertools.product(
            parameters_1, parameters_2
        ):
            parameter = {**parameter_1, **parameter_2}

            def dummy_func(idx, x, y):
                if parameter["vector_func"] == "both":
                    if parameter["method"] == "dynamic_quad":
                        self.assertTrue(isinstance(idx, int))
                        self.assertTrue(isinstance(y, float))
                        self.assertEqual(x.shape, (3,))
                        return 0
                    else:
                        self.assertEqual(y.ndim, 2)
                        self.assertEqual(len(y), 2)
                        self.assertEqual(idx.dtype, int)
                        np.testing.assert_array_equal(x, X)
                        np.testing.assert_array_equal(idx, np.arange(2))
                        return np.zeros_like(y)
                elif parameter["vector_func"]:
                    self.assertEqual(y.ndim, 2)
                    self.assertEqual(len(y), 2)
                    self.assertEqual(idx.dtype, int)
                    np.testing.assert_array_equal(x, X)
                    np.testing.assert_array_equal(idx, np.arange(2))
                    return np.zeros_like(y)
                else:
                    self.assertTrue(isinstance(idx, int))
                    self.assertTrue(isinstance(y, float))
                    self.assertEqual(x.shape, (3,))
                    return 0

            res = _conditional_expect(
                X=X, func=dummy_func, reg=reg, **parameter
            )

            np.testing.assert_array_equal(res, np.zeros(2))

    def test_reshape_distribution(self):
        dist = norm(loc=np.array([0, 0]))
        _reshape_scipy_dist(dist, shape=(2, 1))
        self.assertEqual(dist.kwds["loc"].shape, (2, 1))
        self.assertRaises(TypeError, _reshape_scipy_dist, dist, "illegal")
        self.assertRaises(TypeError, _reshape_scipy_dist, "illegal", (2, 1))


class TestFunctions(unittest.TestCase):
    def setUp(self):
        self.reg = SklearnRegressor(LinearRegression())
        self.X = np.arange(7 * 2).reshape(7, 2)
        self.y = np.arange(7)
        self.mapping = np.array([3, 4, 5])
        self.sample_weight = np.ones_like(self.y)
        self.x_pot = np.array([3, 4])
        self.y_pot = 5

    def test_update_X_y(self):
        X_new, y_new = _update_X_y(
            self.X, self.y, self.y_pot, X_update=self.x_pot
        )

        self.assertEqual(X_new.shape, (8, 2))
        self.assertEqual(y_new.shape, (8,))
        np.testing.assert_equal(X_new[7], self.x_pot)
        self.assertEqual(y_new[7], self.y_pot)

        X_new, y_new = _update_X_y(self.X, self.y, self.y_pot, idx_update=0)

        np.testing.assert_array_equal(X_new, self.X)
        self.assertEqual(y_new[0], 5)

        X_new, y_new = _update_X_y(self.X, self.y, self.y, X_update=self.X)

        np.testing.assert_array_equal(X_new, np.append(self.X, self.X, axis=0))
        np.testing.assert_array_equal(y_new, np.append(self.y, self.y))

        X_new, y_new = _update_X_y(
            self.X, self.y, np.array([3, 4]), idx_update=np.array([0, 2])
        )

        np.testing.assert_array_equal(X_new, self.X)
        self.assertEqual(y_new[0], 3)
        self.assertEqual(y_new[2], 4)

        self.assertRaises(ValueError, _update_X_y, self.X, self.y, self.y_pot)

    def test_update_reg(self):
        self.assertRaises(
            (TypeError, ValueError),
            _update_reg,
            self.reg,
            self.X,
            self.y,
            self.y_pot,
            sample_weight=self.sample_weight,
            mapping=self.mapping,
        )
        self.reg.fit(self.X, self.y)
        reg_new = _update_reg(
            self.reg,
            self.X,
            self.y,
            self.y_pot,
            mapping=self.mapping,
            idx_update=1,
        )
        self.assertTrue(
            np.any(reg_new.predict(self.X) != self.reg.predict(self.X))
        )
        reg_new = _update_reg(
            self.reg,
            self.X,
            self.y,
            self.y_pot,
            mapping=self.mapping,
            idx_update=np.array([1]),
        )
        self.assertTrue(
            np.any(reg_new.predict(self.X) != self.reg.predict(self.X))
        )
        reg_new = _update_reg(
            self.reg,
            self.X,
            self.y,
            self.y_pot,
            mapping=None,
            X_update=np.array([8, 4]),
        )
        self.assertTrue(
            np.any(reg_new.predict(self.X) != self.reg.predict(self.X))
        )
        self.assertRaises(
            ValueError,
            _update_reg,
            self.reg,
            self.X,
            self.y,
            self.y_pot,
            sample_weight=np.arange(7) + 1,
            mapping=None,
            X_update=np.array([8, 4]),
        )

    def test_boostrap_aggregation(self):
        reg_s = _bootstrap_estimators(
            self.reg, self.X, self.y, bootstrap_size=5
        )
        self.assertEqual(len(reg_s), 5)

        reg_s = _bootstrap_estimators(
            self.reg,
            self.X,
            self.y,
            sample_weight=self.sample_weight,
            bootstrap_size=5,
        )
        self.assertEqual(len(reg_s), 5)

        self.assertRaises(
            ValueError,
            _bootstrap_estimators,
            self.reg,
            self.X,
            self.y,
            bootstrap_size=5,
            n_train=-1,
        )
        self.assertRaises(
            ValueError,
            _bootstrap_estimators,
            self.reg,
            self.X,
            self.y,
            bootstrap_size=5,
            n_train=1.9,
        )

    def test_cross_entropy(self):
        X_1 = np.arange(3 * 2).reshape(3, 2)
        y_1 = np.arange(3, dtype=float) + 2
        X_2 = np.arange(5 * 2).reshape(5, 2)
        y_2 = 2 * np.arange(5, dtype=float) - 5
        reg_1 = NICKernelRegressor().fit(X_1, y_1)
        reg_2 = NICKernelRegressor().fit(X_2, y_2)

        result = _cross_entropy(X_eval=X_1, true_reg=reg_1, other_reg=reg_2)
        self.assertEqual(y_1.shape, result.shape)

        for name, val in [
            ("X_eval", "illegal"),
            (
                "true_reg",
                SklearnRegressor(GaussianProcessRegressor()).fit(X_1, y_1),
            ),
            (
                "other_reg",
                SklearnRegressor(GaussianProcessRegressor()).fit(X_1, y_1),
            ),
            ("random_state", "illegal"),
            ("integration_dict", "illegal"),
        ]:
            cross_entropy_dict = dict(
                X_eval=X_1,
                true_reg=reg_1,
                other_reg=reg_2,
                random_state=0,
                integration_dict={},
            )
            cross_entropy_dict[name] = val
            self.assertRaises(
                (TypeError, ValueError), _cross_entropy, **cross_entropy_dict
            )
