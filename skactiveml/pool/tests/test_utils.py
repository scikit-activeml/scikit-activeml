import unittest
from itertools import product

import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.metrics import pairwise_kernels
from sklearn.naive_bayes import GaussianNB

from skactiveml.classifier import ParzenWindowClassifier, SklearnClassifier
from skactiveml.pool.utils import IndexClassifierWrapper
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
        self.y2 = np.array([0, 1, 0, 1])
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
                    np.testing.assert_array_equal(
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
                    np.testing.assert_array_equal(
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
                    np.testing.assert_array_equal(
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
                    np.testing.assert_array_equal(
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
                    np.testing.assert_array_equal(
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
                    np.testing.assert_array_equal(
                        getattr(iclf, pred)(np.arange(4)),
                        getattr(clf, pred)(self.X),
                    )
