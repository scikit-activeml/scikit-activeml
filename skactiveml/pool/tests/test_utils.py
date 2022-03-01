import unittest
from itertools import product

import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.naive_bayes import GaussianNB

from skactiveml.classifier import PWC, SklearnClassifier
from skactiveml.pool.utils import IndexClassifierWrapper
from skactiveml.utils import MISSING_LABEL, is_unlabeled


class TestIndexClassifierWrapper(unittest.TestCase):

    def setUp(self):
        self.X = np.linspace(0, 1, 4).reshape(-1, 1)
        self.y = np.array([0, 1, MISSING_LABEL, MISSING_LABEL])
        self.y2 = np.array([0, 1, 0, 1])
        self.y3 = np.array([0, MISSING_LABEL, MISSING_LABEL, MISSING_LABEL])
        self.clf = PWC(classes=np.unique(self.y))
        self.kwargs = dict(X=self.X, y=self.y, clf=self.clf)
        self.iclf = \
            lambda **kw: IndexClassifierWrapper(self.clf, self.X, self.y, **kw)

    def test_init_param_clf(self):
        self.assertTrue(hasattr(self.iclf(), 'clf'))
        self.assertRaises(TypeError, IndexClassifierWrapper,
                          clf='str', X=self.X, y=self.y)

    def test_init_param_X(self):
        self.assertTrue(hasattr(self.iclf(), 'X'))
        self.assertRaises((ValueError, TypeError), IndexClassifierWrapper,
                          clf=self.clf, X='str', y=self.y)

    def test_init_param_y(self):
        self.assertTrue(hasattr(self.iclf(), 'y'))
        self.assertRaises((ValueError, TypeError), IndexClassifierWrapper,
                          clf=self.clf, X=self.X, y='str')
        self.assertRaises(ValueError, IndexClassifierWrapper,
                          clf=self.clf, X=self.X, y=[0])

    def test_init_param_sample_weight(self):
        self.assertTrue(hasattr(self.iclf(), 'sample_weight'))
        self.assertEqual(self.iclf().sample_weight, None)
        self.assertRaises((ValueError, TypeError), IndexClassifierWrapper,
                          clf=self.clf, X=self.X, y=self.y, sample_weight='s')
        self.assertRaises((ValueError, TypeError), IndexClassifierWrapper,
                          clf=self.clf, X=self.X, y=self.y, sample_weight=[0])

    def test_init_param_fit_base_clf(self):
        self.assertRaises(TypeError, self.iclf, set_base_clf='string')
        self.assertRaises(NotFittedError, self.iclf, set_base_clf=True)

    def test_init_param_ignore_partial_fit(self):
        self.assertTrue(hasattr(self.iclf(), 'ignore_partial_fit'))
        self.assertEqual(self.iclf().ignore_partial_fit, True)

        self.assertRaises(TypeError, self.iclf, ignore_partial_fit='string')

    def test_init_param_use_speed_up(self):
        self.assertTrue(hasattr(self.iclf(), 'use_speed_up'))
        self.assertEqual(self.iclf().use_speed_up, False)

        self.assertRaises(TypeError, self.iclf, use_speed_up='string')

    def test_init_param_missing_label(self):
        self.assertTrue(hasattr(self.iclf(), 'missing_label'))
        self.assertTrue(is_unlabeled([self.iclf().missing_label],
                                     missing_label=MISSING_LABEL))

        self.assertRaises(TypeError, self.iclf, missing_label='string')

    def test_precompute_param_idx_fit(self):
        iclf = self.iclf()
        self.assertRaises((ValueError, TypeError), iclf.precompute, 'str', [0])
        self.assertRaises((ValueError, TypeError), iclf.precompute, [10], [0])

    def test_precompute_param_idx_pred(self):
        iclf = self.iclf()
        self.assertRaises((ValueError, TypeError), iclf.precompute, [0], 'str')
        self.assertRaises((ValueError, TypeError), iclf.precompute, [0], [10])

    def test_precompute_param_fit_params(self):
        iclf = self.iclf(use_speed_up=True)
        self.assertRaises((ValueError, TypeError), iclf.precompute,
                          [0], [0], fit_params=2)
        self.assertRaises((ValueError, TypeError), iclf.precompute,
                          [0], [0], fit_params='wrong_str')

    def test_precompute_param_pred_params(self):
        iclf = self.iclf(use_speed_up=True)
        self.assertRaises((ValueError, TypeError), iclf.precompute,
                          [0], [0], pred_params=2)
        self.assertRaises((ValueError, TypeError), iclf.precompute,
                          [0], [0], pred_params='wrong_str')

    def test_fit_param_idx(self):
        iclf = self.iclf()
        self.assertRaises((ValueError, TypeError), iclf.fit, 0)
        self.assertRaises((ValueError, TypeError), iclf.fit, 'wrong_str')
        self.assertRaises((ValueError, TypeError), iclf.fit, [10])

    def test_fit_param_y(self):
        iclf = self.iclf()
        self.assertRaises((ValueError, TypeError), iclf.fit,
                          [0], y='str')
        self.assertRaises((ValueError, TypeError), iclf.fit,
                          [0], y=[0, 0])

    def test_fit_param_sample_weight(self):
        iclf = self.iclf()
        self.assertRaises((ValueError, TypeError), iclf.fit,
                          [0], sample_weight='str')
        self.assertRaises((ValueError, TypeError), iclf.fit,
                          [0], sample_weight=[0, 0])

    def test_fit_param_set_base_clf(self):
        iclf = self.iclf()
        self.assertRaises(TypeError, iclf.fit, [0], set_base_clf='string')

    def test_partial_fit_param_idx(self):
        iclf = self.iclf().fit([0])
        self.assertRaises((ValueError, TypeError), iclf.partial_fit, 0)
        self.assertRaises((ValueError, TypeError), iclf.partial_fit, 'wrong_str')
        self.assertRaises((ValueError, TypeError), iclf.partial_fit, [10])

    def test_partial_fit_param_y(self):
        iclf = self.iclf().fit([0])
        self.assertRaises((TypeError), iclf.partial_fit,
                          [0], y='str')
        self.assertRaises((TypeError), iclf.partial_fit,
                          [0], y=[0, 0])

    def test_partial_fit_param_sample_weight(self):
        iclf = self.iclf().fit([0])
        self.assertRaises((TypeError), iclf.partial_fit,
                          [0], sample_weight='str')
        self.assertRaises((TypeError), iclf.partial_fit,
                          [0], sample_weight=[0, 0])

    def test_partial_fit_param_use_base_clf(self):
        iclf = self.iclf().fit([0])
        self.assertRaises(TypeError, iclf.partial_fit,
                          [0], use_base_clf='string')

    def test_partial_fit_param_set_base_clf(self):
        iclf = self.iclf().fit([0])
        self.assertRaises(TypeError, iclf.partial_fit,
                          [0], set_base_clf='string')

    def test_fit(self):
        base_clfs = [lambda : PWC(classes=[0,1]),
                     lambda : SklearnClassifier(GaussianNB(), classes=[0,1])]
        speed_ups = [True, False]
        sample_weights = [None, np.linspace(.2, 1, 4)]
        preds = ['predict', 'predict_proba', 'predict_freq']

        params = \
            list(product(base_clfs[:1], speed_ups, sample_weights, preds)) + \
            list(product(base_clfs[1:], speed_ups, sample_weights, preds[:2]))

        for BaseClf, speed_up, sample_weight, pred in params:
            with self.subTest(msg="Test fit via init", BaseClf=BaseClf,
                              speed_up=speed_up, sample_weight=sample_weight,
                              pred=pred):
                for i in range(1, 4):
                    clf = BaseClf()
                    sw_ = None if sample_weight is None else sample_weight[:i]
                    clf.fit(self.X[:i], self.y2[:i],
                            sample_weight=sw_)

                    iclf = IndexClassifierWrapper(
                        BaseClf(), self.X, self.y2,
                        sample_weight=sample_weight, use_speed_up=speed_up
                    )

                    if speed_up:
                        iclf.precompute(np.arange(i), np.arange(4))

                    iclf.fit(np.arange(i))
                    np.testing.assert_array_equal(
                        getattr(iclf, pred)(np.arange(4)),
                        getattr(clf, pred)(self.X)
                    )

            with self.subTest(msg="Test direct fit", BaseClf=BaseClf,
                              speed_up=speed_up, sample_weight=sample_weight,
                              pred=pred):
                for i in range(1, 4):
                    clf = BaseClf()
                    sw_ = None if sample_weight is None else sample_weight[:i]
                    clf.fit(self.X[:i], self.y2[:i],
                            sample_weight=sw_)

                    iclf = IndexClassifierWrapper(
                        BaseClf(), self.X, np.full(4, np.nan),
                        use_speed_up=speed_up
                    )

                    if speed_up:
                        iclf.precompute(np.arange(i), np.arange(4))

                    iclf.fit(np.arange(i), y=self.y2[:i],
                             sample_weight=sw_)
                    np.testing.assert_array_equal(
                        getattr(iclf, pred)(np.arange(4)),
                        getattr(clf, pred)(self.X)
                    )

    def test_partial_fit(self):
        base_clfs = [lambda : PWC(classes=[0,1]),
                     lambda : SklearnClassifier(GaussianNB(), classes=[0,1])]
        speed_ups = [True, False]
        sample_weights = [None, np.linspace(.2, 1, 4)]
        preds = ['predict', 'predict_proba']

        params = list(product(base_clfs[:1], speed_ups, sample_weights, preds))

        for BaseClf, speed_up, sample_weight, pred in params:

            with self.subTest(msg="PWC use base data", BaseClf=str(BaseClf()),
                              speed_up=speed_up, sample_weight=sample_weight,
                              pred=pred):
                iclf = IndexClassifierWrapper(
                    BaseClf(), self.X, self.y2,
                    sample_weight=sample_weight, use_speed_up=speed_up
                )
                if speed_up:
                    iclf.precompute(np.arange(4), np.arange(4))
                init_idx = [0]
                iclf.fit(init_idx)
                all_idx = list(init_idx)
                for add_idx in [[1], [2, 3]]:
                    all_idx += add_idx
                    clf = BaseClf()
                    sw_ = None if sample_weight is None \
                        else sample_weight[all_idx]
                    clf.fit(self.X[all_idx],
                            self.y2[all_idx],
                            sample_weight=sw_)


                    iclf.partial_fit(add_idx)
                    np.testing.assert_array_equal(
                        getattr(iclf, pred)(np.arange(4)),
                        getattr(clf, pred)(self.X)
                    )

        for BaseClf, speed_up, sample_weight, pred in params:

            with self.subTest(msg="PWC use fit data", BaseClf=str(BaseClf()),
                              speed_up=speed_up, sample_weight=sample_weight,
                              pred=pred):
                iclf = IndexClassifierWrapper(
                    BaseClf(), self.X, self.y3,
                    sample_weight=sample_weight, use_speed_up=speed_up
                )
                if speed_up:
                    iclf.precompute(np.arange(4), np.arange(4))
                init_idx = [0]
                iclf.fit(init_idx)
                all_idx = list(init_idx)
                for add_idx in [[1], [2, 3]]:
                    all_idx += add_idx
                    clf = BaseClf()
                    sw_ = None if sample_weight is None \
                        else sample_weight[all_idx]
                    clf.fit(self.X[all_idx],
                            self.y2[all_idx],
                            sample_weight=sw_)

                    sw_add_ = None if sample_weight is None \
                        else sample_weight[add_idx]
                    iclf.partial_fit(add_idx, y=self.y2[add_idx],
                                     sample_weight=sw_add_)
                    np.testing.assert_array_equal(
                        getattr(iclf, pred)(np.arange(4)),
                        getattr(clf, pred)(self.X)
                    )

        for BaseClf, speed_up, sample_weight, pred in params:

            with self.subTest(msg="PWC use fit data with base clf",
                              BaseClf=str(BaseClf()),
                              speed_up=speed_up, sample_weight=sample_weight,
                              pred=pred):
                iclf = IndexClassifierWrapper(
                    BaseClf(), self.X, self.y3,
                    sample_weight=sample_weight, use_speed_up=speed_up
                )
                if speed_up:
                    iclf.precompute(np.arange(4), np.arange(4))
                init_idx = [0]
                iclf.fit(init_idx, set_base_clf=True)
                all_idx = list(init_idx)
                for add_idx in [[1], [2, 3]]:
                    all_idx += add_idx
                    clf = BaseClf()
                    sw_ = None if sample_weight is None \
                        else sample_weight[all_idx]
                    clf.fit(self.X[all_idx],
                            self.y2[all_idx],
                            sample_weight=sw_)

                    sw_add_ = None if sample_weight is None \
                        else sample_weight[add_idx]
                    iclf.partial_fit(add_idx, y=self.y2[add_idx],
                                     sample_weight=sw_add_, use_base_clf=True,
                                     set_base_clf=True)
                    np.testing.assert_array_equal(
                        getattr(iclf, pred)(np.arange(4)),
                        getattr(clf, pred)(self.X)
                    )

        params = \
            list(product(base_clfs[1:], speed_ups, sample_weights, preds[:2]))

        for BaseClf, speed_up, sample_weight, pred in params:

            with self.subTest(msg="NB use fit data", BaseClf=str(BaseClf()),
                              speed_up=speed_up, sample_weight=sample_weight,
                              pred=pred):
                iclf = IndexClassifierWrapper(
                    BaseClf(), self.X, self.y,
                    sample_weight=sample_weight, use_speed_up=speed_up,
                    ignore_partial_fit=False
                )
                clf = BaseClf()
                if speed_up:
                    iclf.precompute(np.arange(4), np.arange(4))
                init_idx = [0, 1]
                iclf.fit(init_idx)
                sw_ = None if sample_weight is None else sample_weight[init_idx]
                clf.fit(self.X[init_idx], self.y2[init_idx], sample_weight=sw_)

                for add_idx in [[2], [3]]:

                    sw_add_ = None if sample_weight is None \
                        else sample_weight[add_idx]
                    iclf.partial_fit(add_idx, y=self.y2[add_idx],
                                     sample_weight=sw_add_)
                    clf.partial_fit(self.X[add_idx], y=self.y2[add_idx],
                                    sample_weight=sw_add_)
                    np.testing.assert_array_equal(
                        getattr(iclf, pred)(np.arange(4)),
                        getattr(clf, pred)(self.X)
                    )

















