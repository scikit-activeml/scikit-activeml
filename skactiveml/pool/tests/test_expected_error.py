import unittest
from copy import deepcopy

import numpy as np
from sklearn import clone
from sklearn.gaussian_process import GaussianProcessClassifier, \
    GaussianProcessRegressor
from sklearn.naive_bayes import GaussianNB

from skactiveml.base import SkactivemlClassifier
from skactiveml.classifier import PWC, SklearnClassifier
from skactiveml.pool import MonteCarloEER, ValueOfInformationEER
from skactiveml.pool._expected_error import IndexClassifierWrapper
from skactiveml.utils import MISSING_LABEL, labeled_indices, \
    ExtLabelEncoder


class TemplateTestEER:
    def setUp(self):
        self.Strategy = self.get_query_strategy()
        self.X = np.array([[1, 2], [5, 8], [8, 4], [5, 4]])
        self.X_cand = np.array([[8, 4], [5, 4], [1, 2]])
        self.candidates = np.array([2, 1, 0], dtype=int)
        self.X_eval = np.array([[2, 1], [3, 7]])
        self.y = [0, 1, 2, 1]
        self.classes = [0, 1, 2]
        self.cost_matrix = np.eye(3)
        self.clf = PWC(classes=self.classes)
        self.clf_partial = SklearnClassifier(
            GaussianNB(), classes=self.classes
        ).fit(self.X, self.y)
        self.kwargs = dict(X=self.X, y=self.y, candidates=self.candidates,
                           clf=self.clf)

        class DummyClf(SkactivemlClassifier):
            def __init__(self, classes=None):
                super().__init__(classes=classes)

            def fit(self, X, y, sample_weight=None):
                X, y, sample_weight = self._validate_data(
                    X=X, y=y, sample_weight=sample_weight)
                return self

            def predict_proba(self, X):
                return np.full(shape=(len(X), len(self.classes_)),
                               fill_value=0.5)

        self.DummyClf = DummyClf

    def test_init_param_cost_matrix(self):
        for cost_matrix in [np.ones((2, 3)), 'string', np.ones((2, 2))]:
            qs = self.Strategy(cost_matrix=cost_matrix)
            self.assertRaises(ValueError, qs.query, **self.kwargs)
        self.assertTrue(hasattr(qs, 'cost_matrix'))

    def test_query_param_clf(self):
        qs = self.Strategy()
        for clf in ['string', GaussianProcessClassifier()]:
            self.assertRaises(
                TypeError, qs.query, candidates=self.candidates, X=self.X,
                y=self.y,
                clf=clf
            )

        for clf in [PWC(missing_label=-1)]:
            with self.subTest(msg=f'clf={clf}', clf=clf):
                self.assertRaises(
                    ValueError, qs.query, candidates=self.candidates, X=self.X,
                    y=self.y,
                    clf=clf
                )

    def test_query_param_sample_weight(self):
        qs = self.Strategy()
        for sw in [self.X_cand, np.empty((len(self.X) - 1)), 'string']:
            with self.subTest(msg=f'sample_weight={sw}'):
                self.assertRaises((IndexError, TypeError, ValueError),
                                  qs.query, **self.kwargs, sample_weight=sw)

    def test_query_param_fit_clf(self):
        qs = self.Strategy()
        for fc in ['string', self.candidates, None]:
            self.assertRaises(TypeError, qs.query, **self.kwargs, fit_clf=fc,
                              msg=f'fit_clf={fc}')

    def test_query_param_ignore_partial_fit(self):
        qs = self.Strategy()
        self.assertRaises(
            TypeError, qs.query, candidates=self.candidates, X=self.X,
            y=self.y,
            clf=self.clf, ignore_partial_fit=None
        )

    def test__concatenate_samples(self):
        X = np.array([[1], [2], [3], [4]])
        y = np.array([0, 1, MISSING_LABEL, MISSING_LABEL])
        ld_idx = np.array([0, 1])
        uld_idx = np.array([2, 3])
        sample_weight = np.array([0.1, 0.2, 0.3, 0.4])
        sample_weight_cand = np.array([0.1, 0.2])
        X_eval = np.array([[5], [6], [7], [8]])
        sample_weight_eval = np.array([0.5, 0.6, 0.7, 0.8])

        qs = self.get_query_strategy()()
        qs.missing_label_ = MISSING_LABEL

        cand = None
        X_full, y_full, w_full, w_eval, idx_train, idx_cand, idx_eval = \
            qs._concatenate_samples(X, y, sample_weight,
                             cand, sample_weight_cand,
                             X_eval, sample_weight_eval)

        np.testing.assert_equal(len(X_full), len(y_full))
        np.testing.assert_equal(len(X_full), len(w_full))
        np.testing.assert_equal(len(X_full), len(w_eval))

        np.testing.assert_array_equal(X, X_full[idx_train])
        np.testing.assert_array_equal(y, y_full[idx_train])
        np.testing.assert_array_equal(sample_weight, w_full[idx_train])

        np.testing.assert_array_equal(X_eval, X_full[idx_eval])
        np.testing.assert_array_equal(sample_weight_eval, w_full[idx_eval])
        np.testing.assert_array_equal(sample_weight_eval, w_eval[idx_eval])

        np.testing.assert_array_equal(X[uld_idx], X_full[idx_cand])
        np.testing.assert_array_equal(y[uld_idx], y_full[idx_cand])
        np.testing.assert_array_equal(sample_weight[uld_idx], w_full[idx_cand])

        cand = np.array([2])
        X_full, y_full, w_full, w_eval, idx_train, idx_cand, idx_eval = \
            qs._concatenate_samples(X, y, None,
                             cand, None,
                             None, None)

        np.testing.assert_array_equal(X, X_full[idx_train])
        np.testing.assert_array_equal(y, y_full[idx_train])


        cand = np.array([2])
        X_full, y_full, w_full, w_eval, idx_train, idx_cand, idx_eval = \
            qs._concatenate_samples(
                X, y, sample_weight, cand, sample_weight_cand,
                X_eval, sample_weight_eval)

        np.testing.assert_equal(len(X_full), len(y_full))
        np.testing.assert_equal(len(X_full), len(w_full))
        np.testing.assert_equal(len(X_full), len(w_eval))

        np.testing.assert_array_equal(X, X_full[idx_train])
        np.testing.assert_array_equal(y, y_full[idx_train])
        np.testing.assert_array_equal(sample_weight, w_full[idx_train])

        np.testing.assert_array_equal(X_eval, X_full[idx_eval])
        np.testing.assert_array_equal(sample_weight_eval, w_full[idx_eval])
        np.testing.assert_array_equal(sample_weight_eval, w_eval[idx_eval])

        np.testing.assert_array_equal(X[cand], X_full[idx_cand])
        np.testing.assert_array_equal(y[cand], y_full[idx_cand])
        np.testing.assert_array_equal(sample_weight[cand], w_full[idx_cand])

        cand = np.array([[0], [9]])
        X_full, y_full, w_full, w_eval, idx_train, idx_cand, idx_eval = \
            qs._concatenate_samples(
                X, y, sample_weight, cand, sample_weight_cand,
                X_eval, sample_weight_eval)

        np.testing.assert_equal(len(X_full), len(y_full))
        np.testing.assert_equal(len(X_full), len(w_full))
        np.testing.assert_equal(len(X_full), len(w_eval))

        np.testing.assert_array_equal(X, X_full[idx_train])
        np.testing.assert_array_equal(y, y_full[idx_train])
        np.testing.assert_array_equal(sample_weight, w_full[idx_train])

        np.testing.assert_array_equal(X_eval, X_full[idx_eval])
        np.testing.assert_array_equal(sample_weight_eval, w_full[idx_eval])
        np.testing.assert_array_equal(sample_weight_eval, w_eval[idx_eval])

        np.testing.assert_array_equal(cand, X_full[idx_cand])
        np.testing.assert_array_equal(np.full((len(cand)), np.nan), y_full[idx_cand])
        np.testing.assert_array_equal(sample_weight_cand, w_full[idx_cand])

    def test_query(self):
        """
        qs = self.Strategy()
        # return_utilities
        L = list(qs.query(**self.kwargs, return_utilities=True))
        self.assertTrue(len(L) == 2)
        L = list(qs.query(**self.kwargs, return_utilities=False))
        self.assertTrue(len(L) == 1)

        # batch_size
        bs = 3
        best_idx = qs.query(**self.kwargs, batch_size=bs)
        self.assertEqual(bs, len(best_idx))
        """
        pass


class TestMonteCarloEER(TemplateTestEER, unittest.TestCase):
    def get_query_strategy(self):
        return MonteCarloEER

    def test_init_param_method(self):
        qs = self.Strategy(method='String')
        self.assertRaises(ValueError, qs.query, **self.kwargs)
        qs = self.Strategy(method=1)
        self.assertRaises(TypeError, qs.query, **self.kwargs)
        self.assertTrue(hasattr(qs, 'method'))

    def test_query_param_sample_weight_candidates(self):
        qs = self.Strategy()
        for swc in ['string', np.empty((len(self.X_cand) - 1))]:
            with self.subTest(msg=f'sample_weight_candidates={swc}'):
                self.assertRaises((ValueError, TypeError),
                                  qs.query, X=self.X, y=self.y,
                                  clf=self.clf, candidates=self.X_cand,
                                  sample_weight=np.ones(len(self.X)),
                                  sample_weight_candidates=swc)

        # TODO: sample weights should be of size X_cand
        #  (why is error not raised?):
        self.assertRaises(
            ValueError, qs.query, X=self.X, y=self.y,
            clf=self.clf, candidates=self.X_cand,
            sample_weight_candidates=np.ones(len(self.X_cand))
        )

    def test_query_param_sample_weight_eval(self):
        qs = self.Strategy()
        for swe in ['string', self.candidates, None,
                    np.empty((len(self.X_eval) - 1))]:
            with self.assertRaises(ValueError,
                                   msg=f'sample_weight_eval={swe}'):
                qs.query(
                    **self.kwargs,
                    X_eval=self.X_eval,
                    sample_weight=np.ones(len(self.X)),
                    sample_weight_candidates=np.ones(len(self.candidates)),
                    sample_weight_eval=swe,
                )


    def test_query_param_X_eval(self):
        qs = self.Strategy()
        for X_eval in ['str', [], np.ones(5)]:
            with self.subTest(msg=f'X_eval={X_eval}', X_eval=X_eval):
                self.assertRaises(
                    ValueError, qs.query, **self.kwargs, X_eval=X_eval,
                )

    def test_query(self):
        super().test_query()
        classes = [0, 1]
        X = [[0], [0], [0], [0]]
        y = [0, MISSING_LABEL, MISSING_LABEL, 1]
        cand = [1, 2]
        cost_matrix = 1 - np.eye(2)

        clf = self.DummyClf(classes=classes)
        #clf = PWC(classes=classes)

        params_list = [
            [
                'log_loss',
                np.full(shape=(1, len(cand)),
                        fill_value=0.5 * np.log(0.5) * len(classes) * len(X))
            ],
            [
                'misclassification_loss',
                np.full(shape=(1, len(cand)), fill_value=-0.5 * len(X))
            ]
        ]

        for method, expected_utils in params_list:
            with self.subTest(msg=method):
                qs = MonteCarloEER(method=method, cost_matrix=cost_matrix)
                qs.query(X, y=np.full(shape=len(X), fill_value=np.nan),
                         clf=clf, fit_clf=True,
                         ignore_partial_fit=True,
                         candidates=cand,
                         return_utilities=True)
                idx, utils = qs.query(X, y, clf, fit_clf=True,
                                      ignore_partial_fit=True,
                                      candidates=cand,
                                      return_utilities=True)

                np.testing.assert_allclose(expected_utils, utils[:, cand])

        # TODO: PWC Test


class TestValueOfInformationEER(TemplateTestEER, unittest.TestCase):
    def get_query_strategy(self):
        return ValueOfInformationEER

    def test_init_param_consider_unlabeled(self):
        qs = ValueOfInformationEER()
        self.assertRaises(
            TypeError, qs.query, clf=self.clf,
            X=self.X, y=self.y, consider_unlabeled='string'
        )

    def test_init_param_consider_labeled(self):
        qs = ValueOfInformationEER()
        self.assertRaises(
            TypeError, qs.query, clf=self.clf,
            X=self.X, y=self.y, consider_labeled='string'
        )

    def test_init_param_candidate_to_labeled(self):
        qs = ValueOfInformationEER()
        self.assertRaises(
            TypeError, qs.query, clf=self.clf,
            X=self.X, y=self.y, candidate_to_labeled='string'
        )

    def test_init_param_subtract_current(self):
        qs = ValueOfInformationEER()
        self.assertRaises(
            TypeError, qs.query, clf=self.clf,
            X=self.X, y=self.y, subtract_current='string'
        )

    def test_query(self):
        super().test_query()
        classes = [0, 1]
        X = [[0], [0], [0], [0]]
        y = [0, MISSING_LABEL, MISSING_LABEL, 1]
        cand = [1, 2]
        cost_matrix = 1 - np.eye(2)
        # clf_partial = SklearnClassifier(
        #     GaussianNB(), classes=classes
        # ).fit(X, y)


        params_list = [
            [
                'kapoor-sub', True, True, True, True,
                [[0, 0]]
            ],
            [
                'kapoor', True, True, True, False,
                np.full(shape=(1, len(cand)), fill_value=-0.5*len(X))
            ],
            [
                'Margeniantu', False, True, False, False,
                np.full(shape=(1, len(cand)), fill_value=
#                        0.25 * (len(classes) - 1) * len(classes) * len(cand))
                        -0.25 * (len(classes) - 1) * len(classes) * len(X))
            ],
            [
                'Joshi-sub', True, False, True, True,
                np.full(shape=(1, len(cand)), fill_value=0)
            ],
            [
                'Joshi', True, False, True, False,
                np.full(shape=(1, len(cand)),
                        fill_value=-0.25 * np.sum(cost_matrix))
            ]
        ]

        # Test with zero labels.
        for msg, consider_unlabeled, consider_labeled, \
            candidate_to_labeled, substract_current, \
            expected_utils in params_list:
            with self.subTest(msg=msg + ": Scenario"):
                qs = ValueOfInformationEER(
                    consider_unlabeled=consider_unlabeled,
                    consider_labeled=consider_labeled,
                    candidate_to_labeled=candidate_to_labeled,
                    subtract_current=substract_current,
                    cost_matrix=cost_matrix
                )
                clf = PWC(classes=classes)
                qs.query(candidates=cand, clf=clf, X=X,
                         y=np.full(shape=len(X), fill_value=np.nan))

        # Test Scenario.
        for msg, consider_unlabeled, consider_labeled, \
            candidate_to_labeled, substract_current, \
            expected_utils in params_list:
            with self.subTest(msg=msg + ": Scenario"):
                qs = ValueOfInformationEER(
                    consider_unlabeled=consider_unlabeled,
                    consider_labeled=consider_labeled,
                    candidate_to_labeled=candidate_to_labeled,
                    subtract_current=substract_current,
                    cost_matrix=cost_matrix
                )
                clf = self.DummyClf()
                qs.query(candidates=cand, clf=clf, X=X, y=y)

                idxs, utils = qs.query(candidates=cand,
                                       clf=self.DummyClf(),
                                       X=X, y=y,
                                       return_utilities=True)
                np.testing.assert_array_equal(expected_utils, utils[:, cand])

        # Test Kapoor
        # qs = ValueOfInformationEER(consider_unlabeled=True,
        #                           consider_labeled=True,
        #                           candidate_to_labeled=True,
        #                           cost_matrix=cost_matrix)
        # qs.query(candidates=candidates, clf=clf_partial, X=X, y=y)

        # idxs, utils = qs.query(candidates=candidates, clf=self.DummyClf(), X=X, y=y,
        #                       return_utilities=True)
        # np.testing.assert_array_equal([[0, 0]], utils[:, 0:2])

        # labeling_cost
        # class DummyClf(SkactivemlClassifier):
        #    def fit(self, X, y, sample_weight=None):
        #        self.classes_ = np.unique(y[labeled_indices(y)])
        #        return self
        #
        #    def predict_proba(self, X):
        #        return np.full(shape=(len(X), len(self.classes_)),
        #                       fill_value=0.5)

        # labeling_cost = 2.345
        # qs = ValueOfInformationEER(cost_matrix=cost_matrix,
        #                           labeling_cost=labeling_cost)
        # idxs, utils = qs.query(candidates=X_cand, clf=DummyClf(), X=X, y=y,
        #                       return_utilities=True)
        # np.testing.assert_array_equal(utils[0], [-labeling_cost, -labeling_cost])
        #
        # labeling_cost = np.array([2.346, 6.234])
        # qs = ValueOfInformationEER(cost_matrix=cost_matrix,
        #                           labeling_cost=labeling_cost)
        # idxs, utils = qs.query(candidates=X_cand, clf=DummyClf(), X=X, y=y,
        #                       return_utilities=True)
        # np.testing.assert_array_equal(utils[0], -labeling_cost)
        #
        # labeling_cost = np.array([[2.346, 6.234]])
        # expected = [-labeling_cost.mean(), -labeling_cost.mean()]
        # qs = ValueOfInformationEER(cost_matrix=cost_matrix,
        #                           labeling_cost=labeling_cost)
        # idxs, utils = qs.query(candidates=X_cand, clf=DummyClf(), X=X, y=y,
        #                       return_utilities=True)
        # np.testing.assert_array_equal(utils[0], expected)
        #
        # labeling_cost = np.array([[2.346, 6.234],
        #                          [3.876, 3.568]])
        # expected = -labeling_cost.mean(axis=1)
        # qs = ValueOfInformationEER(cost_matrix=cost_matrix,
        #                           labeling_cost=labeling_cost)
        # idxs, utils = qs.query(candidates=X_cand, clf=DummyClf(), X=X, y=y,
        #                       return_utilities=True)
        # np.testing.assert_array_equal(utils[0], expected)