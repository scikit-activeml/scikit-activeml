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
            def fit(self, X, y, sample_weight=None):
                self.classes_ = np.unique(y[labeled_indices(y)])
                self._le = ExtLabelEncoder(classes=self.classes_,
                                           missing_label=MISSING_LABEL).fit(y)
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
        for sw in [self.X_cand, np.empty((len(self.X) - 1))]:
            with self.subTest(msg=f'sample_weight={sw}'):
                self.assertRaises(IndexError, qs.query, **self.kwargs,
                                  sample_weight=sw)

        for sw in ['string']:
            with self.subTest(msg=f'sample_weight={sw}'):
                self.assertRaises(TypeError, qs.query, **self.kwargs,
                                  sample_weight=sw)

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

    def test_query(self):
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
        for swc in ['string']:
            with self.subTest(msg=f'sample_weight_candidates={swc}'):
                self.assertRaises(ValueError, qs.query, X=self.X, y=self.y,
                                  clf=self.clf, candidates=self.X_cand,
                                  sample_weight=np.ones(len(self.X)),
                                  sample_weight_candidates=swc)

        for swc in [np.empty((len(self.X_cand) - 1))]:
            with self.subTest(msg=f'sample_weight_candidates={swc}'):
                self.assertRaises(IndexError, qs.query, X=self.X, y=self.y,
                                  clf=self.clf, candidates=self.X_cand,
                                  sample_weight=np.ones(len(self.X)),
                                  sample_weight_candidates=swc)

        self.assertRaises(
            ValueError, qs.query, X=self.X, y=self.y,
            clf=self.clf, candidates=self.X_cand,
            sample_weight_candidates=np.ones(len(self.candidates))
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
        candidates = [0, 1]
        X_cand = np.array([[8, 1], [9, 1]])
        X = np.array([[8, 1], [9, 1], [1, 2], [5, 8], [8, 4]])
        y = np.array([MISSING_LABEL, MISSING_LABEL, 0, 1, MISSING_LABEL])
        cost_matrix = 1 - np.eye(2)

        X = [[1], [2], [3]]
        y = [0, 1, MISSING_LABEL]
        clf = PWC(classes=[0, 1])

        params_list = [
            ['log_loss', np.full(
                shape=(1, len(candidates)),
                fill_value=-0.5 * np.log(0.5) * len(classes) * len(X))
             ], ['misclassification_loss', np.full(
                shape=(1, len(candidates)),
                fill_value=0.5)
                 ]]

        for method, expected_utils in params_list:
            with self.subTest(msg=method):
                qs = MonteCarloEER(method=method, cost_matrix=cost_matrix)
                idx, utils = qs.query(X, y, self.DummyClf(), fit_clf=True,
                                      ignore_partial_fit=True,
                                      candidates=candidates,
                                      return_utilities=True)

                np.testing.assert_allclose(expected_utils, utils[:, 0:2])


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
        candidates = [0, 1]
        X_cand = np.array([[8, 1], [9, 1]])
        X = np.array([[8, 1], [9, 1], [1, 2], [5, 8], [8, 4]])
        y = np.array([MISSING_LABEL, MISSING_LABEL, 0, 1, MISSING_LABEL])
        cost_matrix = 1 - np.eye(2)
        clf_partial = SklearnClassifier(
            GaussianNB(), classes=classes
        ).fit(X, y)
        clf = PWC(classes=[0, 1])

        params_list = [
            ['kapoor', True, True, True, True, [[0, 0]]],
            ['kapoor', True, True, True, False,
             np.full(shape=(1, len(candidates)), fill_value=-0.5*len(X))],
            ['Margeniantu', False, True, False, False,
             np.full(shape=(1, len(candidates)), fill_value=
             0.25 * (len(classes) - 1) * len(classes) * len(candidates))],
            ['Joshi', True, False, True, True,
             np.full(shape=(1, len(candidates)), fill_value=0)],
            ['Joshi', True, False, True, False,
             np.full(shape=(1, len(candidates)), fill_value=
             -0.25 * np.sum(cost_matrix))]]

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
                qs.query(candidates=candidates, clf=clf, X=X,
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
                qs.query(candidates=candidates, clf=clf_partial, X=X, y=y)

                idxs, utils = qs.query(candidates=candidates,
                                       clf=self.DummyClf(),
                                       X=X, y=y,
                                       return_utilities=True)
                np.testing.assert_array_equal(expected_utils, utils[:, 0:2])

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


class TestUpdateIndexClassifier(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[1, 2], [5, 8], [8, 4], [5, 4]])
        self.candidates = np.array([[8, 1], [9, 1], [5, 1]])
        self.y = np.array([0, 1, 2, 1])
        self.sample_weight = np.ones(len(self.X))
        self.classes = [0, 1, 2]
        self.clf = PWC(classes=self.classes)
        self.clf_partial = SklearnClassifier(
            GaussianNB(), classes=self.classes
        ).fit([[5, 6], [1, -1], [9, 9]], [0, 1, 2])

    def test_fit(self):
        fit_idx = [1, 3]
        sample_weight = np.random.random_sample(len(self.X))
        for clf in [self.clf, self.clf_partial]:
            clf = clone(clf).fit(
                self.X[fit_idx], self.y[fit_idx], sample_weight[fit_idx]
            )
            probas_clf = clf.predict_proba(self.X)
            UIclf = IndexClassifierWrapper(clf, self.X, self.y, sample_weight)
            UIclf.fit(fit_idx)
            probas_UIclf = UIclf.predict_proba(range(len(self.X)))
            np.testing.assert_allclose(probas_clf, probas_UIclf)

    def test_fit_as_cand(self):
        fit_idx = [1, 3]
        sample_weight = np.random.random_sample(len(self.X))
        clf = deepcopy(self.clf_partial).partial_fit(
            self.X[fit_idx], self.y[fit_idx], sample_weight[fit_idx]
        )
        UIclf = IndexClassifierWrapper(clf, self.X, self.y, sample_weight)
        UIclf.fit(fit_idx)
        probas_clf = clf.predict_proba(self.X)
        probas_UIclf = UIclf.predict_proba(range(len(self.X)))
        np.testing.assert_allclose(probas_clf, probas_UIclf)
