import unittest
from copy import deepcopy

import numpy as np
from sklearn import clone
from sklearn.gaussian_process import GaussianProcessClassifier, \
    GaussianProcessRegressor
from sklearn.naive_bayes import GaussianNB

from skactiveml.classifier import PWC, SklearnClassifier
from skactiveml.pool import MonteCarloEER, ValueOfInformationEER
from skactiveml.pool._expected_error import UpdateIndexClassifier
from skactiveml.utils import MISSING_LABEL, call_func


class TemplateTestEER:
    def setUp(self):
        self.Strategy = self.get_query_strategy()
        self.X = np.array([[1, 2], [5, 8], [8, 4], [5, 4]])
        self.candidates = np.array([[8, 4], [5, 4]])
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

    def test_init_param_cost_matrix(self):
        for cost_matrix in [np.ones((2, 3)), 'string', np.ones((2, 2))]:
            qs = self.Strategy(cost_matrix=cost_matrix)
            self.assertRaises(ValueError, qs.query, **self.kwargs)
        self.assertTrue(hasattr(qs, 'cost_matrix'))

    def test_query_param_clf(self):
        qs = self.Strategy()
        for clf in ['string', GaussianProcessClassifier()]:
            self.assertRaises(
                TypeError, qs.query, candidates=self.candidates, X=self.X, y=self.y,
                clf=clf
            )

        for clf in [PWC(missing_label=-1)]:
            with self.subTest(msg=f'clf={clf}', clf=clf):
                self.assertRaises(
                    ValueError, qs.query, candidates=self.candidates, X=self.X, y=self.y,
                    clf=clf
                )

    def test_query_param_sample_weight(self):
        qs = self.Strategy()
        for sw in ['string', self.candidates, np.empty((len(self.X) - 1))]:
            self.assertRaises(ValueError, qs.query, **self.kwargs,
                              sample_weight=sw)

    def test_query_param_fit_clf(self):
        qs = self.Strategy()
        for fc in ['string', self.candidates, None]:
            self.assertRaises(TypeError, qs.query, **self.kwargs, fit_clf=fc,
                            msg=f'fit_clf={fc}')

    def test_query_param_ignore_partial_fit(self):
        qs = self.Strategy()
        self.assertRaises(
            TypeError, qs.query, candidates=self.candidates, X=self.X, y=self.y,
            clf=self.clf, ignore_partial_fit=None
        )

    def test_query_param_X_eval(self):
        qs = self.Strategy()
        for X_eval in ['str', [], np.ones(5)]:
            with self.subTest(msg=f'X_eval={X_eval}', X_eval=X_eval):
                self.assertRaises(
                    ValueError, qs.query, **self.kwargs, X_eval=X_eval,
                )

    def test_query_param_sample_weight_eval(self):
        qs = self.Strategy()
        for swe in ['string', self.candidates, None,
                    np.empty((len(self.candidates) - 1))]:
            with self.assertRaises(ValueError, msg=f'sample_weight_eval={swe}'):
                call_func(
                    qs.query,
                    **self.kwargs,
                    X_eval=self.X_eval,
                    sample_weight=np.ones(len(self.X)),
                    sample_weight_candidates=np.ones(len(self.candidates)),
                    sample_weight_eval=swe,
                )

        with self.assertRaises(ValueError, msg=f'sample_weight_eval={swe}'):
            call_func(
                qs.query,
                **self.kwargs,
                X_eval=self.X_eval,
                sample_weight_eval=np.ones(len(self.X_eval))
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
    # TODO
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
        for swc in ['string', self.candidates]:
            self.assertRaises(ValueError, qs.query, **self.kwargs,
                              sample_weight=np.ones(len(self.X)),
                              sample_weight_candidates=swc)

        for swc in [np.empty((len(self.candidates) - 1))]:
            self.assertRaises(IndexError, qs.query, **self.kwargs,
                              sample_weight=np.ones(len(self.X)),
                              sample_weight_candidates=swc)

        self.assertRaises(
            ValueError, qs.query, **self.kwargs,
            sample_weight_candidates=np.ones(len(self.candidates))
        )

    def test_query(self):
        super().test_query()


class TestValueOfInformationEER(TemplateTestEER, unittest.TestCase):
    # TODO
    def get_query_strategy(self):
        return ValueOfInformationEER

    def test_query(self):
        super().test_query()


class TestUpdateIndexClassifier(unittest.TestCase):
    # TODO
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
            UIclf = UpdateIndexClassifier(clf, self.X, self.y, sample_weight)
            UIclf.fit(fit_idx)
            probas_UIclf = UIclf.predict_proba(range(len(self.X)))
            np.testing.assert_allclose(probas_clf, probas_UIclf)

    def test_fit_as_cand(self):
        fit_idx = [1, 3]
        sample_weight = np.random.random_sample(len(self.X))
        clf = deepcopy(self.clf_partial).partial_fit(
            self.X[fit_idx], self.y[fit_idx], sample_weight[fit_idx]
        )
        UIclf = UpdateIndexClassifier(clf, self.X, self.y, sample_weight)
        UIclf.fit(fit_idx)
        probas_clf = clf.predict_proba(self.X)
        probas_UIclf = UIclf.predict_proba(range(len(self.X)))
        np.testing.assert_allclose(probas_clf, probas_UIclf)








