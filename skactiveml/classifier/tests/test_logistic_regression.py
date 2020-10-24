import numpy as np
import unittest

from sklearn.utils.validation import NotFittedError, check_is_fitted
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from skactiveml.classifier._logistic_regression import LogisticRegressionRY


class TestLogisticRegressionRY(unittest.TestCase):

    def setUp(self):
        self.X = np.zeros((2, 1))
        self.y_nan = [['nan', 'nan', 'nan'], ['nan', 'nan', 'nan']]
        self.y = [['tokyo', 'nan', 'paris'], ['tokyo', 'nan', 'nan']]
        self.w = [[2, np.nan, 1], [1, 1, 1]]

    def test_init(self):
        lr = LogisticRegressionRY(tol=1.e-10, solver='CG', annot_prior_full=2,
                                  solver_dict={'disp': True}, random_state=0,
                                  annot_prior_diag=1,
                                  cost_matrix=1 - np.eye(2),
                                  classes=['tokyo', 'paris'],
                                  missing_label='nan', fit_intercept=False)
        self.assertEqual(lr.tol, 1.e-10)
        self.assertEqual(lr.solver, 'CG')
        self.assertEqual(lr.solver_dict, {'disp': True})
        self.assertEqual(lr.random_state, 0)
        self.assertEqual(lr.annot_prior_full, 2)
        self.assertEqual(lr.annot_prior_diag, 1)
        self.assertEqual(lr.missing_label, 'nan')
        self.assertFalse(lr.fit_intercept)
        np.testing.assert_array_equal(lr.classes, ['tokyo', 'paris'])
        self.assertRaises(NotFittedError, check_is_fitted, estimator=lr)

    def test_fit(self):
        lr = LogisticRegressionRY(tol=[0.1], missing_label='nan')
        self.assertRaises(TypeError, lr.fit, X=self.X, y=self.y)
        lr = LogisticRegressionRY(tol=0.0, missing_label='nan')
        self.assertRaises(ValueError, lr.fit, X=self.X, y=self.y)
        lr = LogisticRegressionRY(max_iter=[1], missing_label='nan')
        self.assertRaises(TypeError, lr.fit, X=self.X, y=self.y)
        lr = LogisticRegressionRY(max_iter=0, missing_label='nan')
        self.assertRaises(ValueError, lr.fit, X=self.X, y=self.y)
        lr = LogisticRegressionRY(fit_intercept='Test', missing_label='nan')
        self.assertRaises(TypeError, lr.fit, X=self.X, y=self.y)
        lr = LogisticRegressionRY(annot_prior_full=0, missing_label='nan')
        self.assertRaises(ValueError, lr.fit, X=self.X, y=self.y)
        lr = LogisticRegressionRY(annot_prior_full=[1, 1],
                                  missing_label='nan')
        self.assertRaises(ValueError, lr.fit, X=self.X, y=self.y)
        lr = LogisticRegressionRY(annot_prior_diag=-0.1, missing_label='nan')
        self.assertRaises(ValueError, lr.fit, X=self.X, y=self.y)
        lr = LogisticRegressionRY(annot_prior_diag=[0, 0, -0.1],
                                  missing_label='nan')
        self.assertRaises(ValueError, lr.fit, X=self.X, y=self.y)
        lr = LogisticRegressionRY(weights_prior=[0, 1], missing_label='nan')
        self.assertRaises(TypeError, lr.fit, X=self.X, y=self.y)
        lr = LogisticRegressionRY(weights_prior=-0.1, missing_label='nan')
        self.assertRaises(ValueError, lr.fit, X=self.X, y=self.y)
        lr = LogisticRegressionRY(random_state=0, missing_label='nan',
                                  classes=['tokyo', 'paris'],
                                  solver='Nelder-Mead')
        lr.fit(X=self.X, y=self.y_nan)
        check_is_fitted(lr)
        Alpha_exp = np.ones_like(lr.Alpha_) * 0.5
        W_exp = np.zeros_like(lr.W_)
        np.testing.assert_array_equal(lr.Alpha_, Alpha_exp)
        np.testing.assert_array_equal(lr.W_, W_exp)
        lr.fit(X=self.X, y=self.y, sample_weight=self.w)
        self.assertTrue(np.abs(lr.Alpha_ - Alpha_exp).sum() > 0)
        self.assertTrue(np.abs(lr.W_ - W_exp).sum() > 0)

    def test_predict_proba(self):
        lr = LogisticRegressionRY(random_state=0, missing_label='nan',
                                  classes=['tokyo', 'paris'])
        lr.fit(X=self.X, y=self.y_nan)
        P = lr.predict_proba(X=self.X)
        np.testing.assert_array_equal(P, np.ones_like(P) * 0.5)
        lr.fit(X=self.X, y=self.y, sample_weight=self.w)
        np.testing.assert_array_equal(np.sum(P, axis=1), np.ones(len(P)))

    def test_predict_annot_proba(self):
        lr = LogisticRegressionRY(random_state=0, missing_label='nan',
                                  classes=['tokyo', 'paris'])
        lr.fit(X=self.X, y=self.y_nan)
        P_annot = lr.predict_annot_proba(X=self.X)
        np.testing.assert_array_equal(P_annot, np.ones_like(P_annot) * 0.5)
        lr.fit(X=self.X, y=self.y, sample_weight=self.w)
        self.assertTrue((P_annot <= 1).all())
        self.assertTrue((P_annot >= 0).all())

    def test_on_data_set(self):
        X, y_true = make_blobs(n_samples=300, random_state=0)
        X = StandardScaler().fit_transform(X)
        lr = LogisticRegressionRY(random_state=0, classes=[0, 1, 2])
        y = np.array([y_true, y_true, y_true, y_true], dtype=float).T
        y[0:100, 0] = 0
        y[100:150, 0] = np.nan
        y[90:150, 1] = 1
        y[:, 1] = np.nan
        lr.fit(X, y)
        self.assertTrue(lr.score(X, y_true) > 0.8)
        y = np.full_like(y, fill_value=np.nan)
        lr.fit(X, y)
        self.assertTrue(lr.score(X, y_true) > 0.2)
        lr.fit(X, y_true)
        self.assertTrue(lr.score(X, y_true) > 0.8)
