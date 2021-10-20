import unittest

import numpy as np
from sklearn.utils.validation import check_is_fitted

from skactiveml.classifier import LogisticRegressionRY


class TestLogisticRegressionRY(unittest.TestCase):

    def setUp(self):
        self.X = np.zeros((2, 1))
        self.y_nan = [['nan', 'nan', 'nan'], ['nan', 'nan', 'nan']]
        self.y = np.array([['tokyo', 'nan', 'paris'], ['tokyo', 'nan', 'nan']])
        self.w = np.array([[2, np.nan, 1], [1, 1, 1]])

    def test_init_param_tol(self):
        lr = LogisticRegressionRY()
        self.assertEqual(lr.tol, 1.e-2)
        lr = LogisticRegressionRY(tol=1.e-10)
        self.assertEqual(lr.tol, 1.e-10)
        lr = LogisticRegressionRY(tol=[0.1], missing_label='nan')
        self.assertRaises(TypeError, lr.fit, X=self.X, y=self.y)
        lr = LogisticRegressionRY(tol=0.0, missing_label='nan')
        self.assertRaises(ValueError, lr.fit, X=self.X, y=self.y)

    def test_init_param_solver(self):
        lr = LogisticRegressionRY()
        self.assertEqual(lr.solver, 'Newton-CG')
        lr = LogisticRegressionRY(solver='CG')
        self.assertEqual(lr.solver, 'CG')
        lr = LogisticRegressionRY(missing_label='nan', solver='Test')
        self.assertRaises(ValueError, lr.fit, X=self.X, y=self.y)

    def test_init_param_solver_dict(self):
        lr = LogisticRegressionRY()
        self.assertEqual(lr.solver_dict, None)
        lr = LogisticRegressionRY(solver_dict={'verbose': 0})
        self.assertTrue(isinstance(lr.solver_dict, dict))
        lr = LogisticRegressionRY(missing_label='nan', solver_dict='Test')
        self.assertRaises(ValueError, lr.fit, X=self.X, y=self.y)

    def test_init_param_fit_intercept(self):
        lr = LogisticRegressionRY()
        self.assertTrue(lr.fit_intercept)
        lr = LogisticRegressionRY(fit_intercept=False)
        self.assertFalse(lr.fit_intercept)
        lr = LogisticRegressionRY(missing_label='nan', fit_intercept='Test')
        self.assertRaises(TypeError, lr.fit, X=self.X, y=self.y)

    def test_init_param_max_iter(self):
        lr = LogisticRegressionRY()
        self.assertEqual(lr.max_iter, 100)
        lr = LogisticRegressionRY(max_iter=10)
        self.assertEqual(lr.max_iter, 10)
        lr = LogisticRegressionRY(max_iter=[1], missing_label='nan')
        self.assertRaises(TypeError, lr.fit, X=self.X, y=self.y)
        lr = LogisticRegressionRY(max_iter=0, missing_label='nan')
        self.assertRaises(ValueError, lr.fit, X=self.X, y=self.y)

    def test_init_param_annot_prior_full(self):
        lr = LogisticRegressionRY()
        self.assertEqual(lr.annot_prior_full, 1)
        lr = LogisticRegressionRY(annot_prior_full=2)
        self.assertEqual(lr.annot_prior_full, 2)
        lr = LogisticRegressionRY(annot_prior_full=0, missing_label='nan')
        self.assertRaises(ValueError, lr.fit, X=self.X, y=self.y)
        lr = LogisticRegressionRY(annot_prior_full=[1, 1],
                                  missing_label='nan')
        self.assertRaises(ValueError, lr.fit, X=self.X, y=self.y)

    def test_init_param_annot_prior_diag(self):
        lr = LogisticRegressionRY()
        self.assertEqual(lr.annot_prior_diag, 0)
        lr = LogisticRegressionRY(annot_prior_diag=2)
        self.assertEqual(lr.annot_prior_diag, 2)
        lr = LogisticRegressionRY(annot_prior_diag=-0.1, missing_label='nan')
        self.assertRaises(ValueError, lr.fit, X=self.X, y=self.y)
        lr = LogisticRegressionRY(annot_prior_diag=[0, 0, -0.1],
                                  missing_label='nan')
        self.assertRaises(ValueError, lr.fit, X=self.X, y=self.y)

    def test_init_param_weights_prior(self):
        lr = LogisticRegressionRY()
        self.assertEqual(lr.weights_prior, 1)
        lr = LogisticRegressionRY(weights_prior=0)
        self.assertEqual(lr.annot_prior_diag, 0)
        lr = LogisticRegressionRY(weights_prior=[0, 1], missing_label='nan')
        self.assertRaises(TypeError, lr.fit, X=self.X, y=self.y)
        lr = LogisticRegressionRY(weights_prior=-0.1, missing_label='nan')
        self.assertRaises(ValueError, lr.fit, X=self.X, y=self.y)

    def test_fit(self):
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
        lr.fit(X=self.X, y=self.y[:, 0], sample_weight=self.w[:, 0])
        self.assertEqual(len(lr.Alpha_), 1)

    def test_predict_proba(self):
        lr = LogisticRegressionRY(random_state=0, missing_label='nan',
                                  classes=['tokyo', 'paris'])
        lr.fit(X=self.X, y=self.y_nan)
        P = lr.predict_proba(X=self.X)
        np.testing.assert_array_equal(P, np.ones_like(P) * 0.5)
        lr.fit(X=self.X, y=self.y, sample_weight=self.w)
        np.testing.assert_array_equal(np.sum(P, axis=1), np.ones(len(P)))

    def test_predict(self):
        lr = LogisticRegressionRY(random_state=0, missing_label='nan',
                                  classes=['tokyo', 'paris'])
        lr.fit(X=self.X, y=self.y_nan)
        y_pred = lr.predict(X=self.X)
        np.testing.assert_array_equal(y_pred, ['tokyo', 'paris'])
        lr.fit(X=self.X, y=self.y, sample_weight=self.w)
        y_pred = lr.predict(X=self.X)
        np.testing.assert_array_equal(y_pred, ['tokyo', 'tokyo'])

    def test_predict_annot_perf(self):
        lr = LogisticRegressionRY(random_state=0, missing_label='nan',
                                  classes=['tokyo', 'paris'])
        lr.fit(X=self.X, y=self.y_nan)
        P_annot = lr.predict_annot_perf(X=self.X)
        np.testing.assert_array_equal(P_annot, np.ones_like(P_annot) * 0.5)
        lr.fit(X=self.X, y=self.y, sample_weight=self.w)
        self.assertTrue((P_annot <= 1).all())
        self.assertTrue((P_annot >= 0).all())
