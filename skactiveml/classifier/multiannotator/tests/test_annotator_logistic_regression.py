import unittest

import numpy as np
from sklearn.utils.validation import check_is_fitted
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression

from skactiveml.classifier.multiannotator import AnnotatorLogisticRegression


class TestAnnotatorLogisticRegression(unittest.TestCase):
    def setUp(self):
        self.X = np.zeros((2, 1))
        self.y_nan = [["nan", "nan", "nan"], ["nan", "nan", "nan"]]
        self.y = np.array([["tokyo", "nan", "paris"], ["tokyo", "nan", "nan"]])
        self.w = np.array([[2, np.nan, 1], [1, 1, 1]])

    def test_init_param_n_annotators(self):
        lr = AnnotatorLogisticRegression()
        self.assertEqual(lr.n_annotators, None)
        lr = AnnotatorLogisticRegression(n_annotators=3)
        self.assertEqual(lr.n_annotators, 3)
        lr = AnnotatorLogisticRegression(n_annotators=3.5, missing_label="nan")
        self.assertRaises(TypeError, lr.fit, X=self.X, y=self.y)
        lr = AnnotatorLogisticRegression(n_annotators=0, missing_label="nan")
        self.assertRaises(ValueError, lr.fit, X=self.X, y=self.y)
        lr = AnnotatorLogisticRegression(n_annotators=1, missing_label="nan")
        self.assertRaises(ValueError, lr.fit, X=self.X, y=self.y)

    def test_init_param_tol(self):
        lr = AnnotatorLogisticRegression()
        self.assertEqual(lr.tol, 1.0e-4)
        lr = AnnotatorLogisticRegression(tol=1.0e-10)
        self.assertEqual(lr.tol, 1.0e-10)
        lr = AnnotatorLogisticRegression(tol=[0.1], missing_label="nan")
        self.assertRaises(TypeError, lr.fit, X=self.X, y=self.y)
        lr = AnnotatorLogisticRegression(tol=0.0, missing_label="nan")
        self.assertRaises(ValueError, lr.fit, X=self.X, y=self.y)

    def test_init_param_solver(self):
        lr = AnnotatorLogisticRegression()
        self.assertEqual(lr.solver, "Newton-CG")
        lr = AnnotatorLogisticRegression(solver="CG")
        self.assertEqual(lr.solver, "CG")
        lr = AnnotatorLogisticRegression(missing_label="nan", solver="Test")
        self.assertRaises(ValueError, lr.fit, X=self.X, y=self.y)

    def test_init_param_solver_dict(self):
        lr = AnnotatorLogisticRegression()
        self.assertEqual(lr.solver_dict, None)
        lr = AnnotatorLogisticRegression(solver_dict={"verbose": 0})
        self.assertTrue(isinstance(lr.solver_dict, dict))
        lr = AnnotatorLogisticRegression(
            missing_label="nan", solver_dict="Test"
        )
        self.assertRaises(TypeError, lr.fit, X=self.X, y=self.y)

    def test_init_param_fit_intercept(self):
        lr = AnnotatorLogisticRegression()
        self.assertTrue(lr.fit_intercept)
        lr = AnnotatorLogisticRegression(fit_intercept=False)
        self.assertFalse(lr.fit_intercept)
        lr = AnnotatorLogisticRegression(
            missing_label="nan", fit_intercept="Test"
        )
        self.assertRaises(TypeError, lr.fit, X=self.X, y=self.y)

    def test_init_param_max_iter(self):
        lr = AnnotatorLogisticRegression()
        self.assertEqual(lr.max_iter, 100)
        lr = AnnotatorLogisticRegression(max_iter=10)
        self.assertEqual(lr.max_iter, 10)
        lr = AnnotatorLogisticRegression(max_iter=[1], missing_label="nan")
        self.assertRaises(TypeError, lr.fit, X=self.X, y=self.y)
        lr = AnnotatorLogisticRegression(max_iter=0, missing_label="nan")
        self.assertRaises(ValueError, lr.fit, X=self.X, y=self.y)

    def test_init_param_annot_prior_full(self):
        lr = AnnotatorLogisticRegression()
        self.assertEqual(lr.annot_prior_full, 1)
        lr = AnnotatorLogisticRegression(annot_prior_full=2)
        self.assertEqual(lr.annot_prior_full, 2)
        lr = AnnotatorLogisticRegression(
            annot_prior_full=0, missing_label="nan"
        )
        self.assertRaises(ValueError, lr.fit, X=self.X, y=self.y)
        lr = AnnotatorLogisticRegression(
            annot_prior_full=[1, 1], missing_label="nan"
        )
        self.assertRaises(ValueError, lr.fit, X=self.X, y=self.y)

    def test_init_param_annot_prior_diag(self):
        lr = AnnotatorLogisticRegression()
        self.assertEqual(lr.annot_prior_diag, 0)
        lr = AnnotatorLogisticRegression(annot_prior_diag=2)
        self.assertEqual(lr.annot_prior_diag, 2)
        lr = AnnotatorLogisticRegression(
            annot_prior_diag=-0.1, missing_label="nan"
        )
        self.assertRaises(ValueError, lr.fit, X=self.X, y=self.y)
        lr = AnnotatorLogisticRegression(
            annot_prior_diag=[0, 0, -0.1], missing_label="nan"
        )
        self.assertRaises(ValueError, lr.fit, X=self.X, y=self.y)

    def test_init_param_weights_prior(self):
        lr = AnnotatorLogisticRegression()
        self.assertEqual(lr.weights_prior, 1)
        lr = AnnotatorLogisticRegression(weights_prior=0)
        self.assertEqual(lr.annot_prior_diag, 0)
        lr = AnnotatorLogisticRegression(
            weights_prior=[0, 1], missing_label="nan"
        )
        self.assertRaises(TypeError, lr.fit, X=self.X, y=self.y)
        lr = AnnotatorLogisticRegression(
            weights_prior=-0.1, missing_label="nan"
        )
        self.assertRaises(ValueError, lr.fit, X=self.X, y=self.y)

    def test_fit(self):
        # ---------------------Check trivial use cases.------------------------
        lr = AnnotatorLogisticRegression(
            random_state=0,
            missing_label="nan",
            classes=["tokyo", "paris"],
            solver="Nelder-Mead",
        )
        lr.fit(X=self.X, y=self.y_nan)
        check_is_fitted(lr)
        Alpha_exp = np.ones_like(lr.Alpha_) * 0.5
        W_exp = np.zeros_like(lr.W_)
        np.testing.assert_array_equal(lr.Alpha_, Alpha_exp)
        np.testing.assert_array_equal(lr.W_, W_exp)
        lr.fit(X=self.X, y=self.y, sample_weight=self.w)
        self.assertTrue(np.abs(lr.Alpha_ - Alpha_exp).sum() > 0)
        self.assertTrue(np.abs(lr.W_ - W_exp).sum() > 0)
        self.assertRaises(ValueError, lr.fit, X=self.X, y=self.y[:, 0])
        self.assertRaises(ValueError, lr.fit, X=[], y=[])
        lr.n_annotators = 5
        lr.fit(X=[], y=[])
        y_proba = lr.predict_proba(X=self.X)
        y_proba_exp = np.full((len(self.X), 2), fill_value=0.5)
        np.testing.assert_array_equal(y_proba, y_proba_exp)
        y_perf = lr.predict_annotator_perf(X=self.X)
        y_perf_exp = np.full((len(self.X), 5), fill_value=0.5)
        np.testing.assert_array_equal(y_perf, y_perf_exp)

        # ---------------------Check advanced use cases.-----------------------
        X, y_true = make_blobs(n_samples=200, centers=5, random_state=0)
        classes = np.unique(y_true)
        y_noisy = np.column_stack(
            (y_true, y_true, y_true, (y_true + 1) % 5, np.ones_like(y_true))
        )
        y = np.full(shape=y_noisy.shape, fill_value=-1)

        # Check when there are no labels.
        lr = AnnotatorLogisticRegression(
            missing_label=-1, classes=classes, random_state=0
        )
        lr.fit(X, y)
        probas = lr.predict_proba(X)
        probas_exp = np.full((len(X), len(classes)), 1 / len(classes))
        np.testing.assert_array_equal(probas, probas_exp)
        Alpha_exp = np.full(
            (y.shape[1], len(classes), len(classes)), 1 / len(classes)
        )
        np.testing.assert_array_equal(lr.Alpha_, Alpha_exp)

        # Check training with all labels.
        lr = AnnotatorLogisticRegression(
            missing_label=-1, classes=classes, random_state=0
        )
        lr.fit(X, y_noisy)
        self.assertGreater(lr.score(X, y_true), 0.9)
        Alpha_exp = np.stack([np.eye(len(classes)) for _ in range(y.shape[1])])
        Alpha_exp[-2] = np.array(
            [
                [0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0],
            ]
        )
        Alpha_exp[-1] = np.array(
            [
                [0, 1, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 1, 0, 0, 0],
            ]
        )
        np.testing.assert_array_equal(lr.Alpha_, Alpha_exp)

        # Check effect of prior.
        lr = AnnotatorLogisticRegression(
            missing_label=-1,
            classes=classes,
            random_state=0,
            annot_prior_diag=[0, 0, 0, 0, 10e5],
        )
        lr.fit(X, y)
        Alpha_exp = np.full(
            (y.shape[-1], len(classes), len(classes)), 1 / len(classes)
        )
        Alpha_exp[-1] = np.eye(len(classes))
        np.testing.assert_array_equal(lr.Alpha_, Alpha_exp)
        lr.n_annotators = y.shape[-1]
        lr.fit(X=[], y=[])
        np.testing.assert_array_equal(lr.Alpha_, Alpha_exp)
        y_proba = lr.predict_proba(X=self.X)
        y_proba_exp = np.full((len(self.X), len(classes)), 1 / len(classes))
        np.testing.assert_array_equal(y_proba, y_proba_exp)
        lr.fit(X, y_noisy)
        y_pred = lr.predict(X)
        y_pred_exp = y_noisy[:, -1]
        np.testing.assert_array_equal(y_pred, y_pred_exp)
        np.testing.assert_array_equal(lr.Alpha_, Alpha_exp)
        lr = AnnotatorLogisticRegression(
            missing_label=-1,
            classes=classes,
            random_state=0,
            annot_prior_full=[2, 2, 2, 2, 2],
            annot_prior_diag=[3, 3, 3, 3, 3],
        )
        lr.fit(X, y)
        Alpha_exp = np.full((y.shape[-1], len(classes), len(classes)), 0.125)
        Alpha_exp[:, np.arange(len(classes)), np.arange(len(classes))] = 0.5
        np.testing.assert_array_equal(lr.Alpha_, Alpha_exp)

        # Check training with only labels from one annotator.
        lr = AnnotatorLogisticRegression(
            missing_label=-1,
            classes=classes,
            random_state=0,
        )
        y_1 = y.copy()
        y_1[:, 0] = y_noisy[:, 0]
        lr.fit(X, y_1)
        self.assertGreater(lr.score(X, y_1[:, 0]), 0.9)
        Alpha_exp = np.full(
            (y.shape[1], len(classes), len(classes)), 1 / len(classes)
        )
        Alpha_exp[0] = np.eye(len(classes))
        np.testing.assert_array_equal(Alpha_exp, lr.Alpha_)

        # Check training with a subset of completely unlabeled samples.
        lr = AnnotatorLogisticRegression(
            missing_label=-1,
            classes=classes,
            random_state=0,
        )
        y_2 = y.copy()
        y_2[:50] = y_noisy[:50]
        lr.fit(X, y_2)
        self.assertGreater(lr.score(X, y_true), 0.9)

        # Check training with mixes of labeled and unlabeled samples.
        y_3 = y_noisy.copy()
        mask = np.random.RandomState(0).rand(y_3.shape[0], y_3.shape[1]) < 0.5
        y_3[mask] = -1
        lr.fit(X, y_3)
        self.assertGreater(lr.score(X, y_true), 0.9)

        # Check consistency of fitting with and without bias.
        for fit_intercept in [True, False]:
            lr_sklearn = LogisticRegression(
                solver="newton-cg", random_state=0, fit_intercept=fit_intercept
            )
            lr = AnnotatorLogisticRegression(
                missing_label=-1,
                classes=classes,
                random_state=0,
                fit_intercept=fit_intercept,
            )
            y_sklearn = lr_sklearn.fit(X, y_true).predict(X)
            y_skactiveml = lr.fit(X, y_noisy).predict(X)
            np.testing.assert_array_equal(y_sklearn, y_skactiveml)

    def test_predict_proba(self):
        lr = AnnotatorLogisticRegression(
            random_state=0, missing_label="nan", classes=["tokyo", "paris"]
        )
        lr.fit(X=self.X, y=self.y_nan)
        P = lr.predict_proba(X=self.X)
        np.testing.assert_array_equal(P, np.ones_like(P) * 0.5)
        lr.fit(X=self.X, y=self.y, sample_weight=self.w)
        np.testing.assert_array_equal(np.sum(P, axis=1), np.ones(len(P)))

    def test_predict(self):
        lr = AnnotatorLogisticRegression(
            random_state=0, missing_label="nan", classes=["tokyo", "paris"]
        )
        lr.fit(X=self.X, y=self.y_nan)
        y_pred = lr.predict(X=self.X)
        np.testing.assert_array_equal(y_pred, ["tokyo", "paris"])
        lr.fit(X=self.X, y=self.y, sample_weight=self.w)
        y_pred = lr.predict(X=self.X)
        np.testing.assert_array_equal(y_pred, ["tokyo", "tokyo"])

    def test_predict_annotator_perf(self):
        lr = AnnotatorLogisticRegression(
            random_state=0, missing_label="nan", classes=["tokyo", "paris"]
        )
        lr.fit(X=self.X, y=self.y_nan)
        P_annot = lr.predict_annotator_perf(X=self.X)
        np.testing.assert_array_equal(P_annot, np.ones_like(P_annot) * 0.5)
        lr.fit(X=self.X, y=self.y, sample_weight=self.w)
        self.assertTrue((P_annot <= 1).all())
        self.assertTrue((P_annot >= 0).all())
