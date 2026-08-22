import unittest

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.utils._testing import assert_allclose
from sklearn.utils.validation import NotFittedError, check_is_fitted

from skactiveml.classifier import MixtureModelClassifier
from skactiveml.tests.template_estimator import TemplateClassFrequencyEstimator


class TestMixtureModelClassifier(
    TemplateClassFrequencyEstimator, unittest.TestCase
):
    def setUp(self):
        estimator_class = MixtureModelClassifier
        init_default_params = {
            "missing_label": "nan",
        }
        fit_default_params = {
            "X": np.zeros((3, 1)),
            "y": ["tokyo", "nan", "paris"],
        }
        predict_default_params = {"X": [[1]]}
        super().setUp(
            estimator_class=estimator_class,
            init_default_params=init_default_params,
            fit_default_params=fit_default_params,
            predict_default_params=predict_default_params,
        )
        self.y_nan = ["nan", "nan", "nan"]
        self.w = [2, np.nan, 1]

    def test_init_param_mixture_model(self):
        test_cases = []
        test_cases += [
            (None, None),
            ("Test", TypeError),
            (BayesianGaussianMixture(), None),
        ]
        self._test_param("init", "mixture_model", test_cases)

    def test_init_param_weight_mode(self):
        test_cases = []
        test_cases += [
            ("responsibilities", None),
            ("similarities", None),
            ("Test", ValueError),
        ]
        self._test_param("init", "weight_mode", test_cases)

    def test_fit(self):
        mixture = GaussianMixture(random_state=0, n_components=4)
        cmm = MixtureModelClassifier(
            missing_label="nan",
            mixture_model=mixture,
            classes=[1, 2],
            cost_matrix=1 - np.eye(3),
        )
        self.assertRaises(
            TypeError,
            cmm.fit,
            X=self.fit_default_params["X"],
            y=self.fit_default_params["y"],
        )
        cmm = MixtureModelClassifier(
            missing_label="nan",
            mixture_model=mixture,
            cost_matrix=1 - np.eye(3),
        )
        self.assertRaises(
            ValueError,
            cmm.fit,
            X=self.fit_default_params["X"],
            y=self.fit_default_params["y"],
        )
        cmm = MixtureModelClassifier(missing_label=None, random_state=0)
        self.assertRaises(NotFittedError, check_is_fitted, estimator=cmm)
        cost_matrix = 1 - np.eye(2)
        cmm = MixtureModelClassifier(
            classes=["tokyo", "paris"],
            cost_matrix=cost_matrix,
            missing_label="nan",
        )
        np.testing.assert_array_equal(cost_matrix, cmm.cost_matrix)
        self.assertEqual("nan", cmm.missing_label)
        self.assertEqual(cmm.mixture_model, None)
        np.testing.assert_array_equal(["tokyo", "paris"], cmm.classes)
        mixture = BayesianGaussianMixture(n_components=1).fit(
            X=self.fit_default_params["X"]
        )
        cmm = MixtureModelClassifier(
            mixture_model=mixture,
            classes=["tokyo", "paris", "new york"],
            missing_label="nan",
        )
        self.assertEqual(None, cmm.cost_matrix)
        self.assertFalse(hasattr(cmm, "F_components_"))
        self.assertFalse(hasattr(cmm, "_refit"))
        self.assertFalse(hasattr(cmm, "classes_"))
        cmm.fit(X=self.fit_default_params["X"], y=self.fit_default_params["y"])
        self.assertTrue(hasattr(cmm, "mixture_model_"))
        np.testing.assert_array_equal(
            cmm.classes_, ["new york", "paris", "tokyo"]
        )
        np.testing.assert_array_equal(1 - np.eye(3), cmm.cost_matrix_)
        np.testing.assert_array_equal([[0, 1, 1]], cmm.F_components_)
        cmm.fit(
            X=self.fit_default_params["X"],
            y=self.fit_default_params["y"],
            sample_weight=self.w,
        )
        np.testing.assert_array_equal([[0, 1, 2]], cmm.F_components_)

    def test_multilabel_requires_explicit_mixture_before_fitted_state(self):
        cmm = MixtureModelClassifier(
            classes=[[0, 1], [0, 1]], target_type="multi-label"
        )

        with self.assertRaisesRegex(ValueError, "mixture_model.*multi-label"):
            cmm.fit([[0], [1]], [[0, 1], [1, 0]])

        self.assertFalse(any(name.endswith("_") for name in vars(cmm)))

    def test_multilabel_predictions_in_both_weight_modes(self):
        X = np.array([[-0.1], [0.0], [0.1]])
        y = np.array(
            [["no", "on"], [None, None], ["yes", "off"]],
            dtype=object,
        )
        init_params = {
            "classes": [["no", "yes"], ["off", "on"]],
            "missing_label": None,
            "target_type": "multi-label",
            "class_prior": [[1, 1], [2, 2]],
        }

        for weight_mode in ("responsibilities", "similarities"):
            with self.subTest(weight_mode=weight_mode):
                mixture = GaussianMixture(n_components=1, random_state=0)
                cmm = MixtureModelClassifier(
                    mixture_model=mixture,
                    weight_mode=weight_mode,
                    **init_params,
                ).fit(X, y, sample_weight=[2, np.nan, 3])

                assert_allclose(
                    cmm.predict_freq([[0]]),
                    [[[2, 3], [3, 2]]],
                    atol=1e-10,
                )
                assert_allclose(cmm.predict_proba([[0]]), [[4 / 7, 4 / 9]])
                np.testing.assert_array_equal(
                    cmm.predict([[0]]), [["yes", "off"]]
                )
                self.assertEqual(
                    cmm.target_spec_.classes,
                    (("no", "yes"), ("off", "on")),
                )
                self.assertIn(
                    ("classification", "multi-label", "single-annotator"),
                    cmm._target_capabilities,
                )

    def test_multilabel_per_entry_weights_with_fitted_mixture(self):
        X = np.array([[-0.1], [0.0], [0.1]])
        mixture = GaussianMixture(n_components=1, random_state=0).fit(X)
        cmm = MixtureModelClassifier(
            mixture_model=mixture,
            classes=[["no", "yes"], ["off", "on"]],
            missing_label=None,
            target_type="multi-label",
        ).fit(
            X,
            np.array(
                [["no", "on"], [None, None], ["yes", "off"]],
                dtype=object,
            ),
            sample_weight=[[2, 4], [np.nan, np.nan], [3, 5]],
        )

        assert_allclose(
            cmm.predict_freq([[0]]), [[[2, 3], [5, 4]]], atol=1e-10
        )

    def test_multilabel_cold_start_probabilities_in_both_weight_modes(self):
        X = np.array([[-0.1], [0.0], [0.1]])
        for weight_mode in ("responsibilities", "similarities"):
            with self.subTest(weight_mode=weight_mode):
                cmm = MixtureModelClassifier(
                    mixture_model=GaussianMixture(
                        n_components=1, random_state=0
                    ),
                    weight_mode=weight_mode,
                    classes=[[0, 1], [0, 1]],
                    target_type="multi-label",
                ).fit(
                    X,
                    np.full((3, 2), np.nan),
                )

                np.testing.assert_array_equal(
                    cmm.predict_freq([[0], [1]]), np.zeros((2, 2, 2))
                )
                np.testing.assert_array_equal(
                    cmm.predict_proba([[0], [1]]), np.full((2, 2), 0.5)
                )

    def test_predict_freq(self):
        mixture = BayesianGaussianMixture(n_components=1)
        mixture.fit(
            X=self.fit_default_params["X"], y=self.fit_default_params["y"]
        )
        cmm = MixtureModelClassifier(
            mixture_model=mixture,
            classes=["tokyo", "paris", "new york"],
            missing_label="nan",
        )
        self.assertRaises(
            NotFittedError, cmm.predict_freq, X=self.fit_default_params["X"]
        )
        cmm.fit(X=self.fit_default_params["X"], y=self.y_nan)
        F = cmm.predict_freq(X=self.fit_default_params["X"])
        np.testing.assert_array_equal(
            np.zeros((len(self.fit_default_params["X"]), 3)), F
        )
        cmm.fit(
            X=self.fit_default_params["X"],
            y=self.fit_default_params["y"],
            sample_weight=self.w,
        )
        F = cmm.predict_freq(X=[self.fit_default_params["X"][0]])
        np.testing.assert_array_equal([[0, 1, 2]], F)
        X, y = make_blobs(n_samples=200, centers=2)
        y_nan = np.full_like(y, np.nan, dtype=float)
        mixture = BayesianGaussianMixture(n_components=5)
        cmm = MixtureModelClassifier(
            mixture_model=mixture, classes=[0, 1], weight_mode="similarities"
        )
        self.assertRaises(
            NotFittedError, cmm.predict_freq, X=self.fit_default_params["X"]
        )
        cmm.fit(X=X, y=y_nan)
        F = cmm.predict_freq(X=X)
        np.testing.assert_array_equal(F.shape, [200, 2])
        self.assertEqual(F.sum(), 0)
        cmm.fit(X=X, y=y)
        F = cmm.predict_freq(X=X)
        self.assertTrue(F.sum() > 0)

    def test_predict_proba(self):
        mixture = BayesianGaussianMixture(n_components=1).fit(
            X=self.fit_default_params["X"]
        )
        cmm = MixtureModelClassifier(
            mixture_model=mixture,
            classes=["tokyo", "paris"],
            missing_label="nan",
        )
        self.assertRaises(
            NotFittedError, cmm.predict_proba, X=self.fit_default_params["X"]
        )
        cmm.fit(X=self.fit_default_params["X"], y=self.y_nan)
        P = cmm.predict_proba(X=self.fit_default_params["X"])
        np.testing.assert_array_equal(
            np.ones((len(self.fit_default_params["X"]), 2)) * 0.5, P
        )
        cmm.fit(
            X=self.fit_default_params["X"],
            y=self.fit_default_params["y"],
            sample_weight=self.w,
        )
        P = cmm.predict_proba(X=[self.fit_default_params["X"][0]])
        np.testing.assert_array_equal([[1 / 3, 2 / 3]], P)
        cmm = MixtureModelClassifier(
            mixture_model=mixture,
            missing_label="nan",
            classes=["tokyo", "paris", "new york"],
            class_prior=1,
        )
        cmm.fit(
            X=self.fit_default_params["X"],
            y=self.fit_default_params["y"],
            sample_weight=self.w,
        )
        P = cmm.predict_proba(X=[self.fit_default_params["X"][0]])
        np.testing.assert_array_equal([[1 / 6, 2 / 6, 3 / 6]], P)
        cmm = MixtureModelClassifier(
            mixture_model=mixture,
            missing_label="nan",
            classes=["tokyo", "paris", "new york"],
            class_prior=[0, 0, 1],
        )
        cmm.fit(
            X=self.fit_default_params["X"],
            y=self.fit_default_params["y"],
            sample_weight=self.w,
        )
        P = cmm.predict_proba(X=[self.fit_default_params["X"][0]])
        np.testing.assert_array_equal([[0, 1 / 4, 3 / 4]], P)

    def test_predict(self):
        mixture = BayesianGaussianMixture(n_components=1, random_state=0)
        mixture.fit(X=self.fit_default_params["X"])
        cmm = MixtureModelClassifier(
            mixture_model=mixture,
            classes=["tokyo", "paris", "new york"],
            missing_label="nan",
            random_state=0,
        )
        self.assertRaises(
            NotFittedError, cmm.predict, X=self.fit_default_params["X"]
        )
        cmm.fit(X=self.fit_default_params["X"], y=self.y_nan)
        y = cmm.predict(self.fit_default_params["X"])
        np.testing.assert_array_equal(["paris", "tokyo", "tokyo"], y)
        cmm = MixtureModelClassifier(
            mixture_model=mixture,
            classes=["tokyo", "paris"],
            missing_label="nan",
            random_state=1,
        )
        cmm.fit(X=self.fit_default_params["X"], y=self.y_nan)
        y = cmm.predict(self.fit_default_params["X"])
        np.testing.assert_array_equal(["tokyo", "tokyo", "paris"], y)
        cmm.fit(
            X=self.fit_default_params["X"],
            y=self.fit_default_params["y"],
            sample_weight=self.w,
        )
        y = cmm.predict(self.fit_default_params["X"])
        np.testing.assert_array_equal(["tokyo", "tokyo", "tokyo"], y)
        cmm = MixtureModelClassifier(
            mixture_model=mixture,
            classes=["tokyo", "paris"],
            missing_label="nan",
            cost_matrix=[[0, 1], [10, 0]],
        )
        cmm.fit(X=self.fit_default_params["X"], y=self.fit_default_params["y"])
        y = cmm.predict(self.fit_default_params["X"])
        np.testing.assert_array_equal(["paris", "paris", "paris"], y)
        cmm.fit(
            X=self.fit_default_params["X"],
            y=self.fit_default_params["y"],
            sample_weight=self.w,
        )
        y = cmm.predict(self.fit_default_params["X"])
        np.testing.assert_array_equal(["paris", "paris", "paris"], y)
