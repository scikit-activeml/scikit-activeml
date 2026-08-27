import inspect
import warnings
from copy import deepcopy

from skactiveml.tests.utils import (
    assert_predicts_class_dtype,
    assert_predicts_single_output_shape,
    check_positional_args,
    check_test_param_test_availability,
)
import numpy as np
from sklearn.base import clone
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from skactiveml.utils import (
    MISSING_LABEL,
)


class Dummy:
    def __init__(self):
        pass


class TemplateEstimator:
    def setUp(
        self,
        estimator_class,
        init_default_params,
        fit_default_params=None,
        predict_default_params=None,
        init_default_params_multilabel=None,
        fit_default_params_multilabel=None,
        predict_default_params_multilabel=None,
    ):
        self.super_setUp_has_been_executed = True
        self.estimator_class = estimator_class
        self.test_param_id_offset = 0

        self.init_default_params = {
            "random_state": 42,
            "missing_label": MISSING_LABEL,
        }
        self.init_default_params.update(deepcopy(init_default_params))

        check_positional_args(
            self.estimator_class.__init__,
            "__init__",
            self.init_default_params,
        )

        self.fit_default_params = deepcopy(fit_default_params)
        self.predict_default_params = deepcopy(predict_default_params)
        self.init_default_params_multilabel = deepcopy(
            init_default_params_multilabel
        )
        self.fit_default_params_multilabel = deepcopy(
            fit_default_params_multilabel
        )
        self.predict_default_params_multilabel = deepcopy(
            predict_default_params_multilabel
        )

        check_positional_args(
            self.estimator_class.fit, "fit", self.fit_default_params
        )
        check_positional_args(
            self.estimator_class.predict,
            "predict",
            self.predict_default_params,
        )

    def test_init_param_missing_label(
        self,
        test_cases=None,
        replace_init_params=None,
        replace_fit_params=None,
    ):
        test_cases = [] if test_cases is None else test_cases
        if replace_fit_params is None or "y" not in replace_fit_params:
            ml = self.init_default_params["missing_label"]
            test_cases += [(ml, None), (Dummy, TypeError)]
        self._test_param(
            "init",
            "missing_label",
            test_cases,
            replace_init_params=replace_init_params,
            replace_fit_params=replace_fit_params,
        )

    def test_init_param_random_state(self, test_cases=None):
        test_cases = [] if test_cases is None else test_cases
        test_cases += [(np.nan, ValueError), ("state", ValueError), (1, None)]
        self._test_param("init", "random_state", test_cases)

    def test_fit_param_X(self, test_cases=None):
        test_cases = [] if test_cases is None else test_cases
        test_cases += [
            (np.nan, ValueError),
            ([1], ValueError),
            (np.full_like(self.fit_default_params["X"], 0), None),
        ]
        self._test_param("fit", "X", test_cases)

    def test_fit_param_y(
        self,
        test_cases=None,
        replace_init_params=None,
        replace_fit_params=None,
    ):
        test_cases = [] if test_cases is None else test_cases
        test_cases += [(np.nan, TypeError), ("state", TypeError)]
        self._test_param(
            "fit",
            "y",
            test_cases,
            replace_init_params=replace_init_params,
            replace_fit_params=replace_fit_params,
        )

    def test_fit_param_sample_weight(
        self,
        test_cases=None,
        replace_init_params=None,
        replace_fit_params=None,
    ):
        fit_params = inspect.signature(self.estimator_class.fit).parameters
        if "sample_weight" in fit_params:
            test_cases = [] if test_cases is None else test_cases
            test_cases += [
                (np.nan, ValueError),
                (1, ValueError),
                (np.arange(len(self.fit_default_params["y"])), None),
            ]
            self._test_param(
                "fit",
                "sample_weight",
                test_cases,
                replace_init_params=replace_init_params,
                replace_fit_params=replace_fit_params,
            )

    def test_partial_fit_param_X(
        self, test_cases=None, replace_init_params=None
    ):
        if hasattr(self.estimator_class, "partial_fit"):
            extras_params = deepcopy(self.fit_default_params)
            test_cases = [] if test_cases is None else test_cases
            test_cases += [
                (np.nan, ValueError),
                ([1], ValueError),
                (np.full_like(self.fit_default_params["X"], 0), None),
            ]
            self._test_param(
                "partial_fit",
                "X",
                test_cases,
                replace_init_params=replace_init_params,
                extras_params=extras_params,
                exclude_fit=True,
            )

    def test_partial_fit_param_y(
        self,
        test_cases=None,
        replace_init_params=None,
        replace_fit_params=None,
    ):
        if hasattr(self.estimator_class, "partial_fit"):
            extras_params = deepcopy(self.fit_default_params)
            test_cases = [] if test_cases is None else test_cases
            test_cases += [(np.nan, TypeError), ("state", TypeError)]
            self._test_param(
                "partial_fit",
                "y",
                test_cases,
                replace_init_params=replace_init_params,
                replace_fit_params=replace_fit_params,
                extras_params=extras_params,
                exclude_fit=True,
            )

    def test_partial_fit_param_sample_weight(
        self, test_cases=None, replace_init_params=None
    ):
        if hasattr(self.estimator_class, "partial_fit"):
            extras_params = deepcopy(self.fit_default_params)
            fit_params = inspect.signature(self.estimator_class.fit).parameters
            if "sample_weight" in fit_params:
                test_cases = [] if test_cases is None else test_cases
                test_cases += [
                    (np.nan, ValueError),
                    (1, ValueError),
                    (np.random.rand(len(self.fit_default_params["y"])), None),
                ]
                self._test_param(
                    "partial_fit",
                    "sample_weight",
                    test_cases,
                    replace_init_params=replace_init_params,
                    extras_params=extras_params,
                    exclude_fit=True,
                )

    def test_predict_param_X(self, test_cases=None):
        test_cases = [] if test_cases is None else test_cases
        dtype = None
        if hasattr(self.predict_default_params["X"], "dtype"):
            dtype = self.predict_default_params["X"].dtype
        dummy_zeros = np.zeros(
            np.array(self.predict_default_params["X"]).shape,
            dtype=dtype,
        )
        test_cases += [
            (np.nan, ValueError),
            ("state", ValueError),
            (dummy_zeros, None),
        ]
        self._test_param(
            "predict",
            "X",
            test_cases,
            extras_params=self.predict_default_params,
        )

    def test_init_param_test_assignments(self):
        for param in inspect.signature(
            self.estimator_class.__init__
        ).parameters:
            if param != "self":
                init_params = deepcopy(self.init_default_params)
                init_params[param] = Dummy()
                estimator = self.estimator_class(**init_params)
                self.assertEqual(
                    getattr(estimator, param),
                    init_params[param],
                    msg=f"The parameter `{param}` was not assigned to a class "
                    f"variable when `__init__` was called.",
                )

    def test_param_test_availability(self):
        not_test = ["self"]

        check_test_param_test_availability(
            self,
            self.estimator_class.__init__,
            "init",
            not_test,
            logic_test=False,
        )

        check_test_param_test_availability(
            self, self.estimator_class.fit, "fit", not_test
        )

        check_test_param_test_availability(
            self, self.estimator_class.predict, "predict", not_test
        )

        if hasattr(self.estimator_class, "partial_fit"):
            check_test_param_test_availability(
                self, self.estimator_class.partial_fit, "partial_fit", not_test
            )

    def _test_param(
        self,
        test_func,
        test_param,
        test_cases,
        replace_init_params=None,
        replace_fit_params=None,
        extras_params=None,
        exclude_fit=False,
    ):
        if replace_init_params is None:
            replace_init_params = {}
        if replace_fit_params is None:
            replace_fit_params = {}
        if extras_params is None:
            extras_params = {}

        for i, (test_val, err) in enumerate(test_cases):

            with self.subTest(
                msg=(
                    f"ID: {self.test_param_id_offset}, param: {test_param},"
                    f"expected: {err}"
                ),
                val=str(test_val),
            ):
                self.test_param_id_offset += 1
                init_params = deepcopy(self.init_default_params)
                init_params.update(replace_init_params)

                fit_params = deepcopy(self.fit_default_params)
                fit_params.update(replace_fit_params)

                if f"{test_func}_params" in locals():
                    locals()[f"{test_func}_params"][test_param] = test_val
                else:
                    extras_params[test_param] = test_val

                estimator = self.estimator_class(**init_params)
                if err is None and test_func in ["fit", "init"]:
                    estimator.fit(**fit_params)
                elif test_func in ["fit", "init"]:
                    if not hasattr(estimator, "fit"):
                        if not issubclass(AttributeError, err):
                            estimator.fit
                    else:
                        self.assertRaises(err, estimator.fit, **fit_params)
                else:
                    if not exclude_fit:
                        estimator.fit(**fit_params)
                    func = getattr(estimator, test_func)
                    if err is None:
                        func(**extras_params)
                    else:
                        self.assertRaises(err, func, **extras_params)


class TemplateSkactivemlClassifier(TemplateEstimator):
    # Error expected when fitting on non-integral float class labels. Wrappers
    # forwarding the declared class labels to a `scikit-learn` estimator have
    # to report the estimator's rejection of such a continuous target.
    non_integral_classes_error = None

    def _has_multilabel_defaults(self):
        return self.fit_default_params_multilabel is not None

    def test_init_param_target_type(self):
        self._test_param(
            "init",
            "target_type",
            [
                ("auto", None),
                ("single-output", None),
                ("multi-label", ValueError),
                ("multi-output", ValueError),
                ("invalid", ValueError),
            ],
        )

    def test_fitted_target_spec_matches_declared_capability(self):
        estimator = self.estimator_class(**deepcopy(self.init_default_params))
        estimator.fit(**deepcopy(self.fit_default_params))

        capability = (
            estimator.target_spec_.task,
            estimator.target_spec_.target_type,
            estimator.target_spec_.annotation_type,
        )
        self.assertEqual(estimator.target_type, "auto")
        self.assertIn(capability, estimator._target_capabilities)
        self.assertEqual(capability[0], "classification")
        self.assertNotEqual(capability[1], "multi-output")

    def test_predict_dtype_matches_class_dtype(
        self, replace_init_params=None, replace_fit_params=None
    ):
        """Check that `predict` returns the declared class dtype.

        The label encoder decodes into a dtype that can also represent
        `missing_label`, so integer classes combined with the default
        `missing_label=np.nan` decode to `float64`. Predictions never carry
        `missing_label` and must therefore be narrowed back, so that they
        remain usable where the declared class labels are expected.
        """
        init_params = deepcopy(self.init_default_params)
        init_params["classes"] = [0, 1]
        init_params["missing_label"] = np.nan
        if replace_init_params is not None:
            init_params.update(deepcopy(replace_init_params))

        fit_params = deepcopy(self.fit_default_params)
        fit_params["y"] = [0, np.nan, 1]
        fit_params["X"] = np.zeros((3, 1))
        if replace_fit_params is not None:
            fit_params.update(deepcopy(replace_fit_params))

        predict_params = deepcopy(self.predict_default_params)
        predict_params["X"] = fit_params["X"]

        estimator = self.estimator_class(**init_params)
        self._call_with_target(estimator, "fit", fit_params)
        y_pred = np.asarray(estimator.predict(**predict_params))

        assert_predicts_class_dtype(self, y_pred, estimator.classes_)

    def _get_multilabel_params(self):
        if not self._has_multilabel_defaults():
            return None, None, None

        init_params = deepcopy(self.init_default_params)
        if self.init_default_params_multilabel is not None:
            init_params.update(self.init_default_params_multilabel)

        fit_params = deepcopy(self.fit_default_params_multilabel)
        predict_params = deepcopy(self.predict_default_params_multilabel)
        if predict_params is None:
            predict_params = {"X": fit_params["X"]}

        return init_params, fit_params, predict_params

    def _assert_multilabel_predictions(self, y_pred, classes):
        y_pred = np.asarray(y_pred)
        self.assertEqual(y_pred.ndim, 2)
        self.assertEqual(y_pred.shape[1], len(classes))
        assert_predicts_class_dtype(self, y_pred, classes)
        for j, classes_j in enumerate(classes):
            self.assertTrue(np.isin(y_pred[:, j], classes_j).all())

    def _build_invalid_multilabel_targets(self, y, missing_label):
        y_invalid = np.array(y, copy=True)
        y_invalid = np.atleast_2d(y_invalid)
        y_invalid[0, 0] = missing_label
        return y_invalid

    @staticmethod
    def _target_param_name(method):
        method_params = inspect.signature(method).parameters
        return (
            "Y" if "Y" in method_params and "y" not in method_params else "y"
        )

    def _call_with_target(
        self, estimator, method_name, params, **extra_params
    ):
        method = getattr(estimator, method_name)
        call_params = deepcopy(params)
        target_param = self._target_param_name(method)
        if target_param != "y" and "y" in call_params:
            call_params[target_param] = call_params.pop("y")
        call_params.update(extra_params)
        return method(**call_params)

    def test_init_param_missing_label(
        self, test_cases=None, replace_init_params=None
    ):
        test_cases = [] if test_cases is None else test_cases
        replace_init_params = (
            {} if replace_init_params is None else replace_init_params
        )
        test_cases += [(np.nan, TypeError), ("nan", None), (1, TypeError)]
        replace_init_params["classes"] = ["tokyo", "paris"]
        replace_fit_params = {
            "y": ["tokyo", "nan", "paris"],
            "X": np.zeros((3, 1)),
        }
        super().test_init_param_missing_label(
            test_cases,
            replace_init_params=replace_init_params,
            replace_fit_params=replace_fit_params,
        )

        test_cases = [("state", TypeError), (-1, None)]
        replace_init_params["classes"] = [0, 1]
        replace_fit_params = {"y": [0, -1, 1], "X": np.zeros((3, 1))}
        self._test_param(
            "init",
            "missing_label",
            test_cases,
            replace_init_params=replace_init_params,
            replace_fit_params=replace_fit_params,
        )

        test_cases = [("state", TypeError), (None, None)]
        replace_init_params["classes"] = [0, 1]
        replace_fit_params = {"y": [0, None, 1], "X": np.zeros((3, 1))}
        self._test_param(
            "init",
            "missing_label",
            test_cases,
            replace_init_params=replace_init_params,
            replace_fit_params=replace_fit_params,
            # id_offset=5,
        )

        test_cases = [
            ("state", TypeError),
            (0.0, self.non_integral_classes_error),
        ]
        replace_init_params["classes"] = [0.5, 1.4]
        replace_fit_params = {"y": [0.5, 0, 1.4], "X": np.zeros((3, 1))}
        self._test_param(
            "init",
            "missing_label",
            test_cases,
            replace_init_params=replace_init_params,
            replace_fit_params=replace_fit_params,
        )

        test_cases = [("state", TypeError), (None, ValueError), (np.nan, None)]
        replace_init_params["classes"] = [0, 1]
        replace_fit_params = {"y": [0, np.nan, 1], "X": np.zeros((3, 1))}
        self._test_param(
            "init",
            "missing_label",
            test_cases,
            replace_init_params=replace_init_params,
            replace_fit_params=replace_fit_params,
        )

    def test_init_param_classes(self, test_cases=None):
        test_cases = [] if test_cases is None else test_cases
        test_cases += [
            (np.nan, TypeError),
            ([1, 2], TypeError),
            (["tokyo", "paris"], None),
        ]
        replace_init_params = {"missing_label": "nan"}
        replace_fit_params = {
            "y": ["tokyo", "nan", "paris"],
            "X": np.zeros((3, 1)),
        }
        self._test_param(
            "init",
            "classes",
            test_cases,
            replace_init_params=replace_init_params,
            replace_fit_params=replace_fit_params,
        )
        test_cases = [([1, 2], None), (["tokyo", "paris"], TypeError)]
        replace_init_params = {"missing_label": -1}
        replace_fit_params = {"y": [2, -1, 1], "X": np.zeros((3, 1))}
        self._test_param(
            "init",
            "classes",
            test_cases,
            replace_init_params=replace_init_params,
            replace_fit_params=replace_fit_params,
        )

    def test_init_param_cost_matrix(self):
        test_cases = []
        replace_init_params = {
            "classes": ["tokyo", "paris"],
            "missing_label": "nan",
        }
        replace_fit_params = {
            "y": ["tokyo", "tokyo", "paris"],
            "X": np.zeros((3, 1)),
        }
        test_cost_matrix = 1 - np.eye(len(np.unique(replace_fit_params["y"])))
        test_cases += [
            (None, None),
            (-1, ValueError),
            ([], ValueError),
            (test_cost_matrix, None),
        ]
        self._test_param(
            "init",
            "cost_matrix",
            test_cases,
            replace_init_params=replace_init_params,
            replace_fit_params=replace_fit_params,
        )
        test_cases = [(replace_fit_params["X"], None)]
        replace_init_params["cost_matrix"] = 1 - np.eye(
            len(np.unique(replace_fit_params["y"]))
        )
        self._test_param(
            "predict",
            "X",
            test_cases,
            replace_init_params=replace_init_params,
            replace_fit_params=replace_fit_params,
            # id_offset=4,
        )

    def test_fit_param_X(self, test_cases=None):
        super().test_fit_param_X(test_cases)
        test_cases = [([], None)]
        replace_fit_params = {"y": []}
        replace_init_params = {"classes": [0, 1], "missing_label": -1}
        self._test_param(
            "fit",
            "X",
            test_cases,
            replace_init_params=replace_init_params,
            replace_fit_params=replace_fit_params,
        )
        test_cases = [([], ValueError)]
        replace_init_params["classes"] = None
        self._test_param(
            "fit",
            "X",
            test_cases,
            replace_init_params=replace_init_params,
            replace_fit_params=replace_fit_params,
        )

    def test_fit_param_y(self, test_cases=None):
        test_cases = [] if test_cases is None else test_cases
        test_cases += [
            ([0, 1, 2], TypeError),
            (["tokyo", "nan", "paris"], None),
        ]
        replace_init_params = {
            "classes": ["tokyo", "paris"],
            "missing_label": "nan",
        }
        replace_fit_params = {"X": np.zeros((3, 1))}
        super().test_fit_param_y(
            test_cases,
            replace_init_params=replace_init_params,
            replace_fit_params=replace_fit_params,
        )
        test_cases = [
            ([0, 1, 1], None),
            (["tokyo", "nan", "paris"], TypeError),
        ]
        replace_init_params = {"classes": [0, 1], "missing_label": -1}
        replace_fit_params = {"X": np.zeros((3, 1))}
        self._test_param(
            "fit",
            "y",
            test_cases,
            replace_init_params=replace_init_params,
            replace_fit_params=replace_fit_params,
        )

    def test_partial_fit_param_y(self, test_cases=None):
        if hasattr(self, "partial_fit"):
            test_cases = [] if test_cases is None else test_cases
            test_cases += [
                ([0, 1, 2], TypeError),
                (["tokyo"], ValueError),
                (["nan", "nan", "nan"], None),
            ]
            replace_init_params = {
                "classes": ["tokyo", "paris"],
                "missing_label": "nan",
            }
            replace_fit_params = {"X": np.zeros((3, 1))}
            super().test_partial_fit_param_y(
                test_cases,
                replace_init_params=replace_init_params,
                replace_fit_params=replace_fit_params,
            )
            test_cases = [
                ([0, 1, 2], None),
                (["nan", "nan", "nan"], TypeError),
            ]
            replace_init_params = {"classes": [0, 1, 2], "missing_label": -1}
            replace_fit_params = {"X": np.zeros((3, 1))}
            self._test_param(
                "partial_fit",
                "y",
                test_cases,
                replace_init_params=replace_init_params,
                replace_fit_params=replace_fit_params,
                exclude_fit=True,
            )

    def test_predict_proba_param_X(self, test_cases=None):
        if hasattr(self.estimator_class, "predict_proba"):
            test_cases = [] if test_cases is None else test_cases
            test_cases += [
                (np.nan, ValueError),
                ([[1, 2, 3]], ValueError),
                (self.fit_default_params["X"], None),
            ]
            replace_init_params = {
                "classes": ["tokyo", "paris"],
                "missing_label": "nan",
            }
            replace_fit_params = {
                "X": np.zeros((3, 1)),
                "y": ["tokyo", "nan", "paris"],
            }
            self._test_param(
                "predict_proba",
                "X",
                test_cases,
                extras_params=self.predict_default_params,
                replace_init_params=replace_init_params,
                replace_fit_params=replace_fit_params,
            )

    def test_param_test_availability(self):
        not_test = ["self"]
        check_test_param_test_availability(
            self, self.estimator_class.predict_proba, "predict_proba", not_test
        )
        super().test_param_test_availability()

    def test_fit_multilabel(self):
        if not self._has_multilabel_defaults():
            return

        init_params, fit_params, predict_params = self._get_multilabel_params()
        estimator = self.estimator_class(**init_params)
        self._call_with_target(estimator, "fit", fit_params)
        y_pred = estimator.predict(**predict_params)
        self._assert_multilabel_predictions(y_pred, estimator.classes_)

    def test_fit_multilabel_invalid_rows(self):
        if not self._has_multilabel_defaults():
            return

        init_params, fit_params, _ = self._get_multilabel_params()
        estimator = self.estimator_class(**init_params)
        y_invalid = self._build_invalid_multilabel_targets(
            fit_params["y"], init_params["missing_label"]
        )
        fit_params["y"] = y_invalid
        with self.assertRaises(ValueError):
            self._call_with_target(estimator, "fit", fit_params)

    def test_fit_param_sample_weight_multilabel(self):
        if not self._has_multilabel_defaults():
            return

        fit_params = inspect.signature(self.estimator_class.fit).parameters
        if "sample_weight" not in fit_params:
            return

        init_params, fit_params, _ = self._get_multilabel_params()
        estimator = self.estimator_class(**init_params)
        y = np.asarray(fit_params["y"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._call_with_target(
                estimator,
                "fit",
                fit_params,
                sample_weight=np.ones(len(y), dtype=float),
            )

    def test_partial_fit_multilabel(self):
        if not self._has_multilabel_defaults():
            return

        if not hasattr(self.estimator_class, "partial_fit"):
            return

        init_params, fit_params, predict_params = self._get_multilabel_params()
        estimator = self.estimator_class(**init_params)
        self._call_with_target(estimator, "partial_fit", fit_params)
        y_pred = estimator.predict(**predict_params)
        self._assert_multilabel_predictions(y_pred, estimator.classes_)

    def test_score_multilabel(self):
        if not self._has_multilabel_defaults():
            return

        init_params, fit_params, predict_params = self._get_multilabel_params()
        estimator = self.estimator_class(**init_params)
        self._call_with_target(estimator, "fit", fit_params)
        X = predict_params["X"]
        y_true = fit_params["y"]
        y_pred = estimator.predict(X)
        expected_score = accuracy_score(y_true=y_true, y_pred=y_pred)
        self.assertAlmostEqual(estimator.score(X, y_true), expected_score)


class TemplateMultiAnnotatorClassifier:
    """Shared resolved-target contract for multi-annotator classifiers."""

    def test_init_param_target_type(self):
        for target_type, error in [
            ("auto", None),
            ("single-output", None),
            ("multi-label", ValueError),
            ("multi-output", ValueError),
            ("invalid", ValueError),
        ]:
            with self.subTest(target_type=target_type):
                estimator = clone(
                    self.target_contract_estimator_factory()
                ).set_params(target_type=target_type)
                if error is None:
                    estimator.fit(**deepcopy(self.target_contract_fit_params))
                else:
                    with self.assertRaises(error):
                        estimator.fit(
                            **deepcopy(self.target_contract_fit_params)
                        )

    def test_predict_dtype_matches_class_dtype(self):
        """Check that `predict` returns the declared class dtype.

        Multi-annotator classifiers decode their predictions through the
        same label encoder as single-annotator ones, so their predictions
        are narrowed back to the declared class dtype as well.
        """
        estimator = self.target_contract_estimator_factory()
        fit_params = deepcopy(self.target_contract_fit_params)
        estimator.fit(**fit_params)

        y_pred = np.asarray(estimator.predict(fit_params["X"]))

        assert_predicts_class_dtype(self, y_pred, estimator.classes_)

    def test_multiannotator_target_contract(self):
        estimator = self.target_contract_estimator_factory()
        fit_params = deepcopy(self.target_contract_fit_params)

        self.assertEqual(estimator.target_type, "auto")
        self.assertEqual(
            estimator._target_capabilities,
            frozenset(
                {
                    (
                        "classification",
                        "single-output",
                        "multi-annotator",
                    )
                }
            ),
        )
        estimator.fit(**fit_params)
        self.assertEqual(estimator.target_spec_.task, "classification")
        self.assertEqual(estimator.target_spec_.target_type, "single-output")
        self.assertEqual(
            estimator.target_spec_.annotation_type, "multi-annotator"
        )

        unsupported = clone(estimator).set_params(target_type="multi-label")
        with self.assertRaisesRegex(
            ValueError, "Multi-label targets cannot be combined"
        ):
            unsupported.fit(**fit_params)
        self.assertFalse(hasattr(unsupported, "target_spec_"))
        self.assertFalse(hasattr(unsupported, "classes_"))


class TemplateClassFrequencyEstimator(TemplateSkactivemlClassifier):
    def test_init_param_class_prior(self):
        test_cases = []
        test_cases += [
            (1, None),
            (None, TypeError),
            ([], ValueError),
            ([0, 1], None),
            ([0], ValueError),
        ]
        replace_init_params = {
            "classes": ["tokyo", "paris"],
            "missing_label": "nan",
        }
        replace_fit_params = {
            "X": np.zeros((3, 1)),
            "y": ["tokyo", "nan", "paris"],
        }
        self._test_param(
            "init",
            "class_prior",
            test_cases,
            replace_init_params=replace_init_params,
            replace_fit_params=replace_fit_params,
        )

    def test_param_test_availability(self):
        not_test = ["self"]
        # Check if predict_freq is being tested.
        check_test_param_test_availability(
            self, self.estimator_class.predict_freq, "predict_freq", not_test
        )
        super().test_param_test_availability()

    def test_predict_freq_param_X(self, test_cases=None):
        test_cases = [] if test_cases is None else test_cases
        test_cases += [
            (np.nan, ValueError),
            ([[1, 2, 3]], ValueError),
            (self.fit_default_params["X"], None),
        ]
        replace_init_params = {
            "classes": ["tokyo", "paris"],
            "missing_label": "nan",
        }
        replace_fit_params = {
            "X": np.zeros((3, 1)),
            "y": ["tokyo", "nan", "paris"],
        }
        self._test_param(
            "predict_freq",
            "X",
            test_cases,
            extras_params=self.predict_default_params,
            replace_init_params=replace_init_params,
            replace_fit_params=replace_fit_params,
        )

    def test_sample_proba(self):
        # Setup test cases.
        X, y_full = make_blobs(n_samples=200, centers=4, random_state=0)
        classes = np.unique(y_full)
        pwc = self.estimator_class(
            classes=classes, class_prior=1, missing_label=-1
        )
        y_missing = np.full_like(y_full, fill_value=-1)
        y_partial_missing = y_full.copy()
        y_partial_missing[30:50] = -1
        y_class_0_missing = y_full.copy()
        y_class_0_missing[y_full == 0] = -1

        for y in [y_missing, y_partial_missing, y_class_0_missing, y_full]:
            pwc.fit(X, y)

            for n_samples in [1, 10]:
                # Check shape of probabilities.
                P_sampled = pwc.sample_proba(X, n_samples=n_samples)
                shape_Expected = [n_samples, len(X), len(classes)]
                np.testing.assert_array_equal(P_sampled.shape, shape_Expected)

                # Check normalization of probabilities.
                P_sums = P_sampled.sum(axis=-1)
                P_sums_expected = np.ones_like(P_sums)
                np.testing.assert_allclose(P_sums, P_sums_expected)

        # Check value error if `alphas` as input to dirichlet are zero.
        pwc = self.estimator_class(
            classes=np.unique(y_full), class_prior=0, missing_label=-1
        )
        pwc.fit(X, y_missing)
        self.assertRaises(ValueError, pwc.sample_proba, X=X, n_samples=10)

        pwc.fit(X, y_class_0_missing)
        self.assertRaises(ValueError, pwc.sample_proba, X=X, n_samples=10)


class TemplateSkactivemlRegressor(TemplateEstimator):
    def setUp(
        self,
        estimator_class,
        init_default_params,
        fit_default_params=None,
        predict_default_params=None,
    ):
        super().setUp(
            estimator_class,
            init_default_params,
            fit_default_params,
            predict_default_params,
        )

    def test_init_param_missing_label(self, test_cases=None):
        test_cases = [] if test_cases is None else test_cases
        test_cases += [(1.2, None)]
        # TODO: check_missing_label is only used to check if missing_label
        # is correct but strings are therfore also accepted
        # after fixing this issue add test below again.
        # ("nan", TypeError),
        super().test_init_param_missing_label(test_cases)

    def test_init_param_target_type(self):
        self._test_param(
            "init",
            "target_type",
            [
                ("auto", None),
                ("single-output", None),
                ("multi-output", ValueError),
                ("multi-label", ValueError),
                ("invalid", ValueError),
            ],
        )

    def test_fitted_target_spec_is_single_output_regression(self):
        estimator = self.estimator_class(**self.init_default_params)

        estimator.fit(**self.fit_default_params)

        self.assertEqual(
            estimator._target_capabilities,
            frozenset({("regression", "single-output", "single-annotator")}),
        )
        self.assertEqual(estimator.target_spec_.task, "regression")
        self.assertEqual(estimator.target_spec_.target_type, "single-output")
        self.assertEqual(
            estimator.target_spec_.annotation_type, "single-annotator"
        )
        self.assertIsNone(estimator.target_spec_.classes)

    def test_predict_returns_one_dimensional_targets(
        self, replace_init_params=None, replace_fit_params=None
    ):
        """Check that `predict` describes one target value per sample.

        A single-output regression target is described by one value per
        sample. A wrapped estimator may emit a column instead, which every
        consumer indexing predictions by sample then misreads.
        """
        init_params = deepcopy(self.init_default_params)
        if replace_init_params is not None:
            init_params.update(deepcopy(replace_init_params))

        fit_params = deepcopy(self.fit_default_params)
        if replace_fit_params is not None:
            fit_params.update(deepcopy(replace_fit_params))

        predict_params = deepcopy(self.predict_default_params)

        estimator = self.estimator_class(**init_params)
        estimator.fit(**fit_params)
        y_pred = estimator.predict(**predict_params)

        assert_predicts_single_output_shape(self, y_pred, predict_params["X"])

    def test_fit_param_X(self, test_cases=None):
        super().test_fit_param_X(test_cases)
        test_cases = [([], None)]
        replace_fit_params = {"y": []}
        self._test_param(
            "fit",
            "X",
            test_cases,
            replace_fit_params=replace_fit_params,
        )

    def test_fit_param_y(self, test_cases=None):
        test_cases = [] if test_cases is None else test_cases
        test_cases += [
            ([np.nan] * len(self.fit_default_params["X"]), None),
            ([np.nan] + [1.0] * (len(self.fit_default_params["X"]) - 1), None),
        ]
        replace_init_params = {"missing_label": np.nan}
        super().test_fit_param_y(
            test_cases,
            replace_init_params=replace_init_params,
        )
        # TODO: Test schould throw a TypeError
        # Wrapper classes are failing because of
        # "numpy.core._exceptions._UFuncNoLoopError: ufunc 'add' did not
        # contain a loop with signature matching types
        # (dtype('<U32'), dtype('<U32')) -> None"

        # test_cases = [([1.0, "nan", 0.9], None)]
        replace_init_params = {"missing_label": "nan"}
        self._test_param(
            "fit",
            "y",
            test_cases,
            replace_init_params=replace_init_params,
        )


class TemplateProbabilisticRegressor(TemplateSkactivemlRegressor):
    def setUp(
        self,
        estimator_class,
        init_default_params,
        fit_default_params=None,
        predict_default_params=None,
    ):
        super().setUp(
            estimator_class,
            init_default_params,
            fit_default_params,
            predict_default_params,
        )

    def test_param_test_availability(self):
        not_test = ["self"]
        # Check if predict_target_distribution is being tested.
        check_test_param_test_availability(
            self,
            self.estimator_class.predict_target_distribution,
            "predict_target_distribution",
            not_test,
        )
        super().test_param_test_availability()

    def test_predict_target_distribution_param_X(self):
        test_cases = []
        X = np.array([[0, 1], [1, 0], [2, 3]])
        test_cases += [
            (X, None),
            ("Test", ValueError),
            (np.array([[0], [1]]), ValueError),
        ]
        replace_fit_params = {"X": X}
        extras_params = deepcopy(self.predict_default_params)
        self._test_param(
            "predict_target_distribution",
            "X",
            test_cases,
            extras_params=extras_params,
            replace_fit_params=replace_fit_params,
        )

    def test_predict_param_return_std(self):
        test_cases = []
        test_cases += [(True, None), (1.0, TypeError), ("Test", TypeError)]
        self._test_param(
            "predict",
            "return_std",
            test_cases,
            extras_params=self.predict_default_params,
        )

    def test_predict_param_return_entropy(self):
        test_cases = []
        test_cases += [(True, None), (1.0, TypeError), ("Test", TypeError)]
        self._test_param(
            "predict",
            "return_entropy",
            test_cases,
            extras_params=self.predict_default_params,
        )

    def test_sample_y_param_X(self, test_cases=None):
        test_cases = [] if test_cases is None else test_cases
        test_cases += [
            (np.nan, ValueError),
            ([1], ValueError),
            (np.ones((3, 2)), ValueError),
            (np.zeros((3, 1)), None),
        ]
        extras_params = {"X": [[1]]}
        replace_fit_params = {"X": np.zeros((3, 1)), "y": [0.5, 0.6, np.nan]}
        self._test_param(
            "sample_y",
            "X",
            test_cases,
            extras_params=extras_params,
            replace_fit_params=replace_fit_params,
        )

    def test_sample_y_param_n_samples(self, test_cases=None):
        test_cases = [] if test_cases is None else test_cases
        test_cases += [(-1, ValueError), (1.1, TypeError), (1, None)]
        extras_params = {"X": [[1]]}
        replace_fit_params = {"X": np.zeros((3, 1)), "y": [0.5, 0.6, np.nan]}
        self._test_param(
            "sample_y",
            "n_samples",
            test_cases,
            extras_params=extras_params,
            replace_fit_params=replace_fit_params,
        )

    def test_sample_y_param_random_state(self, test_cases=None):
        test_cases = [] if test_cases is None else test_cases
        test_cases += [(np.nan, ValueError), ("state", ValueError), (1, None)]
        extras_params = {"X": [[1]]}
        replace_fit_params = {"X": np.zeros((3, 1)), "y": [0.5, 0.6, np.nan]}
        self._test_param(
            "sample_y",
            "random_state",
            test_cases,
            extras_params=extras_params,
            replace_fit_params=replace_fit_params,
        )
