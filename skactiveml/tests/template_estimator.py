
#
import inspect
from copy import deepcopy

import numpy as np
from skactiveml.utils import (
    MISSING_LABEL,
)


# TODO: scipy auf 1.9 für python 3.11 und parralel processes auf 5
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
    ):
        self.super_setUp_has_been_executed = True
        self.estimator_class = estimator_class

        self.init_default_params = {"random_state": 42, "missing_label": MISSING_LABEL}
        for key, val in init_default_params.items():
            self.init_default_params[key] = val

        init_params = inspect.signature(self.estimator_class.__init__).parameters
        for key, val in init_params.items():
            if (
                key != "self"
                and val.default == inspect._empty
                and key not in self.init_default_params
            ):
                raise ValueError(
                    f"Missing positional argument `{key}` of `__init__` in "
                    f"`init_default_kwargs`."
                )

        self.fit_default_params = fit_default_params
        self.predict_default_params = predict_default_params
        
        

        fit_params = inspect.signature(self.estimator_class.fit).parameters
        kwargs_var_keyword = list(
            filter(lambda p: p.kind == p.VAR_KEYWORD, fit_params.values())
        )
        # and val not in kwargs_var_keyword

        for key, val in fit_params.items():
            if (
                key != "self"
                and val not in kwargs_var_keyword
                and val.default == inspect._empty
                and self.fit_default_params is not None
                and key not in self.fit_default_params
            ):
                raise ValueError(
                    f"Missing positional argument `{key}` of `fit` in "
                    f"`fit_default_kwargs`."
                )
        # TODO kwargs generell aussortieren in call func wie gemacht werden soll
        predict_params = inspect.signature(self.estimator_class.predict).parameters
        kwargs_var_keyword = list(
            filter(lambda p: p.kind == p.VAR_KEYWORD, predict_params.values())
        )
        # and val not in kwargs_var_keyword
        for key, val in predict_params.items():
            if (
                key != "self"
                and val not in kwargs_var_keyword
                and val.default == inspect._empty
                and self.predict_default_params is not None
                and key not in self.predict_default_params
            ):
                raise ValueError(
                    f"Missing positional argument `{key}` of `predict` in "
                    f"`predict_default_kwargst`."
                )

    def test_init_param_missing_label(self, test_cases=None, replace_init_params=None):
        test_cases = [] if test_cases is None else test_cases
        ml = self.init_default_params["missing_label"]
        test_cases += [(ml, None), (Dummy, TypeError)]
        self._test_param("init", "missing_label", test_cases, replace_init_params=replace_init_params)

    def test_init_param_random_state(self, test_cases=None):
        test_cases = [] if test_cases is None else test_cases
        test_cases += [(np.nan, ValueError), ("state", ValueError), (1, None)]
        self._test_param("init", "random_state", test_cases)

    def test_fit_param_X(self, test_cases=None):
        test_cases = [] if test_cases is None else test_cases
        test_cases += [(np.nan, ValueError), ([1], ValueError), (np.zeros((3, 1)), None)]
        self._test_param("fit", "X", test_cases)

    def test_fit_param_y(self, test_cases=None, replace_init_params=None):
        test_cases = [] if test_cases is None else test_cases
        test_cases += [(np.nan, TypeError), ("state", TypeError)]
        self._test_param("fit", "y", test_cases, replace_init_params=replace_init_params)

    def test_fit_param_sample_weight(self, test_cases=None):
        fit_params = inspect.signature(self.estimator_class.fit).parameters
        if "sample_weight" in fit_params:
            test_cases = [] if test_cases is None else test_cases
            test_cases += [(np.nan, ValueError), (1, ValueError), (np.array([0.5 , 0.0, 1.0]), None)]
            self._test_param("fit", "sample_weight", test_cases)

    def test_partial_fit_param_X(self, test_cases=None, replace_init_params=None, extras_params=None):
        if hasattr(self.estimator_class, "partial_fit"):
            test_cases = [] if test_cases is None else test_cases
            test_cases += [(np.nan, ValueError), ([1], ValueError), (np.zeros((3, 1)), None)]
            self._test_param("partial_fit", "X", test_cases, replace_init_params=replace_init_params, extras_params=extras_params, exclude_fit=True)

    def test_partial_fit_param_y(self, test_cases=None, replace_init_params=None, extras_params=None):
        if hasattr(self.estimator_class, "partial_fit"):
            test_cases = [] if test_cases is None else test_cases
            test_cases += [(np.nan, TypeError), ("state", TypeError)]
            self._test_param("partial_fit", "y", test_cases, replace_init_params=replace_init_params, extras_params=extras_params, exclude_fit=True)

    def test_partial_fit_param_sample_weight(self, test_cases=None, replace_init_params=None, extras_params=None):
        if hasattr(self.estimator_class, "partial_fit"):
            fit_params = inspect.signature(self.estimator_class.fit).parameters
            if "sample_weight" in fit_params:
                test_cases = [] if test_cases is None else test_cases
                test_cases += [(np.nan, ValueError), (1, ValueError), (np.array([0.5 , 0.0, 1.0]), None)]
                self._test_param("partial_fit", "sample_weight", test_cases, replace_init_params=replace_init_params, extras_params=extras_params, exclude_fit=True)

    def test_predict_param_X(self, test_cases=None):
        test_cases = [] if test_cases is None else test_cases
        test_cases += [(np.nan, ValueError), ("state", ValueError), ([[1]], None)]
        self._test_param("predict", "X", test_cases)

    def test_init_param_test_assignments(self):
        for param in inspect.signature(self.estimator_class.__init__).parameters:
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
        not_test = ["self", "kwargs"]

        # Get initial parameters.
        init_params = inspect.signature(self.estimator_class.__init__).parameters
        kwargs_var_keyword = list(
            filter(lambda p: p.kind == p.VAR_KEYWORD, init_params.values())
        )

        # Check init parameters.
        for param, val in init_params.items():
            if param in not_test or val in kwargs_var_keyword:
                continue
            test_func_name = "test_init_param_" + param
            with self.subTest(msg=test_func_name):
                self.assertTrue(
                    hasattr(self, test_func_name),
                    msg=f"'{test_func_name}()' missing in {self.__class__}",
                )

        # Get fit parameters.
        fit_params = inspect.signature(self.estimator_class.fit).parameters
        kwargs_var_keyword = list(
            filter(lambda p: p.kind == p.VAR_KEYWORD, fit_params.values())
        )

        # Check fit parameters.
        for param, val in fit_params.items():
            if param in not_test or val in kwargs_var_keyword:
                continue
            test_func_name = "test_fit_param_" + param
            with self.subTest(msg=test_func_name):
                self.assertTrue(
                    hasattr(self, test_func_name),
                    msg=f"'{test_func_name}()' missing in {self.__class__}",
                )

        # Get predict parameters.
        predict_params = inspect.signature(self.estimator_class.predict).parameters
        kwargs_var_keyword = list(
            filter(lambda p: p.kind == p.VAR_KEYWORD, predict_params.values())
        )

        # Check predict parameters.
        for param, val in predict_params.items():
            if param in not_test or val in kwargs_var_keyword:
                continue
            test_func_name = "test_predict_param_" + param
            with self.subTest(msg=test_func_name):
                self.assertTrue(
                    hasattr(self, test_func_name),
                    msg=f"'{test_func_name}()' missing in {self.__class__}",
                )

        if hasattr(self.estimator_class, "predict_proba"):
            # Get predict_proba parameters.
            predict_proba_params = inspect.signature(self.estimator_class.predict_proba).parameters
            kwargs_var_keyword = list(
                filter(lambda p: p.kind == p.VAR_KEYWORD, predict_proba_params.values())
            )

            # Check predict parameters.
            for param, val in predict_proba_params.items():
                if param in not_test or val in kwargs_var_keyword:
                    continue
                test_func_name = "test_predict_param_" + param
                with self.subTest(msg=test_func_name):
                    self.assertTrue(
                        hasattr(self, test_func_name),
                        msg=f"'{test_func_name}()' missing in {self.__class__}",
                    )

        # Check if fit is being tested.
        with self.subTest(msg="test_fit"):
            self.assertTrue(
                hasattr(self, "test_fit"),
                msg=f"'test_fit' missing in {self.__class__}",
            )

        # Check if predict is being tested.
        with self.subTest(msg="test_predict"):
            self.assertTrue(
                hasattr(self, "test_predict"),
                msg=f"'test_predict' missing in {self.__class__}",
            )
    
    def _test_param(
        self,
        test_func,
        test_param,
        test_cases,
        replace_init_params=None,
        replace_fit_params=None,
        replace_extras_params=None,
        extras_params=None,
        exclude_fit=False,
    ):
        if replace_init_params is None:
            replace_init_params = {}
        if replace_fit_params is None:
            replace_fit_params = {}
        if replace_extras_params is None:
            replace_extras_params = {}
        if extras_params is None:
            extras_params = {}

        for i, (test_val, err) in enumerate(test_cases):
            with self.subTest(msg="Param", id=i, val=test_val):
                init_params = deepcopy(self.init_default_params)
                for key, val in replace_init_params.items():
                    init_params[key] = val

                fit_params = deepcopy(self.fit_default_params)
                for key, val in replace_fit_params.items():
                    fit_params[key] = val

                if f"{test_func}_params" in locals():
                    locals()[f"{test_func}_params"][test_param] = test_val
                else:
                    if test_func == "predict":
                        extras_params = deepcopy(self.predict_default_params)
                    for key, val in replace_extras_params.items():
                        extras_params[key] = val
                    extras_params[test_param] = test_val

                estimator = self.estimator_class(**init_params)
                if err is None and test_func in ["fit", "init"]:
                    estimator.fit(**fit_params)
                elif test_func in ["fit", "init"]:
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

    def test_init_param_missing_label(self, test_cases=None, replace_init_params=None):
        test_cases = [] if test_cases is None else test_cases
        replace_init_params = {} if replace_init_params is None else replace_init_params
        test_cases += [(np.nan, TypeError), ("nan", None),  (1, TypeError)]
        super().test_init_param_missing_label(test_cases)
        test_cases = [("state", TypeError), (-1, None)]
        replace_init_params["classes"] = [0, 1]
        replace_fit_params = {"y": [0, 0, 1]}
        self._test_param("init", "missing_label", test_cases, replace_init_params=replace_init_params, replace_fit_params=replace_fit_params)

    def test_init_param_classes(self, test_cases=None):
        test_cases = [] if test_cases is None else test_cases
        test_cases += [(np.nan, TypeError), ([1,2], TypeError), (["tokyo", "paris"], None)]
        self._test_param("init", "classes", test_cases)

    def test_init_param_cost_matrix(self):
        test_cases = []
        replace_init_params = {"classes": ["tokyo", "paris"]}
        test_cases += [(None, None), (-1, ValueError), ([], ValueError), (1 - np.eye(3), None)]
        self._test_param("init", "cost_matrix", test_cases, replace_init_params=replace_init_params)

    def test_fit_param_y(self, test_cases=None):
        test_cases = [] if test_cases is None else test_cases
        test_cases += [([0,1,2], TypeError), (["tokyo", "nan", "paris"], None)]
        replace_init_params = {"classes": ["tokyo", "paris"]}
        return super().test_fit_param_y(test_cases, replace_init_params=replace_init_params) 
    
    def test_partial_fit_param_X(self, test_cases=None, extras_params=None):
        extras_params = deepcopy(self.fit_default_params)
        return super().test_partial_fit_param_X(test_cases, extras_params=extras_params)

    def test_partial_fit_param_y(self, test_cases=None):
        test_cases = [] if test_cases is None else test_cases
        test_cases += [([0,1,2], TypeError), (["tokyo"], ValueError), [["nan", "nan", "nan"], None]]
        replace_init_params = {"classes": ["tokyo", "paris"]}
        extras_params = deepcopy(self.fit_default_params)
        return super().test_partial_fit_param_y(test_cases, replace_init_params=replace_init_params, extras_params=extras_params) 
    
    def test_partial_fit_param_sample_weight(self, test_cases=None, extras_params=None):
        extras_params = deepcopy(self.fit_default_params)
        return super().test_partial_fit_param_sample_weight(test_cases, extras_params=extras_params)

    def test_predict_proba_param_X(self, test_cases=None):
        if hasattr(self.estimator_class, "predict_proba"):
            test_cases = [] if test_cases is None else test_cases
            test_cases += [(np.nan, ValueError), ([[1,2,3]], ValueError), ([[1],[2],[3]], None)]
            self._test_param("predict_proba", "X", test_cases, extras_params=self.predict_default_params)

    def test_param_test_availability(self):
        with self.subTest(msg="test_predict_proba"):
            self.assertTrue(
                hasattr(self, "test_predict_proba"),
                msg=f"'test_predict_proba' missing in {self.__class__}",
            )
        return super().test_param_test_availability()


class TemplateClassFrequencyEstimator(TemplateSkactivemlClassifier):
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

    def test_init_param_class_prior(self):
        test_cases = []
        test_cases += [(1, None), (None, TypeError), ([], ValueError), ([0, 1], None)]
        self._test_param("init", "class_prior", test_cases)

    def test_param_test_availability(self):
        # Check if predict_proba is being tested.
        with self.subTest(msg="test_predict_freq"):
            self.assertTrue(
                hasattr(self, "test_predict_freq"),
                msg=f"'test_predict_freq' missing in {self.__class__}",
            )
        return super().test_param_test_availability()
    
    def test_predict_freq_param_X(self, test_cases=None):
        if hasattr(self.estimator_class, "predict_freq"):
            test_cases = [] if test_cases is None else test_cases
            test_cases += [(np.nan, ValueError), ([[1,2,3]], ValueError), ([[1],[2],[3]], None)]
            self._test_param("predict_freq", "X", test_cases, extras_params=self.predict_default_params)


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
        test_cases = [(1.2, None)] 
        #TODO: check_missing_label is only used to check if missing_label
        # is correct but strings are therfore also accepted
        # after fixing this issue add test below again.
        # ("nan", TypeError),
        return super().test_init_param_missing_label(test_cases)

    def test_fit_param_y(self, test_cases=None):
        test_cases = [] if test_cases is None else test_cases
        test_cases += [([np.nan,np.nan,np.nan], None), ("state", TypeError), (np.nan, TypeError), ([1.0, 1.1, 0.9], None)]
        return super().test_fit_param_y(test_cases)
    
    def test_partial_fit_param_X(self, test_cases=None, replace_init_params=None, extras_params=None):
        extras_params = deepcopy(self.fit_default_params)
        return super().test_partial_fit_param_X(test_cases, replace_init_params=replace_init_params, extras_params=extras_params)

    def test_partial_fit_param_y(self,test_cases=None, replace_init_params=None):
        test_cases = [] if test_cases is None else test_cases
        extras_params = deepcopy(self.fit_default_params)
        return super().test_partial_fit_param_y(test_cases, replace_init_params=replace_init_params, extras_params=extras_params) 
    
    def test_partial_fit_param_sample_weight(self, test_cases=None, replace_init_params=None, extras_params=None):
        extras_params = deepcopy(self.fit_default_params)
        return super().test_partial_fit_param_sample_weight(test_cases, extras_params=extras_params, replace_init_params=replace_init_params)


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
        # Check if predict_proba is being tested.
        with self.subTest(msg="test_predict_target_distribution"):
            self.assertTrue(
                hasattr(self, "test_predict_target_distribution"),
                msg=f"'test_predict_target_distribution' missing in {self.__class__}",
            )
        return super().test_param_test_availability()

    def test_predict_target_distribution_X(self):
        test_cases = []
        X = np.array([[0, 1], [1, 0], [2, 3]])
        test_cases += [(X, None), ("Test", ValueError)]
        extras_params = deepcopy(self.predict_default_params)
        self._test_param("predict_target_distribution", "X", test_cases, extras_params=extras_params)

    def test_predict_param_return_std(self):
        test_cases = []
        test_cases += [(True, None), (1.0, TypeError), ("Test", TypeError)]
        self._test_param("predict", "return_std", test_cases)

    def test_predict_param_return_entropy(self):
        test_cases = []
        test_cases += [(True, None), (1.0, TypeError), ("Test", TypeError)]
        self._test_param("predict", "return_entropy", test_cases)

    def test_sample_y_param_X(self, test_cases=None):
        test_cases = [] if test_cases is None else test_cases
        test_cases += [(np.nan, ValueError), ([1], ValueError), (np.zeros((3, 1)), None)]
        extras_params = {"X": [[1]]}
        self._test_param("sample_y", "X", test_cases, extras_params=extras_params)
    
    def test_sample_y_param_n_samples(self, test_cases=None):
        test_cases = [] if test_cases is None else test_cases
        test_cases += [(-1, ValueError), (1.1, TypeError), (1, None)]
        extras_params = {"X": [[1]]}
        self._test_param("sample_y", "n_samples", test_cases, extras_params=extras_params)

    def test_sample_y_param_random_state(self, test_cases=None):
        test_cases = [] if test_cases is None else test_cases
        test_cases += [(np.nan, ValueError), ("state", ValueError), (1, None)]
        extras_params = {"X": [[1]]}
        self._test_param("sample_y", "random_state", test_cases, extras_params=extras_params)
