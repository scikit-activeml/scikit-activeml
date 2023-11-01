import inspect
from copy import deepcopy

import numpy as np
from numpy.random import RandomState

from skactiveml.utils import call_func


class Dummy:
    def __init__(self):
        pass


class TemplateBudgetManager:
    def setUp(
        self,
        bm_class,
        init_default_params,
        query_by_utility_params,
    ):
        self.super_setUp_has_been_executed = True
        self.bm_class = bm_class
        init_params = inspect.signature(self.bm_class.__init__).parameters
        self.init_default_params = {"budget": 0.1}
        if "random_state" in init_params:
            self.init_default_params["random_state"] = 42

        self.query_by_utility_params = query_by_utility_params

        for key, val in init_default_params.items():
            self.init_default_params[key] = val

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
        query_by_utility_params = inspect.signature(
            self.bm_class.query_by_utility
        ).parameters
        for key, val in query_by_utility_params.items():
            if (
                key != "self"
                and val.default == inspect._empty
                and self.query_by_utility_params is not None
                and key not in self.query_by_utility_params
            ):
                raise ValueError(
                    f"Missing positional argument `{key}` of `query_by_utility` in "
                    f"`query_by_utility_default_kwargs_clf`."
                )

    def test_init_param_random_state(self, test_cases=None):
        init_params = inspect.signature(self.bm_class.__init__).parameters
        if "random_state" in init_params:
            test_cases = [] if test_cases is None else test_cases
            test_cases += [
                (np.nan, ValueError),
                ("state", ValueError),
                (1, None),
            ]
            self._test_param("init", "random_state", test_cases)

    def test_init_param_budget(self, test_cases=None):
        # budget must be defined as a float with a range of: 0 < budget <= 1
        test_cases = [] if test_cases is None else test_cases
        test_cases += [
            (np.nan, ValueError),
            ("state", TypeError),
            (0.0, ValueError),
            (0.1, None),
            (1.1, ValueError),
        ]
        self._test_param("init", "budget", test_cases)

    def test_init_param_test_assignments(self):
        for param in inspect.signature(self.bm_class.__init__).parameters:
            if param != "self":
                init_params = deepcopy(self.init_default_params)
                init_params[param] = Dummy()
                qs = self.bm_class(**init_params)
                self.assertEqual(
                    getattr(qs, param),
                    init_params[param],
                    msg=f"The parameter `{param}` was not assigned to a class "
                    f"variable when `__init__` was called.",
                )

    def test_param_test_availability(self):
        not_test = ["self", "kwargs"]

        # Get initial parameters.
        init_params = inspect.signature(self.bm_class.__init__).parameters
        init_params = list(init_params.keys())

        # Check init parameters.
        for param in np.setdiff1d(init_params, not_test):
            test_func_name = "test_init_param_" + param
            with self.subTest(msg=test_func_name):
                self.assertTrue(
                    hasattr(self, test_func_name),
                    msg=f"'{test_func_name}()' missing in {self.__class__}",
                )

        # Get query_by_utility parameters.
        query_by_utility_params = inspect.signature(
            self.bm_class.query_by_utility
        ).parameters
        query_by_utility_params = list(query_by_utility_params.keys())

        # Check query_by_utility parameters.
        for param in np.setdiff1d(query_by_utility_params, not_test):
            test_func_name = "test_query_by_utility_param_" + param
            with self.subTest(msg=test_func_name):
                self.assertTrue(
                    hasattr(self, test_func_name),
                    msg=f"'{test_func_name}()' missing in {self.__class__}",
                )

        # Check if query_by_utility is being tested.
        with self.subTest(msg="test_query_by_utility"):
            self.assertTrue(
                hasattr(self, "test_query_by_utility"),
                msg=f"'test_query_by_utility' missing in {self.__class__}",
            )

    def test_query_by_utility_param_utilities(self, test_cases=None):
        test_cases = [] if test_cases is None else test_cases
        test_cases += [
            (np.array([0.1]), None),
            (Dummy, TypeError),
            ("state", TypeError),
            (0.0, TypeError),
            ([0.1], TypeError),
            (["string"], TypeError),
        ]
        self._test_param("query_by_utility", "utilities", test_cases)

    def _test_param(
        self,
        test_func,
        test_param,
        test_cases,
        replace_init_params=None,
        replace_query_by_utility_params=None,
    ):
        if replace_init_params is None:
            replace_init_params = {}
        if replace_query_by_utility_params is None:
            replace_query_by_utility_params = {}

        for i, (test_val, err) in enumerate(test_cases):
            with self.subTest(msg="Param", id=i, val=test_val):
                init_params = deepcopy(self.init_default_params)
                for key, val in replace_init_params.items():
                    init_params[key] = val

                query_by_utility_params = deepcopy(
                    self.query_by_utility_params
                )
                for key, val in replace_query_by_utility_params.items():
                    query_by_utility_params[key] = val
                update_params = {}

                locals()[f"{test_func}_params"][test_param] = test_val

                bm = self.bm_class(**init_params)
                if err is None:
                    bm.query_by_utility(**query_by_utility_params)
                elif test_func in ["query_by_utility", "init"]:
                    self.assertRaises(
                        err, bm.query_by_utility, **query_by_utility_params
                    )
                else:
                    func = getattr(bm, test_func)
                    self.assertRaises(err, func, **update_params)

    def test_query_by_utility(
        self,
        expected_output=None,
        utilities=None,
    ):
        if expected_output is None:
            raise ValueError("Test need to override expected_output")
        random_state = np.random.RandomState(0)
        init_params = deepcopy(self.init_default_params)
        init_params_list = inspect.signature(self.bm_class.__init__).parameters
        if "random_state" in init_params_list:
            init_params["random_state"] = random_state
        if utilities is None:
            random = RandomState(0)
            utilities = random.rand(50)
        bm = self.bm_class(**init_params)
        bm2 = self.bm_class(**init_params)
        utilities_update = []
        bm1_outputs = []

        for u in utilities:
            output = bm.query_by_utility(np.array([u]))
            bm1_outputs.extend(output)
            utilities_update.append(u)
            budget_manager_param_dict1 = {"utilities": utilities_update}
            call_func(
                bm.update,
                candidates=np.array([u]),
                queried_indices=output,
                **budget_manager_param_dict1,
            )
        bm2_outputs = bm2.query_by_utility(np.array(utilities))
        budget_manager_param_dict2 = {"utilities": utilities}
        call_func(
            bm.update,
            candidates=np.array(utilities),
            queried_indices=bm2_outputs,
            **budget_manager_param_dict2,
        )
        self.assertEqual(len(bm1_outputs), len(bm2_outputs))
        if len(expected_output) == 0:
            self.assertEqual(len(expected_output), len(bm2_outputs))
        else:
            self.assertEqual(expected_output, bm2_outputs)

    def test_update_before_query_by_utility(
        self,
    ):
        init_params = deepcopy(self.init_default_params)
        init_params_list = inspect.signature(self.bm_class.__init__).parameters
        if "random_state" in init_params_list:
            init_params["random_state"] = np.random.RandomState(0)
        bm = self.bm_class(**init_params)
        bm2 = self.bm_class(**init_params)
        bm2_outputs = []
        utilities_update = []
        utilities = np.array([0.2, 0.6, 0.8, 0.9, 0.1])
        candidat = np.array([0.3])
        for u in utilities:
            output = bm2.query_by_utility(np.array([u]))
            bm2_outputs.extend(output)
            utilities_update.append(u)
            budget_manager_param_dict2 = {"utilities": utilities_update}
            call_func(
                bm2.update,
                candidates=np.array([u]),
                queried_indices=output,
                **budget_manager_param_dict2,
            )
        budget_manager_param_dict1 = {"utilities": utilities}
        call_func(
            bm.update,
            candidates=np.array(utilities),
            queried_indices=bm2_outputs,
            **budget_manager_param_dict1,
        )
        output1 = bm.query_by_utility(candidat)
        output2 = bm2.query_by_utility(candidat)
        self.assertEqual(len(output1), len(output2))
