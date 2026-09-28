import inspect
from copy import deepcopy

import numpy as np
from numpy.random import RandomState

from skactiveml.utils import call_func

from skactiveml.tests.utils import (
    assert_state_unchanged,
    check_positional_args,
    check_test_param_test_availability,
)


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
        self.init_default_params.update(deepcopy(init_default_params))
        self.update_params = {
            "candidates": [[0], [1]],
            "queried_indices": [0],
        }
        if "utilities" in inspect.signature(self.bm_class.update).parameters:
            self.update_params["utilities"] = np.array([0.2, 0.8])

        check_positional_args(
            self.bm_class.__init__,
            "__init__",
            self.init_default_params,
        )

        check_positional_args(
            self.bm_class.query_by_utility,
            "query_by_utility",
            self.query_by_utility_params,
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

    def test_set_params_budget_rejects_invalid_budgets(self):
        for budget, err in [
            (1, TypeError),
            (True, TypeError),
            (np.float32(1.0), TypeError),
            (np.array(1.0), TypeError),
            (1.5, ValueError),
            (0.0, ValueError),
            (-0.1, ValueError),
        ]:
            with self.subTest(budget=budget):
                bm = self.bm_class(**deepcopy(self.init_default_params))
                bm.update(**deepcopy(self.update_params))
                bm.set_params(budget=budget)
                before = deepcopy(bm)
                with self.assertRaises(err):
                    bm.query_by_utility(
                        **deepcopy(self.query_by_utility_params)
                    )
                assert_state_unchanged(self, bm, before)
                with self.assertRaises(err):
                    bm.update(**deepcopy(self.update_params))
                assert_state_unchanged(self, bm, before)

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

        # Check init parameters.
        check_test_param_test_availability(
            self,
            self.bm_class.__init__,
            "init",
            not_test,
            logic_test=False,
        )

        # Check query_by_utility parameters and if the function is being
        # tested.
        check_test_param_test_availability(
            self, self.bm_class.query_by_utility, "query_by_utility", not_test
        )
        check_test_param_test_availability(
            self, self.bm_class.update, "update", not_test, logic_test=False
        )

    def test_query_by_utility_param_utilities(self, test_cases=None):
        test_cases = [] if test_cases is None else test_cases
        test_cases += [
            (np.array([0.1]), None),
            (np.array([np.nan]), None),
            (Dummy, TypeError),
            ("state", TypeError),
            (0.0, TypeError),
            ([0.1], TypeError),
            (["string"], TypeError),
        ]
        self._test_param("query_by_utility", "utilities", test_cases)

    def test_update_param_candidates(self, test_cases=None):
        test_cases = [] if test_cases is None else test_cases
        test_cases += [
            (Dummy, TypeError),
            (None, TypeError),
            (0, TypeError),
            ([[0], [1]], None),
        ]
        self._test_param("update", "candidates", test_cases)

    def test_update_param_queried_indices(self, test_cases=None):
        test_cases = [] if test_cases is None else test_cases
        test_cases += [
            ("string", IndexError),
            (Dummy, IndexError),
            (0, IndexError),
            ([-1], IndexError),
            ([2], IndexError),
            ([0.5], IndexError),
            ([0.0], None),
            ([True], IndexError),
            ([[0]], IndexError),
            ([np.nan], IndexError),
            (np.array([2**64 - 1], dtype=np.uint64), IndexError),
            ([], None),
            ([1], None),
            ([1, 0, 1], None),
        ]
        self._test_param("update", "queried_indices", test_cases)

    def test_update_integer_valued_float_indices(self):
        candidates = np.zeros((2049, 1))
        utilities = np.zeros(len(candidates))
        indices = np.array([0, len(candidates) - 1])
        for dtype in [np.float16, np.float32, np.float64]:
            with self.subTest(dtype=dtype):
                bm = self.bm_class(**deepcopy(self.init_default_params))
                expected = deepcopy(bm)
                for manager, queried_indices in [
                    (bm, indices.astype(dtype)),
                    (expected, indices),
                ]:
                    call_func(
                        manager.update,
                        candidates=candidates,
                        queried_indices=queried_indices,
                        utilities=utilities,
                    )
                assert_state_unchanged(self, bm, expected)

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
            with self.subTest(msg="Param", id=i, val=str(test_val)):
                init_params = deepcopy(self.init_default_params)
                for key, val in replace_init_params.items():
                    init_params[key] = val

                query_by_utility_params = deepcopy(
                    self.query_by_utility_params
                )
                for key, val in replace_query_by_utility_params.items():
                    query_by_utility_params[key] = val
                update_params = deepcopy(self.update_params)

                locals()[f"{test_func}_params"][test_param] = test_val

                bm = self.bm_class(**init_params)
                if test_func == "update":
                    for initialized in [False, True]:
                        with self.subTest(initialized=initialized):
                            if initialized:
                                bm.update(**deepcopy(self.update_params))
                            before = deepcopy(bm)
                            if err is not None:
                                self.assertRaises(
                                    err, bm.update, **update_params
                                )
                                assert_state_unchanged(self, bm, before)
                            else:
                                self.assertIs(bm.update(**update_params), bm)
                                if test_param == "queried_indices":
                                    reference_params = deepcopy(update_params)
                                    reference_params["queried_indices"] = (
                                        np.unique(test_val).astype(int)
                                    )
                                    before.update(**reference_params)
                                    assert_state_unchanged(self, bm, before)
                elif err is None:
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
        expected_output,
        utilities=None,
    ):
        if expected_output is None:
            raise ValueError("Test need to override expected_output")
        random_state = np.random.RandomState(0)
        init_params = deepcopy(self.init_default_params)
        init_params_list = inspect.signature(self.bm_class.__init__).parameters
        if "random_state" in init_params_list:
            init_params["random_state"] = random_state
        utilities_nan = np.full(50, np.nan)
        if utilities is None:
            random = RandomState(0)
            utilities = random.rand(50)
            utilities = np.append(utilities, utilities_nan)
        bm = self.bm_class(**init_params)
        bm2 = self.bm_class(**init_params)
        bm1_outputs = []

        for i, u in enumerate(utilities):
            output = bm.query_by_utility(np.array([u]))
            bm1_outputs.extend(i + idx for idx in output)
            budget_manager_param_dict1 = {"utilities": np.array([u])}
            call_func(
                bm.update,
                candidates=np.array([u]),
                queried_indices=output,
                **budget_manager_param_dict1,
            )
        bm2_outputs = bm2.query_by_utility(np.array(utilities))
        budget_manager_param_dict2 = {"utilities": utilities}
        call_func(
            bm2.update,
            candidates=np.array(utilities),
            queried_indices=bm2_outputs,
            **budget_manager_param_dict2,
        )
        self.assertEqual(bm1_outputs, list(bm2_outputs))
        assert_state_unchanged(self, bm, bm2)
        if len(expected_output) == 0:
            self.assertEqual(len(expected_output), len(bm2_outputs))
        else:
            self.assertEqual(expected_output, bm2_outputs)
        output = bm.query_by_utility(utilities_nan)
        self.assertEqual(0, len(output))

    def test_query_by_utility_never_queries_nan(self):
        utilities = RandomState(0).rand(60)
        utilities[RandomState(1).rand(60) < 0.5] = np.nan
        for budget in [0.1, 1.0]:
            with self.subTest(budget=budget):
                init_params = deepcopy(self.init_default_params)
                init_params["budget"] = budget
                bm = self.bm_class(**init_params)
                bm2 = self.bm_class(**init_params)
                bm1_outputs = []
                for i, u in enumerate(utilities):
                    output = bm.query_by_utility(np.array([u]))
                    bm1_outputs.extend(i + idx for idx in output)
                    call_func(
                        bm.update,
                        candidates=np.array([u]),
                        queried_indices=output,
                        utilities=np.array([u]),
                    )
                bm2_outputs = bm2.query_by_utility(utilities)
                call_func(
                    bm2.update,
                    candidates=utilities,
                    queried_indices=bm2_outputs,
                    utilities=utilities,
                )
                self.assertEqual(bm1_outputs, list(bm2_outputs))
                assert_state_unchanged(self, bm, bm2)
                self.assertFalse(np.isnan(utilities[bm2_outputs]).any())

    def test_query_by_utility_preserves_state(self):
        random_states = (
            [0, np.random.RandomState(0), None]
            if "random_state" in self.init_default_params
            else [None]
        )
        original_global_state = np.random.get_state()
        try:
            for random_state in random_states:
                for initialize_with in ("query", "update"):
                    with self.subTest(
                        random_state=random_state,
                        initialize_with=initialize_with,
                    ):
                        init_params = deepcopy(self.init_default_params)
                        init_params["budget"] = 0.5
                        if "random_state" in init_params:
                            init_params["random_state"] = deepcopy(
                                random_state
                            )
                        bm = self.bm_class(**init_params)
                        if initialize_with == "query":
                            bm.query_by_utility(
                                **deepcopy(self.query_by_utility_params)
                            )
                        else:
                            bm.update(**deepcopy(self.update_params))
                        before = deepcopy(bm)
                        global_before = np.random.get_state()
                        for utilities in (
                            np.array([0.8]),
                            np.array([0.2, 0.6, 0.8, 0.9, 0.1, np.nan]),
                        ):
                            params = deepcopy(self.query_by_utility_params)
                            params["utilities"] = utilities
                            first = bm.query_by_utility(**params)
                            assert_state_unchanged(self, bm, before)
                            repeated = bm.query_by_utility(**params)
                            np.testing.assert_array_equal(first, repeated)
                            assert_state_unchanged(self, bm, before)
                            assert_state_unchanged(
                                self,
                                np.random.get_state(),
                                global_before,
                                name="global_random_state",
                            )
        finally:
            np.random.set_state(original_global_state)

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
        utilities = np.array([0.2, 0.6, 0.8, 0.9, 0.1])
        candidate = np.array([0.3])
        for i, u in enumerate(utilities):
            output = bm2.query_by_utility(np.array([u]))
            bm2_outputs.extend(i + idx for idx in output)
            budget_manager_param_dict2 = {"utilities": np.array([u])}
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
        assert_state_unchanged(self, bm, bm2)
        output1 = bm.query_by_utility(candidate)
        output2 = bm2.query_by_utility(candidate)
        self.assertEqual(output1, output2)
