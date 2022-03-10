import inspect
import unittest
from importlib import import_module
from os import path

import numpy as np

from skactiveml.base import BudgetManager
from skactiveml.stream import budgetmanager
from skactiveml.utils import call_func


class TestBudgetManager(unittest.TestCase):
    def setUp(self):
        self.budget_managers = {}
        for bm_name in budgetmanager.__all__:
            bm = getattr(budgetmanager, bm_name)
            attr_is_budget_manager = (
                inspect.isclass(bm)
                and issubclass(bm, BudgetManager)
                and not inspect.isabstract(bm)
            )
            if attr_is_budget_manager:
                self.budget_managers[bm_name] = bm

    def test_budget_managers(self):
        # Create data set for testing.
        rand = np.random.RandomState(0)
        random_state = rand.randint(2 ** 31 - 1)
        utilities = rand.rand(100)

        for bm_name, bm_class in self.budget_managers.items():
            bm_kwargs = {}
            bm_init_sig = inspect.signature(bm_class.__init__)
            bm_init_params = bm_init_sig.parameters.keys()
            if "random_state" in bm_init_params:
                bm_kwargs["random_state"] = random_state
            bm = bm_class(**bm_kwargs)
            bm2 = bm_class(**bm_kwargs)
            for t, u in enumerate(utilities):
                queried_indices = call_func(
                    bm.query_by_utility,
                    utilities=u.reshape([1, -1]),
                )

                for i in range(3):
                    queried_indices2 = call_func(
                        bm2.query_by_utility,
                        utilities=u.reshape([1, -1]),
                    )
                self.assertEqual(len(queried_indices), len(queried_indices2))
                call_func(
                    bm.update,
                    candidates=u.reshape([1, -1]),
                    queried_indices=queried_indices,
                    utilities=u.reshape([1, -1]),
                )
                call_func(
                    bm2.update,
                    candidates=u.reshape([1, -1]),
                    queried_indices=queried_indices,
                    utilities=u.reshape([1, -1]),
                )

    def test_param(self):
        not_test = ["self", "kwargs"]
        for bm_name in self.budget_managers:
            with self.subTest(msg="Param Test", bm_name=bm_name):
                # Get initial parameters.
                bm_class = self.budget_managers[bm_name]
                init_sig = inspect.signature(bm_class)
                init_params = init_sig.parameters.keys()
                init_params = list(init_params)

                # Get query parameters.
                query_sig = inspect.signature(bm_class.query_by_utility)
                query_params = query_sig.parameters.keys()
                query_params = list(query_params)

                # Check initial parameters.
                values = [Dummy() for i in range(len(init_params))]
                qs_obj = bm_class(*values)
                for param, value in zip(init_params, values):
                    self.assertTrue(
                        hasattr(qs_obj, param),
                        msg=f'"{param}" not tested for __init__()',
                    )
                    self.assertEqual(getattr(qs_obj, param), value)

                # Get class to check.
                class_filename = path.basename(inspect.getfile(bm_class))[:-3]
                mod = (
                        "skactiveml.stream.budgetmanager.tests.test"
                        + class_filename
                )
                mod = import_module(mod)
                test_class_name = "Test" + bm_class.__name__
                msg = f"{bm_class} has no test called {test_class_name}."
                self.assertTrue(hasattr(mod, test_class_name), msg=msg)
                test_obj = getattr(mod, test_class_name)

                # Check init parameters.
                for param in np.setdiff1d(init_params, not_test):
                    test_func_name = "test_init_param_" + param
                    self.assertTrue(
                        hasattr(test_obj, test_func_name),
                        msg=f"'{test_func_name}()' missing for parameter"
                            f" '{param}' of {bm_name}.__init__()",
                    )

                # Check query parameters.
                for param in np.setdiff1d(query_params, not_test):
                    test_func_name = "test_query_param_" + param
                    msg = (
                        f"'{test_func_name}()' missing for parameter "
                        f"'{param}' of {bm_name}.query()"
                    )
                    self.assertTrue(hasattr(test_obj, test_func_name), msg)


class Dummy:
    def __init__(self):
        pass
