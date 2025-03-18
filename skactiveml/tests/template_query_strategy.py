import inspect
import warnings
from copy import deepcopy

import numpy as np
from numpy.random import RandomState
from sklearn import clone
import sklearn.datasets

from skactiveml.tests.utils import (
    check_positional_args,
    check_test_param_test_availability,
)

from skactiveml.exceptions import MappingError
from skactiveml.utils import (
    MISSING_LABEL,
    is_unlabeled,
    is_labeled,
    unlabeled_indices,
    call_func,
)

from sklearn.naive_bayes import GaussianNB


class Dummy:
    def __init__(self):
        pass


class TemplateQueryStrategy:
    def setUp(
        self,
        qs_class,
        init_default_params,
        query_default_params_clf=None,
        query_default_params_reg=None,
    ):
        self.super_setUp_has_been_executed = True
        self.qs_class = qs_class

        self.init_default_params = {"random_state": 42}
        self.init_default_params.update(deepcopy(init_default_params))

        check_positional_args(
            self.qs_class.__init__,
            "__init__",
            self.init_default_params,
        )

        self.query_default_params_clf = query_default_params_clf
        self.query_default_params_reg = query_default_params_reg

        if (
            self.query_default_params_clf is None
            and self.query_default_params_reg is None
        ):
            raise ValueError(
                "The query strategies must support either "
                "classification or regression. Hence, at least "
                "one parameter of `query_default_params_clf` "
                "and `query_default_params_reg` must be not None. "
                "Use emtpy dictionary to use default values."
            )
        if self.query_default_params_clf is not None:
            check_positional_args(
                self.qs_class.query,
                "query",
                self.query_default_params_clf,
                kwargs_name="query_default_kwargs_clf",
            )
        if self.query_default_params_reg is not None:
            check_positional_args(
                self.qs_class.query,
                "query",
                self.query_default_params_reg,
                kwargs_name="query_default_kwargs_reg",
            )

    def test_init_param_random_state(self, test_cases=None):
        test_cases = [] if test_cases is None else test_cases
        test_cases += [(np.nan, ValueError), ("state", ValueError), (1, None)]
        self._test_param("init", "random_state", test_cases)

    def test_query_param_fit_clf(self, test_cases=None, fit_values=None):
        self._fit_test(
            test_cases=test_cases, fit_values=fit_values, model_type="clf"
        )

    def test_query_param_fit_reg(self, test_cases=None, fit_values=None):
        self._fit_test(
            test_cases=test_cases, fit_values=fit_values, model_type="reg"
        )

    def _fit_test(self, test_cases, model_type, fit_values=None):
        fit_values = [False, True] if fit_values is None else fit_values
        query_params = inspect.signature(self.qs_class.query).parameters
        if f"fit_{model_type}" in query_params:
            # custom test cases are not necessary
            test_cases = [] if test_cases is None else test_cases
            test_cases += [(np.nan, TypeError), ("state", TypeError)]
            self._test_param("query", f"fit_{model_type}", test_cases)

            # check if model remains the same for both options
            for fit_type in fit_values:
                with self.subTest(msg="Model consistency"):
                    if model_type == "clf":
                        query_params = self.query_default_params_clf
                    elif model_type == "reg":
                        query_params = self.query_default_params_reg
                    else:
                        raise ValueError(
                            "Only 'reg' or 'clf' is allowed as `model_type`."
                        )
                    mdl = deepcopy(query_params[f"{model_type}"])
                    if not fit_type:
                        mdl.fit(query_params["X"], query_params["y"])
                    query_params = deepcopy(query_params)
                    query_params[f"{model_type}"] = deepcopy(mdl)
                    query_params[f"fit_{model_type}"] = fit_type

                    qs = self.qs_class(**self.init_default_params)
                    qs.query(**query_params)
                    self.assertTrue(
                        _cmp_object_dict(
                            query_params[f"{model_type}"].__dict__,
                            mdl.__dict__,
                        ),
                        msg=f"{model_type} changed after calling query for "
                        f"`fit_{model_type}={fit_type}`.",
                    )

    def test_query_param_clf(self, test_cases=None):
        self._model_comparison(test_cases=test_cases, model_type="clf")

    def test_query_param_reg(self, test_cases=None):
        self._model_comparison(test_cases=test_cases, model_type="reg")

    def _model_comparison(self, test_cases, model_type):
        query_params = inspect.signature(self.qs_class.query).parameters
        if f"{model_type}" in query_params:
            # custom test cases are necessary as model_type usually has
            # specific properties for query strategies
            if test_cases is None:
                raise NotImplementedError(
                    f"The test function `test_query_param_{model_type}` "
                    f"should be implemented for every query strategy as they "
                    f"probably have specific demands. If the query strategy "
                    f"supports every {model_type}, please call "
                    f"`super().test_query_param_{model_type}(test_cases=[])`."
                )
            test_cases += [
                (np.nan, TypeError),
                ("state", TypeError),
                (Dummy(), TypeError),
                (GaussianNB(), TypeError),
            ]
            self._test_param("query", f"{model_type}", test_cases)

            # check if model remains the same
            with self.subTest(msg=f"{model_type} consistency"):
                if model_type == "clf":
                    query_params = self.query_default_params_clf
                elif model_type == "reg":
                    query_params = self.query_default_params_reg
                else:
                    raise ValueError(
                        "Only 'reg' or 'clf' is allowed as `model_type`."
                    )
                mdl = deepcopy(query_params[f"{model_type}"])
                query_params = deepcopy(query_params)
                query_params[f"{model_type}"] = deepcopy(mdl)

                qs = self.qs_class(**self.init_default_params)
                qs.query(**query_params)
                self.assertTrue(
                    _cmp_object_dict(
                        query_params[f"{model_type}"].__dict__, mdl.__dict__
                    ),
                    msg=f"`{model_type}` changed after calling query.",
                )

    def test_init_param_test_assignments(self):
        for param in inspect.signature(self.qs_class.__init__).parameters:
            if param != "self":
                init_params = deepcopy(self.init_default_params)
                init_params[param] = Dummy()
                qs = self.qs_class(**init_params)
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
            self.qs_class.__init__,
            "init",
            not_test,
            logic_test=False,
        )

        # Check query parameters and check if query is being tested.
        check_test_param_test_availability(
            self, self.qs_class.query, "query", not_test
        )

    def _test_param(
        self,
        test_func,
        test_param,
        test_cases,
        replace_init_params=None,
        replace_query_params=None,
        exclude_clf=False,
        exclude_reg=False,
    ):
        if replace_init_params is None:
            replace_init_params = {}
        if replace_query_params is None:
            replace_query_params = {}

        for i, (test_val, err) in enumerate(test_cases):
            with self.subTest(msg="Param", id=i, val=test_val):
                init_params = deepcopy(self.init_default_params)
                for key, val in replace_init_params.items():
                    init_params[key] = val

                for query_params, exclude_case in [
                    (self.query_default_params_clf, exclude_clf),
                    (self.query_default_params_reg, exclude_reg),
                ]:
                    if not (query_params is None or exclude_case):
                        query_params = deepcopy(query_params)
                        for key, val in replace_query_params.items():
                            query_params[key] = val

                        locals()[f"{test_func}_params"][test_param] = test_val

                        qs = self.qs_class(**init_params)
                        if err is None:
                            qs.query(**query_params)
                        else:
                            if not hasattr(qs, "query"):
                                if not issubclass(AttributeError, err):
                                    qs.query
                            else:
                                self.assertRaises(
                                    err, qs.query, **query_params
                                )


class TemplatePoolQueryStrategy(TemplateQueryStrategy):
    def setUp(
        self,
        qs_class,
        init_default_params,
        query_default_params_clf=None,
        query_default_params_reg=None,
    ):
        if "missing_label" not in init_default_params:
            init_default_params["missing_label"] = MISSING_LABEL
        super().setUp(
            qs_class,
            init_default_params,
            query_default_params_clf,
            query_default_params_reg,
        )
        self.y_shape = list(
            self.query_default_params_clf["y"].shape
            if self.query_default_params_clf is not None
            else self.query_default_params_reg["y"].shape
        )

    def test_init_param_missing_label(self, test_cases=None):
        test_cases = [] if test_cases is None else test_cases
        ml = self.init_default_params["missing_label"]
        test_cases += [(ml, None), (Dummy, TypeError)]
        self._test_param("init", "missing_label", test_cases)

    def test_query_param_X(self, test_cases=None):
        test_cases = [] if test_cases is None else test_cases
        test_cases += [
            ("string", (ValueError, TypeError)),
            (Dummy, (ValueError, TypeError)),
        ]
        self._test_param("query", "X", test_cases)

        for exclude_clf, exclude_reg, query_params in [
            (False, True, self.query_default_params_clf),
            (True, False, self.query_default_params_reg),
        ]:
            if query_params is not None:
                X = query_params["X"]
                test_cases += [(X, None), (np.vstack([X, X]), ValueError)]
                self._test_param(
                    "query",
                    "X",
                    test_cases,
                    exclude_clf=exclude_clf,
                    exclude_reg=exclude_reg,
                )

    def test_query_param_y(self, test_cases=None):  # TODO more cases
        test_cases = [] if test_cases is None else test_cases
        test_cases += [(np.nan, TypeError), (Dummy, TypeError)]
        self._test_param("query", "y", test_cases)

        if self.query_default_params_clf is not None:
            y = self.query_default_params_clf["y"]
            test_cases = [(y, None), (np.vstack([y, y]), ValueError)]
            self._test_param("query", "y", test_cases, exclude_reg=True)

            for ml, classes, t, err in [
                (np.nan, [1.0, 2.0], float, None),
                (0, [1, 2], int, None),
                (None, [1, 2], object, None),
                (None, ["A", "B"], object, None),
                ("", ["A", "B"], str, None),
            ]:
                replace_init_params = {"missing_label": ml}
                if "classes" in self.init_default_params:
                    replace_init_params["classes"] = classes
                if "query_strategy" in self.init_default_params:
                    query_strategy = clone(
                        self.init_default_params["query_strategy"]
                    )
                    query_strategy.missing_label = ml
                    replace_init_params["query_strategy"] = query_strategy
                replace_query_params = {}
                if "clf" in self.query_default_params_clf:
                    clf = clone(self.query_default_params_clf["clf"])
                    clf.missing_label = ml
                    clf.classes = classes
                    replace_query_params["clf"] = clf
                if "ensemble" in self.query_default_params_clf:
                    ensemble = clone(self.query_default_params_clf["ensemble"])
                    ensemble.missing_label = ml
                    ensemble.classes = classes
                    replace_query_params["ensemble"] = ensemble
                replace_y = np.full_like(y, ml, dtype=t)
                replace_y[0] = classes[0]
                replace_y[1] = classes[1]
                test_cases = [(replace_y, err)]
                self._test_param(
                    "query",
                    "y",
                    test_cases,
                    replace_init_params=replace_init_params,
                    replace_query_params=replace_query_params,
                    exclude_reg=True,
                )
        if self.query_default_params_reg is not None:
            y = self.query_default_params_reg["y"]
            test_cases = [(y, None), (np.vstack([y, y]), ValueError)]
            self._test_param("query", "y", test_cases, exclude_clf=True)
            y_string = np.full(len(y), "test")
            test_cases = [(y_string, TypeError)]
            self._test_param("query", "y", test_cases, exclude_clf=True)

    def test_query_param_candidates(self, test_cases=None):  # TODO more cases
        test_cases = [] if test_cases is None else test_cases

        for exclude_clf, exclude_reg, query_params in [
            (False, True, self.query_default_params_clf),
            (True, False, self.query_default_params_reg),
        ]:
            if query_params is not None:
                ulbd_idx = unlabeled_indices(query_params["y"])
                cases = test_cases + [
                    (np.nan, ValueError),
                    (Dummy, TypeError),
                    ([ulbd_idx[0]], None),
                ]
                self._test_param(
                    "query",
                    "candidates",
                    cases,
                    exclude_clf=exclude_clf,
                    exclude_reg=exclude_reg,
                )

    def test_query_param_sample_weight(self, test_cases=None):
        query_params = inspect.signature(self.qs_class.query).parameters
        if "sample_weight" in query_params:
            # custom test cases are not necessary
            test_cases = [] if test_cases is None else test_cases
            test_cases += [
                (np.nan, (ValueError, TypeError)),
                (Dummy, (ValueError, TypeError)),
                (None, None),
            ]
            self._test_param("query", "sample_weight", test_cases)

            for exclude_clf, exclude_reg, query_params in [
                (False, True, self.query_default_params_clf),
                (True, False, self.query_default_params_reg),
            ]:
                if query_params is not None:
                    y = query_params["y"]
                    test_cases = [
                        (np.ones(len(y)), None),
                        (np.ones(len(y) + 1), ValueError),
                    ]
                    self._test_param(
                        "query",
                        "sample_weight",
                        test_cases,
                        exclude_clf=exclude_clf,
                        exclude_reg=exclude_reg,
                    )

    def test_query_param_utility_weight(
        self, test_cases=None
    ):  # TODO more cases
        query_params_list = inspect.signature(self.qs_class.query).parameters
        if "utility_weight" in query_params_list:
            # custom test cases are not necessary
            test_cases = [] if test_cases is None else test_cases
            test_cases += [
                (0, (ValueError, TypeError)),
                (1.2, (ValueError, TypeError)),
                (1, (ValueError, TypeError)),
            ]
            self._test_param("query", "utility_weight", test_cases)

            init_params = deepcopy(self.init_default_params)
            init_params["random_state"] = np.random.RandomState(0)
            qs = self.qs_class(**init_params)

            for query_params in [
                self.query_default_params_clf,
                self.query_default_params_reg,
            ]:
                if query_params is not None:
                    query_params = deepcopy(query_params)
                    query_params["return_utilities"] = True
                    if "utility_weight" in query_params.keys():
                        del query_params["utility_weight"]

                    ml = self.init_default_params["missing_label"]
                    unld_idx = is_unlabeled(query_params["y"], ml)

                    query_idx1, utils1 = qs.query(**query_params)

                    utility_weight = np.random.rand(len(unld_idx))
                    query_params["utility_weight"] = utility_weight
                    query_idx2, utils2 = qs.query(**query_params)
                    np.testing.assert_allclose(utils1 * utility_weight, utils2)

                    try:
                        query_params["candidates"] = query_params["X"][
                            unld_idx
                        ]
                        query_params["utility_weight"] = utility_weight[
                            unld_idx
                        ]
                        query_idx3, utils3 = qs.query(**query_params)

                        np.testing.assert_allclose(
                            (utils1 * utility_weight)[:, unld_idx], utils3
                        )

                        test_cases = [
                            (0, (ValueError, TypeError)),
                            (1.2, (ValueError, TypeError)),
                            (utility_weight, (ValueError, TypeError)),
                        ]
                        self._test_param(
                            "query",
                            "utility_weight",
                            test_cases,
                            replace_init_params=init_params,
                            replace_query_params=query_params,
                        )

                    except MappingError:
                        pass

    def test_query_param_batch_size(self, test_cases=None):  # TODO more cases
        test_cases = [] if test_cases is None else test_cases
        test_cases += [(0, ValueError), (1.2, TypeError), (1, None)]
        self._test_param("query", "batch_size", test_cases)

    def test_query_param_return_utilities(
        self, test_cases=None
    ):  # TODO more cases
        test_cases = [] if test_cases is None else test_cases
        test_cases += [("string", TypeError), (Dummy, TypeError), (True, None)]
        self._test_param("query", "return_utilities", test_cases)

    def test_query_reproducibility(self):
        # checks if the results stays the same with same random state
        init_params = deepcopy(self.init_default_params)
        init_params["random_state"] = np.random.RandomState(0)

        qs = self.qs_class(**init_params)

        for query_params in [
            self.query_default_params_clf,
            self.query_default_params_reg,
        ]:
            if query_params is not None:
                query_params = deepcopy(query_params)
                query_params["return_utilities"] = True
                id1, u1 = qs.query(**query_params)
                id2, u2 = qs.query(**query_params)

                self.assertEqual(len(u1[0]), len(query_params["X"]))
                np.testing.assert_array_equal(id1, id2)
                np.testing.assert_allclose(u1, u2)


class TemplateSingleAnnotatorPoolQueryStrategy(TemplatePoolQueryStrategy):
    def test_query_al_cycles(self):
        budget = 1
        init_params = deepcopy(self.init_default_params)
        qs = self.qs_class(**init_params)

        for query_params in [
            self.query_default_params_clf,
            self.query_default_params_reg,
        ]:
            if query_params is not None:
                query_params = deepcopy(query_params)

                missing_label = self.init_default_params["missing_label"]
                lbld_idx = is_labeled(query_params["y"], missing_label)
                unld_idx = is_unlabeled(query_params["y"], missing_label)
                y_true = deepcopy(query_params["y"])
                y_true[unld_idx] = y_true[lbld_idx][0]

                for init_labels in [0, 1, sum(unld_idx) - 1]:
                    y = np.full(y_true.shape, fill_value=missing_label)
                    y[0:init_labels] = y_true[0:init_labels]

                    with self.subTest(init_labels=init_labels):
                        for b in range(budget):
                            query_params["y"] = y
                            query_id = qs.query(**query_params)
                            query_params["y"][query_id] = y_true[query_id]

    def test_query_batch_variation(self):
        init_params = deepcopy(self.init_default_params)
        qs = self.qs_class(**init_params)

        for query_params in [
            self.query_default_params_clf,
            self.query_default_params_reg,
        ]:
            if query_params is not None:
                query_params = deepcopy(query_params)
                missing_label = self.init_default_params["missing_label"]
                max_batch_size = int(
                    sum(is_unlabeled(query_params["y"], missing_label))
                )
                batch_size = min(5, max_batch_size)
                self.assertTrue(batch_size > 1, msg="Too few unlabeled")

                query_params["batch_size"] = batch_size
                query_params["return_utilities"] = True
                query_ids, utils = qs.query(**query_params)

                self.assertEqual(len(query_ids), batch_size)
                self.assertEqual(len(utils), batch_size)
                self.assertEqual(len(utils[0]), len(query_params["X"]))
                n_labeled = sum(is_labeled(query_params["y"], missing_label))
                self.assertEqual(sum(np.isnan(utils[0])), n_labeled)

                query_params["batch_size"] = max_batch_size + 1
                query_params["return_utilities"] = True
                self.assertWarns(Warning, qs.query, **query_params)

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    ids, utilities = qs.query(**query_params)
                    self.assertEqual(len(ids), max_batch_size)

    def test_query_candidate_variation(self):
        init_params = deepcopy(self.init_default_params)
        qs = self.qs_class(**init_params)
        missing_label = self.init_default_params["missing_label"]

        for query_params in [
            self.query_default_params_clf,
            self.query_default_params_reg,
        ]:
            if query_params is not None:
                query_params = deepcopy(query_params)
                query_params["candidates"] = None
                query_params["return_utilities"] = True

                query_idx1, utils1 = qs.query(**query_params)

                unld_idx = unlabeled_indices(query_params["y"], missing_label)
                query_params["candidates"] = unld_idx
                query_idx2, utils2 = qs.query(**query_params)

                unld_idx2 = unld_idx[0:1]
                query_params["candidates"] = unld_idx2
                query_idx3, utils3 = qs.query(**query_params)

                np.testing.assert_allclose(utils1, utils2)
                utils3_copy = np.full_like(utils1, fill_value=np.nan)
                utils3_copy[0, unld_idx2] = utils3[0, unld_idx2]
                np.testing.assert_allclose(utils3, utils3_copy)

                try:
                    query_params["candidates"] = query_params["X"][unld_idx]
                    query_idx4, utils4 = qs.query(**query_params)

                    np.testing.assert_allclose(utils1[0][unld_idx], utils4[0])
                except MappingError:
                    pass


class TemplateSingleAnnotatorStreamQueryStrategy(TemplateQueryStrategy):
    def setUp(
        self,
        qs_class,
        init_default_params,
        query_default_params_clf=None,
        query_default_params_reg=None,
    ):
        super().setUp(
            qs_class,
            init_default_params,
            query_default_params_clf,
            query_default_params_reg,
        )
        self.update_params = {
            "candidates": [[]],
            "queried_indices": [],
        }

    def test_query_param_clf(self, test_cases=None):
        # _model_comparison checks for the availability of the classifier
        self.query_default_params_clf["fit_clf"] = True
        self._model_comparison(test_cases=test_cases, model_type="clf")

    def test_query_param_reg(self, test_cases=None):
        # _model_comparison checks for the availability of the regressor
        query_params = inspect.signature(self.qs_class.query).parameters
        if "fit_reg" in query_params:
            self.query_default_params_reg["fit_reg"] = True
        self._model_comparison(test_cases=test_cases, model_type="reg")

    def test_init_param_budget(self, test_cases=None):
        test_cases = [] if test_cases is None else test_cases
        test_cases += [
            (None, None),
            (0.5, None),
            (Dummy, TypeError),
            (0.0, ValueError),
            (1.1, ValueError),
            ("0.0", TypeError),
            (1, TypeError),
        ]
        self._test_param("init", "budget", test_cases)

    def test_init_param_budget_manager(self, test_cases=None):
        query_params = inspect.signature(self.qs_class.__init__).parameters
        if "budget_manager" in query_params:
            test_cases = [] if test_cases is None else test_cases
            test_cases += [(None, None), (0.5, TypeError), (Dummy, TypeError)]
            self._test_param("init", "budget_manager", test_cases)

    def test_query_param_X(self, test_cases=None):
        query_params = inspect.signature(self.qs_class.query).parameters
        if "X" in query_params:
            test_cases = [] if test_cases is None else test_cases
            test_cases += [
                ("string", (ValueError, TypeError)),
                (Dummy, (ValueError, TypeError)),
            ]
            self._test_param("query", "X", test_cases)

            for exclude_clf, exclude_reg, query_params in [
                (False, True, self.query_default_params_clf),
                (True, False, self.query_default_params_reg),
            ]:
                if query_params is not None:
                    if not exclude_clf:
                        replace_query_params = {"fit_clf": True}
                    else:
                        replace_query_params = {"fit_reg": True}
                    X = query_params["X"]
                    test_cases += [(X, None), (np.vstack([X, X]), ValueError)]
                    self._test_param(
                        "query",
                        "X",
                        test_cases,
                        exclude_clf=exclude_clf,
                        exclude_reg=exclude_reg,
                        replace_query_params=replace_query_params,
                    )

    def test_query_param_y(self, test_cases=None):  # TODO more cases
        query_params = inspect.signature(self.qs_class.query).parameters
        if "y" in query_params:
            test_cases = [] if test_cases is None else test_cases
            test_cases += [(np.nan, TypeError), (Dummy, TypeError)]
            self._test_param("query", "y", test_cases)

            if self.query_default_params_clf is not None:
                y = self.query_default_params_clf["y"]
                test_cases = [(y, None), (np.vstack([y, y]), ValueError)]
                self._test_param("query", "y", test_cases, exclude_reg=True)

                for ml, classes, t, err in [
                    (np.nan, [1.0, 2.0], float, None),
                    (0, [1, 2], int, None),
                    (None, [1, 2], object, None),
                    (None, ["A", "B"], object, None),
                    ("", ["A", "B"], str, None),
                ]:
                    replace_init_params = {}
                    replace_query_params = {"fit_clf": True}
                    if "classes" in self.init_default_params:
                        replace_init_params["classes"] = classes
                    if "clf" in self.query_default_params_clf:
                        clf = clone(self.query_default_params_clf["clf"])
                        clf.missing_label = ml
                        clf.classes = classes
                        replace_query_params["clf"] = clf
                    else:
                        replace_query_params = None
                    replace_y = np.full_like(y, ml, dtype=t)
                    replace_y[0] = classes[0]
                    replace_y[1] = classes[1]
                    test_cases = [(replace_y, err)]
                    self._test_param(
                        "query",
                        "y",
                        test_cases,
                        replace_init_params=replace_init_params,
                        replace_query_params=replace_query_params,
                        exclude_reg=True,
                    )

            if self.query_default_params_reg is not None:
                y = self.query_default_params_reg["y"]
                replace_query_params = {"fit_reg": True}
                test_cases = [(y, None), (np.vstack([y, y]), ValueError)]
                self._test_param(
                    "query",
                    "y",
                    test_cases,
                    exclude_clf=True,
                    replace_query_params=replace_query_params,
                )
                y_string = np.full(len(y), "test")
                test_cases = [(y_string, TypeError)]
                self._test_param(
                    "query",
                    "y",
                    test_cases,
                    exclude_clf=True,
                    replace_query_params=replace_query_params,
                )

    def test_query_param_candidates(self, test_cases=None):  # TODO more cases
        test_cases = [] if test_cases is None else test_cases

        for exclude_clf, exclude_reg, query_params in [
            (False, True, self.query_default_params_clf),
            (True, False, self.query_default_params_reg),
        ]:
            if query_params is not None:
                ulbd_idx = query_params["candidates"]
                cases = test_cases + [
                    (np.nan, ValueError),
                    (Dummy, ValueError),
                    ([ulbd_idx[0]], None),
                ]
                self._test_param(
                    "query",
                    "candidates",
                    cases,
                    exclude_clf=exclude_clf,
                    exclude_reg=exclude_reg,
                )

    def test_query_param_sample_weight(self, test_cases=None):
        query_params = inspect.signature(self.qs_class.query).parameters
        if "sample_weight" in query_params:
            # custom test cases are not necessary
            test_cases = [] if test_cases is None else test_cases
            test_cases += [
                (np.nan, (ValueError, TypeError)),
                (Dummy, (ValueError, TypeError)),
                (None, None),
            ]
            self._test_param("query", "sample_weight", test_cases)

            for exclude_clf, exclude_reg, query_params in [
                (False, True, self.query_default_params_clf),
                (True, False, self.query_default_params_reg),
            ]:
                if query_params is not None:
                    if not exclude_clf:
                        replace_query_params = {"fit_clf": True}
                    else:
                        replace_query_params = {"fit_reg": True}
                    y = query_params["y"]
                    test_cases = [
                        (np.ones(len(y)), None),
                        (np.ones(len(y) + 1), ValueError),
                    ]
                    self._test_param(
                        "query",
                        "sample_weight",
                        test_cases,
                        replace_query_params=replace_query_params,
                        exclude_clf=exclude_clf,
                        exclude_reg=exclude_reg,
                    )

    def test_query(
        self,
        expected_output,
        expected_utilities,
        budget_manager_param_dict=None,
        X=None,
        y=None,
        candidates=None,
        queried_indices=None,
    ):
        if expected_output is None or expected_utilities is None:
            raise ValueError(
                "Test need to override expected_output and expected_utilities"
            )
        for exclude_clf, exclude_reg, query_params in [
            (False, True, self.query_default_params_clf),
            (True, False, self.query_default_params_reg),
        ]:
            if query_params is None:
                continue
            # initialise query stategies to compare expectes_output
            init_params = deepcopy(self.init_default_params)
            init_params["random_state"] = np.random.RandomState(0)
            qs = self.qs_class(**init_params)
            qs2 = self.qs_class(**init_params)
            # if no candidates are given generate a dataset with a fixed seed
            if candidates is None:
                if X is not None or y is not None:
                    raise ValueError(
                        "override candidates or X and y need to be None"
                    )
                init_train_length = 4
                random_state = RandomState(0)
                X_all, y_centers = sklearn.datasets.make_blobs(
                    n_samples=20,
                    centers=3,
                    random_state=random_state,
                    shuffle=True,
                )
                y_all = y_centers % 2
                X = X_all[:init_train_length]
                y = y_all[:init_train_length]
                candidates = X_all[4:]
                if queried_indices is None:
                    queried_indices = np.arange(0, init_train_length)
            # add candidates as well as X and y to the default query_params
            query_default_params = deepcopy(self.query_default_params_clf)
            query_params = inspect.signature(self.qs_class.query).parameters
            if "clf" in query_params or "reg" in query_params:
                query_default_params["X"] = X
                query_default_params["y"] = y
                if not exclude_clf:
                    query_default_params["fit_clf"] = True
                if not exclude_reg:
                    query_default_params["fit_reg"] = True
            query_default_params["candidates"] = candidates
            query_default_params["return_utilities"] = True

            # update query as to already have queried the initial samples
            # as well as test if update can be called before query
            if X is not None:
                call_func(
                    qs.update,
                    candidates=X,
                    queried_indices=queried_indices,
                    budget_manager_param_dict=budget_manager_param_dict,
                )
                call_func(
                    qs2.update,
                    candidates=X,
                    queried_indices=queried_indices,
                    budget_manager_param_dict=budget_manager_param_dict,
                )
            else:
                call_func(
                    qs.update,
                    candidates=candidates,
                    queried_indices=queried_indices,
                    budget_manager_param_dict=budget_manager_param_dict,
                )
                call_func(
                    qs2.update,
                    candidates=candidates,
                    queried_indices=queried_indices,
                    budget_manager_param_dict=budget_manager_param_dict,
                )
            # use qs and qs2 to compare if query is not changed without update
            qs_output, utilities = qs.query(**query_default_params)
            for i in range(3):
                qs_output2, utilities2 = qs2.query(**query_default_params)

            # Test if all query strategie outputs and utilities are the same
            np.testing.assert_almost_equal(expected_utilities, utilities)
            self.assertFalse(isinstance(list, type(qs_output)))
            if len(expected_output) == 0:
                self.assertEqual(len(expected_output), len(qs_output))
                self.assertEqual(len(qs_output2), len(qs_output))
            else:
                np.testing.assert_array_equal(
                    np.array(expected_output), np.array(qs_output)
                )
                np.testing.assert_array_equal(
                    np.array(qs_output2), np.array(qs_output)
                )
            np.testing.assert_almost_equal(utilities, utilities2)

    def test_update_before_query(
        self,
    ):
        for exclude_clf, exclude_reg, query_params in [
            (False, True, self.query_default_params_clf),
            (True, False, self.query_default_params_reg),
        ]:
            if query_params is None:
                continue
            init_params = deepcopy(self.init_default_params)
            init_params["random_state"] = np.random.RandomState(0)
            qs = self.qs_class(**init_params)
            qs2 = self.qs_class(**init_params)
            X = [[0, 0], [0, 1], [1, 0], [1, 1], [0.75, 0.75], [0.5, 0.5]]
            y_true = [0, 0, 1, 1, 1, 0]
            query_default_params1 = deepcopy(self.query_default_params_clf)
            query_params = inspect.signature(self.qs_class.query).parameters
            utilities = []
            X_queue = []
            y_queue = []
            qs_outputs = []
            for i, x in enumerate(X):
                if ("clf" in query_params or "reg" in query_params) and i > 0:
                    X_queue.append(X[i - 1])
                    y_queue.append(y_true[i - 1])
                    query_default_params1["X"] = X_queue
                    query_default_params1["y"] = y_queue
                    if not exclude_clf:
                        query_default_params1["fit_clf"] = True
                    if not exclude_reg:
                        query_default_params1["fit_reg"] = True
                query_default_params1["candidates"] = np.array(
                    np.array(x).reshape([1, -1])
                )
                query_default_params1["return_utilities"] = True
                qs_output, u = qs.query(**query_default_params1)
                budget_manager_param_dict1 = {"utilities": u}
                qs_outputs.extend(qs_output)
                call_func(
                    qs.update,
                    candidates=np.array(x).reshape([1, -1]),
                    queried_indices=qs_output,
                    budget_manager_param_dict=budget_manager_param_dict1,
                )
                utilities.extend(u)

            budget_manager_param_dict1 = {"utilities": np.array(utilities)}
            call_func(
                qs2.update,
                candidates=np.array(X),
                queried_indices=qs_outputs,
                budget_manager_param_dict=budget_manager_param_dict1,
            )
            query_default_params1["candidates"] = X
            _, expected_utilities = qs.query(**query_default_params1)
            _, utilities = qs2.query(**query_default_params1)
            np.testing.assert_almost_equal(expected_utilities, utilities)

    def test_query_param_return_utilities(self, test_cases=None):
        test_cases = [] if test_cases is None else test_cases
        test_cases += [("string", TypeError), (Dummy, TypeError), (True, None)]
        self._test_param("query", "return_utilities", test_cases)

    def test_query_reproducibility(self):
        # checks if the results stays the same with same random state
        init_params = deepcopy(self.init_default_params)
        init_params["random_state"] = np.random.RandomState(0)

        qs = self.qs_class(**init_params)

        for query_params in [
            self.query_default_params_clf,
            self.query_default_params_reg,
        ]:
            if query_params is not None:
                query_params = deepcopy(query_params)
                query_params["return_utilities"] = True
                id1, u1 = qs.query(**query_params)
                id2, u2 = qs.query(**query_params)

                self.assertEqual(len(u1), len(query_params["candidates"]))
                np.testing.assert_array_equal(id1, id2)
                np.testing.assert_allclose(u1, u2)

    def test_update_param_candidates(self, test_cases=None):
        test_cases = [] if test_cases is None else test_cases
        test_cases += [(Dummy, TypeError), ([[]], None), ([[0]], None)]
        self._test_param("update", "candidates", test_cases)

    def test_update_param_queried_indices(self, test_cases=None):
        test_cases = [] if test_cases is None else test_cases
        test_cases += [
            ("string", IndexError),
            (Dummy, IndexError),
            ([], None),
            ([0], None),
        ]
        self._test_param("update", "queried_indices", test_cases)

    def _test_param(
        self,
        test_func,
        test_param,
        test_cases,
        replace_init_params=None,
        replace_query_params=None,
        exclude_clf=False,
        exclude_reg=False,
    ):
        if replace_init_params is None:
            replace_init_params = {}
        if replace_query_params is None:
            replace_query_params = {}

        for i, (test_val, err) in enumerate(test_cases):
            with self.subTest(msg="Param", id=i, val=test_val):
                init_params = deepcopy(self.init_default_params)
                for key, val in replace_init_params.items():
                    init_params[key] = val

                for query_params, exclude_case in [
                    (self.query_default_params_clf, exclude_clf),
                    (self.query_default_params_reg, exclude_reg),
                ]:
                    if not (query_params is None or exclude_case):
                        query_params = deepcopy(query_params)
                        for key, val in replace_query_params.items():
                            query_params[key] = val
                        update_params = deepcopy(self.update_params)

                        locals()[f"{test_func}_params"][test_param] = test_val

                        qs = self.qs_class(**init_params)
                        if err is None:
                            qs.query(**query_params)
                        elif test_func in ["query", "init"]:
                            self.assertRaises(err, qs.query, **query_params)
                        else:
                            func = getattr(qs, test_func)
                            self.assertRaises(err, func, **update_params)


def _cmp_object_dict(d1, d2):
    keys = np.union1d(d1.keys(), d2.keys())[0]
    for key in keys:
        if key not in d1.keys() or key not in d2.keys():
            return False
        if hasattr(d1[key], "__dict__") ^ hasattr(d1[key], "__dict__"):
            return False
        if hasattr(d1[key], "__dict__") and hasattr(d1[key], "__dict__"):
            if not _cmp_object_dict(d1[key].__dict__, d2[key].__dict__):
                return False
        try:
            if np.issubdtype(type(d1[key]), np.number) and np.issubdtype(
                type(d1[key]), np.number
            ):
                if np.isnan(d1[key]) == np.isnan(d2[key]):
                    pass
                elif np.isnan(d1[key]) ^ np.isnan(d2[key]):
                    return False
                else:
                    if not d1[key].__eq__(d2[key]):
                        return False
        except NotImplementedError:
            pass
        except Exception:
            return False
    return True
