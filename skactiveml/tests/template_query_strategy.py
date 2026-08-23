import inspect
import warnings
from copy import deepcopy

import numpy as np
from numpy.random import RandomState
from sklearn import clone
import sklearn.datasets
from sklearn.linear_model import SGDClassifier
from sklearn.multioutput import MultiOutputClassifier

from skactiveml.tests.utils import (
    assert_no_query_state,
    check_positional_args,
    check_test_param_test_availability,
)

from skactiveml.base import _TaskAgnosticPoolQueryStrategy
from skactiveml.exceptions import MappingError
from skactiveml.classifier import ParzenWindowClassifier, SklearnClassifier
from skactiveml.classifier.multiannotator import AnnotatorEnsembleClassifier
from skactiveml.utils import (
    MISSING_LABEL,
    is_unlabeled,
    is_labeled,
    labeled_indices,
    resolve_target_spec,
    unlabeled_indices,
    call_func,
)
from skactiveml.utils._target import _class_vocabulary_key
from skactiveml.utils._validation import _has_nested_classes

from sklearn.base import BaseEstimator
from sklearn.naive_bayes import GaussianNB

# Distinct binary class vocabularies covering the shared custom-vocabulary
# contract of multi-label pool query strategies. Each vocabulary is given in
# canonical, i.e. sorted, order, so that its second entry is the positive
# class of the corresponding label output.
_MULTILABEL_STRING_VOCABULARIES = (
    ("no", "yes"),
    ("off", "on"),
)


class Dummy:
    def __init__(self):
        pass


def _relabel_multilabel_target(
    y, source_classes, vocabularies, *, missing_label
):
    """Relabel a multi-label target into other binary class vocabularies.

    Each observation is mapped by its position in the source vocabulary of its
    label output onto the canonically, i.e. sorted, ordered target vocabulary.
    Canonically ordered source classes therefore preserve the meaning of every
    observation, i.e. negative stays negative and positive stays positive.

    Parameters
    ----------
    y : array-like of shape (n_samples, n_outputs)
        The multi-label target to relabel.
    source_classes : tuple of tuple
        The binary class vocabulary of `y` per label output.
    vocabularies : tuple of tuple
        The binary class vocabulary to relabel onto per label output.
    missing_label : scalar or str or None
        Value representing a missing observation in `y`.

    Returns
    -------
    y_relabeled : numpy.ndarray of shape (n_samples, n_outputs)
        The object-valued relabeled target, using `None` as its missing label.
    """
    y = np.asarray(y)
    is_lbld = is_labeled(y, missing_label, target_type="multi-label")
    y_relabeled = np.full(y.shape, None, dtype=object)
    for output_idx, source_classes_i in enumerate(source_classes):
        target_classes_i = sorted(vocabularies[output_idx])
        for class_idx, class_label in enumerate(source_classes_i):
            is_class = is_lbld & (y[:, output_idx] == class_label)
            y_relabeled[is_class, output_idx] = target_classes_i[class_idx]
    return y_relabeled


def _fully_observed_target(y, missing_label, target_type):
    """Return a copy of `y` in which every observation is observed.

    The missing observations are filled with an already observed value, so
    that the class vocabulary and, for a multi-label target, the all-or-nothing
    row contract are preserved.

    Parameters
    ----------
    y : array-like of shape (n_samples,) or (n_samples, n_outputs)
        The partially observed target to complete.
    missing_label : scalar or str or None
        Value representing a missing observation in `y`.
    target_type : str
        The resolved target type of `y`.

    Returns
    -------
    y_observed : numpy.ndarray of shape (n_samples,) or (n_samples, n_outputs)
        The completed target, i.e., an exhausted candidate pool.
    """
    y_observed = np.array(y, copy=True)
    is_lbld = is_labeled(y_observed, missing_label, target_type=target_type)
    is_ulbld = is_unlabeled(y_observed, missing_label, target_type=target_type)
    y_observed[is_ulbld] = y_observed[is_lbld][0]
    return y_observed


def _with_component_params(params, *, missing_label, classes=None):
    """Apply a missing label and class vocabularies to all components.

    Every component carried by the parameters, e.g. a wrapped query strategy
    or a classifier query argument, must agree with the strategy about the
    missing label. Class vocabularies are replaced only where a component
    declares one vocabulary per label output, so that a vocabulary of an
    auxiliary single-output problem is never overwritten.
    """
    params = deepcopy(params)
    for key, value in params.items():
        if key == "missing_label":
            params[key] = missing_label
        else:
            params[key] = _component_with_params(value, missing_label, classes)
    return params


def _component_with_params(value, missing_label, classes):
    """Return one component with the given declarations applied.

    Only the parameters a component declares itself are replaced. A component
    nested more deeply, e.g. the strategy wrapped by a wrapped strategy, or a
    collection of components is not covered by any current fixture and fails
    loudly through the missing-label checks, so such a fixture requires an
    explicit override.
    """
    if not isinstance(value, BaseEstimator):
        return value
    component_params = value.get_params(deep=False)
    replacements = {}
    if "missing_label" in component_params:
        replacements["missing_label"] = missing_label
    if classes is not None and _has_nested_classes(
        component_params.get("classes")
    ):
        replacements["classes"] = [list(classes_i) for classes_i in classes]
    return clone(value).set_params(**replacements)


class TemplateQueryStrategy:
    def setUp(
        self,
        qs_class,
        init_default_params,
        init_default_params_multilabel=None,
        query_default_params_clf=None,
        query_default_params_reg=None,
        query_default_params_clf_multilabel=None,
    ):
        self.super_setUp_has_been_executed = True
        self.qs_class = qs_class
        self.supports_multilabel_batch_variation = getattr(
            self, "supports_multilabel_batch_variation", True
        )

        self.init_default_params = {"random_state": 42}
        self.init_default_params.update(deepcopy(init_default_params))

        check_positional_args(
            self.qs_class.__init__,
            "__init__",
            self.init_default_params,
        )

        self.query_default_params_clf = query_default_params_clf
        self.query_default_params_reg = query_default_params_reg
        self.init_default_params_multilabel = deepcopy(
            init_default_params_multilabel
        )
        self.query_default_params_clf_multilabel = (
            query_default_params_clf_multilabel
        )

        if (
            self.query_default_params_clf is None
            and self.query_default_params_reg is None
            and self.query_default_params_clf_multilabel is None
        ):
            raise ValueError(
                "The query strategies must support either "
                "classification or regression. Hence, at least "
                "one parameter of `query_default_params_clf` "
                "and `query_default_params_reg` or "
                "`query_default_params_clf_multilabel` must be not None. "
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
        if self.query_default_params_clf_multilabel is not None:
            check_positional_args(
                self.qs_class.query,
                "query",
                self.query_default_params_clf_multilabel,
                kwargs_name="query_default_kwargs_clf_multilabel",
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

    def _default_query_params(self, model_type):
        if model_type == "clf":
            return (
                self.query_default_params_clf
                if self.query_default_params_clf is not None
                else self.query_default_params_clf_multilabel
            )
        if model_type == "reg":
            return self.query_default_params_reg
        raise ValueError("Only 'reg' or 'clf' is allowed as `model_type`.")

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
                    query_params = self._default_query_params(model_type)
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
                query_params = self._default_query_params(model_type)
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
            with self.subTest(msg="Param", id=i, val=str(test_val)):
                init_params = deepcopy(self.init_default_params)
                for key, val in replace_init_params.items():
                    init_params[key] = val

                query_param_cases = [
                    (self.query_default_params_clf, exclude_clf),
                    (self.query_default_params_reg, exclude_reg),
                ]
                if (
                    self.query_default_params_clf is None
                    and self.query_default_params_reg is None
                ):
                    query_param_cases.append(
                        (
                            self.query_default_params_clf_multilabel,
                            exclude_clf,
                        )
                    )
                for query_params, exclude_case in query_param_cases:
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
        init_default_params_multilabel=None,
        query_default_params_clf=None,
        query_default_params_reg=None,
        query_default_params_clf_multilabel=None,
    ):
        if "missing_label" not in init_default_params:
            init_default_params["missing_label"] = MISSING_LABEL
        super().setUp(
            qs_class,
            init_default_params,
            init_default_params_multilabel,
            query_default_params_clf,
            query_default_params_reg,
            query_default_params_clf_multilabel,
        )
        if self.query_default_params_clf is not None:
            default_query_params = self.query_default_params_clf
        elif self.query_default_params_reg is not None:
            default_query_params = self.query_default_params_reg
        else:
            default_query_params = self.query_default_params_clf_multilabel
        self.y_shape = list(default_query_params["y"].shape)

    def test_init_param_missing_label(self, test_cases=None):
        test_cases = [] if test_cases is None else test_cases
        ml = self.init_default_params["missing_label"]
        test_cases += [(ml, None), (Dummy, TypeError)]
        self._test_param("init", "missing_label", test_cases)

    def test_init_param_target_type(self):
        self.assertIn(
            "target_type",
            inspect.signature(self.qs_class.__init__).parameters,
        )
        strategy = self.qs_class(**deepcopy(self.init_default_params))
        self.assertEqual(strategy.target_type, "auto")
        self.assertIsInstance(strategy._target_capabilities, frozenset)
        self.assertTrue(strategy._target_capabilities)
        self.assertFalse(
            any(
                target_type == "multi-output"
                for _, target_type, _ in strategy._target_capabilities
            )
        )
        self._test_param(
            "init",
            "target_type",
            [("invalid", ValueError)],
        )

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
                if "estimator" in self.query_default_params_clf:
                    estimator = clone(
                        self.query_default_params_clf["estimator"]
                    )
                    estimator.missing_label = ml
                    estimator.classes = classes
                    replace_query_params["estimator"] = estimator
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
                ulbd_idx = unlabeled_indices(
                    query_params["y"],
                    missing_label=self.init_default_params["missing_label"],
                )
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

        def strip_random_state_inplace(obj):
            if isinstance(obj, dict):
                obj.pop("random_state", None)
                for v in obj.values():
                    strip_random_state_inplace(v)
            elif isinstance(obj, (list, tuple)):
                for item in obj:
                    strip_random_state_inplace(item)

        strip_random_state_inplace(init_params)
        init_params["random_state"] = np.random.RandomState(0)
        qs1 = self.qs_class(**init_params)
        qs2 = self.qs_class(**init_params)

        for query_params in [
            self.query_default_params_clf,
            self.query_default_params_reg,
        ]:
            if query_params is not None:
                query_params = deepcopy(query_params)
                query_params["return_utilities"] = True
                id1, u1 = qs1.query(**query_params)
                id1_again, u1_again = qs1.query(**query_params)
                id2, u2 = qs2.query(**query_params)

                self.assertEqual(len(u1[0]), len(query_params["X"]))
                np.testing.assert_array_equal(id1, id1_again)
                np.testing.assert_allclose(u1, u1_again)
                np.testing.assert_array_equal(id1, id2)
                np.testing.assert_allclose(u1, u2)

    def test_query_multilabel_invalid_rows(self):
        # Partially observed multi-label rows are rejected, for the ordinary
        # as well as for a custom class vocabulary.
        if self.query_default_params_clf_multilabel is None:
            return

        fixtures = [
            (
                "default",
                self._multilabel_init_params(),
                deepcopy(self.query_default_params_clf_multilabel),
            ),
            (
                "custom-vocabulary",
                *self._multilabel_custom_vocabulary_params(
                    self._multilabel_string_vocabularies()
                ),
            ),
        ]
        for name, init_params, query_params in fixtures:
            with self.subTest(vocabularies=name):
                missing_label = init_params["missing_label"]
                y = np.array(query_params["y"], copy=True)
                observed_idx = labeled_indices(
                    y, missing_label, target_type="multi-label"
                )
                y[observed_idx[0], 0] = missing_label
                query_params["y"] = y

                qs = self.qs_class(**init_params)
                self.assertRaises(ValueError, qs.query, **query_params)

    def test_query_param_sample_weight_multilabel(self):
        if self.query_default_params_clf_multilabel is None:
            return

        query_signature = inspect.signature(self.qs_class.query).parameters
        if "sample_weight" not in query_signature:
            return

        base_query_params = deepcopy(self.query_default_params_clf_multilabel)
        y = np.asarray(base_query_params["y"])
        qs = self.qs_class(**self._multilabel_init_params())

        query_params = deepcopy(base_query_params)
        query_params["sample_weight"] = np.ones(len(y))
        qs.query(**query_params)

        query_params = deepcopy(base_query_params)
        query_params["sample_weight"] = np.ones(len(y) + 1)
        self.assertRaises(ValueError, qs.query, **query_params)

    def test_query_multilabel_proba_format_contract(self):
        # The public multilabel probability formats of `SklearnClassifier`
        # ("array" and "list") must be interchangeable at the acquisition
        # boundary of a query strategy, i.e., a strategy must never assume the
        # native representation of the wrapped estimator.
        if self.query_default_params_clf_multilabel is None:
            return

        base_params = deepcopy(self.query_default_params_clf_multilabel)
        estimator_key = self._multilabel_proba_format_estimator_key(
            base_params
        )
        if estimator_key is None:
            return

        proba_formats = ["auto", "array", "list"]
        results = {}
        for proba_format in proba_formats:
            with self.subTest(proba_format=proba_format):
                query_params = deepcopy(base_params)
                query_params[estimator_key].set_params(
                    proba_format=proba_format
                )
                query_params["return_utilities"] = True
                if self.supports_multilabel_batch_variation:
                    # Cover acquisition logic that indexes probabilities only
                    # from the second selected sample of a batch onwards.
                    query_params["batch_size"] = 2
                qs = self.qs_class(**self._multilabel_init_params())
                results[proba_format] = qs.query(**query_params)
        if len(results) < len(proba_formats):
            # A failing query is already reported by its own subtest.
            return

        for proba_format in ["array", "list"]:
            with self.subTest(proba_format=proba_format):
                np.testing.assert_array_equal(
                    results["auto"][0],
                    results[proba_format][0],
                    err_msg=f"`proba_format='{proba_format}'` selects other "
                    f"samples than `proba_format='auto'`.",
                )
                np.testing.assert_allclose(
                    results["auto"][1],
                    results[proba_format][1],
                    equal_nan=True,
                    err_msg=f"`proba_format='{proba_format}'` yields other "
                    f"utilities than `proba_format='auto'`.",
                )

    @staticmethod
    def _multilabel_proba_format_estimator_key(query_params):
        # Locates the wrapper whose public multilabel probability format can
        # be varied. Strategies without such an estimator, e.g. purely
        # representation-based ones, are not covered by this contract.
        for key in ["clf", "estimator"]:
            candidate = query_params.get(key)
            if isinstance(candidate, SklearnClassifier) and "proba_format" in (
                candidate.get_params()
            ):
                return key
        return None

    def _multilabel_init_params(self):
        init_params = deepcopy(self.init_default_params)
        if self.init_default_params_multilabel is not None:
            init_params.update(deepcopy(self.init_default_params_multilabel))
        if (
            "target_type"
            in inspect.signature(self.qs_class.__init__).parameters
        ):
            init_params["target_type"] = "multi-label"
        return init_params

    def test_query_multilabel_custom_class_vocabularies(self):
        # A multi-label target may use its own binary class vocabulary per
        # label output, including string labels. Every strategy must therefore
        # operate on encoded targets instead of assuming that raw labels are
        # numeric or that they are `{0, 1}`.
        if self.query_default_params_clf_multilabel is None:
            return

        vocabulary_cases = [
            ("numeric", self._multilabel_numeric_vocabularies()),
            ("string", self._multilabel_string_vocabularies()),
        ]
        results = {}
        for name, vocabularies in vocabulary_cases:
            with self.subTest(vocabularies=name):
                init_params, query_params = (
                    self._multilabel_custom_vocabulary_params(vocabularies)
                )
                y = query_params["y"]
                missing_label = init_params["missing_label"]
                unld_idx = unlabeled_indices(
                    y, missing_label, target_type="multi-label"
                )
                lbld_idx = labeled_indices(
                    y, missing_label, target_type="multi-label"
                )
                query_params["return_utilities"] = True

                qs = self.qs_class(**init_params)
                query_indices, utilities = qs.query(**query_params)

                self.assertIn(query_indices[0], unld_idx)
                self.assertTrue(
                    np.isfinite(utilities[0, query_indices[0]]),
                    msg=f"Non-finite utility of the selected candidate for "
                    f"the {name}-valued class vocabularies {vocabularies}.",
                )
                # Only labeled samples are excluded via `np.nan`, i.e. a
                # candidate that a strategy deliberately excludes must remain
                # comparable through `-np.inf` instead of becoming invalid.
                self.assertFalse(np.isnan(utilities[0, unld_idx]).any())
                self.assertFalse(np.isposinf(utilities).any())
                self.assertTrue(np.isnan(utilities[0, lbld_idx]).all())
                results[name] = (query_indices, utilities)

        if len(results) < len(vocabulary_cases):
            # A failing query is already reported by its own subtest.
            return

        # Semantically equivalent vocabularies must not change acquisition.
        np.testing.assert_array_equal(
            results["numeric"][0],
            results["string"][0],
            err_msg="String-valued class vocabularies select other samples "
            "than their equivalent numeric encoding.",
        )
        np.testing.assert_allclose(
            results["numeric"][1],
            results["string"][1],
            equal_nan=True,
            err_msg="String-valued class vocabularies yield other utilities "
            "than their equivalent numeric encoding.",
        )

    def _multilabel_n_outputs(self):
        """Return the number of label outputs of the multi-label fixture."""
        return np.asarray(self.query_default_params_clf_multilabel["y"]).shape[
            1
        ]

    def _multilabel_string_vocabularies(self):
        """Return one distinct string class vocabulary per label output."""
        return tuple(
            (
                _MULTILABEL_STRING_VOCABULARIES[output_idx]
                if output_idx < len(_MULTILABEL_STRING_VOCABULARIES)
                else (f"neg-{output_idx}", f"pos-{output_idx}")
            )
            for output_idx in range(self._multilabel_n_outputs())
        )

    def _multilabel_numeric_vocabularies(self):
        """Return the ordinary numeric class vocabulary per label output."""
        return tuple((0, 1) for _ in range(self._multilabel_n_outputs()))

    def _multilabel_heterogeneous_vocabularies(self):
        """Return per-output vocabularies whose dtypes deliberately differ."""
        vocabularies = list(self._multilabel_string_vocabularies())
        vocabularies[-1] = (0, 1)
        return tuple(vocabularies)

    def test_query_multilabel_rejects_heterogeneous_vocabularies(self):
        # One array holds every label output, so vocabularies of different
        # dtypes cannot be represented. A strategy resolving a class
        # vocabulary has to reject them through the shared resolution, and
        # must not commit query state while doing so.
        #
        # A task-agnostic strategy is exempt by construction rather than by
        # fixture: it takes no estimator, so it resolves the target type alone
        # and never receives a class vocabulary to reject. Deriving the
        # exemption from the class keeps it from widening silently when a
        # fixture changes where it declares its vocabularies.
        if self.query_default_params_clf_multilabel is None:
            return
        if self._multilabel_n_outputs() < 2:
            return
        if issubclass(self.qs_class, _TaskAgnosticPoolQueryStrategy):
            return

        init_params, query_params = self._multilabel_custom_vocabulary_params(
            self._multilabel_heterogeneous_vocabularies()
        )
        qs = self.qs_class(**init_params)

        with self.assertRaisesRegex(
            ValueError, "one dtype across all label outputs"
        ):
            qs.query(**query_params)

        assert_no_query_state(self, qs)

    def _multilabel_custom_vocabulary_params(self, vocabularies):
        """Build the multi-label fixture of one custom class vocabulary.

        The default implementation relabels the target of
        `query_default_params_clf_multilabel` into the given per-output binary
        class vocabularies and adapts the declared vocabularies and the missing
        label of every component. An object-valued target is used, which
        requires `missing_label=None` for the strategy as well as for each of
        its components.

        Override this hook if a strategy needs auxiliary inputs, e.g.
        embeddings, logits, or a discriminator, that cannot be derived from the
        default multi-label query parameters. Overriding replaces the fixture;
        it must never skip the custom-vocabulary contract.

        Parameters
        ----------
        vocabularies : tuple of tuple
            One binary class vocabulary per label output in canonical, i.e.
            sorted, order.

        Returns
        -------
        init_params : dict
            Initialization parameters of the query strategy.
        query_params : dict
            Query parameters using the given class vocabularies.
        """
        query_params = deepcopy(self.query_default_params_clf_multilabel)
        query_params["y"] = _relabel_multilabel_target(
            query_params["y"],
            source_classes=self._multilabel_source_classes(query_params),
            vocabularies=vocabularies,
            missing_label=self.init_default_params["missing_label"],
        )
        # An object-valued target admits `None` as its only missing label.
        init_params = _with_component_params(
            self._multilabel_init_params(), missing_label=None
        )
        query_params = _with_component_params(
            query_params, missing_label=None, classes=vocabularies
        )
        return init_params, query_params

    def _multilabel_source_classes(self, query_params):
        """Resolve the class vocabularies of the default fixture."""
        declared_classes = None
        for value in query_params.values():
            classes = getattr(value, "classes", None)
            if not _has_nested_classes(classes):
                continue
            if declared_classes is not None and _class_vocabulary_key(
                classes
            ) != _class_vocabulary_key(declared_classes):
                raise AssertionError(
                    "The default multi-label query parameters declare "
                    "conflicting class vocabularies, so they cannot be "
                    "relabeled. Override "
                    "`_multilabel_custom_vocabulary_params`."
                )
            declared_classes = classes
        target_spec = resolve_target_spec(
            query_params["y"],
            task="classification",
            target_type="multi-label",
            annotation_type="single-annotator",
            classes=declared_classes,
            missing_label=self.init_default_params["missing_label"],
        )
        return target_spec.classes


class TemplateSingleAnnotatorPoolQueryStrategy(TemplatePoolQueryStrategy):
    def _test_fitted_multilabel_classifier_rejection(
        self,
        *,
        estimator_param="clf",
        fit_param="fit_clf",
        ensemble=False,
    ):
        X = np.array([[0.0], [1.0], [2.0], [3.0]])
        y = np.array([[0, 1], [1, 0], [-1, -1], [-1, -1]])
        classifier = SklearnClassifier(
            MultiOutputClassifier(
                SGDClassifier(loss="log_loss", random_state=0)
            ),
            classes=[[0, 1], [0, 1]],
            missing_label=-1,
            target_type="multi-label",
        ).fit(X, y)
        estimator = (
            [classifier, deepcopy(classifier)] if ensemble else classifier
        )
        init_params = deepcopy(self.init_default_params)
        init_params["missing_label"] = -1
        strategy = self.qs_class(**init_params)
        query_params = {
            "X": X,
            "y": y,
            estimator_param: estimator,
            fit_param: False,
        }

        with self.assertRaisesRegex(
            ValueError,
            rf"{type(strategy).__name__} does not support target capability",
        ):
            strategy.query(**query_params)

        assert_no_query_state(self, strategy)

    def _test_classification_target_contract(
        self,
        expected_capabilities,
        *,
        estimator_param="clf",
        fit_param="fit_clf",
    ):
        strategy = self.qs_class(**deepcopy(self.init_default_params))

        self.assertEqual(strategy.target_type, "auto")
        self.assertEqual(strategy._target_capabilities, expected_capabilities)

        query_params = deepcopy(self.query_default_params_clf_multilabel)
        estimator = clone(query_params[estimator_param])
        estimator.set_params(target_type="multi-label")
        estimator.fit(query_params["X"], query_params["y"])
        query_params[estimator_param] = estimator
        query_params[fit_param] = False
        conflicting = clone(strategy).set_params(target_type="single-output")

        with self.assertRaisesRegex(ValueError, "conflicts"):
            conflicting.query(**query_params)

        assert_no_query_state(self, conflicting)

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

                    with self.subTest(init_labels=str(init_labels)):
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

    def test_query_multilabel_batch_variation(self):
        if self.query_default_params_clf_multilabel is None:
            return
        if not self.supports_multilabel_batch_variation:
            return

        init_params = self._multilabel_init_params()
        qs = self.qs_class(**init_params)
        query_params = deepcopy(self.query_default_params_clf_multilabel)
        missing_label = self.init_default_params["missing_label"]
        max_batch_size = int(
            sum(
                is_unlabeled(
                    query_params["y"],
                    missing_label,
                    target_type="multi-label",
                )
            )
        )
        batch_size = min(5, max_batch_size)
        self.assertTrue(batch_size > 1, msg="Too few unlabeled")

        query_params["batch_size"] = batch_size
        query_params["return_utilities"] = True
        query_ids, utils = qs.query(**query_params)

        self.assertEqual(len(query_ids), batch_size)
        self.assertEqual(len(utils), batch_size)
        self.assertEqual(len(utils[0]), len(query_params["X"]))
        n_labeled = sum(
            is_labeled(
                query_params["y"],
                missing_label,
                target_type="multi-label",
            )
        )
        self.assertEqual(sum(np.isnan(utils[0])), n_labeled)

        query_params["batch_size"] = max_batch_size + 1
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

    def _test_exhausted_candidate_pool(
        self, init_params, query_params, target_type
    ):
        """Check the exhausted-pool contract for one acquisition fixture."""
        missing_label = self.init_default_params["missing_label"]
        n_features = np.asarray(query_params["X"]).shape[1]
        n_samples = len(query_params["X"])
        y_observed = _fully_observed_target(
            query_params["y"], missing_label, target_type
        )
        cases = [
            ("fully labeled pool", y_observed, None, n_samples),
            (
                "empty index array",
                query_params["y"],
                np.array([], int),
                n_samples,
            ),
            (
                "empty candidate array",
                query_params["y"],
                np.empty((0, n_features)),
                0,
            ),
        ]

        for name, y, candidates, n_utilities in cases:
            with self.subTest(case=name, target_type=target_type):
                qs = self.qs_class(**deepcopy(init_params))
                params = deepcopy(query_params)
                params["y"] = y
                params["candidates"] = candidates
                params["return_utilities"] = True

                with self.assertWarnsRegex(UserWarning, "exhausted"):
                    query_indices, utilities = qs.query(**params)

                self.assertEqual(query_indices.shape, (0,))
                self.assertTrue(np.issubdtype(query_indices.dtype, np.integer))
                self.assertEqual(utilities.shape, (0, n_utilities))

                params["return_utilities"] = False
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    query_indices = qs.query(**params)
                self.assertEqual(query_indices.shape, (0,))

    def test_query_exhausted_candidate_pool(self):
        # An exhausted candidate pool is a valid acquisition state that is
        # answered with an empty batch instead of an error about array shapes.
        init_params = deepcopy(self.init_default_params)
        for query_params in [
            self.query_default_params_clf,
            self.query_default_params_reg,
        ]:
            if query_params is not None:
                self._test_exhausted_candidate_pool(
                    init_params, query_params, "single-output"
                )

    def test_query_multilabel_exhausted_candidate_pool(self):
        if self.query_default_params_clf_multilabel is None:
            return

        self._test_exhausted_candidate_pool(
            self._multilabel_init_params(),
            self.query_default_params_clf_multilabel,
            "multi-label",
        )

    def test_query_multilabel_candidate_variation(self):
        if self.query_default_params_clf_multilabel is None:
            return

        init_params = self._multilabel_init_params()
        qs = self.qs_class(**init_params)
        missing_label = self.init_default_params["missing_label"]
        query_params = deepcopy(self.query_default_params_clf_multilabel)
        query_params["candidates"] = None
        query_params["return_utilities"] = True

        query_idx1, utils1 = qs.query(**query_params)

        unld_idx = unlabeled_indices(
            query_params["y"],
            missing_label,
            target_type="multi-label",
        )
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
            self.assertEqual(query_idx4.shape, (1,))
            self.assertEqual(utils4.shape, (1, len(unld_idx)))
        except MappingError:
            pass


class TemplateMultilabelOnlySingleAnnotatorPoolQueryStrategy(
    TemplateSingleAnnotatorPoolQueryStrategy
):
    """Shared target-contract tests for multi-label-only strategies."""

    def _query_multilabel_only_strategy(self, strategy, y, clf, **kwargs):
        query_params = deepcopy(self.query_default_params_clf_multilabel)
        query_params.update(y=y, clf=clf, **kwargs)
        return strategy.query(**query_params)

    def test_query_requires_multilabel_y(self):
        y = np.array([0.0, 1.0, 0.0, np.nan, np.nan, np.nan, np.nan, np.nan])
        clf = SklearnClassifier(estimator=GaussianNB())
        strategy = self.qs_class(**deepcopy(self.init_default_params))

        with self.assertRaisesRegex(
            ValueError,
            rf"{type(strategy).__name__} does not support target capability",
        ):
            self._query_multilabel_only_strategy(strategy, y, clf)

        assert_no_query_state(self, strategy)

    def test_target_contract(self):
        strategy = self.qs_class(**deepcopy(self.init_default_params))

        self.assertEqual(strategy.target_type, "auto")
        self.assertEqual(
            strategy._target_capabilities,
            frozenset({("classification", "multi-label", "single-annotator")}),
        )
        self.assertNotIn(
            ("classification", "multi-label", "multi-annotator"),
            strategy._target_capabilities,
        )

    def test_query_rejects_fitted_target_spec_conflict_before_state(self):
        query_params = self.query_default_params_clf_multilabel
        clf = clone(query_params["clf"]).fit(
            query_params["X"], query_params["y"]
        )
        init_params = deepcopy(self.init_default_params)
        init_params["target_type"] = "single-output"
        strategy = self.qs_class(**init_params)

        with self.assertRaisesRegex(ValueError, "conflicts"):
            self._query_multilabel_only_strategy(
                strategy, query_params["y"], clf, fit_clf=False
            )

        assert_no_query_state(self, strategy)

    def test_query_reuses_fitted_target_spec_without_class_evidence(self):
        query_params = self.query_default_params_clf_multilabel
        clf = SklearnClassifier(
            estimator=MultiOutputClassifier(GaussianNB()),
            classes=None,
            target_type="multi-label",
            proba_format="array",
            random_state=0,
        ).fit(query_params["X"], query_params["y"])
        established_spec = clf.target_spec_
        y_query = np.array(
            [
                [0.0, 1.0],
                [0.0, 1.0],
                *[[np.nan, np.nan] for _ in range(6)],
            ]
        )

        strategy = self.qs_class(**deepcopy(self.init_default_params))
        query_idx, utilities = self._query_multilabel_only_strategy(
            strategy,
            y_query,
            clf,
            fit_clf=False,
            return_utilities=True,
        )

        self.assertEqual(established_spec.classes, ((0.0, 1.0),) * 2)
        self.assertIs(clf.target_spec_, established_spec)
        self.assertIn(query_idx[0], range(2, len(query_params["X"])))
        self.assertTrue(np.isnan(utilities[0, :2]).all())
        self.assertFalse(hasattr(strategy, "target_spec_"))

    def test_query_resolves_explicit_multilabel_without_classes(self):
        query_params = self.query_default_params_clf_multilabel
        clf = SklearnClassifier(
            estimator=MultiOutputClassifier(GaussianNB()),
            classes=None,
            target_type="multi-label",
            proba_format="array",
            random_state=0,
        )

        strategy = self.qs_class(**deepcopy(self.init_default_params))
        query_idx = self._query_multilabel_only_strategy(
            strategy, query_params["y"], clf
        )

        unlabeled_idx = unlabeled_indices(
            query_params["y"],
            missing_label=self.init_default_params["missing_label"],
            target_type="multi-label",
        )
        self.assertIn(query_idx[0], unlabeled_idx)
        self.assertEqual(clf.target_type, "multi-label")
        self.assertFalse(hasattr(clf, "target_spec_"))

    def test_query_supports_custom_binary_vocabularies(self):
        missing_label = -1
        y = np.array(
            [
                [0, 0],
                [0, 5],
                [2, 5],
                *[[missing_label, missing_label] for _ in range(5)],
            ]
        )
        clf = SklearnClassifier(
            estimator=MultiOutputClassifier(GaussianNB()),
            classes=[[0, 2], [0, 5]],
            missing_label=missing_label,
            target_type="multi-label",
            proba_format="array",
            random_state=0,
        )
        init_params = deepcopy(self.init_default_params)
        init_params.update(missing_label=missing_label, random_state=0)
        strategy = self.qs_class(**init_params)

        query_idx, utilities = self._query_multilabel_only_strategy(
            strategy, y, clf, return_utilities=True
        )

        self.assertIn(query_idx[0], range(3, len(y)))
        self.assertTrue(np.isfinite(utilities[0, 3:]).all())

    def test_query_rejects_partially_observed_rows_before_state(self):
        query_params = self.query_default_params_clf_multilabel
        clf = clone(query_params["clf"]).fit(
            query_params["X"], query_params["y"]
        )
        y = query_params["y"].copy()
        y[1, 0] = np.nan
        strategy = self.qs_class(**deepcopy(self.init_default_params))

        with self.assertRaisesRegex(ValueError, "no mixing"):
            self._query_multilabel_only_strategy(
                strategy, y, clf, fit_clf=False
            )

        assert_no_query_state(self, strategy)

    def test_query_rejects_other_target_capabilities_before_state(self):
        query_params = self.query_default_params_clf_multilabel
        multi_output_y = np.array(
            [
                [0.0, 0.0],
                [1.0, 1.0],
                [2.0, 0.0],
                *[[np.nan, np.nan] for _ in range(5)],
            ]
        )
        multi_output_clf = SklearnClassifier(
            estimator=MultiOutputClassifier(GaussianNB()),
            classes=[[0, 1, 2], [0, 1]],
            target_type="multi-output",
        )
        multiannotator_clf = AnnotatorEnsembleClassifier(
            estimators=[
                ("pwc-0", ParzenWindowClassifier(classes=[0, 1])),
                ("pwc-1", ParzenWindowClassifier(classes=[0, 1])),
            ],
            classes=[0, 1],
        ).fit(query_params["X"], query_params["y"])

        cases = [
            (multi_output_y, multi_output_clf, "SklearnClassifier"),
            (
                query_params["y"],
                multiannotator_clf,
                self.qs_class.__name__,
            ),
        ]
        for y, clf, component in cases:
            with self.subTest(component=component):
                strategy = self.qs_class(**deepcopy(self.init_default_params))
                with self.assertRaisesRegex(
                    ValueError,
                    rf"{component} does not support target capability",
                ):
                    self._query_multilabel_only_strategy(
                        strategy, y, clf, fit_clf=False
                    )
                assert_no_query_state(self, strategy)


class TemplateSingleAnnotatorStreamQueryStrategy(TemplateQueryStrategy):
    def setUp(
        self,
        qs_class,
        init_default_params,
        query_default_params_clf=None,
        query_default_params_reg=None,
    ):
        super().setUp(
            qs_class=qs_class,
            init_default_params=init_default_params,
            query_default_params_clf=query_default_params_clf,
            query_default_params_reg=query_default_params_reg,
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

            # Test if all query strategy outputs and utilities are the same
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

        def strip_random_state_inplace(obj):
            if isinstance(obj, dict):
                obj.pop("random_state", None)
                for v in obj.values():
                    strip_random_state_inplace(v)
            elif isinstance(obj, (list, tuple)):
                for item in obj:
                    strip_random_state_inplace(item)

        strip_random_state_inplace(init_params)
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
            with self.subTest(msg="Param", id=i, val=str(test_val)):
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
