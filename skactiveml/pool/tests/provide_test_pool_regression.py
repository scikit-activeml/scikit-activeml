from itertools import product

import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor

from skactiveml.base import ProbabilisticRegressor
from skactiveml.classifier import ParzenWindowClassifier
from skactiveml.regressor import (
    NICKernelRegressor,
    SklearnRegressor,
    SklearnNormalRegressor,
)
from skactiveml.utils import (
    MISSING_LABEL,
    unlabeled_indices,
    call_func,
    is_unlabeled,
)


def provide_test_regression_query_strategy_init_random_state(
    test_instance, qs_class, init_dict=None, query_dict=None
):

    # initialisation
    if init_dict is None:
        init_dict = {}
    if query_dict is None:
        query_dict = get_default_query_dict()

    X_s, y_s = get_list_of_regression_test_data()

    init_dict["missing_label"] = MISSING_LABEL
    for name, value in [("candidates", None), ("return_utilities", True)]:
        query_dict[name] = value

    # testing
    for (poss_random_state, (X, y)) in product(
        [0, np.random.RandomState(5)], zip(X_s, y_s)
    ):
        init_dict["random_state"] = poss_random_state
        for name, value in [("X", X), ("y", y), ("batch_size", 1)]:
            query_dict[name] = value

        qs = call_func(qs_class, **init_dict)
        # query the utilities
        results = [call_func(qs.query, **query_dict)[1] for _ in range(2)]
        for i in range(len(results) - 1):
            np.testing.assert_array_equal(results[i], results[i + 1])

    for illegal_random_state in ["illegal_random_state", dict]:
        init_dict["random_state"] = illegal_random_state
        for name, value in [("X", X_s[0]), ("y", y_s[0])]:
            query_dict[name] = value
            query_dict[name] = value

        qs = call_func(qs_class, **init_dict)
        test_instance.assertRaises(
            (TypeError, ValueError), call_func, qs.query, **query_dict
        )


def provide_test_regression_query_strategy_init_missing_label(
    test_instance,
    qs_class,
    init_dict=None,
    query_dict=None,
    missing_label_params_query_dict=None,
):

    # initialisation
    if init_dict is None:
        init_dict = {}
    if query_dict is None:
        query_dict = get_default_query_dict()
    if missing_label_params_query_dict is None:
        missing_label_params_query_dict = []

    init_dict["random_state"] = 0
    for name, value in [
        ("candidates", None),
        ("return_utilities", False),
    ]:
        query_dict[name] = value

    # testing
    for poss_missing_label in [MISSING_LABEL, -1]:
        X, y = get_regression_test_data(
            number=0, missing_label=poss_missing_label
        )
        init_dict["missing_label"] = poss_missing_label
        batch_size = int(
            np.sum(is_unlabeled(y, missing_label=poss_missing_label))
        )
        for name, value in [("X", X), ("y", y), ("batch_size", batch_size)]:
            query_dict[name] = value
        qs = call_func(qs_class, **init_dict)

        for param in missing_label_params_query_dict:
            query_dict[param].missing_label = poss_missing_label

        indices = call_func(qs.query, **query_dict)
        np.testing.assert_array_equal(
            indices.sort(), unlabeled_indices(y).sort()
        )

    for illegal_missing_label in [dict, []]:
        init_dict["missing_label"] = illegal_missing_label
        X, y = (np.arange(5 * 2).reshape(5, 2), np.arange(5))
        for name, value in [("X", X), ("y", y), ("batch_size", 1)]:
            query_dict[name] = value
        qs = call_func(qs_class, **init_dict)

        test_instance.assertRaises(
            TypeError, call_func, qs.query, **query_dict
        )


def provide_test_regression_query_strategy_init_integration_dict(
    test_instance,
    qs_class,
    init_dict=None,
    query_dict=None,
    integration_dict_name=None,
):
    # initialisation
    if init_dict is None:
        init_dict = get_default_init_dict()
    if query_dict is None:
        query_dict = get_default_query_dict()
    if integration_dict_name is None:
        integration_dict_name = "integration_dict"

    X, y = get_regression_test_data()
    update_query_dict_for_one_batch(query_dict, X, y)
    qs = call_func(qs_class, **init_dict)

    # testing
    for poss_integration_dict in [
        {"method": "assume_linear"},
        {"method": "monte_carlo"},
    ]:
        init_dict[integration_dict_name] = poss_integration_dict
        qs = call_func(qs_class, **init_dict)
        call_func(qs.query, **query_dict)

    for illegal_integration_dict in ["illegal", dict]:
        init_dict[integration_dict_name] = illegal_integration_dict
        qs = call_func(qs_class, **init_dict)
        test_instance.assertRaises(
            (TypeError, ValueError), call_func, qs.query, **query_dict
        )


def provide_test_regression_query_strategy_query_X(
    test_instance, qs_class, init_dict=None, query_dict=None
):
    # initialisation
    if init_dict is None:
        init_dict = get_default_init_dict()
    if query_dict is None:
        query_dict = get_default_query_dict()

    # testing
    y = np.arange(5, dtype=float)
    y[0:2] = MISSING_LABEL
    qs = call_func(qs_class, **init_dict)

    # correct argument
    X = np.arange(5).reshape(5, 1)
    update_query_dict_for_one_batch(query_dict, X, y)
    call_func(qs.query, **query_dict)

    # wrong shape dimension, wrong shape form, None and str not allowed
    for X_illegal in [
        np.arange(5),
        np.arange(3).reshape(3, 1),
        None,
        "illegal",
    ]:
        query_dict["X"] = X_illegal
        test_instance.assertRaises(
            (TypeError, ValueError), call_func, qs.query, **query_dict
        )


def provide_test_regression_query_strategy_query_y(
    test_instance, qs_class, init_dict=None, query_dict=None
):
    # initialisation
    if init_dict is None:
        init_dict = get_default_init_dict()
    if query_dict is None:
        query_dict = get_default_query_dict()

    # testing
    X = np.arange(5).reshape(5, 1)
    qs = call_func(qs_class, **init_dict)

    # correct arguments
    y = np.arange(5, dtype=float)
    for missing_label_indices in [np.arange(k + 1) for k in range(len(X))]:
        y[missing_label_indices] = MISSING_LABEL
        update_query_dict_for_one_batch(query_dict, X, y)
        indices = call_func(qs.query, **query_dict)

    # wrong shape dimension, wrong shape form, None and str not allowed
    for y_illegal in [
        np.arange(5 * 2).reshape(5, 2),
        np.arange(3),
        None,
        "illegal",
    ]:
        for name, value in [("X", X), ("y", y_illegal), ("batch_size", 1)]:
            query_dict[name] = value

        test_instance.assertRaises(
            (TypeError, ValueError), call_func, qs.query, **query_dict
        )


def provide_test_regression_query_strategy_query_reg(
    test_instance,
    qs_class,
    init_dict=None,
    query_dict=None,
    is_probabilistic=False,
):
    # initialisation
    if init_dict is None:
        init_dict = get_default_init_dict()
    if query_dict is None:
        query_dict = {"fit_reg": True}

    X, y = get_regression_test_data()
    update_query_dict_for_one_batch(query_dict, X, y)
    qs = call_func(qs_class, **init_dict)

    if is_probabilistic:
        possible_regs = [
            NICKernelRegressor(),
            SklearnNormalRegressor(GaussianProcessRegressor()),
        ]
        illegal_regs = [
            "illegal",
            ParzenWindowClassifier(),
            SklearnRegressor(GaussianProcessRegressor()),
        ]
    else:
        possible_regs = [
            NICKernelRegressor(),
            SklearnRegressor(GaussianProcessRegressor()),
        ]
        illegal_regs = ["illegal", ParzenWindowClassifier()]

    # correct arguments
    for poss_reg in possible_regs:
        query_dict["reg"] = poss_reg
        call_func(qs.query, **query_dict)

    for illegal_reg in illegal_regs:
        query_dict["reg"] = illegal_reg
        test_instance.assertRaises(
            TypeError, call_func, qs.query, **query_dict
        )


def provide_test_regression_query_strategy_query_fit_reg(
    test_instance, qs_class, init_dict=None, query_dict=None
):
    # initialisation
    call_status_dict = {"fit": False}
    if init_dict is None:
        init_dict = get_default_init_dict()
    if query_dict is None:

        class SpyRegressor(NICKernelRegressor):
            def fit(self, *args, **kwargs):
                call_status_dict["fit"] = True
                return super().fit(*args, **kwargs)

        query_dict = {"reg": SpyRegressor()}

    X, y = get_regression_test_data()
    query_dict["reg"].fit(X, y)
    update_query_dict_for_one_batch(query_dict, X, y)
    qs = call_func(qs_class, **init_dict)

    # correct arguments

    for poss_fit_reg in [True, False]:
        query_dict["fit_reg"] = poss_fit_reg
        call_status_dict["fit"] = False
        call_func(qs.query, **query_dict)
        test_instance.assertTrue(not poss_fit_reg or call_status_dict["fit"])

    # illegal arguments

    for illegal_fit_reg in ["illegal", 15]:
        query_dict["fit_reg"] = illegal_fit_reg
        test_instance.assertRaises(
            TypeError, call_func, qs.query, **query_dict
        )


def provide_test_regression_query_strategy_query_sample_weight(
    test_instance, qs_class, init_dict=None, query_dict=None
):
    # initialisation
    call_status_dict = {"used_sample_weight": False}
    if init_dict is None:
        init_dict = get_default_init_dict()
    if query_dict is None:

        class SpyRegressor(NICKernelRegressor):
            def fit(self, *args, sample_weight=None, **kwargs):
                if (sample_weight is not None) or (
                    len(args) >= 3 and args[2] is not None
                ):
                    call_status_dict["used_sample_weight"] = True
                return super().fit(*args, **kwargs)

        query_dict = {"reg": SpyRegressor(), "fit_reg": True}

    X, y = get_regression_test_data()
    update_query_dict_for_one_batch(query_dict, X, y)
    qs = call_func(qs_class, **init_dict)

    # correct arguments
    for poss_sample_weight in [None, np.arange(len(y)) + 1]:
        query_dict["sample_weight"] = poss_sample_weight
        call_func(qs.query, **query_dict)
        if poss_sample_weight is not None:
            test_instance.assertTrue(call_status_dict["used_sample_weight"])

    # illegal arguments
    for illegal_sample_weight in ["illegal", dict, np.arange(len(y) - 1)]:
        query_dict["sample_weight"] = illegal_sample_weight
        test_instance.assertRaises(
            (TypeError, ValueError), call_func, **query_dict
        )


def provide_test_regression_query_strategy_query_candidates(
    test_instance, qs_class, init_dict=None, query_dict=None
):
    # initialisation
    if init_dict is None:
        init_dict = get_default_init_dict()
    if query_dict is None:
        query_dict = get_default_query_dict()

    X, y = get_regression_test_data()
    for name, value in [
        ("X", X),
        ("y", y),
        ("batch_size", 1),
        ("return_utilities", True),
    ]:
        query_dict[name] = value

    qs = call_func(qs_class, **init_dict)
    lbld_indices = unlabeled_indices(y)

    # correct arguments

    utilities_s = []
    for poss_candidates in [None, lbld_indices, X[lbld_indices]]:
        query_dict["candidates"] = poss_candidates
        utilities_s.append(call_func(qs.query, **query_dict)[1])

    np.testing.assert_array_equal(utilities_s[0], utilities_s[1])
    np.testing.assert_allclose(utilities_s[1][:, lbld_indices], utilities_s[2])

    test_instance.assertEqual(utilities_s[1].shape, (1, len(X)))
    test_instance.assertEqual(utilities_s[2].shape, (1, len(lbld_indices)))

    # illegal arguments

    for illegal_candidates in [np.arange(4) + 7, "illegal", dict]:
        query_dict["candidates"] = illegal_candidates
        test_instance.assertRaises(
            (ValueError, TypeError), call_func, qs.query, **query_dict
        )


def provide_test_regression_query_strategy_query_X_eval(
    test_instance, qs_class, init_dict=None, query_dict=None
):
    # initialisation
    if init_dict is None:
        init_dict = get_default_init_dict()
    if query_dict is None:
        query_dict = get_default_query_dict()

    X, y = get_regression_test_data()
    update_query_dict_for_one_batch(query_dict, X, y)

    qs = call_func(qs_class, **init_dict)

    # correct arguments

    for poss_X_eval in [None, np.arange(3).reshape(-1, 1)]:
        query_dict["X_eval"] = poss_X_eval
        call_func(qs.query, **query_dict)

    # illegal arguments

    for illegal_X_eval in [np.arange(3), "illegal", dict]:
        query_dict["X_eval"] = illegal_X_eval
        test_instance.assertRaises(
            (ValueError, TypeError), call_func, qs.query, **query_dict
        )


def provide_test_regression_query_strategy_query_batch_size(
    test_instance, qs_class, init_dict=None, query_dict=None
):
    # initialisation
    if init_dict is None:
        init_dict = get_default_init_dict()
    if query_dict is None:
        query_dict = get_default_query_dict()

    X = np.arange(15).reshape(-1, 1)
    y = 2 * np.arange(15, dtype=float) + 3
    y[0:7] = MISSING_LABEL
    for name, value in [
        ("X", X),
        ("y", y),
        ("return_utilities", True),
    ]:
        query_dict[name] = value
    qs = call_func(qs_class, **init_dict)

    # correct arguments
    for poss_batch_size in [1, 2, 3, 7]:
        query_dict["batch_size"] = poss_batch_size
        indices, utilities = call_func(qs.query, **query_dict)
        test_instance.assertEqual(indices.shape, (poss_batch_size,))
        test_instance.assertEqual(utilities.shape, (poss_batch_size, len(X)))

    # illegal arguments
    for illegal_batch_size in [0, "illegal", dict]:
        query_dict["batch_size"] = illegal_batch_size
        test_instance.assertRaises(
            (TypeError, ValueError), call_func, qs.query, **query_dict
        )


def provide_test_regression_query_strategy_query_return_utilities(
    test_instance, qs_class, init_dict=None, query_dict=None
):
    # initialisation
    if init_dict is None:
        init_dict = get_default_init_dict()
    if query_dict is None:
        query_dict = get_default_query_dict()

    X, y = get_regression_test_data()
    update_query_dict_for_one_batch(query_dict, X, y)
    qs = call_func(qs_class, **init_dict)

    for poss_return_utilities in [True, False]:
        query_dict["return_utilities"] = poss_return_utilities
        re_val = call_func(qs.query, **query_dict)
        test_instance.assertEqual(
            isinstance(re_val, tuple), poss_return_utilities
        )

    for illegal_return_utilities in ["illegal", dict, 5]:
        query_dict["return_utilities"] = illegal_return_utilities
        test_instance.assertRaises(
            TypeError, call_func, qs.query, **query_dict
        )


def provide_test_regression_query_strategy_change_dependence(
    test_instance, qs_class, init_dict=None, query_dict=None, reg_name="reg"
):
    # initialisation
    if init_dict is None:
        init_dict = get_default_init_dict()
    if query_dict is None:
        query_dict = get_default_query_dict()

    X, y = get_regression_test_data()
    update_query_dict_for_one_batch(query_dict, X, y)
    qs = call_func(qs_class, **init_dict)

    class ZeroRegressor(ProbabilisticRegressor):
        def predict_target_distribution(self, X):
            return norm(loc=np.zeros(len(X)))

        def fit(self, *args, **kwargs):
            return self

    query_dict[reg_name] = ZeroRegressor()
    query_dict["return_utilities"] = True

    utilities = call_func(qs.query, **query_dict)[1][0]
    np.testing.assert_almost_equal(
        np.zeros_like(y),
        np.where(is_unlabeled(utilities), 0, utilities),
    )


def get_list_of_regression_test_data(missing_label=MISSING_LABEL):
    X_s = [
        np.arange(7).reshape(7, 1),
        0.3 * np.arange(5 * 3).reshape(5, 3),
    ]
    y_s = [
        np.arange(7, dtype=float) * 4 + 1,
        4.5 * np.arange(5, dtype=float) ** 2 - 7,
    ]
    y_s[0][[0, 1, 4]] = missing_label
    y_s[1][[2, 3, 4]] = missing_label
    return X_s, y_s


def get_regression_test_data(number=0, missing_label=MISSING_LABEL):
    X_s, y_s = get_list_of_regression_test_data(missing_label)
    return X_s[number], y_s[number]


def get_default_query_dict():
    return {"reg": NICKernelRegressor(), "fit_reg": True}


def get_default_init_dict(missing_label=MISSING_LABEL):
    return {"missing_label": missing_label, "random_state": 0}


def update_query_dict_for_one_batch(query_dict, X, y):
    for name, value in [
        ("X", X),
        ("y", y),
        ("batch_size", 1),
        ("return_utilities", False),
    ]:
        query_dict[name] = value
