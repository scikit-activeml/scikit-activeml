import itertools

import numpy as np
from sklearn.utils import check_array

from skactiveml.base import SkactivemlConditionalEstimator
from skactiveml.utils import check_type, check_random_state, check_scalar
from skactiveml.utils._functions import reshape_dist
from skactiveml.utils._validation import check_callable


def conditional_expect(
    X,
    func,
    cond_est,
    method=None,
    n_monte_carlo=10,
    scipy_dict=None,
    random_state=None,
    include_x=False,
    include_idx=False,
    vector_func=False,
):
    f"""Calculates the conditional expectation, i.e. E[func(Y)|X=x_eval], where
    Y | X ~ cond_est, for x_eval in `X_eval`.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The samples where the expectation should be evaluated.
    func : callable
        The function that transforms the random variable.
    cond_est: SkactivemlConditionalEstimator
        Distribution over which the expectation is calculated.
    method: string, default='monte_carlo'
        The method by which the expectation is computed:
        -'monte_carlo' basic monte carlo integration using random sampling.
        -'assume_linear' assumes E[func(Y)|X=x_eval] ~= func(E[Y|X=x_eval]) and
          thereby computes only the function applied to the expected value.
        -'scipy' uses `scipy` function `expect` on the `rv_continuous` random
          variable of `cond_est`.
    n_monte_carlo: int, optional (default=10)
        The number of monte carlo samples used.
    scipy_dict: dict, optional (default=None)
        Further arguments for using `scipy's` `expect`
    random_state: numeric | np.random.RandomState, optional (default=None)
        Random state for fixing the number generation.
    include_x: bool, optional (default=False)
        If `include_x` is `True`, `func` also takes the x value.
    include_idx: bool, optional (default=False)
        If `include_idx` is `True`, `func` also takes the index of the x value.
    vector_func: bool, optional (default=False)
        If `vector_func` is `True`, the integration values are passed as a whole
        to the function `func`.


    Returns
    -------
    expectation : numpy.ndarray of shape (n_1, ..., n_i-1, n_i+1, ..., n_m)
        The conditional expectation for each value applied
    """

    X = check_array(X, allow_nd=True)

    check_type(cond_est, "cond_est", SkactivemlConditionalEstimator)
    check_type(method, "method", "monte_carlo", "assume_linear", "scipy", "quantile")
    check_type(n_monte_carlo, "n_monte_carlo", int)
    check_type(scipy_dict, "scipy_args", dict, None)
    check_type(include_idx, "include_idx", bool)
    check_type(include_x, "include_x", bool)
    check_type(vector_func, "vector_func", bool)
    check_callable(func, "func", n_free_parameters=1 + include_idx + include_x)

    if scipy_dict is None:
        scipy_dict = {}

    random_state = check_random_state(random_state)

    def arg_filter(idx_y, x_y, y):
        ret = tuple()
        if include_idx:
            ret += (idx_y,)
        if include_x:
            ret += (x_y,)
        ret += (y,)
        return ret

    expectation = np.zeros(len(X))

    if vector_func:
        if method == "monte_carlo":
            potential_y_vals = cond_est.sample_y(
                X=X, n_rv_samples=n_monte_carlo, random_state=random_state
            )
            output = func(*arg_filter(np.arange(len(X)), X, potential_y_vals))
            expectation = np.average(output, axis=1)
        elif method == "assume_linear":
            y_val = cond_est.predict(X).reshape(-1, 1)
            expectation = func(*arg_filter(np.arange(len(X)), X, y_val))
        elif method == "scipy":
            for idx, x in enumerate(X):
                cond_dist = cond_est.estimate_conditional_distribution([x])
                expectation[idx] = cond_dist.expect(
                    lambda y: func(
                        *arg_filter(np.arange(len(X)), X, np.full((len(X), 1), y))
                    )[idx],
                    **scipy_dict,
                )
        elif method == "quantile":
            cond_dist = reshape_dist(
                cond_est.estimate_conditional_distribution(X), shape=(-1, 1)
            )
            split_val = np.arange(1, n_monte_carlo + 1) / (n_monte_carlo + 1)
            y_val = cond_dist.ppf(split_val.reshape(1, -1))
            output = func(*arg_filter(np.arange(len(X)), X, y_val))
            expectation = np.average(output, axis=1)
    else:
        if method == "monte_carlo":
            for idx, x in enumerate(X):
                potential_y_vals = cond_est.sample_y(
                    X=[x], n_rv_samples=n_monte_carlo, random_state=random_state
                )[0]
                output = np.zeros_like(potential_y_vals)
                for i, y_val in enumerate(potential_y_vals):
                    output[i] = func(*arg_filter(idx, x, y_val))

                expectation[idx] = np.average(output)
        elif method == "assume_linear":
            for idx, x in enumerate(X):
                y_val = cond_est.predict([x])[0]
                expectation[idx] = func(*arg_filter(idx, x, y_val))
        elif method == "scipy":
            for idx, x in enumerate(X):
                cond_dist = cond_est.estimate_conditional_distribution([x])
                expectation[idx] = cond_dist.expect(
                    lambda y: func(*arg_filter(idx, x, y)), **scipy_dict
                )

    return expectation
