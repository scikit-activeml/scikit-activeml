import numpy as np
import scipy
from scipy import integrate
from scipy.special import roots_hermitenorm
from sklearn.utils import check_array

from ....base import TargetDistributionEstimator
from ....utils._validation import (
    check_type,
    check_random_state,
    check_scalar,
    check_callable,
)


def conditional_expect(
    X,
    func,
    reg,
    method=None,
    quantile_method=None,
    n_integration_samples=10,
    quad_dict=None,
    random_state=None,
    include_x=False,
    include_idx=False,
    vector_func=False,
):
    f"""Calculates the conditional expectation, i.e. E[func(Y)|X=x_eval], where
    Y | X ~ reg.predict_target_distribution, for x_eval in `X_eval`.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The samples where the expectation should be evaluated.
    func : callable
        The function that transforms the random variable.
    reg: TargetDistributionEstimator
        Predicts the target distribution over which the expectation is calculated.
    method: string, optional, optional (default=None)
        The method by which the expectation is computed.
        -'assume_linear' assumes E[func(Y)|X=x_eval] ~= func(E[Y|X=x_eval]) and
          thereby only takes the function value at the expected y value.
        -'monte_carlo' Basic monte carlo integration. Taking the average
          of randomly drawn samples. `n_integration_samples` specifies the
          number of monte carlo samples.
        -'quantile' Uses the quantile function to transform the integration
          space into the interval from 0 to 1 and than uses the method from
          'quantile_method' to calculate the integral. The number of integration
          points is specified by `n_integration_samples`.
        -'gauss_hermite' Uses Gauss-Hermite quadrature. This assumes Y | X
          to be gaussian distributed. The number of evaluation  points is given
          by `n_integration_samples`.
        -'dynamic_quad' uses `scipy's` function `expect` on the `rv_continuous`
          random variable of `reg`, which in turn uses a dynamic gaussian
          quadrature routine for calculating the integral. Performance is worse
          using a vector function.
        If `method is None` 'gauss_hermite' is used.
    quantile_method: string, optional (default=None)
        Specifies the integration methods used after the quantile
        transformation.
        -'trapezoid' Trapezoidal method for integration using evenly spaced
          samples.
        -'simpson' Simpson method for integration using evenly spaced samples.
        -'average' Taking the average value for integration using evenly spaced
          samples.
        -'romberg' Romberg method for integration. If `n_integration_samples` is
          not equal to `2**k + 1` for a natural number k, the number of
          samples used for integration is put to the smallest such number greater
          than `n_integration_samples`.
        -'quadrature' Gaussian quadrature method for integration.
        If `quantile_method is None` quadrature is used.
    n_integration_samples: int, optional (default=10)
        The number of integration samples used in 'quantile' and 'monte_carlo'.
    quad_dict: dict, optional (default=None)
        Further arguments for using `scipy's` `expect`
    random_state: numeric | np.random.RandomState, optional (default=None)
        Random state for fixing the number generation.
    include_x: bool, optional (default=False)
        If `include_x` is `True`, `func` also takes the x value.
    include_idx: bool, optional (default=False)
        If `include_idx` is `True`, `func` also takes the index of the x value.
    vector_func: bool or str, optional (default=False)
        If `vector_func` is `True`, the integration values are passed as a whole
        to the function `func`. If `vector_func` is 'both', the integration
        values might or might not be passed as a whole. The integration values
        if passed as a whole are of the form (n_samples, n_integration), where
        n_integration denotes the number of integration values.


    Returns
    -------
    expectation : numpy.ndarray of shape (n_samples)
        The conditional expectation for each value applied.
    """

    X = check_array(X, allow_nd=True)

    check_type(reg, "reg", TargetDistributionEstimator)
    check_type(
        method,
        "method",
        "monte_carlo",
        "assume_linear",
        "dynamic_quad",
        "gauss_hermite",
        "quantile",
        None,
    )
    check_type(
        quantile_method,
        "quantile_method",
        "trapezoid",
        "simpson",
        "average",
        "romberg",
        "quadrature",
        None,
    )
    check_scalar(n_integration_samples, "n_monte_carlo", int, min_val=1)
    check_type(quad_dict, "scipy_args", dict, None)
    check_type(include_idx, "include_idx", bool)
    check_type(include_x, "include_x", bool)
    check_type(vector_func, "vector_func", bool, "both")
    check_callable(func, "func", n_free_parameters=1 + include_idx + include_x)

    if method is None:
        method = "gauss_hermite"
    if quantile_method is None:
        quantile_method = "quadrature"
    if quad_dict is None:
        quad_dict = {}
    if method == "quantile" and quantile_method == "romberg":
        # n_integration_samples need to be of the form 2**k + 1
        n_integration_samples = 2 ** int(np.log2(n_integration_samples) + 1) + 1
    is_optional = vector_func == "both"
    if is_optional:
        vector_func = True

    random_state = check_random_state(random_state)

    def arg_filter(idx_y, x_y, y):
        ret = tuple()
        if include_idx:
            ret += (idx_y,)
        if include_x:
            ret += (x_y,)
        ret += (y,)
        return ret

    def evaluate_func(inner_potential_y):
        if vector_func:
            inner_output = func(*arg_filter(np.arange(len(X)), X, inner_potential_y))
        else:
            inner_output = np.zeros_like(inner_potential_y)
            for idx_x, inner_x in enumerate(X):
                for idx_y, y_val in enumerate(inner_potential_y[idx_x]):
                    inner_output[idx_x, idx_y] = func(
                        *arg_filter(idx_x, inner_x, y_val)
                    )
        return inner_output

    expectation = np.zeros(len(X))

    if method in ["assume_linear", "monte_carlo"]:
        if method == "assume_linear":
            potential_y = reg.predict(X).reshape(-1, 1)
        else:  # method equals "monte_carlo"
            potential_y = reg.sample_y(
                X=X, n_rv_samples=n_integration_samples, random_state=random_state
            )
        expectation = np.average(evaluate_func(potential_y), axis=1)
    elif method == "quantile":
        if quantile_method in ["trapezoid", "simpson", "average", "romberg"]:
            eval_points = np.arange(1, n_integration_samples + 1) / (
                n_integration_samples + 1
            )
            cond_dist = reshape_dist(reg.predict_target_distribution(X), shape=(-1, 1))
            potential_y = cond_dist.ppf(eval_points.reshape(1, -1))
            output = evaluate_func(potential_y)

            if quantile_method == "trapezoid":
                expectation = integrate.trapezoid(
                    output, dx=1 / n_integration_samples, axis=1
                )
            elif quantile_method == "simpson":
                expectation = integrate.simpson(
                    output, dx=1 / n_integration_samples, axis=1
                )
            elif quantile_method == "average":
                expectation = np.average(output, axis=-1)
            else:  # quantile_method equals "romberg"
                expectation = integrate.romb(
                    output, dx=1 / n_integration_samples, axis=1
                )
        else:  # quantile_method equals "quadrature"

            def fixed_quad_function_wrapper(inner_eval_points):
                inner_cond_dist = reshape_dist(
                    reg.predict_target_distribution(X), shape=(-1, 1)
                )
                inner_potential_y = inner_cond_dist.ppf(
                    inner_eval_points.reshape(1, -1)
                )

                return evaluate_func(inner_potential_y)

            expectation, _ = integrate.fixed_quad(
                fixed_quad_function_wrapper, 0, 1, n=n_integration_samples
            )
    elif method == "gauss_hermite":
        unscaled_potential_y, weights = roots_hermitenorm(n_integration_samples)
        cond_mean, cond_std = reg.predict(X, return_std=True)
        potential_y = (
            cond_std[:, np.newaxis] * unscaled_potential_y[np.newaxis, :]
            + cond_mean[:, np.newaxis]
        )
        output = evaluate_func(potential_y)
        expectation = (
            1 / (2 * np.pi) ** (1 / 2) * np.sum(weights[np.newaxis, :] * output, axis=1)
        )
    else:  # method equals "quad"
        for idx, x in enumerate(X):
            cond_dist = reg.predict_target_distribution([x])

            def quad_function_wrapper(y):
                if is_optional or not vector_func:
                    return func(*arg_filter(idx, x, y))
                else:
                    return func(
                        *arg_filter(np.arange(len(X)), X, np.full((len(X), 1), y))
                    )[idx]

            expectation[idx] = cond_dist.expect(
                quad_function_wrapper,
                **quad_dict,
            )

    return expectation


def reshape_dist(dist, shape=None):
    """Reshapes the parameters "loc", "scale", "df" of a distribution, if they
    exist.

    Parameters
    ----------
    dist : scipy.stats._distn_infrastructure.rv_frozen
        The distribution.
    shape : tuple, optional (default = None)
        The new shape.

    Returns
    -------
    dist : scipy.stats._distn_infrastructure.rv_frozen
        The reshaped distribution.
    """
    check_type(dist, "dist", scipy.stats._distn_infrastructure.rv_frozen)
    check_type(shape, "shape", tuple, None)
    for idx, item in enumerate(shape):
        check_type(item, f"shape[{idx}]", int)

    for argument in ["loc", "scale", "df"]:
        if argument in dist.kwds:
            # check if shapes are compatible
            dist.kwds[argument].shape = shape

    return dist
