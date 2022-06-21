import numpy as np
import scipy
from scipy import integrate
from scipy.special import roots_hermitenorm
from sklearn import clone
from sklearn.utils import check_array, column_or_1d, check_consistent_length

from skactiveml.base import (
    ProbabilisticRegressor,
    SkactivemlClassifier,
    SkactivemlRegressor,
)
from skactiveml.utils import (
    check_indices,
    check_X_y,
)
from skactiveml.utils._validation import (
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
    """Calculates the conditional expectation, i.e. E[func(Y)|X=x_eval], where
    Y | X ~ reg.predict_target_distribution, for x_eval in `X_eval`.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The samples where the expectation should be evaluated.
    func : callable
        The function that transforms the random variable.
    reg: ProbabilisticRegressor
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
        The number of integration samples used in 'quantile', 'monte_carlo' and
        'gauss-hermite'.
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

    check_type(reg, "reg", ProbabilisticRegressor)
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
        n_integration_samples = (
            2 ** int(np.log2(n_integration_samples) + 1) + 1
        )
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
            inner_output = func(
                *arg_filter(np.arange(len(X)), X, inner_potential_y)
            )
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
                X=X,
                n_samples=n_integration_samples,
                random_state=random_state,
            )
        expectation = np.average(evaluate_func(potential_y), axis=1)
    elif method == "quantile":
        if quantile_method in ["trapezoid", "simpson", "average", "romberg"]:
            eval_points = np.arange(1, n_integration_samples + 1) / (
                n_integration_samples + 1
            )
            cond_dist = _reshape_scipy_dist(
                reg.predict_target_distribution(X), shape=(-1, 1)
            )
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
                inner_cond_dist = _reshape_scipy_dist(
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
        unscaled_potential_y, weights = roots_hermitenorm(
            n_integration_samples
        )
        cond_mean, cond_std = reg.predict(X, return_std=True)
        potential_y = (
            cond_std[:, np.newaxis] * unscaled_potential_y[np.newaxis, :]
            + cond_mean[:, np.newaxis]
        )
        output = evaluate_func(potential_y)
        expectation = (
            1
            / (2 * np.pi) ** (1 / 2)
            * np.sum(weights[np.newaxis, :] * output, axis=1)
        )
    else:  # method equals "dynamic_quad"
        for idx, x in enumerate(X):
            cond_dist = reg.predict_target_distribution([x])

            def quad_function_wrapper(y):
                if is_optional or not vector_func:
                    return func(*arg_filter(idx, x, y))
                else:
                    return func(
                        *arg_filter(
                            np.arange(len(X)), X, np.full((len(X), 1), y)
                        )
                    )[idx]

            expectation[idx] = cond_dist.expect(
                quad_function_wrapper,
                **quad_dict,
            )

    return expectation


def _reshape_scipy_dist(dist, shape):
    """Reshapes the parameters "loc", "scale", "df" of a distribution, if they
    exist.

    Parameters
    ----------
    dist : scipy.stats._distn_infrastructure.rv_frozen
        The distribution.
    shape : tuple
        The new shape.

    Returns
    -------
    dist : scipy.stats._distn_infrastructure.rv_frozen
        The reshaped distribution.
    """
    check_type(dist, "dist", scipy.stats._distn_infrastructure.rv_frozen)
    check_type(shape, "shape", tuple)
    for idx, item in enumerate(shape):
        check_type(item, f"shape[{idx}]", int)

    for argument in ["loc", "scale", "df"]:
        if argument in dist.kwds:
            # check if shapes are compatible
            dist.kwds[argument].shape = shape

    return dist


def _update_X_y(X, y, y_update, idx_update=None, X_update=None):
    """Update the training data by the updating samples/labels.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training data set.
    y : array-like of shape (n_samples)
        Labels of the training data set.
    idx_update : array-like of shape (n_updates) or int
        Index of the samples or sample to be updated.
    X_update : array-like of shape (n_updates, n_features) or (n_features)
        Samples to be updated or sample to be updated.
    y_update : array-like of shape (n_updates) or numeric
        Updating labels or updating label.

    Returns
    -------
    X_new : np.ndarray of shape (n_new_samples, n_features)
        The new training data set.
    y_new : np.ndarray of shape (n_new_samples)
        The new labels.
    """

    X = check_array(X)
    y = column_or_1d(check_array(y, force_all_finite=False, ensure_2d=False))
    check_consistent_length(X, y)

    if isinstance(y_update, (int, float)):
        y_update = np.array([y_update])
    else:
        y_update = check_array(
            y_update,
            force_all_finite=False,
            ensure_2d=False,
            ensure_min_samples=0,
        )
        y_update = column_or_1d(y_update)

    if idx_update is not None:
        if isinstance(idx_update, (int, np.integer)):
            idx_update = np.array([idx_update])
        idx_update = check_indices(idx_update, A=X, unique="check_unique")
        check_consistent_length(y_update, idx_update)
        X_new = X.copy()
        y_new = y.copy()
        y_new[idx_update] = y_update
        return X_new, y_new
    elif X_update is not None:
        X_update = check_array(X_update, ensure_2d=False)
        if X_update.ndim == 1:
            X_update = X_update.reshape(1, -1)
        check_consistent_length(X.T, X_update.T)
        check_consistent_length(y_update, X_update)
        X_new = np.append(X, X_update, axis=0)
        y_new = np.append(y, y_update, axis=0)
        return X_new, y_new
    else:
        raise ValueError("`idx_update` or `X_update` must not be `None`")


def _update_reg(
    reg,
    X,
    y,
    y_update,
    sample_weight=None,
    idx_update=None,
    X_update=None,
    mapping=None,
):
    """Update the regressor by the updating samples, depending on
    the mapping. Chooses `X_update` if `mapping is None` and updates
    `X[mapping[idx_update]]` otherwise.

    Parameters
    ----------
    reg : SkactivemlRegressor
        The regressor to be updated.
    X : array-like of shape (n_samples, n_features)
        Training data set.
    y : array-like of shape (n_samples)
        Labels of the training data set.
    y_update : array-like of shape (n_updates) or numeric
        Updating labels or updating label.
    sample_weight : array-like of shape (n_samples), optional (default = None)
        Sample weight of the training data set. If
    idx_update : array-like of shape (n_updates) or int
        Index of the samples or sample to be updated.
    X_update : array-like of shape (n_updates, n_features) or (n_features)
        Samples to be updated or sample to be updated.
    mapping : array-like of shape (n_candidates), optional (default = None)
        The deciding mapping.

    Returns
    -------
    reg_new : SkaktivemlRegressor
        The updated regressor.
    """

    if sample_weight is not None and mapping is None:
        raise ValueError(
            "If `sample_weight` is not `None` a mapping "
            "between candidates and the training dataset must "
            "exist."
        )

    if mapping is not None:
        if isinstance(idx_update, (int, np.integer)):
            check_indices([idx_update], A=mapping, unique="check_unique")
        else:
            check_indices(idx_update, A=mapping, unique="check_unique")
        X_new, y_new = _update_X_y(
            X, y, y_update, idx_update=mapping[idx_update]
        )
    else:
        X_new, y_new = _update_X_y(X, y, y_update, X_update=X_update)

    reg_new = clone(reg).fit(X_new, y_new, sample_weight)
    return reg_new


def bootstrap_estimators(
    est,
    X,
    y,
    k_bootstrap=5,
    n_train=0.5,
    sample_weight=None,
    random_state=None,
):
    """Train the estimator on bootstraps of `X` and `y`.

    Parameters
    ----------
    est : SkactivemlClassifier or SkactivemlRegressor
        The estimator to be be trained.
    X : array-like of shape (n_samples, n_features)
        Training data set, usually complete, i.e. including the labeled and
        unlabeled samples.
    y : array-like of shape (n_samples)
        Labels of the training data set.
    k_bootstrap : int, optional (default=5)
        The number of trained bootstraps.
    n_train : int or float, optional (default=0.5)
        The size of each bootstrap training data set.
    sample_weight: array-like of shape (n_samples), optional (default=None)
        Weights of training samples in `X`.
    random_state : numeric | np.random.RandomState (default=None)
        The random state to use. If `random_state is None` random
        `random_state` is used.

    Returns
    -------
    bootstrap_est : list of SkactivemlClassifier or list of SkactivemlRegressor
        The estimators trained on different bootstraps.
    """

    check_X_y(X=X, y=y, sample_weight=sample_weight)
    check_scalar(k_bootstrap, "k_bootstrap", int, min_val=1)
    check_scalar(
        n_train,
        "n_train",
        (int, float),
        min_val=0,
        max_val=1,
        min_inclusive=False,
    )
    check_type(est, "est", SkactivemlClassifier, SkactivemlRegressor)
    random_state = check_random_state(random_state)

    bootstrap_est = [clone(est) for _ in range(k_bootstrap)]
    sample_indices = np.arange(len(X))
    subsets_indices = [
        random_state.choice(sample_indices, size=int(len(X) * n_train + 1))
        for _ in range(k_bootstrap)
    ]

    for est_b, subset_indices in zip(bootstrap_est, subsets_indices):
        X_for_learner = X[subset_indices]
        y_for_learner = y[subset_indices]
        if sample_weight is None:
            est_b.fit(X_for_learner, y_for_learner)
        else:
            weight_for_learner = sample_weight[subset_indices]
            est_b.fit(X_for_learner, y_for_learner, weight_for_learner)

    return bootstrap_est
