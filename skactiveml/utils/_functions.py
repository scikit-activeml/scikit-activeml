import inspect

import numpy as np
import scipy
from scipy.stats import rankdata
from sklearn import clone
from sklearn.utils.validation import (
    check_array,
    column_or_1d,
    check_consistent_length,
)

from ._validation import check_indices, check_type


def call_func(f_callable, only_mandatory=False, **kwargs):
    """Calls a function with the given parameters given in kwargs if they
    exist as parameters in f_callable.

    Parameters
    ----------
    f_callable : callable
        The function or object that is to be called
    only_mandatory : boolean
        If True only mandatory parameters are set.
    kwargs : kwargs
        All parameters that could be used for calling f_callable.

    Returns
    -------
    called object
    """
    params = inspect.signature(f_callable).parameters
    param_keys = params.keys()
    if only_mandatory:
        param_keys = list(
            filter(lambda k: params[k].default == inspect._empty, param_keys)
        )

    vars = dict(filter(lambda e: e[0] in param_keys, kwargs.items()))

    return f_callable(**vars)


def update_X_y(X, y, y_update, idx_update=None, X_update=None):
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
            y_update, force_all_finite=False, ensure_2d=False, ensure_min_samples=0
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


def update_reg(
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
    idx_update : int, optional (default = None)
        Index of the sample to be updated.
    X_update : (n_features), optional (default = None)
        Sample to be updated.
    mapping : array-like of shape (n_candidates), optional (default = None)
        The deciding mapping.

    Returns
    -------
    reg_new : SkaktivemlRegressor
        The updated regressor.
    """

    if sample_weight is not None and mapping is not None:
        raise ValueError(
            "If `sample_weight` is not `None`a mapping "
            "between candidates and the training dataset must "
            "exist."
        )

    if mapping is not None:
        if isinstance(idx_update, (int, np.integer)):
            check_indices([idx_update], A=mapping, unique="check_unique")
        else:
            check_indices(idx_update, A=mapping, unique="check_unique")
        X_new, y_new = update_X_y(X, y, y_update, idx_update=mapping[idx_update])
    else:
        X_new, y_new = update_X_y(X, y, y_update, X_update=X_update)

    reg_new = clone(reg).fit(X_new, y_new, sample_weight)
    return reg_new


def bootstrap_estimators(
    est,
    X,
    y,
    k_bootstrap,
    n_train,
    sample_weight=None,
    random_state=None,
):
    learners = [clone(est) for _ in range(k_bootstrap)]
    sample_indices = np.arange(len(X))
    subsets_indices = [
        random_state.choice(sample_indices, size=int(len(X) * n_train))
        for _ in range(k_bootstrap)
    ]

    for learner, subset_indices in zip(learners, subsets_indices):
        X_for_learner = X[subset_indices]
        y_for_learner = y[subset_indices]
        if sample_weight is None:
            learner.fit(X_for_learner, y_for_learner)
        else:
            weight_for_learner = sample_weight[subset_indices]
            learner.fit(X_for_learner, y_for_learner, weight_for_learner)

    return learners


def reshape_dist(dist, shape=None):
    """Reshapes the parameters of a distribution.

    Parameters
    ----------
    dist : scipy.stats._distn_infrastructure.rv_frozen
        The distribution.
    shape : tuple, optional (default = None)
        The new shape.

    Returns
    -------
    dist : scipy.stats._distn_infrastructure.rv_frozen
        The reshape distribution.
    """
    check_type(dist, "dist", scipy.stats._distn_infrastructure.rv_frozen)
    check_type(shape, "shape", tuple, None)
    for idx, item in enumerate(shape):
        check_type(item, f"shape[{idx}]", int)

    for argument in ["loc", "scale", "df"]:
        if argument in dist.kwds:
            dist.kwds[argument].shape = shape

    return dist
