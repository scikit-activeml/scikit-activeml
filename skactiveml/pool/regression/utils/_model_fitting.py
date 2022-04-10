import numpy as np
from sklearn import clone
from sklearn.utils import check_array, column_or_1d, check_consistent_length

from skactiveml.base import SkactivemlClassifier, SkactivemlRegressor
from skactiveml.utils._validation import (
    check_indices,
    check_X_y,
    check_scalar,
    check_type,
    check_random_state,
)


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
    k_bootstrap : int
        The number of trained bootstraps.
    n_train : int or float
        The size of each bootstrap training data set.
    sample_weight: array-like of shape (n_samples), optional (default=None)
        Weights of training samples in `X`.
    random_state : numeric | np.random.RandomState (default=None)
        The random state to use. If `random_state is None` random
        `random_state` is used.

    Returns
    -------
    bootstrap_est : list of SkactivemlClassifier or SkactivemlRegressor
        The estimators trained on different bootstraps.
    """

    check_X_y(X=X, y=y, sample_weight=sample_weight)
    check_scalar(k_bootstrap, "k_bootstrap", int, min_val=1)
    check_scalar(
        n_train, "n_train", (int, float), min_val=0, max_val=1, min_inclusive=False
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
