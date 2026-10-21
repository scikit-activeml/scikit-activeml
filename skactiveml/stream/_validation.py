"""Shared validation and budget-manager helpers for stream query strategies."""

import numpy as np
from sklearn.utils import check_array, check_consistent_length

from ..utils._label_dtype import _as_label_array


def _validate_X_y_sample_weight(X, y, sample_weight):
    """Check `X`, `y`, and `sample_weight` and their lengths.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features) or None
        Training data set used to fit the classifier.
    y : array-like of shape (n_samples,) or None
        Labels of the training data set (possibly including unlabeled ones
        indicated by `clf.missing_label`).
    sample_weight : array-like of shape (n_samples,) or None
        Weights of training samples in `X`.

    Returns
    -------
    X : np.ndarray of shape (n_samples, n_features) or None
        Checked training data set.
    y : np.ndarray of shape (n_samples,) or None
        Checked labels of the input samples `X`.
    sample_weight : np.ndarray of shape (n_samples,) or None
        Checked weights of training samples in `X`.
    """
    if sample_weight is not None:
        sample_weight = np.array(sample_weight)
        check_consistent_length(sample_weight, y)
    if X is not None and y is not None:
        X = check_array(X)
        y = _as_label_array(y)
        check_consistent_length(X, y)
    return X, y, sample_weight


def _sync_default_budget_manager(strategy):
    """Apply `budget` to `budget_manager_` if it is the default one.

    Its other parameters keep the values it was created with.

    Parameters
    ----------
    strategy : skactiveml.base.SingleAnnotatorStreamQueryStrategy
        The query strategy owning `budget_manager_`.
    """
    if strategy.budget_manager is None:
        strategy._validate_budget()
        if strategy.budget_manager_.budget != strategy.budget:
            strategy.budget_manager_.set_params(budget=strategy.budget)
