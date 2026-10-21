"""Shared validation and budget-manager helpers for stream query strategies."""

from copy import deepcopy

import numpy as np
from sklearn.utils import check_array, check_consistent_length

from ..base import BudgetManager
from ..utils import check_budget_manager, check_type
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


def _validate_budget_manager(
    strategy, default_budget_manager_class, default_kwargs=None, *, seeded=True
):
    """Create `budget_manager_` once and apply `budget` to a default one.

    A default budget manager keeps the other parameters it was created with.

    Parameters
    ----------
    strategy : skactiveml.base.SingleAnnotatorStreamQueryStrategy
        The query strategy owning `budget_manager_`.
    default_budget_manager_class : type
        The budget manager class used if `strategy.budget_manager` is `None`.
    default_kwargs : dict or None, default=None
        Parameters of the default budget manager.
    seeded : bool, default=True
        Whether to validate the strategy's `random_state_` and pass a seed
        drawn from a copy of it to the default budget manager, if its class
        accepts `random_state`.
    """
    if not hasattr(strategy, "budget_manager_"):
        if seeded:
            strategy._validate_random_state()
            random_seed = deepcopy(strategy.random_state_).randint(2**31 - 1)
            default_kwargs = {
                **(default_kwargs or {}),
                "random_state": random_seed,
            }
        check_type(
            strategy.budget_manager,
            "budget_manager_",
            BudgetManager,
            type(None),
        )
        strategy.budget_manager_ = check_budget_manager(
            strategy.budget,
            strategy.budget_manager,
            default_budget_manager_class,
            default_kwargs,
        )
    if strategy.budget_manager is None:
        strategy._validate_budget()
        if strategy.budget_manager_.budget != strategy.budget:
            strategy.budget_manager_.set_params(budget=strategy.budget)
