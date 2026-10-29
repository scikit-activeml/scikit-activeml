"""Shared target resolution helpers for pool-based query strategies."""

from sklearn import clone

from ..base import SkactivemlClassifier, SkactivemlRegressor
from ..utils import (
    check_equal_missing_label,
    check_type,
    resolve_target_spec,
)
from ..utils._label import _check_labels
from ..utils._target import (
    _check_refit_capability,
    _check_target_matches_spec,
    _declared_target_type,
    _predicted_target_spec,
    _check_target_spec_capability,
)


def _resolve_estimator_target_spec(
    strategy,
    estimator,
    y,
):
    """Resolve one estimator-backed strategy target specification.

    Callers check that the missing labels of `strategy` and `estimator` are
    equal.
    """
    if isinstance(estimator, SkactivemlClassifier):
        task = "classification"
    elif isinstance(estimator, SkactivemlRegressor):
        task = "regression"
    else:  # pragma: no cover - callers validate their estimator type first.
        raise TypeError("`estimator` must be a scikit-activeml estimator.")

    if hasattr(estimator, "target_spec_"):
        estimator_target_spec = estimator.target_spec_
        target_spec = _predicted_target_spec(estimator_target_spec)
        _check_target_matches_spec(
            y, target_spec, missing_label=strategy.missing_label
        )
    else:
        target_spec = resolve_target_spec(
            y,
            task=task,
            target_type=_declared_target_type(y, estimator),
            annotation_type="single-annotator",
            classes=(
                estimator._declared_classes
                if task == "classification"
                else None
            ),
            missing_label=strategy.missing_label,
        )
        estimator_target_spec = target_spec

    _check_target_spec_capability(
        type(estimator).__name__,
        estimator_target_spec,
        estimator._target_capabilities,
    )
    _check_target_spec_capability(
        type(strategy).__name__, target_spec, strategy._target_capabilities
    )
    _check_labels(
        y,
        strategy.missing_label,
        target_type=target_spec.target_type,
        task=target_spec.task,
    )
    return target_spec


def _fit_and_resolve_estimator_target_spec(
    strategy,
    estimator,
    X,
    y,
    *,
    fit_estimator,
    sample_weight,
    estimator_name,
    fit_name,
    estimator_types,
):
    """Validate, optionally fit, and resolve an estimator-backed query."""
    check_type(estimator, estimator_name, *estimator_types)
    check_equal_missing_label(estimator.missing_label, strategy.missing_label)
    check_type(fit_estimator, fit_name, bool)
    if fit_estimator:
        _check_refit_capability(estimator)
        # A fresh fit resolves from constructor settings, including explicit
        # classes, rather than freezing the previous fit's inferred vocabulary.
        estimator = clone(estimator)
        if sample_weight is None:
            estimator = estimator.fit(X, y)
        else:
            estimator = estimator.fit(X, y, sample_weight)
    target_spec = _resolve_estimator_target_spec(strategy, estimator, y)
    return estimator, target_spec
