"""Shared target reconciliation helpers for pool-based query strategies.

Every helper in this module operates on plain values only, i.e., target
declarations, estimators, and capability sets. Helpers that would need to know
how a component stores those values (for instance how a wrapper reaches its
wrapped strategy or its query arguments) belong to that component instead. The
orchestration of wrapper target reconciliation is therefore owned by each
wrapper, i.e., by `skactiveml.pool._wrapper._TargetPreservingWrapper` and by
`skactiveml.pool.multiannotator._wrapper.SingleAnnotatorWrapper`. Each collects
its own declarations and authorities and then delegates the wrapper-agnostic
resolution, comparison, and capability checks to this module.
"""

from sklearn import clone

from ..base import SkactivemlClassifier, SkactivemlRegressor
from ..utils import (
    check_equal_missing_label,
    check_type,
    is_unlabeled,
    resolve_target_spec,
)
from ..utils._target import (
    _class_vocabulary_key,
    _resolve_task_agnostic_target_type,
    _validate_target_semantics,
    _check_target_spec_capability,
)

_ALLOWED_DECLARED_TARGET_TYPES = frozenset(
    {"auto", "single-output", "multi-label", "multi-output"}
)


def _resolve_estimator_target_spec(
    strategy, estimator, y, *, missing_label_checked=False
):
    """Resolve one estimator-backed strategy target specification."""
    if isinstance(estimator, SkactivemlClassifier):
        task = "classification"
        classes = getattr(estimator, "classes_", estimator.classes)
    elif isinstance(estimator, SkactivemlRegressor):
        task = "regression"
        classes = None
    else:  # pragma: no cover - callers validate their estimator type first.
        raise TypeError("`estimator` must be a scikit-activeml estimator.")

    if not missing_label_checked:
        check_equal_missing_label(
            estimator.missing_label, strategy.missing_label
        )
    strategy_target_type = strategy.target_type
    _validate_target_semantics(
        task,
        strategy_target_type,
        "single-annotator",
        allow_auto=True,
    )

    if hasattr(estimator, "target_spec_"):
        target_spec = estimator.target_spec_
        resolve_target_spec(
            y,
            task=target_spec.task,
            target_type=target_spec.target_type,
            annotation_type=target_spec.annotation_type,
            classes=target_spec.classes,
            missing_label=strategy.missing_label,
        )
    else:
        estimator_target_type = getattr(estimator, "target_type", "auto")
        _validate_target_semantics(
            task,
            estimator_target_type,
            "single-annotator",
            allow_auto=True,
        )
        if (
            strategy_target_type != "auto"
            and estimator_target_type != "auto"
            and strategy_target_type != estimator_target_type
        ):
            raise ValueError(
                f"{type(strategy).__name__}'s explicit `target_type` "
                "conflicts with the estimator's explicit `target_type`."
            )
        target_spec = resolve_target_spec(
            y,
            task=task,
            target_type=(
                estimator_target_type
                if estimator_target_type != "auto"
                else strategy_target_type
            ),
            annotation_type="single-annotator",
            classes=classes,
            missing_label=strategy.missing_label,
        )

    _check_target_spec_capability(
        type(estimator).__name__,
        target_spec,
        estimator._target_capabilities,
    )

    if strategy_target_type != "auto" and (
        strategy_target_type != target_spec.target_type
    ):
        raise ValueError(
            f"{type(strategy).__name__}'s explicit `target_type` conflicts "
            "with the fitted estimator's target specification."
        )
    _check_target_spec_capability(
        type(strategy).__name__, target_spec, strategy._target_capabilities
    )
    is_unlabeled(
        y,
        missing_label=strategy.missing_label,
        target_type=target_spec.target_type,
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
        # A fresh fit resolves from constructor settings, including explicit
        # classes, rather than freezing the previous fit's inferred vocabulary.
        estimator = clone(estimator)
    target_spec = _resolve_estimator_target_spec(
        strategy, estimator, y, missing_label_checked=True
    )
    if fit_estimator:
        if sample_weight is None:
            estimator = estimator.fit(X, y)
        else:
            estimator = estimator.fit(X, y, sample_weight)
        target_spec = _resolve_estimator_target_spec(strategy, estimator, y)
    return estimator, target_spec


def _collect_declared_authorities(authority_params, query_kwargs):
    """Discover the estimators declared as target authorities.

    Discovery is deliberately restricted to the parameter names a strategy
    declares through `_target_authority_params`, so that query arguments
    holding an estimator for an auxiliary problem are never mistaken for a
    semantic authority for `y`.

    Parameters
    ----------
    authority_params : iterable of str
        Declared names of `query` parameters carrying a target authority.
    query_kwargs : dict
        The keyword arguments forwarded to the wrapped query strategy.

    Returns
    -------
    authorities : list of skactiveml estimators
        The discovered estimators, ordered by declaration and, for sequence
        valued arguments, by position. Authorities that will be refitted are
        represented by unfitted clones, so preflight uses their constructor
        declarations. The canonical `fit_clf`, `fit_reg`, `fit_ensemble`, and
        `fit_estimator` arguments default to `True`, as in the query methods.
        Other authority roles retain their supplied estimators.
    """
    authorities = []
    seen_authorities = set()
    for name in authority_params:
        if name not in query_kwargs:
            continue
        refit = False
        if name in ("clf", "reg", "ensemble", "estimator"):
            fit_name = f"fit_{name}"
            refit = query_kwargs.get(fit_name, True)
            check_type(refit, fit_name, bool)
        value = query_kwargs[name]
        values = value if isinstance(value, (list, tuple)) else (value,)
        for item in values:
            is_authority = isinstance(
                item, (SkactivemlClassifier, SkactivemlRegressor)
            )
            if is_authority and id(item) not in seen_authorities:
                seen_authorities.add(id(item))
                authorities.append(clone(item) if refit else item)
    return authorities


def _reconcile_target_declarations(
    declarations, authorities, y, *, missing_label, owner_name
):
    """Reconcile target declarations and authorities into one target type.

    This helper is shared because it consumes already collected declarations
    and authorities, i.e., it never asks how a component stores them. Only the
    collection of both is component specific and therefore owned by the
    caller.

    Parameters
    ----------
    declarations : list of (str, str)
        Declared target types with the name of their declaring component, in
        deterministic order.
    authorities : list of skactiveml estimators
        The estimators whose target semantics are authoritative for `y`, in
        deterministic order.
    y : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Target observations, including values equal to `missing_label`.
    missing_label : scalar or str or None
        Value representing a missing target observation.
    owner_name : str
        Name of the component reporting declaration conflicts.

    Returns
    -------
    target_type : str
        The resolved target type.
    target_spec : TargetSpec or None
        The resolved target specification, or `None` if no authority assigned
        a prediction task.
    """
    _check_declared_target_types(declarations)

    fitted_specs = [
        authority.target_spec_
        for authority in authorities
        if hasattr(authority, "target_spec_")
    ]
    if fitted_specs:
        # Fitted specifications outrank every declaration, so conflicting
        # declarations are reported against the resolved specification below.
        target_spec = fitted_specs[0]
        if any(spec != target_spec for spec in fitted_specs[1:]):
            raise ValueError(
                "Supplied fitted estimators have conflicting target "
                "specifications."
            )
        resolve_target_spec(
            y,
            task=target_spec.task,
            target_type=target_spec.target_type,
            annotation_type=target_spec.annotation_type,
            classes=target_spec.classes,
            missing_label=missing_label,
        )
        target_type = target_spec.target_type
    else:
        declared_target_type = _resolve_declared_target_type(
            declarations, authorities, owner_name
        )
        tasks = {_authority_task(authority) for authority in authorities}
        if len(tasks) > 1:
            raise ValueError(
                "Supplied estimators have conflicting classification and "
                "regression tasks."
            )
        if len(tasks) == 1:
            task = next(iter(tasks))
            _validate_target_semantics(
                task,
                declared_target_type,
                "single-annotator",
                allow_auto=True,
            )
            target_spec = resolve_target_spec(
                y,
                task=task,
                target_type=declared_target_type,
                annotation_type="single-annotator",
                classes=(
                    _resolve_authority_classes(authorities)
                    if task == "classification"
                    else None
                ),
                missing_label=missing_label,
            )
            target_type = target_spec.target_type
        else:
            target_type = _resolve_task_agnostic_target_type(
                y,
                target_type=declared_target_type,
                missing_label=missing_label,
            )
            target_spec = None

    for component_target_type, component_name in declarations:
        if component_target_type != "auto" and (
            component_target_type != target_type
        ):
            raise ValueError(
                f"{component_name}'s explicit `target_type` conflicts with "
                "the resolved target specification."
            )

    return target_type, target_spec


def _check_declared_target_types(declarations):
    """Raise for a declared target type outside the accepted vocabulary."""
    for target_type, _ in declarations:
        if target_type not in _ALLOWED_DECLARED_TARGET_TYPES:
            raise ValueError(
                "`target_type` must be one of {'auto', 'single-output', "
                "'multi-label', 'multi-output'}."
            )


def _resolve_declared_target_type(declarations, authorities, owner_name):
    """Resolve one explicit target type from declarations and authorities."""
    explicit_target_types = {
        target_type
        for target_type in (
            *(value for value, _ in declarations),
            *(
                getattr(authority, "target_type", "auto")
                for authority in authorities
            ),
        )
        if target_type != "auto"
    }
    if len(explicit_target_types) > 1:
        conflicting = ", ".join(
            repr(value) for value in sorted(explicit_target_types)
        )
        raise ValueError(
            f"{owner_name}'s target declaration conflicts with another "
            f"declared or supplied `target_type`: {conflicting}."
        )
    return next(iter(explicit_target_types), "auto")


def _authority_task(authority):
    """Return the prediction task an estimator authority implies."""
    return (
        "classification"
        if isinstance(authority, SkactivemlClassifier)
        else "regression"
    )


def _resolve_authority_classes(authorities):
    """Resolve one class vocabulary shared by all classifier authorities."""
    resolved_classes = None
    resolved_key = None
    for authority in authorities:
        classes = getattr(authority, "classes_", authority.classes)
        if classes is None:
            continue
        key = _class_vocabulary_key(classes)
        if resolved_key is None:
            resolved_classes, resolved_key = classes, key
        elif key != resolved_key:
            raise ValueError(
                "Supplied estimators have conflicting class vocabularies: "
                f"{resolved_classes!r} and {classes!r}."
            )
    return resolved_classes


def _check_resolved_target_capability(
    component, target_type, target_spec, capabilities
):
    """Check a resolved target against a component's capabilities.

    This helper is shared because it needs the resolved target and the
    capability set only, both of which every component already exposes. It
    covers the task-agnostic case as well, where no authority assigned a
    prediction task and therefore no exact semantic triple exists.

    Parameters
    ----------
    component : str
        Name of the component whose capabilities are checked.
    target_type : str
        The resolved target type.
    target_spec : TargetSpec or None
        The resolved target specification, or `None` for a task-agnostic
        resolution.
    capabilities : set of (str, str, str)
        The component's declared target capabilities.
    """
    if target_spec is not None:
        _check_target_spec_capability(component, target_spec, capabilities)
    elif not any(capability[1] == target_type for capability in capabilities):
        supported = ", ".join(repr(value) for value in sorted(capabilities))
        raise ValueError(
            f"{component} does not support target type {target_type!r}. "
            f"Supported capabilities are: {supported}."
        )
