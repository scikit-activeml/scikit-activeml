from sklearn import clone

from ..base import SkactivemlClassifier, SkactivemlRegressor
from ..utils import (
    check_equal_missing_label,
    check_type,
    is_unlabeled,
    resolve_target_spec,
)
from ..utils._target import (
    _resolve_task_agnostic_target_type,
    _validate_target_semantics,
    check_target_capability,
)


def _resolve_estimator_target_spec(strategy, estimator, y):
    """Resolve one estimator-backed strategy target specification."""
    if isinstance(estimator, SkactivemlClassifier):
        task = "classification"
        classes = getattr(estimator, "classes_", estimator.classes)
    elif isinstance(estimator, SkactivemlRegressor):
        task = "regression"
        classes = None
    else:  # pragma: no cover - callers validate their estimator type first.
        raise TypeError("`estimator` must be a scikit-activeml estimator.")

    strategy_target_type = strategy.target_type
    _validate_target_semantics(
        task,
        strategy_target_type,
        "single-annotator",
        allow_auto=True,
    )

    if hasattr(estimator, "target_spec_"):
        target_spec = estimator.target_spec_
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
        check_target_capability(
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
    check_target_capability(
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
        if sample_weight is None:
            estimator = clone(estimator).fit(X, y)
        else:
            estimator = clone(estimator).fit(X, y, sample_weight)
    target_spec = _resolve_estimator_target_spec(strategy, estimator, y)
    return estimator, target_spec


def _resolve_wrapper_target_type(wrapper, y, query_kwargs):
    """Resolve the target structure a strategy wrapper must preserve."""
    allowed_target_types = {
        "auto",
        "single-output",
        "multi-label",
        "multi-output",
    }
    target_declarations = [(wrapper.target_type, type(wrapper).__name__)]
    wrapped_strategy = wrapper.query_strategy
    seen_strategies = set()
    while (
        wrapped_strategy is not None
        and id(wrapped_strategy) not in seen_strategies
    ):
        seen_strategies.add(id(wrapped_strategy))
        target_declarations.append(
            (
                getattr(wrapped_strategy, "target_type", "auto"),
                type(wrapped_strategy).__name__,
            )
        )
        wrapped_strategy = getattr(wrapped_strategy, "query_strategy", None)

    for target_type, _ in target_declarations:
        if target_type not in allowed_target_types:
            raise ValueError(
                "`target_type` must be one of {'auto', 'single-output', "
                "'multi-label', 'multi-output'}."
            )

    estimators = []
    for value in query_kwargs.values():
        values = value if isinstance(value, (list, tuple)) else (value,)
        estimators.extend(
            item
            for item in values
            if isinstance(item, (SkactivemlClassifier, SkactivemlRegressor))
        )

    fitted_specs = [
        estimator.target_spec_
        for estimator in estimators
        if hasattr(estimator, "target_spec_")
    ]
    if fitted_specs:
        target_spec = fitted_specs[0]
        if any(spec != target_spec for spec in fitted_specs[1:]):
            raise ValueError(
                "Supplied fitted estimators have conflicting target "
                "specifications."
            )
        target_type = target_spec.target_type
        task = target_spec.task
    else:
        estimator_target_types = [
            getattr(estimator, "target_type", "auto")
            for estimator in estimators
        ]
        tasks = {
            (
                "classification"
                if isinstance(estimator, SkactivemlClassifier)
                else "regression"
            )
            for estimator in estimators
        }
        explicit_target_types = {
            target_type
            for target_type in (
                *(value for value, _ in target_declarations),
                *estimator_target_types,
            )
            if target_type != "auto"
        }
        if len(explicit_target_types) > 1:
            raise ValueError(
                f"{type(wrapper).__name__}'s target declaration conflicts "
                "with the wrapped strategy or supplied estimator."
            )
        declared_target_type = next(iter(explicit_target_types), "auto")
        if len(tasks) == 1:
            task = next(iter(tasks))
            _validate_target_semantics(
                task,
                declared_target_type,
                "single-annotator",
                allow_auto=True,
            )
            estimator = estimators[0]
            target_spec = resolve_target_spec(
                y,
                task=task,
                target_type=declared_target_type,
                annotation_type="single-annotator",
                classes=(
                    getattr(estimator, "classes_", estimator.classes)
                    if task == "classification"
                    else None
                ),
                missing_label=wrapper.missing_label,
            )
            target_type = target_spec.target_type
        elif len(tasks) > 1:
            raise ValueError(
                "Supplied estimators have conflicting classification and "
                "regression tasks."
            )
        else:
            task = None
            target_type = _resolve_task_agnostic_target_type(
                y,
                target_type=declared_target_type,
                missing_label=wrapper.missing_label,
            )
            target_spec = None

    for component_target_type, component_name in target_declarations:
        if component_target_type != "auto" and (
            component_target_type != target_type
        ):
            raise ValueError(
                f"{component_name}'s explicit `target_type` conflicts with "
                "the resolved target specification."
            )

    capabilities = wrapper._target_capabilities
    if capabilities:
        if target_spec is not None:
            check_target_capability(
                type(wrapper.query_strategy).__name__,
                target_spec,
                capabilities,
            )
        elif not any(
            capability[1] == target_type for capability in capabilities
        ):
            supported = ", ".join(
                repr(value) for value in sorted(capabilities)
            )
            raise ValueError(
                f"{type(wrapper.query_strategy).__name__} does not support "
                f"target type {target_type!r}. Supported capabilities are: "
                f"{supported}."
            )

    is_unlabeled(
        y,
        missing_label=wrapper.missing_label,
        target_type=target_type,
    )
    return target_type
