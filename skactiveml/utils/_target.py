"""Resolution of target semantics."""

from dataclasses import dataclass

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_array

from ._label import MISSING_LABEL, is_unlabeled
from ._validation import (
    _has_nested_classes,
    check_classes,
    check_classifier_params,
)

_NAN_CLASS_HASH_KEY = object()


@dataclass(frozen=True, eq=False)
class TargetSpec:
    """Immutable description of resolved target semantics.

    Parameters
    ----------
    task : "classification" or "regression"
        The prediction task.
    target_type : "single-output" or "multi-label" or "multi-output"
        The resolved structure of one sample's target.
    annotation_type : "single-annotator" or "multi-annotator"
        Whether observations come from one or multiple annotators.
    classes : tuple or None
        The normalized immutable class vocabulary for classification, or
        `None` for regression. Single-output classification stores one tuple;
        multi-label classification stores one binary tuple per label output.
    """

    task: str
    target_type: str
    annotation_type: str
    classes: tuple | None

    def __post_init__(self):
        _validate_target_semantics(
            self.task,
            self.target_type,
            self.annotation_type,
            allow_auto=False,
        )
        if self.task == "classification" and self.classes is None:
            raise ValueError(
                "`classes` is required for classification targets."
            )
        if self.task == "regression" and self.classes is not None:
            raise ValueError(
                "`classes` is not accepted for regression targets."
            )
        if self.classes is not None:
            check_classes(self.classes)
            has_nested_classes = _check_class_vocabulary_structure(
                self.target_type, self.classes
            )
            if has_nested_classes:
                normalized_classes = tuple(
                    _normalize_class_vocabulary(
                        classes_i, [], f"`classes[{output_idx}]`"
                    )
                    for output_idx, classes_i in enumerate(self.classes)
                )
            else:
                normalized_classes = _normalize_class_vocabulary(
                    self.classes, [], "`classes`"
                )
            object.__setattr__(self, "classes", normalized_classes)

    def __eq__(self, other):
        if not isinstance(other, TargetSpec):
            return NotImplemented
        return (
            self.task == other.task
            and self.target_type == other.target_type
            and self.annotation_type == other.annotation_type
            and _class_vocabularies_equal(self.classes, other.classes)
        )

    def __hash__(self):
        return hash(
            (
                self.task,
                self.target_type,
                self.annotation_type,
                _class_vocabulary_hash_key(self.classes),
            )
        )


def _class_vocabularies_equal(classes_a, classes_b):
    if classes_a is None or classes_b is None:
        return classes_a is classes_b
    if len(classes_a) != len(classes_b):
        return False
    return all(
        (
            _class_vocabularies_equal(class_a, class_b)
            if isinstance(class_a, tuple) and isinstance(class_b, tuple)
            else class_a == class_b
            or (_is_nan_class(class_a) and _is_nan_class(class_b))
        )
        for class_a, class_b in zip(classes_a, classes_b)
    )


def _class_vocabulary_key(classes):
    """Return a comparable key for a declared class vocabulary.

    Unlike `_class_vocabulary_hash_key`, this helper also accepts the not yet
    normalized vocabularies that estimators expose through `classes` or
    `classes_`, i.e., arbitrarily nested sequences and arrays.

    Parameters
    ----------
    classes : array-like or tuple of array-like or None
        The declared class vocabulary.

    Returns
    -------
    key : tuple or None
        A key comparing equal exactly for equal vocabularies.
    """
    if classes is None:
        return None
    return _class_vocabulary_hash_key(_as_nested_tuple(classes))


def _as_nested_tuple(values):
    """Convert nested sequences of class labels into nested tuples."""
    return tuple(
        (
            _as_nested_tuple(value)
            if isinstance(value, (list, tuple, np.ndarray))
            else value
        )
        for value in values
    )


def _class_vocabulary_hash_key(classes):
    if classes is None:
        return None
    return tuple(
        (
            _class_vocabulary_hash_key(value)
            if isinstance(value, tuple)
            else _NAN_CLASS_HASH_KEY if _is_nan_class(value) else value
        )
        for value in classes
    )


def _is_nan_class(value):
    return bool(value != value)


def _validate_target_semantics(
    task, target_type, annotation_type, *, allow_auto
):
    if task not in {"classification", "regression"}:
        raise ValueError(
            "`task` must be either 'classification' or 'regression'."
        )

    target_types = {"single-output", "multi-label", "multi-output"}
    if allow_auto:
        target_types.add("auto")
    if target_type not in target_types:
        allowed = (
            "{'auto', 'single-output', 'multi-label', 'multi-output'}"
            if allow_auto
            else "{'single-output', 'multi-label', 'multi-output'}"
        )
        raise ValueError(f"`target_type` must be one of {allowed}.")

    if annotation_type not in {"single-annotator", "multi-annotator"}:
        raise ValueError(
            "`annotation_type` must be either 'single-annotator' or "
            "'multi-annotator'."
        )
    if task == "regression" and target_type == "multi-label":
        raise ValueError(
            "`target_type='multi-label'` requires classification."
        )
    if (
        target_type in {"multi-label", "multi-output"}
        and annotation_type == "multi-annotator"
    ):
        target_name = (
            "Multi-label" if target_type == "multi-label" else "Multi-output"
        )
        raise ValueError(
            f"{target_name} targets cannot be combined with "
            "`annotation_type='multi-annotator'`."
        )


def _check_class_vocabulary_structure(target_type, classes):
    """Validate flat or nested classes against a resolved target type."""
    if classes is None:
        return False

    has_nested_classes = _has_nested_classes(classes)
    if target_type == "single-output" and has_nested_classes:
        raise ValueError(
            "Single-output classification requires a flat class vocabulary."
        )
    if target_type in {"multi-label", "multi-output"}:
        if not has_nested_classes:
            vocabulary_kind = (
                "nested binary class vocabularies"
                if target_type == "multi-label"
                else "nested class vocabularies"
            )
            raise ValueError(
                f"{target_type.capitalize()} classification requires "
                f"{vocabulary_kind}."
            )
        if target_type == "multi-label" and not all(
            len(classes_i) == 2 for classes_i in classes
        ):
            raise ValueError(
                "Each multi-label class vocabulary must contain exactly two "
                "classes."
            )
    return has_nested_classes


def resolve_target_spec(
    y,
    *,
    task,
    target_type="auto",
    annotation_type="single-annotator",
    classes=None,
    missing_label=MISSING_LABEL,
):
    """Resolve declared intent and target evidence to a target specification.

    Parameters
    ----------
    y : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Target observations, including values equal to `missing_label`.
    task : "classification" or "regression"
        The prediction task.
    target_type : "auto" or "single-output" or "multi-label" or \
            "multi-output", default="auto"
        Declared target type. The returned specification is always concrete.
    annotation_type : "single-annotator" or "multi-annotator", \
            default="single-annotator"
        Declared annotation type.
    classes : array-like or tuple of array-like, default=None
        Optional classification class vocabularies. A flat vocabulary
        describes single-output classification. Nested binary vocabularies
        describe multi-label classification; nested non-binary vocabularies
        describe recognized multi-output classification semantics.
    missing_label : scalar or str or None, default=np.nan
        Value representing a missing target observation.

    Returns
    -------
    target_spec : TargetSpec
        The resolved immutable target specification.
    """
    _validate_target_semantics(
        task, target_type, annotation_type, allow_auto=True
    )
    if task == "regression" and classes is not None:
        raise ValueError("`classes` is not accepted for regression targets.")
    if task == "classification" and classes is not None:
        check_classifier_params(classes, missing_label)

    y = _check_target_array(y)

    if target_type == "auto":
        if annotation_type == "multi-annotator":
            if classes is not None and _has_nested_classes(classes):
                raise ValueError(
                    "Nested class vocabularies contradict "
                    "`annotation_type='multi-annotator'`."
                )
            target_type = "single-output"
        elif task == "regression":
            target_type = (
                "single-output"
                if y.ndim == 1 or y.shape[1] == 1
                else "multi-output"
            )
        elif classes is not None:
            if _has_nested_classes(classes):
                target_type = (
                    "multi-label"
                    if all(len(classes_i) == 2 for classes_i in classes)
                    else "multi-output"
                )
            else:
                target_type = "single-output"
        elif y.ndim == 1:
            target_type = "single-output"
        else:
            raise ValueError(
                "Two-dimensional targets with `target_type='auto'` are "
                "ambiguous; declare `target_type` or `classes`."
            )

    _check_class_vocabulary_structure(target_type, classes)

    if target_type == "multi-label":
        return _resolve_multilabel(
            y,
            classes=classes,
            missing_label=missing_label,
        )
    if target_type == "multi-output":
        return _resolve_multioutput(
            y,
            task=task,
            classes=classes,
            missing_label=missing_label,
        )
    _validate_single_output_shape(y, task, annotation_type)

    normalized_classes = None
    if task == "classification":
        observed = np.asarray(y)[~is_unlabeled(y, missing_label=missing_label)]
        declared = observed if classes is None else np.asarray(classes)
        if classes is None and len(observed) == 0:
            raise ValueError(
                "No class label is observed and `classes` is not defined."
            )
        normalized_classes = _normalize_class_vocabulary(
            declared, observed, "`classes`"
        )

    return TargetSpec(
        task=task,
        target_type="single-output",
        annotation_type=annotation_type,
        classes=normalized_classes,
    )


def _resolve_task_agnostic_target_type(
    y,
    *,
    target_type="auto",
    missing_label=MISSING_LABEL,
):
    """Resolve target structure without assigning a prediction task."""
    _validate_target_semantics(
        "classification",
        target_type,
        "single-annotator",
        allow_auto=True,
    )
    y = _check_target_array(y)

    if target_type == "auto":
        if y.ndim != 1:
            raise ValueError(
                "Two-dimensional targets with `target_type='auto'` are "
                "ambiguous for a task-agnostic strategy; declare "
                "`target_type`."
            )
        return "single-output"

    if target_type == "single-output":
        if y.ndim == 2 and y.shape[1] == 1:
            return target_type
        if y.ndim != 1:
            raise ValueError(
                "Single-output targets must be one-dimensional or a column "
                "vector."
            )
        return target_type

    if target_type == "multi-label":
        if y.ndim != 2:
            raise ValueError("Multi-label targets must be two-dimensional.")
        is_unlabeled(
            y,
            missing_label=missing_label,
            target_type="multi-label",
        )
        return target_type

    if y.ndim != 2 or y.shape[1] < 2:
        raise ValueError(
            "Multi-output targets must be two-dimensional with at least two "
            "target columns."
        )
    return target_type


def _check_target_array(y):
    y = check_array(
        y,
        ensure_2d=False,
        ensure_all_finite=False,
        ensure_min_samples=0,
        dtype=None,
    )
    if y.ndim == 0:
        raise TypeError("`y` must be a one- or two-dimensional array-like.")
    return y


def _validate_single_output_shape(y, task, annotation_type):
    if annotation_type == "multi-annotator":
        if y.ndim != 2:
            raise ValueError(
                "Multi-annotator targets must be two-dimensional."
            )
        return

    if task == "classification" and y.ndim != 1:
        raise ValueError(
            "Single-output, single-annotator classification targets must be "
            "one-dimensional."
        )
    if task == "regression" and y.ndim == 2 and y.shape[1] != 1:
        raise ValueError(
            "Single-output regression targets must be one-dimensional or a "
            "column vector."
        )


def _resolve_multioutput(y, *, task, classes, missing_label):
    if y.ndim != 2 or y.shape[1] < 2:
        raise ValueError(
            "Multi-output targets must be two-dimensional with at least two "
            "target columns."
        )

    normalized_classes = None
    if task == "classification":
        if classes is not None and len(classes) != y.shape[1]:
            raise ValueError(
                "Multi-output `classes` must contain one vocabulary per "
                "target column."
            )

        missing = is_unlabeled(y, missing_label=missing_label)
        normalized_classes = []
        for output_idx in range(y.shape[1]):
            observed = y[:, output_idx][~missing[:, output_idx]]
            declared = observed if classes is None else classes[output_idx]
            if len(declared) == 0:
                raise ValueError(
                    "No class label is observed for multi-output target "
                    f"column {output_idx} and `classes` is not defined."
                )
            normalized_classes.append(
                _normalize_class_vocabulary(
                    declared, observed, f"`classes[{output_idx}]`"
                )
            )
        normalized_classes = tuple(normalized_classes)

    return TargetSpec(
        task=task,
        target_type="multi-output",
        annotation_type="single-annotator",
        classes=normalized_classes,
    )


def _resolve_multilabel(y, *, classes, missing_label):
    if y.ndim != 2:
        raise ValueError("Multi-label targets must be two-dimensional.")

    is_missing = is_unlabeled(
        y, missing_label=missing_label, target_type="multi-label"
    )
    observed_y = y[~is_missing]

    if classes is None:
        normalized_classes = []
        for output_idx in range(y.shape[1]):
            classes_i = _normalize_class_vocabulary(
                observed_y[:, output_idx],
                observed_y[:, output_idx],
                f"`classes[{output_idx}]`",
            )
            if len(classes_i) != 2:
                raise ValueError(
                    "Each multi-label output must expose exactly two observed "
                    "classes when `classes=None`; output "
                    f"{output_idx} exposes {len(classes_i)}."
                )
            normalized_classes.append(classes_i)
    else:
        if len(classes) != y.shape[1]:
            raise ValueError(
                "Multi-label `classes` must contain one vocabulary per "
                "target column."
            )
        normalized_classes = []
        for output_idx, declared in enumerate(classes):
            classes_i = _normalize_class_vocabulary(
                declared,
                observed_y[:, output_idx],
                f"`classes[{output_idx}]`",
            )
            normalized_classes.append(classes_i)

    return TargetSpec(
        task="classification",
        target_type="multi-label",
        annotation_type="single-annotator",
        classes=tuple(normalized_classes),
    )


def check_target_capability(component, target_spec, capabilities):
    """Raise when a component does not declare an exact target capability."""
    capability = (
        target_spec.task,
        target_spec.target_type,
        target_spec.annotation_type,
    )
    _check_target_capability(component, capability, capabilities)


def _check_target_capability(component, capability, capabilities):
    """Raise when a component does not declare an exact semantic triple."""
    if capability not in capabilities:
        supported = ", ".join(repr(value) for value in sorted(capabilities))
        raise ValueError(
            f"{component} does not support target capability {capability!r}. "
            f"Supported capabilities are: {supported}."
        )


def _normalize_class_vocabulary(declared, observed, name):
    normalized = tuple(LabelEncoder().fit(declared).classes_)
    if not np.isin(observed, normalized).all():
        _raise_unknown_class_error(observed, normalized, name)
    return normalized


def _raise_unknown_class_error(observed, declared, name):
    observed_dtype = np.asarray(observed).dtype
    declared_dtype = np.asarray(declared).dtype
    observed_is_numeric = np.issubdtype(observed_dtype, np.number)
    declared_is_numeric = np.issubdtype(declared_dtype, np.number)
    if observed_is_numeric != declared_is_numeric:
        raise TypeError(
            f"The labels in `y` are not type-compatible with {name}."
        )
    raise ValueError(f"`y` contains labels outside {name}.")
