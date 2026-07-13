"""Resolution of target semantics."""

from dataclasses import dataclass

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_array

from ._label import MISSING_LABEL, is_unlabeled
from ._validation import check_classifier_params


@dataclass(frozen=True)
class TargetSpec:
    """Immutable description of resolved target semantics.

    Parameters
    ----------
    task : {"classification", "regression"}
        The prediction task.
    target_type : {"single-output", "multi-label", "multi-output"}
        The resolved structure of one sample's target.
    annotation_type : {"single-annotator", "multi-annotator"}
        Whether observations come from one or multiple annotators.
    classes : tuple or None
        The normalized immutable class vocabulary for classification, or
        `None` for regression.
    """

    task: str
    target_type: str
    annotation_type: str
    classes: tuple | None

    def __post_init__(self):
        if self.classes is not None:
            object.__setattr__(self, "classes", _freeze_classes(self.classes))


def _freeze_classes(classes):
    return tuple(
        (
            _freeze_classes(value)
            if isinstance(value, (list, tuple, np.ndarray))
            else value
        )
        for value in classes
    )


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
    task : {"classification", "regression"}
        The prediction task.
    target_type : {"auto", "single-output", "multi-label", "multi-output"}, \
            default="auto"
        Declared target type. The returned specification is always concrete.
    annotation_type : {"single-annotator", "multi-annotator"}, \
            default="single-annotator"
        Declared annotation type.
    classes : array-like or tuple of array-like, default=None
        Optional classification class vocabularies.
    missing_label : scalar or str or None, default=np.nan
        Value representing a missing target observation.

    Returns
    -------
    target_spec : TargetSpec
        The resolved immutable target specification.
    """
    if task not in {"classification", "regression"}:
        raise ValueError(
            "`task` must be either 'classification' or 'regression'."
        )
    if target_type not in {
        "auto",
        "single-output",
        "multi-label",
        "multi-output",
    }:
        raise ValueError(
            "`target_type` must be one of {'auto', 'single-output', "
            "'multi-label', 'multi-output'}."
        )
    if annotation_type not in {"single-annotator", "multi-annotator"}:
        raise ValueError(
            "`annotation_type` must be either 'single-annotator' or "
            "'multi-annotator'."
        )
    if task == "regression" and target_type == "multi-label":
        raise ValueError(
            "`target_type='multi-label'` requires classification."
        )
    if task == "classification" and classes is not None:
        check_classifier_params(classes, missing_label)

    y = check_array(
        y,
        ensure_2d=False,
        ensure_all_finite=False,
        ensure_min_samples=0,
        dtype=None,
    )

    if target_type == "multi-label":
        return _resolve_multilabel(
            y,
            task=task,
            annotation_type=annotation_type,
            classes=classes,
            missing_label=missing_label,
        )

    if target_type == "auto":
        if classes is not None and _has_nested_classes(classes):
            if all(len(classes_i) == 2 for classes_i in classes):
                return _resolve_multilabel(
                    y,
                    task=task,
                    annotation_type=annotation_type,
                    classes=classes,
                    missing_label=missing_label,
                )
            target_type = "multi-output"
        elif classes is not None or y.ndim == 1:
            target_type = "single-output"
        else:
            raise ValueError(
                "Two-dimensional targets with `target_type='auto'` are "
                "ambiguous; declare `target_type` or `classes`."
            )

    if target_type == "multi-output":
        normalized_classes = None
        if task == "classification" and classes is not None:
            normalized_classes = tuple(
                tuple(LabelEncoder().fit(classes_i).classes_)
                for classes_i in classes
            )
        return TargetSpec(
            task=task,
            target_type="multi-output",
            annotation_type=annotation_type,
            classes=normalized_classes,
        )
    if target_type != "single-output":
        raise ValueError(f"Cannot resolve `target_type='{target_type}'`.")

    normalized_classes = None
    if task == "classification":
        observed = np.asarray(y)[~is_unlabeled(y, missing_label=missing_label)]
        declared = observed if classes is None else np.asarray(classes)
        if classes is None and len(observed) == 0:
            raise ValueError(
                "No class label is observed and `classes` is not defined."
            )
        if classes is not None and not np.isin(observed, declared).all():
            _raise_unknown_class_error(observed, declared, "`classes`")
        normalized_classes = tuple(LabelEncoder().fit(declared).classes_)

    return TargetSpec(
        task=task,
        target_type="single-output",
        annotation_type=annotation_type,
        classes=normalized_classes,
    )


def _has_nested_classes(classes):
    if isinstance(classes, (str, bytes)):
        return False
    try:
        values = list(classes)
    except TypeError:
        return False
    return bool(values) and all(
        not isinstance(value, (str, bytes)) and hasattr(value, "__iter__")
        for value in values
    )


def _resolve_multilabel(y, *, task, annotation_type, classes, missing_label):
    if task != "classification":
        raise ValueError(
            "`target_type='multi-label'` requires classification."
        )
    if annotation_type != "single-annotator":
        raise ValueError(
            "Multi-label multi-annotator targets are not supported."
        )
    if y.ndim != 2:
        raise ValueError("Multi-label targets must be two-dimensional.")

    is_missing = is_unlabeled(
        y, missing_label=missing_label, target_type="multi-label"
    )
    observed_y = y[~is_missing]

    if classes is None:
        normalized_classes = []
        for output_idx in range(y.shape[1]):
            classes_i = LabelEncoder().fit(observed_y[:, output_idx]).classes_
            if len(classes_i) != 2:
                raise ValueError(
                    "Each multi-label output must expose exactly two observed "
                    "classes when `classes=None`; output "
                    f"{output_idx} exposes {len(classes_i)}."
                )
            normalized_classes.append(tuple(classes_i))
    else:
        if len(classes) != y.shape[1]:
            raise ValueError(
                "Multi-label `classes` must contain one vocabulary per "
                "target column."
            )
        normalized_classes = []
        for output_idx, declared in enumerate(classes):
            classes_i = LabelEncoder().fit(declared).classes_
            if len(classes_i) != 2:
                raise ValueError(
                    "Each multi-label class vocabulary must contain exactly "
                    f"two classes; output {output_idx} contains "
                    f"{len(classes_i)}."
                )
            if not np.isin(observed_y[:, output_idx], classes_i).all():
                _raise_unknown_class_error(
                    observed_y[:, output_idx],
                    classes_i,
                    f"`classes[{output_idx}]`",
                )
            normalized_classes.append(tuple(classes_i))

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
    if capability not in capabilities:
        supported = ", ".join(repr(value) for value in sorted(capabilities))
        raise ValueError(
            f"{component} does not support target capability {capability!r}. "
            f"Supported capabilities are: {supported}."
        )


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
