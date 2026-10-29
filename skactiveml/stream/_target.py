"""Shared target resolution helpers for stream-based query strategies.

The classifier passed to `query` determines the target specification.
"""

import numpy as np
from sklearn.base import clone

from ..base import SkactivemlClassifier
from ..utils import check_type, resolve_target_spec
from ..utils._target import (
    _check_refit_capability,
    _check_target_matches_spec,
    _check_target_spec_capability,
    _declared_target_type,
    _predicted_target_spec,
)
from ..utils._validation import _has_nested_classes


def _resolve_clf_target_spec(strategy, clf, y):
    """Resolve the target specification `clf` is the authority for.

    Parameters
    ----------
    strategy : skactiveml.base.SingleAnnotatorStreamQueryStrategy
        The query strategy whose capabilities are checked.
    clf : skactiveml.base.SkactivemlClassifier
        The classifier passed to `query`, or a clone of it.
    y : array-like of shape (n_samples,) or (n_samples, n_outputs) or \
            None
        Labels of the training data set, if any.

    Returns
    -------
    target_spec : skactiveml.utils.TargetSpec
        The resolved target specification.

    Raises
    ------
    ValueError
        If `clf` or the strategy does not support the target specification,
        if `y` does not match a fitted `clf`, or if an unfitted `clf` declares
        no `classes` and `y` provides no class evidence.
    """
    clf_target_spec = getattr(clf, "target_spec_", None)
    if clf_target_spec is None:
        classes = clf._declared_classes
        if y is None:
            # Only the declared vocabulary is evidence. Its nesting
            # decides the number of label outputs of the empty labels.
            n_outputs = (
                len(classes)
                if classes is not None and _has_nested_classes(classes)
                else None
            )
            y = np.full(
                (0,) if n_outputs is None else (0, n_outputs),
                clf.missing_label,
            )
        target_spec = resolve_target_spec(
            y,
            task="classification",
            target_type=_declared_target_type(y, clf),
            annotation_type="single-annotator",
            classes=classes,
            missing_label=clf.missing_label,
        )
        clf_target_spec = target_spec
    else:
        target_spec = _predicted_target_spec(clf_target_spec)
        if y is not None:
            _check_target_matches_spec(
                y, target_spec, missing_label=clf.missing_label
            )
    _check_target_spec_capability(
        type(clf).__name__, clf_target_spec, clf._target_capabilities
    )
    _check_target_spec_capability(
        type(strategy).__name__, target_spec, strategy._target_capabilities
    )
    return target_spec


def _fit_and_resolve_clf_target_spec(
    strategy, clf, X, y, *, fit_clf, sample_weight
):
    """Validate, optionally fit, and resolve the classifier of a stream query.

    Parameters
    ----------
    strategy : skactiveml.base.SingleAnnotatorStreamQueryStrategy
        The query strategy whose capabilities are checked.
    clf : skactiveml.base.SkactivemlClassifier
        Model implementing the methods `fit` and `predict_proba`.
    X : array-like of shape (n_samples, n_features) or None
        Training data set used to fit the classifier.
    y : array-like of shape (n_samples,) or (n_samples, n_outputs) or \
            None
        Labels of the training data set (possibly including unlabeled ones
        indicated by `clf.missing_label`).
    fit_clf : bool
        Defines whether a clone of the classifier is fitted on `X`, `y`, and
        `sample_weight`.
    sample_weight : array-like of shape (n_samples,) or None
        Weights of training samples in `X`.

    Returns
    -------
    clf : skactiveml.base.SkactivemlClassifier
        The checked classifier, i.e., the fitted clone if `fit_clf` is `True`.
    target_spec : skactiveml.utils.TargetSpec
        The resolved target specification.
    """
    check_type(clf, "clf", SkactivemlClassifier)
    check_type(fit_clf, "fit_clf", bool)
    if fit_clf:
        _check_refit_capability(clf)
        clf = clone(clf)
        if sample_weight is None:
            clf = clf.fit(X, y)
        else:
            clf = clf.fit(X, y, sample_weight)
    target_spec = _resolve_clf_target_spec(strategy, clf, y)
    return clf, target_spec
