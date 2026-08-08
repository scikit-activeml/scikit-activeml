"""
Wrapper for scikit-learn classifiers to deal with missing labels and labels
from multiple annotators.
"""

# Author: Marek Herde <marek.herde@uni-kassel.de>
import warnings
from collections import deque
from copy import deepcopy
from dataclasses import dataclass
import inspect

import numpy as np
from sklearn.base import MetaEstimatorMixin, is_classifier
from sklearn.utils import get_tags
from sklearn.utils.validation import (
    check_is_fitted,
    check_array,
    has_fit_parameter,
    column_or_1d,
)

from sklearn.utils import check_consistent_length
from sklearn.exceptions import NotFittedError

from ..base import SkactivemlClassifier, _resolve_own_fitted_attribute
from ..utils._target import _check_target_capability, check_target_capability
from ..utils import (
    rand_argmin,
    MISSING_LABEL,
    ExtLabelEncoder,
    is_labeled,
    check_random_state,
    check_equal_missing_label,
    check_classifier_params,
    check_type,
    check_scalar,
    match_signature,
    check_n_features,
    _has_nested_classes,
    resolve_target_spec,
)

# used to defer import of capymoa as it may result in an error with pytest
import importlib

successful_skorch_torch_import = False
try:
    import torch
    from torch import nn
    from skactiveml.base import SkorchMixin
    from skactiveml.utils import make_criterion_tuple_aware

    successful_skorch_torch_import = True
except ImportError:  # pragma: no cover
    pass

spec = importlib.util.find_spec("capymoa")
successful_capymoa_import = spec is not None

successful_river_import = False
try:
    from inspect import signature
    import river
    import river.base
    import pandas as pd

    successful_river_import = True
except ImportError:  # pragma: no cover
    pass


def _declares_multilabel_support(estimator):
    """Check whether `estimator` declares multi-label classification support.

    The decision is made from the estimator's own declarations only, never by
    fitting or predicting on synthetic data, because such a probe can mutate
    the estimator and report false negatives caused by estimator parameters,
    data constraints, or minimum sample requirements.

    Exposing `predict_proba` is not sufficient, since it says nothing about
    whether an estimator accepts a two-dimensional target. Neither is the
    presence of a tags object, because both tags default to negative. An
    estimator therefore has to declare its capability positively.

    Parameters
    ----------
    estimator : object
        The estimator whose declared capabilities are inspected.

    Returns
    -------
    declares_multilabel_support : bool
        `True`, if `estimator` is a classifier that exposes `predict_proba`
        and positively declares either the `scikit-learn` multi-output or the
        multi-label tag, and `False` otherwise.
    """
    if not is_classifier(estimator) or not hasattr(estimator, "predict_proba"):
        return False
    tags = get_tags(estimator)
    return bool(
        getattr(tags.target_tags, "multi_output", False)
        or getattr(tags.classifier_tags, "multi_label", False)
    )


def _multilabel_capability_error(component, estimator):
    """Compose the error message for a rejected multi-label `estimator`."""
    return (
        f"'{component}' does not support multi-label classification with "
        f"the estimator '{estimator}'. A multi-label estimator must be a "
        f"scikit-learn classifier, implement 'predict_proba', and positively "
        f"declare either 'target_tags.multi_output' or "
        f"'classifier_tags.multi_label' through its estimator tags. "
        f"Estimators declaring neither are rejected because they are not "
        f"guaranteed to accept a two-dimensional target. Either wrap the "
        f"estimator, e.g., via 'sklearn.multioutput.MultiOutputClassifier', "
        f"or use "
        f"'target_type=\"single-output\"'."
    )


#: The class labels of one binary indicator column of a multi-label target.
_INDICATOR_CLASSES = (0, 1)


@dataclass(frozen=True)
class _FittedTargetEvidence:
    """Target evidence that a pre-fitted estimator publishes about itself.

    A pre-fitted estimator's learned class vocabulary determines the meaning
    of its predictions and probability columns. This value describes that
    meaning as the estimator itself reports it, so that declared target
    semantics can be reconciled with it before fitted wrapper state is
    published. It is read from fitted attributes only, never by calling a
    prediction method on invented input.

    Parameters
    ----------
    kind : str
        The kind of target evidence the estimator publishes, one of:

        - `"single-output"`: one flat learned class vocabulary, describing one
          categorical assignment per sample;
        - `"label-vocabularies"`: one learned class vocabulary per label
          output, as published by `sklearn.multioutput.MultiOutputClassifier`
          and a multi-output `sklearn.ensemble.RandomForestClassifier`;
        - `"label-outputs"`: flat learned classes identifying label outputs,
          published together with explicit fitted multi-label metadata as by
          `sklearn.multiclass.OneVsRestClassifier`; or
        - `"unknown"`: no learned class vocabulary at all.
    classes : numpy.ndarray or tuple of numpy.ndarray or None
        The learned class vocabulary for `"single-output"`, one learned class
        vocabulary per label output for `"label-vocabularies"`, and `None`
        otherwise, because flat label-output identifiers are no class
        vocabulary and an unknown vocabulary cannot be described.
    n_label_outputs : int or None
        The number of label outputs for `"label-vocabularies"` and
        `"label-outputs"`, and `None` otherwise.
    """

    kind: str
    classes: np.ndarray | tuple | None
    n_label_outputs: int | None

    @property
    def describes_label_outputs(self):
        """Whether the estimator's metadata describes several label outputs."""
        return self.kind in {"label-vocabularies", "label-outputs"}


def _discover_fitted_target_evidence(estimator):
    """Read the target evidence a pre-fitted `estimator` publishes.

    Parameters
    ----------
    estimator : object
        The pre-fitted estimator whose fitted attributes are inspected.

    Returns
    -------
    evidence : _FittedTargetEvidence
        The target evidence published by `estimator`.
    """
    classes = getattr(estimator, "classes_", None)
    if classes is None:
        return _FittedTargetEvidence("unknown", None, None)
    if _has_nested_classes(classes):
        vocabularies = tuple(np.asarray(classes_j) for classes_j in classes)
        return _FittedTargetEvidence(
            "label-vocabularies", vocabularies, len(vocabularies)
        )
    classes = np.asarray(classes)
    if bool(getattr(estimator, "multilabel_", False)):
        # A fitted multi-label estimator such as `OneVsRestClassifier`
        # publishes one flat identifier per label output instead of one class
        # vocabulary per output.
        return _FittedTargetEvidence("label-outputs", None, len(classes))
    return _FittedTargetEvidence("single-output", classes, None)


def _class_column(class_label, declared_classes):
    """Locate one class label in a declared class vocabulary.

    A learned class label is never `numpy.nan`, because `scikit-learn` rejects
    such a target, so the labels are compared by plain equality.

    Parameters
    ----------
    class_label : scalar
        The class label to locate, e.g., one an estimator learned during its
        own fitting.
    declared_classes : array-like of shape (n_classes,)
        The class vocabulary of one output declared through the wrapper.

    Returns
    -------
    class_index : int or None
        The column `class_label` occupies in `declared_classes`, or `None` if
        that vocabulary does not contain it.
    """
    for class_index, declared_class in enumerate(declared_classes):
        if bool(declared_class == class_label):
            return class_index
    return None


def _format_class(class_label):
    """Render one class label readably for an error message."""
    return (
        class_label.item()
        if isinstance(class_label, np.generic)
        else class_label
    )


def _format_classes(classes):
    """Render a class vocabulary readably for an error message."""
    return [_format_class(class_label) for class_label in classes]


def _classes_missing_from(class_labels, declared_classes):
    """Collect class labels that a declared class vocabulary lacks.

    Parameters
    ----------
    class_labels : array-like of shape (n_class_labels,)
        The class labels an estimator predicts for one output.
    declared_classes : array-like of shape (n_classes,)
        The class vocabulary of the same output declared through the wrapper.

    Returns
    -------
    missing_classes : list
        The class labels that `declared_classes` does not contain, compared by
        class identity rather than position or count.
    """
    return [
        class_label
        for class_label in class_labels
        if _class_column(class_label, declared_classes) is None
    ]


def _map_proba_columns(P, learned_classes, declared_classes, output_idx=None):
    """Map probability columns onto a declared class vocabulary.

    An estimator's probability columns follow its own learned class vocabulary,
    which may be ordered differently from the declared one and may omit
    declared classes. The columns are therefore mapped by class identity rather
    than by position, so that equally wide vocabularies are never silently
    reinterpreted. Declared classes the estimator never learned receive
    zero-filled columns.

    Parameters
    ----------
    P : numpy.ndarray of shape (n_samples, n_learned_classes)
        The probabilities of one output as returned by the wrapped estimator.
    learned_classes : array-like of shape (n_learned_classes,)
        The class labels the estimator learned for that output, in the order of
        its probability columns.
    declared_classes : array-like of shape (n_classes,)
        The class vocabulary of the same output declared through the wrapper.
    output_idx : int or None, default=None
        Index of the label output, or `None` for single-output classification.

    Returns
    -------
    P_mapped : numpy.ndarray of shape (n_samples, n_classes)
        The probabilities in the column order of `declared_classes`.

    Raises
    ------
    ValueError
        If the columns of `P` do not correspond to `learned_classes`, or if a
        learned class label is not contained in `declared_classes`.
    """
    name = "`predict_proba`" if output_idx is None else f"P[{output_idx}]"
    output = "" if output_idx is None else f" of output {output_idx}"
    if P.shape[1] != len(learned_classes):
        raise ValueError(
            f"{name} has {P.shape[1]} columns but the fitted estimator "
            f"reports {len(learned_classes)} classes{output}."
        )
    P_mapped = np.zeros((len(P), len(declared_classes)), dtype=float)
    for column, learned_class in enumerate(learned_classes):
        class_index = _class_column(learned_class, declared_classes)
        if class_index is None:
            raise ValueError(
                f"Class {_format_class(learned_class)!r} learned by the "
                f"wrapped estimator is not contained in the declared "
                f"classes{output}."
            )
        P_mapped[:, class_index] = P[:, column]
    return P_mapped


def _check_fitted_target_evidence(estimator, evidence, target_spec):
    """Reconcile a target specification with a pre-fitted estimator.

    Declared target semantics may extend a pre-fitted estimator's learned
    class vocabulary, but they can neither reinterpret its learned classes nor
    change the number of outputs it predicts.

    An estimator publishing no learned class vocabulary is accepted, because
    its own metadata then contradicts nothing. The structure of what it
    predicts is validated where it becomes observable, i.e., when `predict` or
    `predict_proba` is called.

    Parameters
    ----------
    estimator : object
        The pre-fitted estimator, named in the raised errors.
    evidence : _FittedTargetEvidence
        The target evidence published by `estimator`.
    target_spec : skactiveml.utils.TargetSpec
        The resolved target specification that is about to be published.

    Raises
    ------
    ValueError
        If the resolved target semantics contradict the target evidence.
    """
    if evidence.kind == "unknown":
        return
    if target_spec.target_type != "multi-label":
        if evidence.describes_label_outputs:
            raise ValueError(
                f"The pre-fitted estimator '{estimator}' predicts "
                f"{evidence.n_label_outputs} label outputs, so it cannot be "
                f"declared as a single-output classifier with the class "
                f"vocabulary {_format_classes(target_spec.classes)}. Declare "
                f"one binary class vocabulary per label output."
            )
        _check_learned_classes(
            estimator, evidence.classes, target_spec.classes
        )
        return

    n_outputs = len(target_spec.classes)
    if not evidence.describes_label_outputs:
        raise ValueError(
            f"The pre-fitted estimator '{estimator}' learned one categorical "
            f"class assignment per sample from the class vocabulary "
            f"{_format_classes(evidence.classes)}, so it cannot be declared "
            f"as a multi-label classifier with {n_outputs} label outputs. A "
            f"pre-fitted multi-label estimator has to publish either one "
            f"class vocabulary per label output, e.g., "
            f"`sklearn.multioutput.MultiOutputClassifier`, or explicit fitted "
            f"multi-label metadata, e.g., "
            f"`sklearn.multiclass.OneVsRestClassifier`. Otherwise, fit the "
            f"estimator through this wrapper, or declare "
            f"`target_type='single-output'`."
        )
    if evidence.n_label_outputs != n_outputs:
        raise ValueError(
            f"The pre-fitted estimator '{estimator}' learned "
            f"{evidence.n_label_outputs} label outputs, but {n_outputs} class "
            f"vocabularies are declared. A pre-fitted estimator's number of "
            f"outputs cannot be changed through `classes`."
        )
    if evidence.kind == "label-vocabularies":
        for output_idx, learned_classes in enumerate(evidence.classes):
            _check_learned_classes(
                estimator,
                learned_classes,
                target_spec.classes[output_idx],
                output_idx=output_idx,
            )
        return
    # A fitted multi-label estimator publishing label-output identifiers
    # predicts one binary indicator per output, so its outputs cannot carry
    # declared class labels other than the indicator values themselves.
    for output_idx, declared_classes in enumerate(target_spec.classes):
        if _classes_missing_from(_INDICATOR_CLASSES, declared_classes):
            raise ValueError(
                f"The pre-fitted estimator '{estimator}' predicts a binary "
                f"indicator per label output, so every declared class "
                f"vocabulary has to consist of the indicator values "
                f"{list(_INDICATOR_CLASSES)}. Output {output_idx} declares "
                f"{_format_classes(declared_classes)}."
            )


def _check_learned_classes(
    estimator, learned_classes, declared_classes, output_idx=None
):
    """Check that a declared class vocabulary contains the learned classes.

    Parameters
    ----------
    estimator : object
        The pre-fitted estimator, named in the raised error.
    learned_classes : array-like of shape (n_learned_classes,)
        The class labels `estimator` learned for one output.
    declared_classes : tuple
        The class vocabulary of the same output declared through the wrapper.
    output_idx : int or None, default=None
        Index of the label output, or `None` for single-output classification.

    Raises
    ------
    ValueError
        If a learned class label is missing from `declared_classes`.
    """
    missing_classes = _classes_missing_from(learned_classes, declared_classes)
    if not missing_classes:
        return
    output = "" if output_idx is None else f" for label output {output_idx}"
    raise ValueError(
        f"The pre-fitted estimator '{estimator}' learned the class labels "
        f"{_format_classes(missing_classes)}{output} that are not contained "
        f"in the declared class vocabulary "
        f"{_format_classes(declared_classes)}. Its learned classes determine "
        f"the meaning of its predictions and probability columns, so "
        f"`classes` can only extend them with additional classes, which then "
        f"receive zero-filled probability columns."
    )


def _prior_matrix_from_counts(counts, n_samples):
    counts = np.asarray(counts, dtype=float)
    total = counts.sum()
    k = counts.size

    if total == 0:
        return np.full((n_samples, k), 1.0 / k, dtype=float)

    row = counts / total
    return np.tile(row, (n_samples, 1))


class SklearnClassifier(SkactivemlClassifier, MetaEstimatorMixin):
    """Sklearn Classifier

    Implementation of a wrapper class for `scikit-learn` [1]_ classifiers such
    that

    - missing labels can be handled, e.g., by filtering them,
    - classes can be fixed at initialization, e.g., to have consistent
      probabilistic outputs even when there are no observed labels for each
      class,
    - cost-sensitive decisions can be made, e.g., to consider different types
      of misclassification costs.

    Parameters
    ----------
    estimator : sklearn.base.ClassifierMixin
        The `scikit-learn` classifier to be wrapped. A `predict_proba`
        method is only required when `predict_proba` or `cost_matrix`
        based prediction is used.
    include_unlabeled_samples : bool, default=False
        - If `False`, only labeled samples are passed to the `fit` method of
          the `estimator`.
        - If `True`, all samples including the unlabeled ones are passed to
          the `fit` method of the `estimator`. Ensure that your `estimator`
          is able to handle unlabeled samples marked by `missing_label`.
          Otherwise, `missing_label` is interpreted as a regular class label.
          Note that semi-supervised classifiers of `sklearn` expect
          `missing_label=-1`.
    classes : array-like of shape (n_classes,), or a list of such \
            array-likes, default=None
        - A flat vocabulary describes single-output classification.
        - Nested binary vocabularies describe multi-label classification, one
          class vocabulary per label output. With explicit
          `target_type="multi-label"`, vocabularies can instead be resolved
          from `y` when `classes=None` and both classes are observed in every
          output.
    missing_label : scalar or string or np.nan or None, default=np.nan
        Value to represent a missing label.
    cost_matrix : array-like of shape (n_classes, n_classes)
        Cost matrix with `cost_matrix[i,j]` indicating cost of predicting class
        `classes[j]` for a sample of class `classes[i]`. Can be only set, if
        `classes` is not `None` and in the case of single output problems.
    random_state : int or RandomState instance or None, default=None
        Determines random number for `predict` method. Pass an int for
        reproducible results across multiple method calls.
    proba_format : "auto" or "list" or "array", default="auto"
    Output format of ``predict_proba``.

    - Single-output: always returns an array of shape `(n_samples, n_classes)`.
    - Multilabel (2D targets with binary classes per output):
        * 'list'  -> list of `(n_samples, 2)` arrays
        * 'array' -> array of shape `(n_samples, n_outputs)` with
          `P(y=pos_label)`
    target_type : "auto" or "single-output" or "multi-label", default="auto"
        Declared target type. Single-output classification is always supported.
        Multi-label classification requires an estimator that implements
        `predict_proba` and positively declares either
        `target_tags.multi_output` or `classifier_tags.multi_label`, e.g.,
        `sklearn.multioutput.MultiOutputClassifier`,
        `sklearn.multiclass.OneVsRestClassifier`, or
        `sklearn.ensemble.RandomForestClassifier`. An explicit
        `"multi-label"` can resolve binary per-label vocabularies from
        observed targets when `classes` is `None`.

    Attributes
    ----------
    target_spec_ : skactiveml.utils.TargetSpec
        Immutable target specification established by a successful fit. Use
        its `classes` field for canonical class ordering.

    Notes
    -----
    A pre-fitted `estimator` already published the target semantics of its own
    predictions, so its learned classes are reconciled with the declared ones
    by class identity before any fitted attribute of this wrapper is published.
    Declared `classes` may extend the learned class vocabulary, whose
    additional classes then receive zero-filled probability columns in the
    declared order, but they can neither reinterpret learned classes nor change
    the number of predicted outputs. A flat `classes_` published together with
    fitted multi-label metadata, e.g., by
    `sklearn.multiclass.OneVsRestClassifier`, identifies binary indicator
    outputs instead of describing one class vocabulary, so one binary indicator
    vocabulary per output has to be declared through `classes`. A pre-fitted
    estimator publishing neither one class vocabulary per label output nor
    such metadata cannot be declared multi-label, because its flat learned
    vocabulary is indistinguishable from single-output classification.

    Attributes this wrapper does not hold itself are read from the wrapped
    `estimator`. The fitted attributes it resolves itself, e.g. `classes_` and
    `target_spec_`, never are: around a pre-fitted `estimator` they resolve
    this wrapper's own target semantics on first access, and they raise the
    usual not-fitted error while no such semantics exist. A pre-fitted
    estimator's learned classes stay readable as `estimator.classes_`, which
    states whose vocabulary they are.

    Two degenerate training cases are part of this wrapper's contract and make
    it predict the observed class label distribution instead of raising: an
    empty labeled training subset, and an `estimator` rejecting a labeled
    training subset that carries fewer than two distinct classes in at least
    one output. Both set `is_fitted_` to `False` and emit a warning. Every
    other `estimator` failure is raised.

    References
    ----------
    .. [1] Fabian Pedregosa, Gaël Varoquaux, Alexandre Gramfort, Vincent
       Michel, Bertrand Thirion, Olivier Grisel, Mathieu Blondel, Peter
       Prettenhofer, Ron Weiss, Vincent Dubourg, Jake Vanderplas, Alexandre
       Passos, David Cournapeau, Matthieu Brucher, Matthieu Perrot, and Édouard
       Duchesnay. 2011. Scikit-learn: Machine Learning in Python. J. Mach.
       Learn. Res. 12, 2011, 2825–2830.
    """

    #: Fitted attributes this wrapper resolves itself, which `__getattr__`
    #: therefore never forwards to the wrapped `estimator`. `n_features_in_`
    #: is deliberately absent: it describes the input data rather than the
    #: target, both objects agree on it, and `partial_fit` reads it through
    #: `hasattr` to decide whether to reset the feature count.
    _own_fitted_attributes = frozenset(
        {
            "_label_counts",
            "_le",
            "check_X_dict_",
            "classes_",
            "cost_matrix_",
            "estimator_",
            "is_fitted_",
            "random_state_",
            "target_spec_",
        }
    )

    def __init__(
        self,
        estimator,
        include_unlabeled_samples=False,
        classes=None,
        missing_label=MISSING_LABEL,
        cost_matrix=None,
        random_state=None,
        proba_format="auto",
        target_type="auto",
    ):
        super().__init__(
            classes=classes,
            missing_label=missing_label,
            cost_matrix=cost_matrix,
            random_state=random_state,
            target_type=target_type,
        )
        self.estimator = estimator
        self.include_unlabeled_samples = include_unlabeled_samples
        self.proba_format = proba_format

    @property
    def _target_capabilities(self):
        capabilities = {
            ("classification", "single-output", "single-annotator")
        }
        if _declares_multilabel_support(self.estimator):
            capabilities.add(
                ("classification", "multi-label", "single-annotator")
            )
        return frozenset(capabilities)

    def _resolve_target_spec(self, y, classes=None):
        target_spec = resolve_target_spec(
            y,
            task="classification",
            target_type=self.target_type,
            annotation_type="single-annotator",
            classes=self.classes if classes is None else classes,
            missing_label=self.missing_label,
        )
        if target_spec.target_type == "multi-label" and not (
            _declares_multilabel_support(self.estimator)
        ):
            # The generic capability check below reports the unsupported
            # semantic triple, which does not explain what is missing.
            raise ValueError(
                _multilabel_capability_error(
                    type(self).__name__, self.estimator
                )
            )
        check_target_capability(
            type(self).__name__, target_spec, self._target_capabilities
        )
        return target_spec

    @match_signature("estimator", "fit")
    def fit(self, X, y=None, sample_weight=None, **fit_kwargs):
        """Fit the model using `X` as training data and `y` as class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, ...)
            The feature matrix representing the samples.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Labels of the training data set (possibly including unlabeled
            ones indicated by `missing_label`). For multilabel
            problems, a row `y[i]` must either contain only observed
            labels or only `missing_label` values, i.e., no mixing
            within a row. Note that `Y` (capitalized) is only accepted if the
            wrapped estimator exposes this parameter name in its `fit`
            signature.
        sample_weight : array-like of shape (n_samples,) or \
                (n_samples, n_outputs)
            It contains the weights of the training samples' class labels.
            Only supported if the wrapped `sklearn` classifier can handle
            sample weights.
        fit_kwargs : dict-like
            Further parameters as input to the `fit` method of the `estimator`.

        Returns
        -------
        self: SklearnClassifier,
            The `SklearnClassifier` object fitted on the training data.
        """
        y = self._extract_target_arg(y, fit_kwargs)
        return self._fit(
            fit_function="fit",
            X=X,
            y=y,
            sample_weight=sample_weight,
            **fit_kwargs,
        )

    @match_signature("estimator", "partial_fit")
    def partial_fit(self, X, y=None, sample_weight=None, **fit_kwargs):
        """Partially fitting the model using `X` as training data and `y` as
        class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, ...)
            The feature matrix representing the samples.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Labels of the training data set (possibly including unlabeled
            ones indicated by `missing_label`). For multilabel
            problems, a row `y[i]` must either contain only observed
            labels or only `missing_label` values, i.e., no mixing
            within a row. Note that `Y` (capitalized) is only accepted if the
            wrapped estimator exposes this parameter name in its
            `partial_fit` signature.
        sample_weight : array-like of shape (n_samples,) or \
                (n_samples, n_outputs)
            It contains the weights of the training samples' class labels.
            Only supported if the wrapped `sklearn` classifier can handle
            sample weights.
        fit_kwargs : dict-like
            Further parameters as input to the `partial_fit` method of the
            `estimator`.

        Returns
        -------
        self : SklearnClassifier,
            The `SklearnClassifier` object fitted on the training data.
        """
        y = self._extract_target_arg(y, fit_kwargs)
        return self._fit(
            fit_function="partial_fit",
            X=X,
            y=y,
            sample_weight=sample_weight,
            **fit_kwargs,
        )

    @match_signature("estimator", "predict")
    def predict(self, X, **predict_kwargs):
        """Return class label predictions for the input data `X`.

        Parameters
        ----------
        X : array-like of shape (n_samples, ...)
            Input samples.
        predict_kwargs : dict-like
            Further parameters as input to the `predict` method of the
            `estimator`.

        Returns
        -------
        y_pred : numpy.ndarray of shape (n_samples,) or (n_samples, n_outputs)
            Predicted class labels of the input samples.
        """
        check_is_fitted(self)
        predict_dict = {"ensure_min_samples": 1, "ensure_min_features": 1}
        X = check_array(X, **(self.check_X_dict_ | predict_dict))
        check_n_features(self, X, reset=False)
        if self.is_fitted_:
            if self.cost_matrix is None:
                y_pred = self.estimator_.predict(X, **predict_kwargs)
                if self._is_multilabel_target():
                    y_pred = self._check_multilabel_predictions(y_pred)
                else:
                    y_pred = y_pred.astype(self.classes_.dtype)
            else:
                P = self.predict_proba(X)
                costs = np.dot(P, self.cost_matrix_)
                y_pred = rand_argmin(
                    costs, random_state=self.random_state_, axis=1
                )
                y_pred = self._le.inverse_transform(y_pred)
        else:
            p = self.predict_proba([X[0]])
            if self._is_multilabel_target():
                if isinstance(p, np.ndarray) and p.ndim == 2:
                    # Uniform sampling in (0, 1).
                    rand_p = self.random_state_.random((len(X), len(p[0])))
                    y_enc_pred = (rand_p < p).astype(np.int64)
                else:
                    y_enc_pred = [
                        self.random_state_.choice(
                            len(p_[0]), size=len(X), p=p_[0]
                        )
                        for p_ in p
                    ]
                    y_enc_pred = np.column_stack(y_enc_pred)
            else:
                y_enc_pred = self.random_state_.choice(
                    np.arange(len(p[0])), len(X), replace=True, p=p[0]
                )
            y_pred = self._le.inverse_transform(y_enc_pred)
        return y_pred

    @match_signature("estimator", "predict_proba")
    def predict_proba(self, X, **predict_proba_kwargs):
        """Return probability estimates for the input data `X`.

        Parameters
        ----------
        X : array-like of shape (n_samples, ...)
            Input samples.
        predict_proba_kwargs : dict-like
            Further parameters as input to the `predict_proba` method of the
            `estimator`.

        Returns
        -------
        P : numpy.ndarray of shape (n_samples, n_classes), \
                numpy.ndarray of shape (n_samples, n_outputs), or list of \
                numpy.ndarray
            The class probabilities of the input samples. For single-output
            classification, the return value has shape
            `(n_samples, n_classes)`. For multilabel classification,
            `proba_format='array'` returns shape `(n_samples, n_outputs)` with
            positive-class probabilities and `proba_format='list'` returns one
            `(n_samples, 2)` array per output.
        """
        # Input parameter checks.
        check_is_fitted(self)
        predict_dict = {"ensure_min_samples": 1, "ensure_min_features": 1}
        X = check_array(X, **(self.check_X_dict_ | predict_dict))
        check_n_features(self, X, reset=False)
        n_samples = len(X)
        proba_format = self._resolve_proba_format()

        if self.is_fitted_:
            # Obtain class probabilities if wrapped classifier was successfully
            # fitted.
            P = self.estimator_.predict_proba(X, **predict_proba_kwargs)

            if not self._is_multilabel_target():
                # Single output classification.
                P = self._normalize_single_output_proba(P, n_samples=n_samples)
                # Fall through to the label-count prior fallback if the
                # fitted estimator yielded NaN probabilities.
                if not np.any(np.isnan(P)):
                    return P
            else:
                # Multi-label targets correspond to one binary class
                # vocabulary per output.
                n_outputs = len(self.classes_)
                if isinstance(P, list):
                    P_list = self._normalize_multilabel_proba_list(
                        P, n_samples=n_samples
                    )
                else:
                    P_ml = self._check_multilabel_proba_array(P)
                    P_list = [
                        np.column_stack([1 - P_ml[:, j], P_ml[:, j]])
                        for j in range(n_outputs)
                    ]
                # Fall through to the label-count prior fallback if the
                # fitted estimator yielded NaN probabilities.
                if not any(np.any(np.isnan(P_j)) for P_j in P_list):
                    if proba_format == "array":
                        # Binary per output: return positive-class
                        # probabilities of shape (n_samples, n_outputs).
                        return np.column_stack(
                            [P_list[j][:, 1] for j in range(n_outputs)]
                        )
                    return P_list

        # Fallback, if fitting of the underlying estimator failed.
        warnings.warn(
            f"Since the 'estimator' could not be fitted when calling the "
            f"`fit` method, the class label distribution "
            f"`_label_counts={self._label_counts}` is used to make the "
            f"predictions."
        )

        # Fallback for single output.
        if not self._is_multilabel_target():
            return _prior_matrix_from_counts(self._label_counts, len(X))

        if proba_format == "array":
            # Binary per task: return (n_samples, n_outputs) with P(y=1).
            return np.column_stack(
                [
                    _prior_matrix_from_counts(counts_j, len(X))[:, 1]
                    for counts_j in self._label_counts
                ]
            )

        # List format: return one probability matrix per label.
        return [
            _prior_matrix_from_counts(counts_j, len(X))
            for counts_j in self._label_counts
        ]

    def _fit(self, fit_function, X, y, sample_weight=None, **fit_kwargs):
        """Fit or partially fit this wrapper as a single transaction.

        The snapshot taken here covers the entire fit, not only the estimator
        call: a rejection raised while validating the estimator, the class
        vocabulary, or the target specification rolls the wrapper back just as
        a failing estimator fit does. Without that, a failing re-fit would
        leave an already fitted wrapper reporting metadata from the abandoned
        attempt, e.g. an `n_features_in_` that contradicts its `estimator_`.

        The degenerate training cases are not failures and keep their state,
        because `_validate_and_fit` returns `self` for them rather than
        raising. The transaction also covers this wrapper only: as
        `_restore_attributes` documents, a `partial_fit` that already mutated
        `estimator_` in place cannot be rolled back.

        Parameters
        ----------
        fit_function : "fit" or "partial_fit"
            Name of the estimator method to call.
        X : array-like of shape (n_samples, ...)
            The feature matrix representing the samples.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Labels of the training data set, possibly including unlabeled ones.
        sample_weight : array-like of shape (n_samples,), default=None
            It contains the weights of the training samples' class labels.
        fit_kwargs : dict-like
            Further parameters as input to `fit_function` of the `estimator`.

        Returns
        -------
        self : SklearnClassifier
            The wrapper fitted on the training data.
        """
        attributes_before = dict(self.__dict__)
        try:
            return self._validate_and_fit(
                fit_function=fit_function,
                X=X,
                y=y,
                sample_weight=sample_weight,
                **fit_kwargs,
            )
        except Exception:
            self._restore_attributes(attributes_before)
            raise

    def _validate_and_fit(
        self, fit_function, X, y, sample_weight=None, **fit_kwargs
    ):
        """Validate the inputs and fit the estimator, committing state freely.

        This method may write fitted attributes before a later step rejects
        the call, because its only caller `_fit` rolls them back. See `_fit`
        for the parameters and the transactional guarantee.
        """
        is_incremental = fit_function == "partial_fit"
        supplied_classes = (
            fit_kwargs.get("classes") if is_incremental else None
        )
        if supplied_classes is not None and self.classes is not None:
            configured_spec = self._resolve_fitting_target_spec(
                y, classes=self.classes
            )
            self._resolve_fitting_target_spec(
                y,
                established_spec=configured_spec,
                classes=supplied_classes,
            )
        target_spec = self._resolve_target_spec_for_fit(
            y,
            is_incremental=is_incremental,
            classes=supplied_classes,
        )

        # Check input parameters.
        self.check_X_dict_ = {
            "ensure_min_samples": 0,
            "ensure_min_features": 0,
            "allow_nd": True,
            "dtype": None,
        }
        X, y, sample_weight = self._validate_data(
            X=X,
            y=y,
            sample_weight=sample_weight,
            check_X_dict=self.check_X_dict_,
            reset=fit_function == "fit" or not hasattr(self, "n_features_in_"),
            target_spec=target_spec,
        )
        self.target_spec_ = target_spec

        # Check whether estimator is a valid classifier.
        if not is_classifier(estimator=self.estimator):
            raise TypeError(
                "'{}' must be a scikit-learn "
                "classifier.".format(self.estimator)
            )

        # Check boolean flag.
        check_type(
            self.include_unlabeled_samples,
            "include_unlabeled_samples",
            bool,
        )

        # Check whether estimator can deal with cost matrix.
        if self.cost_matrix is not None and not hasattr(
            self.estimator, "predict_proba"
        ):
            raise ValueError(
                "'cost_matrix' can be only set, if 'estimator'"
                "implements 'predict_proba'."
            )
        if hasattr(self, "estimator_"):
            if fit_function != "partial_fit":
                self.estimator_ = deepcopy(self.estimator)
        else:
            self.estimator_ = deepcopy(self.estimator)

        # Include unlabeled samples, if requested, e.g., when wrapping
        # semi-supervised classifiers from sklearn.
        if self.include_unlabeled_samples:
            is_included = np.full_like(y, fill_value=True, dtype=bool)
        else:
            is_included = is_labeled(
                y=y,
                missing_label=-1,
                target_type=target_spec.target_type,
            )

        # Count labels per class.
        if self._is_multilabel_target():
            self._label_counts = [
                np.array(
                    [
                        np.sum(y[is_included, j] == class_idx)
                        for class_idx in range(len(classes_j))
                    ],
                    dtype=int,
                )
                for j, classes_j in enumerate(self._le.classes_)
            ]
        else:
            self._label_counts = [
                np.sum(y[is_included] == c)
                for c in range(len(self._le.classes_))
            ]

        X_train = X[is_included]
        y_train = y[is_included].astype(np.int64)

        # Degenerate training cases are part of the wrapper contract, e.g.,
        # during an active learning cold start. An empty labeled training
        # subset is recognized before the estimator is called at all.
        if len(y_train) == 0:
            self._fall_back_to_label_prior("there is no labeled data")
            return self

        y_train_inv = self._decode_labeled_targets(y_train)
        sample_weight_train = (
            None if sample_weight is None else sample_weight[is_included]
        )
        try:
            self._call_estimator_fit(
                fit_function=fit_function,
                X=X_train,
                y=y_train_inv,
                sample_weight=sample_weight_train,
                **fit_kwargs,
            )
        except Exception as error:
            # Only the second degenerate case is absorbed: many estimators
            # reject a training subset carrying fewer than two distinct
            # classes although others fit it, so the failure is discovered
            # rather than predicted.
            if self._has_degenerate_training_classes(y_train):
                self._fall_back_to_label_prior(
                    f"the estimator raised '{error}' on a labeled training "
                    f"subset carrying fewer than two classes in at least one "
                    f"output"
                )
                return self

            # Every other failure states a genuinely broken estimator
            # contract and must not be hidden behind prior-only predictions.
            # The message is built before propagating, because `_fit` restores
            # the pre-call state and with it the reported `estimator_`.
            message = (
                f"Calling '{fit_function}' of the estimator "
                f"'{self.estimator_}' failed on {len(y_train)} labeled "
                f"samples covering at least two classes per output. This is "
                f"not one of the degenerate training cases for which the "
                f"class label distribution is used as a fallback."
            )
            raise RuntimeError(message) from error

        self.is_fitted_ = True
        return self

    def _restore_attributes(self, attributes):
        """Restore every attribute of this wrapper to a pre-call snapshot.

        A failing fit must not leave a wrapper that reports fitted state it
        cannot serve. Because the attributes are restored as a whole rather
        than selectively deleted, an already fitted wrapper keeps exactly its
        pre-call values, and a previously unfitted one stays unfitted.

        Note that this restores this wrapper only. A `partial_fit` mutates
        `estimator_` in place, so a failing incremental update can leave the
        wrapped estimator itself in an implementation-defined state.

        Parameters
        ----------
        attributes : dict
            Snapshot of `self.__dict__` taken before the failing call.
        """
        self.__dict__.clear()
        self.__dict__.update(attributes)

    def _decode_labeled_targets(self, y_train):
        """Decode a labeled training subset into its declared class dtype.

        The label encoder decodes into a dtype that can also represent
        `missing_label`, e.g., `object` for `missing_label=None`. A labeled
        training subset never carries `missing_label`, so its decoded labels
        are narrowed back to the dtype of the declared classes. Without this
        narrowing, a wrapped estimator rejects an `object` target that only
        contains ordinary class labels.

        Parameters
        ----------
        y_train : numpy.ndarray of shape (n_labeled,) or \
                (n_labeled, n_outputs)
            The encoded class labels of the labeled training subset.

        Returns
        -------
        y_train_inv : numpy.ndarray of shape (n_labeled,) or \
                (n_labeled, n_outputs)
            The decoded class labels passed on to the wrapped estimator.
        """
        y_train_inv = self._le.inverse_transform(y_train)
        if self._is_multilabel_target():
            class_dtypes = [classes_j.dtype for classes_j in self.classes_]
        else:
            class_dtypes = [self.classes_.dtype]
        return y_train_inv.astype(np.result_type(*class_dtypes), copy=False)

    def _has_degenerate_training_classes(self, y_train):
        """Check whether an encoded training subset lacks two classes.

        Parameters
        ----------
        y_train : numpy.ndarray of shape (n_labeled,) or \
                (n_labeled, n_outputs)
            The encoded class labels of the labeled training subset.

        Returns
        -------
        is_degenerate : bool
            `True`, if fewer than two distinct classes are observed, for
            multi-label classification in at least one output, and `False`
            otherwise.
        """
        if self._is_multilabel_target():
            return any(
                len(np.unique(y_train[:, j])) < 2
                for j in range(y_train.shape[1])
            )
        return len(np.unique(y_train)) < 2

    def _fall_back_to_label_prior(self, reason):
        """Mark this wrapper as unfitted to predict the label distribution.

        Parameters
        ----------
        reason : str
            Description of the degenerate training case, completing the
            sentence "The 'estimator' could not be fitted because ...".
        """
        self.is_fitted_ = False
        warnings.warn(
            f"The 'estimator' could not be fitted because {reason}. "
            f"Therefore, the class labels of the samples are counted and "
            f"will be used to make predictions. The class label distribution "
            f"is `_label_counts={self._label_counts}`."
        )

    def __sklearn_is_fitted__(self):
        if "is_fitted_" in self.__dict__:
            return True

        try:
            check_is_fitted(self.estimator)
        except NotFittedError:
            return False

        estimator = deepcopy(self.estimator)
        evidence = _discover_fitted_target_evidence(self.estimator)

        # The target specification is resolved, validated, and reconciled with
        # the pre-fitted estimator before any fitted attribute is written, such
        # that a failing target contract leaves this wrapper exactly in its
        # pre-call state.
        label_state = self._resolve_label_state(self._prefit_classes(evidence))
        _check_fitted_target_evidence(
            self.estimator, evidence, label_state["target_spec"]
        )
        self._commit_label_state(label_state)
        self._initialize_label_counts_from_classes()

        # set attributes that would be set by the fit function
        self.is_fitted_ = True
        self.estimator_ = estimator
        self.check_X_dict_ = {
            "ensure_min_samples": 0,
            "ensure_min_features": 0,
            "allow_nd": True,
            "dtype": None,
        }

        return True

    def __getattr__(self, item):
        if item in self._own_fitted_attributes:
            return _resolve_own_fitted_attribute(self, item)
        if "estimator_" in self.__dict__:
            return getattr(self.estimator_, item)
        else:
            return getattr(self.estimator, item)

    def _prefit_classes(self, evidence):
        """Determine the class vocabulary of a pre-fitted `estimator`.

        The declared `classes` take precedence, because the label state is
        resolved from the returned vocabulary and also validated against it.
        Reconciliation with the estimator's own learned classes is a separate
        step, so that it can report their contradiction rather than a generic
        unknown-class error.

        Parameters
        ----------
        evidence : _FittedTargetEvidence
            The target evidence published by the pre-fitted `estimator`.

        Returns
        -------
        classes : array-like of shape (n_classes,), or a list of such \
                array-likes
            The declared `classes` if they are given, and the estimator's
            learned class vocabulary otherwise.

        Raises
        ------
        ValueError
            If neither this wrapper nor the pre-fitted `estimator` declares a
            class vocabulary, because the estimator's predictions cannot be
            interpreted without one.
        """
        if self.classes is not None:
            return self.classes
        if evidence.classes is not None:
            return evidence.classes
        if evidence.kind == "label-outputs":
            example = [list(_INDICATOR_CLASSES)] * evidence.n_label_outputs
            raise ValueError(
                f"The pre-fitted estimator '{self.estimator}' declares "
                f"multi-label classification with "
                f"{evidence.n_label_outputs} label outputs, so its flat "
                f"learned classes identify those outputs instead of one class "
                f"vocabulary. Declare one binary indicator vocabulary per "
                f"label output, i.e., `classes={example}`."
            )
        raise ValueError(
            f"The pre-fitted estimator '{self.estimator}' exposes no learned "
            f"class vocabulary through `classes_`, so its predictions cannot "
            f"be interpreted. Declare `classes`."
        )

    def _resolve_label_state(self, classes):
        """Resolve the label state for `classes` without committing it.

        Every target resolution and validation step is performed on local
        state only, such that a failure leaves this wrapper untouched.

        Parameters
        ----------
        classes : array-like of shape (n_classes,), or a list of such \
                array-likes
            Class vocabulary to resolve the target specification from.

        Returns
        -------
        label_state : dict
            Resolved label state to be committed via `_commit_label_state`.
        """
        random_state = check_random_state(self.random_state)
        effective_classes = (
            self.classes if self.classes is not None else classes
        )
        y_dummy = (
            np.empty(
                (0, len(effective_classes)),
                dtype=np.asarray(self.missing_label).dtype,
            )
            if _has_nested_classes(effective_classes)
            else classes
        )
        target_spec = self._resolve_target_spec(
            y_dummy, classes=effective_classes
        )
        le = ExtLabelEncoder(
            classes=target_spec.classes,
            missing_label=self.missing_label,
            target_type=target_spec.target_type,
        )
        le.fit(y_dummy)
        if target_spec.target_type == "multi-label":
            cost_matrix = None
        else:
            cost_matrix = (
                1 - np.eye(len(le.classes_))
                if self.cost_matrix is None
                else self.cost_matrix
            )
        check_classifier_params(le.classes_, self.missing_label, cost_matrix)
        return {
            "random_state": random_state,
            "target_spec": target_spec,
            "le": le,
            "cost_matrix": cost_matrix,
        }

    def _commit_label_state(self, label_state):
        """Write a label state resolved by `_resolve_label_state`."""
        self.random_state_ = label_state["random_state"]
        self.target_spec_ = label_state["target_spec"]
        self._le = label_state["le"]
        self.classes_ = self._le.classes_
        self.cost_matrix_ = label_state["cost_matrix"]

    def _initialize_label_counts_from_classes(self):
        """Initialize the per-class label counts with zeros.

        This is used for pre-fitted estimators, where no labels are observed
        through the wrapper's own `fit` method. The resulting all-zero counts
        make the label-count prior fallback of `predict` and `predict_proba`
        default to a uniform class distribution.
        """
        if self._is_multilabel_target():
            self._label_counts = [
                np.zeros(len(classes_j), dtype=int)
                for classes_j in self.classes_
            ]
        else:
            self._label_counts = [0 for _ in self.classes_]

    @staticmethod
    def _extract_target_arg(y, fit_kwargs):
        if y is not None and "Y" in fit_kwargs:
            raise TypeError("Pass only one of `y` and `Y`.")
        return y if y is not None else fit_kwargs.pop("Y", None)

    @staticmethod
    def _target_parameter_name(fit_method):
        fit_params = inspect.signature(fit_method).parameters
        if "y" in fit_params:
            return "y"
        if "Y" in fit_params:
            return "Y"
        raise TypeError(
            "The wrapped estimator's fit method must accept either `y` "
            "or `Y` as target parameter."
        )

    def _call_estimator_fit(
        self, fit_function, X, y, sample_weight=None, **fit_kwargs
    ):
        fit_method = getattr(self.estimator_, fit_function)
        fit_params = inspect.signature(fit_method).parameters
        target_param = self._target_parameter_name(fit_method)
        call_kwargs = dict(fit_kwargs)
        call_kwargs["X"] = X
        call_kwargs[target_param] = y

        if fit_function == "partial_fit" and "classes" in fit_params:
            call_kwargs["classes"] = self.classes_

        if sample_weight is not None and (
            "sample_weight" in fit_params
            or any(
                param.kind == inspect.Parameter.VAR_KEYWORD
                for param in fit_params.values()
            )
        ):
            call_kwargs["sample_weight"] = sample_weight

        return fit_method(**call_kwargs)

    def _check_multilabel_predictions(self, y_pred):
        """Validate the class label predictions of a multi-label estimator.

        Parameters
        ----------
        y_pred : array-like
            The predictions returned by the wrapped estimator's `predict`.

        Returns
        -------
        y_pred : numpy.ndarray of shape (n_samples, n_outputs)
            The validated predictions.

        Raises
        ------
        ValueError
            If the predictions do not describe one class label per output.
        """
        n_outputs = len(self.classes_)
        y_pred = np.asarray(y_pred)
        if y_pred.ndim != 2 or y_pred.shape[1] != n_outputs:
            raise ValueError(
                f"Expected `predict` of the wrapped estimator to return "
                f"shape `(n_samples, {n_outputs})` for multi-label "
                f"classification, got {y_pred.shape}."
            )
        return y_pred

    def _check_multilabel_proba_array(self, P):
        """Validate a positive-class probability array of a multi-label fit.

        Parameters
        ----------
        P : array-like
            The probabilities returned by the wrapped estimator's
            `predict_proba`, which are not given as one matrix per output.

        Returns
        -------
        P_ml : numpy.ndarray of shape (n_samples, n_outputs)
            The validated positive-class probabilities.

        Raises
        ------
        ValueError
            If the probabilities follow neither documented multi-label
            probability contract.
        """
        n_outputs = len(self.classes_)
        P_ml = np.asarray(P, dtype=float)
        if P_ml.ndim != 2 or P_ml.shape[1] != n_outputs:
            raise ValueError(
                f"Expected `predict_proba` of the wrapped estimator to "
                f"return positive-class probabilities of shape "
                f"`(n_samples, {n_outputs})`, or one `(n_samples, 2)` array "
                f"per output, for multi-label classification, got "
                f"{P_ml.shape}."
            )
        return P_ml

    def _normalize_single_output_proba(self, P, n_samples):
        """Align a single-output probability array to `classes_`.

        The wrapped estimator's probability columns follow its own learned
        class vocabulary, which may be ordered differently from `classes_` and
        may omit declared classes. The columns are therefore mapped by class
        identity rather than by position, so that equally wide vocabularies
        are not silently reinterpreted. Declared classes the estimator never
        learned receive zero-filled columns.

        Parameters
        ----------
        P : array-like
            The probabilities returned by the wrapped estimator's
            `predict_proba`.
        n_samples : int
            Number of samples the probabilities are expected to have.

        Returns
        -------
        P : numpy.ndarray of shape (n_samples, n_classes)
            The probabilities in the column order of `classes_`.

        Raises
        ------
        ValueError
            If the probabilities can be reconciled with neither the learned
            nor the declared class vocabulary.
        """
        P = np.asarray(P, dtype=float)
        if P.ndim != 2 or P.shape[0] != n_samples:
            raise ValueError(
                f"Expected `predict_proba` of the wrapped estimator to return "
                f"shape `({n_samples}, n_classes)` for single-output "
                f"classification, got {P.shape}."
            )

        est_classes = getattr(self.estimator_, "classes_", None)
        if est_classes is None:
            if P.shape[1] != len(self.classes_):
                raise ValueError(
                    f"`predict_proba` returned {P.shape[1]} columns but "
                    f"{len(self.classes_)} classes are declared, and the "
                    f"wrapped estimator does not expose `classes_` to map "
                    f"them. Provide an estimator exposing its learned "
                    f"classes, declare `classes` matching the estimator, or "
                    f"return probabilities for all declared classes."
                )
            return P

        return _map_proba_columns(P, np.asarray(est_classes), self.classes_)

    def _normalize_multilabel_proba_list(self, P, n_samples):
        """Align a list of per-output probability matrices to `classes_`.

        The wrapped estimator may return, for each output, a probability
        matrix whose columns follow its own class order and may omit classes
        that were not observed during fitting. This method maps each such
        matrix onto the declared classes of the corresponding output, filling
        the probabilities of unobserved classes with zeros.

        Parameters
        ----------
        P : list of array-like
            One probability matrix of shape `(n_samples, n_classes)` per
            output, as returned by the wrapped estimator's `predict_proba`.
        n_samples : int
            Number of samples the probability matrices are expected to have.

        Returns
        -------
        P_list : list of numpy.ndarray
            One probability matrix of shape `(n_samples, n_classes)` per
            output, with columns aligned to `self.classes_`.

        Raises
        ------
        ValueError
            If the number of outputs, the shape of a matrix, or the class
            mapping of an output cannot be reconciled with `self.classes_`.
        """
        n_outputs = len(self.classes_)
        if len(P) != n_outputs:
            raise ValueError(
                f"Expected {n_outputs} outputs from `predict_proba`, got "
                f"{len(P)}."
            )

        P_list = []
        for j, P_j in enumerate(P):
            P_j = np.asarray(P_j, dtype=float)
            classes_j = np.asarray(self.classes_[j])
            if P_j.ndim != 2:
                raise ValueError(
                    f"Expected P[{j}] to be of shape "
                    f"(n_samples, n_classes), got {P_j.shape}."
                )
            if P_j.shape[0] != n_samples:
                raise ValueError(
                    f"Expected P[{j}] to contain {n_samples} samples, got "
                    f"{P_j.shape[0]}."
                )

            est_classes_j = self._estimator_classes_for_output(j, n_outputs)
            if est_classes_j is None:
                if P_j.shape[1] != len(classes_j):
                    raise ValueError(
                        f"P[{j}] has {P_j.shape[1]} columns but output {j} "
                        f"declares {len(classes_j)} classes, and the wrapped "
                        f"estimator does not expose per-output classes to map "
                        f"them. Provide an estimator that exposes `classes_` "
                        f"(or per-output `estimators_`), declare `classes` "
                        f"matching the estimator, or return probabilities for "
                        f"all declared classes."
                    )
                P_list.append(P_j)
                continue

            P_list.append(
                _map_proba_columns(
                    P_j,
                    np.asarray(est_classes_j),
                    classes_j,
                    output_idx=j,
                )
            )

        return P_list

    def _estimator_classes_for_output(self, output_idx, n_outputs):
        """Determine the class labels of the wrapped estimator for one output.

        The aggregated `classes_` attribute is preferred because it holds the
        per-output class vector for both `MultiOutputClassifier` and native
        multilabel estimators. The `estimators_[output_idx].classes_` attribute
        is only used as a fallback, since for ensembles the `estimators_` are
        base learners trained on all outputs, so this attribute is a list of
        per-output arrays rather than this output's class vector.

        Parameters
        ----------
        output_idx : int
            Index of the output whose class labels are requested.
        n_outputs : int
            Total number of outputs of the multilabel problem.

        Returns
        -------
        classes_j : numpy.ndarray of shape (n_classes,) or None
            The class labels of the wrapped estimator for the output
            `output_idx`, or `None` if they cannot be determined.
        """
        # Collect class candidates in priority order.
        candidates = []
        est_classes = getattr(self.estimator_, "classes_", None)
        if est_classes is not None and len(est_classes) == n_outputs:
            candidates.append(est_classes[output_idx])
        if (
            hasattr(self.estimator_, "estimators_")
            and len(self.estimator_.estimators_) == n_outputs
            and hasattr(self.estimator_.estimators_[output_idx], "classes_")
        ):
            candidates.append(self.estimator_.estimators_[output_idx].classes_)

        for candidate in candidates:
            classes_j = np.asarray(candidate)
            # Accept only a flat vector of scalar class labels, rejecting
            # scalars (ndim 0) and list-of-arrays specifications.
            if classes_j.ndim == 1 and all(np.ndim(c) == 0 for c in classes_j):
                return classes_j

        return None

    def _resolve_proba_format(self):
        check_type(self.proba_format, "proba_format", str)
        if self.proba_format not in ["auto", "list", "array"]:
            raise ValueError(
                "`proba_format` must be one of {'auto', 'list', 'array'}."
            )
        if not self._is_multilabel_target():
            return "array"

        is_multilabel = all(len(classes_j) == 2 for classes_j in self.classes_)
        if self.proba_format == "auto":
            return "array" if is_multilabel else "list"
        return self.proba_format

    def _is_multilabel_target(self):
        return (
            "target_spec_" in self.__dict__
            and self.target_spec_.target_type == "multi-label"
        )


class SlidingWindowClassifier(SkactivemlClassifier, MetaEstimatorMixin):
    """Sliding Window Classifier

    Implementation of a wrapper class for `SkactivemlClassifier` such that the
    number of training samples can be limited to the latest `window_size`
    samples. Furthermore, saves `X`, `y` and `sample_weight`, enabling the use
    of a `partial_fit` for any classifier.

    Parameters
    ----------
    estimator : sklearn.base.SkactivemlClassifier
        The classifier to be wrapped. If this classifier already implements a
        `partial_fit`, this method will be overwritten by this wrapper using
        the sliding window approach.
    classes : array-like of shape (n_classes,), default=None
        Holds the label for each class. If `None`, `classes` are determined
        during the fit.
    missing_label : scalar or string or np.nan or None, default=np.nan
        Value to represent a missing label.
    cost_matrix : array-like of shape (n_classes, n_classes)
        Cost matrix with `cost_matrix[i,j]` indicating cost of predicting class
        `classes[j]` for a sample of class `classes[i]`. Can be only set, if
        `classes` is not none.
    window_size : int, default=None,
        Value to represent the estimator sliding window size for X, y and
        sample weight. If `None` the window is unrestricted in its size.
    only_labeled : bool, default=False
        If `True`, unlabeled samples are discarded.
    random_state : int or RandomState instance or None, default=None
        Determines random number for `predict` method. Pass an int for
        reproducible results across multiple method calls.
    target_type : "auto" or "single-output", default="auto"
        Declared target type. It must remain compatible with the wrapped
        estimator across incremental updates. The wrapper supports only
        single-output classification.

    Notes
    -----
    Attributes this wrapper does not hold itself are read from the wrapped
    `estimator`, `classes_` among them: the wrapper resolves no class
    vocabulary of its own, so the one its `SkactivemlClassifier` resolved is
    the wrapper's own answer. The fitted attributes it does hold itself, e.g.
    `target_spec_` and the sliding window in `X_train_`, are never read from
    the `estimator` and raise the usual not-fitted error before a fit.
    """

    #: Fitted attributes this wrapper holds itself, which `__getattr__`
    #: therefore never forwards to the wrapped `estimator`. `classes_` is
    #: deliberately absent: this wrapper resolves no class vocabulary of its
    #: own, so the one its `SkactivemlClassifier` estimator resolved is the
    #: wrapper's own answer.
    _own_fitted_attributes = frozenset(
        {
            "X_train_",
            "check_X_dict_",
            "estimator_",
            "random_state_",
            "sample_weight_train_",
            "target_spec_",
            "y_train_",
        }
    )

    def __init__(
        self,
        estimator,
        classes=None,
        missing_label=MISSING_LABEL,
        cost_matrix=None,
        window_size=None,
        only_labeled=False,
        random_state=None,
        target_type="auto",
    ):
        super().__init__(
            classes=classes,
            missing_label=missing_label,
            cost_matrix=cost_matrix,
            random_state=random_state,
            target_type=target_type,
        )
        self.estimator = estimator
        self.only_labeled = only_labeled
        self.window_size = window_size

    @match_signature("estimator", "fit")
    def fit(self, X, y, sample_weight=None, **fit_kwargs):
        """Fit the model using `X` as training data and `y` as class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, ...)
            The feature matrix representing the samples.
        y : array-like of shape (n_samples,)
            It contains the class labels of the training samples. Missing
            labels are represented by the attribute `self.missing_label_`.
        sample_weight : array-like of shape (n_samples,), default=None
            It contains the weights of the training samples' class labels.
        fit_kwargs : dict-like
            Further parameters as input to the `fit` method of the `estimator`.

        Returns
        -------
        self: SlidingWindowClassifier,
            The `SlidingWindowClassifier` is fitted on the training data.
        """
        # Check whether estimator is a valid classifier.
        if not isinstance(self.estimator, SkactivemlClassifier):
            raise TypeError(
                "'{}' must be a SkactivemlClassifier"
                "classifier.".format(self.estimator)
            )
        self.check_X_dict_ = {
            "ensure_min_samples": 0,
            "ensure_min_features": 0,
            "allow_nd": True,
            "dtype": None,
        }
        X, y, sample_weight = self._validate_data(
            X=X,
            y=y,
            sample_weight=sample_weight,
            check_X_dict=self.check_X_dict_,
            established_spec=None,
        )

        self._add_samples("fit", X, y, sample_weight)
        X_train = np.array(self.X_train_)
        y_train = np.array(self.y_train_)
        sample_weight_train = None
        if self.sample_weight_train_ is not None:
            sample_weight_train = np.array(
                self.sample_weight_train_, dtype=float
            )
        return self._fit(
            X=X_train,
            y=y_train,
            sample_weight=sample_weight_train,
            **fit_kwargs,
        )

    @match_signature("estimator", "fit")
    def partial_fit(self, X, y, sample_weight=None, **fit_kwargs):
        """Partially fitting the model using `X` as training data and `y` as
        class labels. If `base_estimator` has no `partial_fit` function use
        `fit` with the sliding window for X, y and sample_weight.

        Parameters
        ----------
        X : array-like of shape (n_samples, ...)
            The feature matrix representing the samples.
        y : array-like of shape (n_samples,)
            It contains the class labels of the training samples. Missing
            labels are represented by the attribute `self.missing_label_`.
        sample_weight : array-like of shape (n_samples,), default=None
            It contains the weights of the training samples' class labels.
        fit_kwargs : dict-like
            Further parameters as input to the `fit` method of the `estimator`.

        Returns
        -------
        self : SlidingWindowClassifier,
            The SlidingWindowClassifier is fitted on the training data.
        """
        # Check whether estimator is a valid classifier.
        if not isinstance(self.estimator, SkactivemlClassifier):
            raise TypeError(
                "'{}' must be a SkactivemlClassifier.".format(self.estimator)
            )
        self.check_X_dict_ = {
            "ensure_min_samples": 0,
            "ensure_min_features": 0,
            "allow_nd": True,
            "dtype": None,
        }

        X, y, sample_weight = self._validate_data(
            X=X,
            y=y,
            sample_weight=sample_weight,
            check_X_dict=self.check_X_dict_,
            established_spec=getattr(self, "target_spec_", None),
        )

        self._add_samples("partial_fit", X, y, sample_weight)
        X_train = np.array(self.X_train_)
        y_train = np.array(self.y_train_)
        sample_weight_train = None
        if self.sample_weight_train_ is not None:
            sample_weight_train = np.array(
                self.sample_weight_train_, dtype=float
            )
        return self._fit(
            X=X_train,
            y=y_train,
            sample_weight=sample_weight_train,
            **fit_kwargs,
        )

    def _add_samples(self, fit_func, X, y, sample_weight=None):
        if not hasattr(self, "X_train_"):
            self.X_train_ = deque(maxlen=self.window_size)
        if not hasattr(self, "y_train_"):
            self.y_train_ = deque(maxlen=self.window_size)
        if not hasattr(self, "sample_weight_train_"):
            self.sample_weight_train_ = deque(maxlen=self.window_size)
        if self.only_labeled:
            is_lbld = is_labeled(y, self.missing_label)
            X = X[is_lbld]
            y = y[is_lbld]
            if sample_weight is not None:
                sample_weight = sample_weight[is_lbld]
            else:
                sample_weight = None
        # reset the window if fit is called otherwise extend the window with
        # the given data
        if fit_func == "fit":
            self.X_train_ = deque(maxlen=self.window_size)
            self.y_train_ = deque(maxlen=self.window_size)
            self.sample_weight_train_ = deque(maxlen=self.window_size)
        self.X_train_.extend(X)
        self.y_train_.extend(y)
        if sample_weight is not None:
            self.sample_weight_train_.extend(sample_weight)
        else:
            self.sample_weight_train_ = None

    def _fit(self, X, y, sample_weight=None, **fit_kwargs):
        # Check whether estimator can deal with cost matrix.
        if self.cost_matrix is not None and not hasattr(
            self.estimator, "predict_proba"
        ):
            raise ValueError(
                "'cost_matrix' can be only set, if 'estimator'"
                "implements 'predict_proba'."
            )

        if hasattr(self, "estimator_"):
            self.estimator_ = deepcopy(self.estimator)
        else:
            self.estimator_ = deepcopy(self.estimator)

        if self.estimator_.classes is None:
            self.estimator_.set_params(classes=self.target_spec_.classes)

        if has_fit_parameter(self.estimator, "sample_weight"):
            fit_kwargs["sample_weight"] = sample_weight

        self.estimator_.fit(X=X, y=y, **fit_kwargs)

        return self

    def _validate_data(
        self,
        X,
        y,
        sample_weight=None,
        check_X_dict=None,
        established_spec=None,
    ):
        # super._validate_data is not called because a partial-fit window may
        # contain only a subset of the established class vocabulary.
        outer_classes = (
            self.classes
            if self.classes is not None
            else self.estimator.classes
        )
        target_spec = self._resolve_fitting_target_spec(
            y,
            established_spec=established_spec,
            classes=outer_classes,
        )
        inner_classes = (
            self.estimator.classes
            if self.estimator.classes is not None
            else target_spec.classes
        )
        self.estimator._resolve_fitting_target_spec(
            y,
            established_spec=target_spec,
            classes=inner_classes,
        )
        if self.window_size is not None:
            check_scalar(
                self.window_size,
                "window_size",
                int,
                min_val=0,
                min_inclusive=False,
            )
        check_type(self.only_labeled, "only_labeled", bool)

        check_y_dict = {
            "ensure_min_samples": 0,
            "ensure_min_features": 0,
            "ensure_2d": False,
            "ensure_all_finite": False,
            "dtype": None,
        }

        # Check input parameters.
        y = check_array(y, **check_y_dict)
        y = column_or_1d(y)
        if len(y) == 0:
            check_X_dict["ensure_2d"] = False
        X = check_array(X, **check_X_dict)
        check_consistent_length(X, y)
        if sample_weight is not None:
            sample_weight = check_array(sample_weight, **check_y_dict)
            sample_weight = column_or_1d(sample_weight)
            if len(y) != len(sample_weight):
                raise ValueError(
                    f"`y` has the length {len(y)} and `sample_weight` has the "
                    f"shape {sample_weight.shape}. Both need to have "
                    f"the same one-dimensional shape."
                )

        # Check common classifier parameters.
        check_classifier_params(
            self.classes, self.missing_label, self.cost_matrix
        )

        if (
            self.cost_matrix is not None
            and self.estimator.cost_matrix is not None
            and not np.array_equiv(
                self.cost_matrix, self.estimator.cost_matrix
            )
        ):
            raise ValueError(
                "'cost_matrix' and estimator.cost_matrix must be equal. "
                "Got {} is not equal to {}.".format(
                    self.cost_matrix, self.estimator.cost_matrix
                )
            )
        # self.missing_label is not testet completly and
        # needs to be checked for the general test.
        # if general test is removed, remove this check.
        _ = is_labeled(y, missing_label=self.missing_label)

        check_equal_missing_label(
            self.missing_label,
            self.estimator.missing_label,
        )
        # if self.classes=None or self.estimator.classes=None then no checks
        # are done if general test is removed it should be checked again
        # Store and check random state.
        self.random_state_ = check_random_state(self.random_state)
        self.target_spec_ = target_spec

        return X, y, sample_weight

    @match_signature("estimator", "predict")
    def predict(self, X, **predict_kwargs):
        """Return class label predictions for the input data `X`.

        Parameters
        ----------
        X : array-like of shape (n_samples, ...)
            Input samples.
        predict_kwargs : dict-like
            Further parameters as input to the `predict` method of the
            `estimator`.

        Returns
        -------
        y_pred : numpy.ndarray shape (n_samples,)
            Predicted class labels of the input samples.
        """
        check_is_fitted(self)
        return self.estimator_.predict(X, **predict_kwargs)

    @match_signature("estimator", "predict_proba")
    def predict_proba(self, X, **predict_proba_kwargs):
        """Return probability estimates for the input data `X`.

        Parameters
        ----------
        X : array-like of shape (n_samples, ...)
            Input samples.
        predict_proba_kwargs : dict-like
            Further parameters as input to the `predict_proba` method of the
            `estimator`.

        Returns
        -------
        P : numpy.ndarray shape (n_samples, classes)
            The class probabilities of the input samples `X`. Classes are
            ordered according to the attribute `self.classes_`.
        """
        check_is_fitted(self)
        proba = self.estimator_.predict_proba(X, **predict_proba_kwargs)
        return proba

    @match_signature("estimator", "predict_freq")
    def predict_freq(self, X, **predict_freq_kwargs):
        """Return class frequency estimates for the test samples `X`.

        Parameters
        ----------
        X : array-like of shape (n_samples, ...)
            Test samples whose class frequencies are to be estimated.

        Returns
        -------
        F : numpy.ndarray of shape (n_samples, classes)
            The class frequency estimates of the test samples `X`. Classes are
            ordered according to the attribute `self.classes_`.
        """
        check_is_fitted(self)
        freq = self.estimator_.predict_freq(X, **predict_freq_kwargs)
        return freq

    def __getattr__(self, item):
        if item in self._own_fitted_attributes:
            return _resolve_own_fitted_attribute(self, item)
        if "estimator_" in self.__dict__ and hasattr(self.estimator_, item):
            return getattr(self.estimator_, item)
        else:
            raise AttributeError(f"{item} does not exist")


if successful_skorch_torch_import:

    class SkorchClassifier(SkactivemlClassifier, SkorchMixin):
        """SkorchClassifier

        Implement a classification wrapper class to make it possible to use
        `torch` with `skactiveml`. This is achieved by providing a wrapper
        around `torch` that has a `skactiveml` interface and can handle
        missing labels. This wrapper is based on the open-source library
        `skorch` [1]_.

        Notes
        -----
        - Adjust your `criterion` and `module.forward` outputs consistently.
          See the documentation of the parameters `forward_outputs` and
          `criterion_output_keys` for further details.
        - Beyond multiclass classification, only multilabel classification is
          supported, which corresponds to multiple binary classification tasks.

        Parameters
        ----------
        module : torch.nn.Module.__class__ or torch.nn.Module
            A PyTorch `torch.nn.Module`. In general, the uninstantiated class
            should be passed, although instantiated modules will also work.
        criterion : torch.nn.Module or torch.nn.Module.__class__, \
                default=None
            The loss (criterion) used to optimize the module.

            - If `None`, the torch.nn.Module is set to
              `torch.nn.CrossEntropyLoss` in the case of a single output
              classification problem and `torch.nn.BCEWithLogitsLoss` in the
              case of a multioutput classification problem.
            - If a class (subclass of `torch.nn.Module`) is passed
              (e.g. `torch.nn.CrossEntropyLoss`), it is instantiated
              internally.
            - If an instance is passed (e.g. `torch.nn.CrossEntropyLoss()`),
              that instance (or a wrapped copy of it) is used.

        forward_outputs : dict[str, tuple[int, Callable | None]] or None,\
                default=None
            Dictionary that describes how to get and post-process the outputs
            of `module.forward` for prediction. This parameter replaces the
            functionality of `predict_nonlinearity` in a `skorch.net.NeuralNet`
            (see documentation of `neural_net_param_dict`).

            Given `raw_outputs = module.forward(x)`, each entry
            `name -> (idx, transform)` in `forward_outputs` is interpreted as:

            - `idx` : int
              Index into `raw_outputs` (0-based).
            - `transform` : callable or `None`
              If not `None`, it is applied to the selected raw tensor
              `raw_outputs[idx]`. Otherwise, the raw tensor is used.

            This allows multiple named outputs to reference the same raw tensor
            with different transforms, for example::

                forward_outputs = {
                    "proba":  (0, torch.nn.Softmax(dim=-1)),  # probabilities
                    "logits": (0, None),                      # raw scores
                    "emb":    (1, None),                      # embeddings
                }

            The first entry in `forward_outputs` defines the primary
            scores used for prediction:

            - In `predict_proba`, the transformed first output is
              interpreted as class probabilities `P`.
            - In `predict`, the class probabilities `P` returned by
              `predict_proba` are used to infer class label predictions.

            If `forward_outputs` is `None`, a sensible default is chosen
            for common single-output classifiers based on the `criterion`:

            - If `criterion` is `torch.nn.CrossEntropyLoss`, it is
              assumed that `module.forward` returns logits and the
              effective mapping is::

                  {"proba": (0, torch.nn.Softmax(dim=-1))}

            - If `criterion` is `torch.nn.NLLLoss`, it is assumed that
              `module.forward` returns log-probabilities and the effective
              mapping is::

                  {"proba": (0, torch.exp)}

            - For all other criteria, a single-output module is assumed to
              already produce values in probability space, and the effective
              mapping is::

                  {"proba": (0, None)}

        criterion_output_keys : str or sequence of str or None, default=None
            Name or names of the forward outputs that are passed to the
            loss / criterion during training. Use this when
            `module.forward` returns multiple outputs
            (e.g. `(logits, embeddings, ...)`), but the criterion expects
            a single tensor input or a specific tuple of inputs.

            The names must refer to keys of the effective `forward_outputs`
            mapping. If `criterion_output_keys` is not `None` and
            `forward_outputs` is `None`, a `ValueError` is raised
            because the names cannot be resolved.

            - If a `str`, the corresponding named output of
              `module.forward` (i.e., the raw tensor selected via its
              index in `forward_outputs` before applying the transform)
              is passed to the criterion (e.g. `"logits"` to use only the
              class scores).
            - If a sequence of `str`, the selected named outputs are passed to
              the criterion in that order. Each raw forward output index may
              appear at most once: using multiple names that resolve to the
              same underlying index (e.g. `"proba"` and `"logits"` both
              pointing to index 0) is not allowed and results in a
              `ValueError`.
            - If `None`, the first output defined by the effective
              `forward_outputs` mapping is used as criterion input.

            To pass all distinct forward outputs to the criterion in the
            same order as `forward_outputs`, choose one representative name
            per raw output index and set, for example::

                # assuming that each key refers to a different raw index
                criterion_output_keys = tuple(forward_outputs.keys())

            If `forward_outputs` contains multiple names that refer to the
            same raw output index (aliases such as `"proba"` and `"logits"`
            both mapping to index 0), you must select at most one name per
            raw index in `criterion_output_keys`.
        neural_net_param_dict : dict, default=None
            Additional arguments for `skorch.net.NeuralNet`. If
            `neural_net_param_dict` is `None`, no additional arguments are
            added. `module`, `criterion`, and `predict_nonlinearity` are not
            allowed in this dictionary.
        sample_dtype : str or type, default=np.float32
            Data type to which input samples are cast inside the estimator. If
            set to `None`, the input dtype is preserved.
        target_dtype : str or type, default=None
            Data type used to cast the internally encoded targets `y_enc`.
            These encoded targets are integers in the range
            `[0, n_classes - 1]`. Missing labels are encoded as `-1`.

            - If `None`, infer a suitable dtype from the loss criterion; if
              inference fails, default to `np.int64`.
            - Otherwise, cast targets via
              `y_enc.astype(target_dtype, copy=False)`.
        target_type : "auto" or "single-output" or "multi-label", \
                default="auto"
            Declared target type. Multi-label classification is supported.
        include_unlabeled_samples : bool, default=False
            - If `False`, only labeled samples are passed to the `fit` method
              of the estimator.
            - If `True`, all samples including the unlabeled ones are passed to
              the `fit` method of the estimator. Ensure that the `criterion`
              is able to handle unlabeled samples marked by `missing_label`.
              Otherwise, `missing_label` is interpreted as a regular class
              label.
        classes : array-like of shape (n_classes,) or a list of such \
            array-likes, default=None
        - If `classes` is not nested (`None` or one-dimensional), a single task
          problem is assumed such that `y` can be shape `(n_samples,)` or
          `(n_samples, n_annotators)`.
        - If `classes` is nested (list of array-like objects), multilabel
          classification is assumed in this wrapper and `y` must be shape
          `(n_samples, n_tasks)` with `n_tasks == len(classes)`. Each task
          must be binary.
        missing_label : scalar or str or np.nan or None, default=np.nan
            Value to represent a missing label.
        cost_matrix : array-like of shape (n_classes, n_classes), default=None
            Cost matrix with `cost_matrix[i, j]` indicating the cost of
            predicting class `classes[j]` for a sample of class
            `classes[i]`. Can only be set if `classes` is not `None` and for
            single output problems.
        random_state : int or RandomState instance or None, default=None
            Determines random number generation for methods that rely on
            randomness (e.g. `predict` for stochastic models). Pass an int for
            reproducible results across multiple method calls.

        References
        ----------
        .. [1] Marian Tietz, Thomas J. Fan, Daniel Nouri, Benjamin Bossan, and
           skorch Developers. skorch: A scikit-learn compatible neural network
           library that wraps PyTorch, July 2017.
        """

        def __init__(
            self,
            module,
            criterion=None,
            forward_outputs=None,
            criterion_output_keys=None,
            neural_net_param_dict=None,
            sample_dtype=np.float32,
            include_unlabeled_samples=False,
            classes=None,
            cost_matrix=None,
            missing_label=MISSING_LABEL,
            random_state=None,
            target_dtype=None,
            target_type="auto",
        ):
            super(SkorchClassifier, self).__init__(
                classes=classes,
                missing_label=missing_label,
                cost_matrix=cost_matrix,
                random_state=random_state,
            )
            self.module = module
            self.criterion = criterion
            self.forward_outputs = forward_outputs
            self.criterion_output_keys = criterion_output_keys
            self.neural_net_param_dict = neural_net_param_dict
            self.sample_dtype = sample_dtype
            self.target_dtype = target_dtype
            self.target_type = target_type
            self._resolve_target_spec_on_validate = True
            self.include_unlabeled_samples = include_unlabeled_samples

        @property
        def _target_capabilities(self):
            return frozenset(
                {
                    ("classification", "single-output", "single-annotator"),
                    ("classification", "multi-label", "single-annotator"),
                }
            )

        def fit(self, X, y, **fit_params):
            """Initialize and fit the module.

            If the module was already initialized, by calling fit, the module
            will be re-initialized (unless `warm_start` is True).

            Parameters
            ----------
            X : matrix-like, shape (n_samples, ...)
                Training data set, usually complete, i.e. including the labeled
                and unlabeled samples
            y : array-like of shape (n_samples,) or (n_samples, n_outputs)
                Labels of the training data set (possibly including unlabeled
                ones indicated by self.missing_label). For multioutput
                problems, a row `y[i]` must be either contain only observed
                labels or only `missing_label` values, i.e., no mixing
                within a row.
            fit_params : dict-like
                Further parameters as input to the 'fit' method of the
                `skorch.net.NeuralNet`.

            Returns
            -------
            self: SkorchClassifier,
                `SkorchClassifier` object fitted on the training data.
            """
            return self._fit("fit", X, y, **fit_params)

        def partial_fit(self, X, y, **fit_params):
            """Fit the module without re-initialization.

            If the module was already initialized, by calling `partial_fit`,
            the module will not be re-initialized again.

            Parameters
            ----------
            X : matrix-like, shape (n_samples, ...)
                Training data set, usually complete, i.e. including the labeled
                and unlabeled samples
            y : array-like of shape (n_samples,) or (n_samples, n_outputs)
                Labels of the training data set (possibly including unlabeled
                ones indicated by self.missing_label). For multioutput
                problems, a row `y[i]` must either contain only observed
                labels or only `missing_label` values, i.e., no mixing
                within a row.
            fit_params : dict-like
                Further parameters as input to the 'partial_fit' method of the
                `skorch.net.NeuralNet`.

            Returns
            -------
            self: SkorchClassifier
                `SkorchClassifier` object fitted on the training data.
            """
            return self._fit("partial_fit", X, y, **fit_params)

        def predict(self, X, extra_outputs=None):
            """Return class predictions for the test samples `X`.

            By default, this method returns only the predicted classes
            `y_pred`. The predictions are obtained via the class probabilities
            `P` outputted by `predict_proba`. If `extra_outputs` is provided,
            a tuple is returned whose first element is `y_pred` and whose
            remaining elements are the requested additional forward outputs,
            in the order specified by `extra_outputs`.

            Parameters
            ----------
            X : array-like of shape (n_samples, ...)
                Test samples.
            extra_outputs : None or str or or sequence of str, default=None
                Names of additional outputs to return next to `y_pred`. The
                names must be a subset of the keys of the effective
                `forward_outputs` mapping.

                For example, if::

                    self.forward_outputs = {
                        "proba":  (0, torch.nn.Softmax(dim=-1)),
                        "logits": (0, None),
                        "emb":    (1, None),
                    }

                then valid values for `extra_outputs` include `"emb"` or
                `["emb", "logits"]`.

                - If `extra_outputs is None`, only `y_pred` is returned.
                - If `extra_outputs` is a string, e.g. `"emb"`, the
                  return value is `(y_pred, emb)`.
                - If `extra_outputs` is a sequence of strings, the return
                  value is `(y_pred, out_1, out_2, ...)`, where `out_i`
                  corresponds to the i-th name in `extra_outputs`.

            Returns
            -------
            y_pred : numpy.ndarray of shape (n_samples,) \
                    or (n_samples, n_outputs)
                Predicted class labels of the test samples.
            *extras : numpy.ndarray, optional
                Additional outputs. Only present if `extra_outputs` is not
                `None`. In that case, the method returns a single tuple whose
                first element is `y_pred` and whose remaining elements
                (`extras`) correspond to the requested forward outputs in the
                order given by `extra_outputs`.
            """
            return super().predict(
                X=X,
                extra_outputs=extra_outputs,
            )

        def predict_proba(self, X, extra_outputs=None):
            """Return class probability estimates for the test samples `X`.

            By default, this method returns only the predicted class
            probabilities `P`. If `extra_outputs` is provided, a tuple is
            returned whose first element is `y_pred` and whose remaining
            elements are the requested additional forward outputs, in the
            order specified by `extra_outputs`.

            Parameters
            ----------
            X : array-like of shape (n_samples, ...)
                Test samples.
            extra_outputs : None or str or sequence of str, default=None
                Names of additional outputs to return next to `P`. The names
                must be a subset of the keys of the effective `forward_outputs`
                mapping.

                For example, if::

                    self.forward_outputs = {
                        "proba":  (0, torch.nn.Softmax(dim=-1)),
                        "logits": (0, None),
                        "emb":    (1, None),
                    }

                then valid values for `extra_outputs` include `"emb"` or
                `["emb", "logits"]`.

                - If `extra_outputs is None`, only `P` is returned.
                - If `extra_outputs` is a string, e.g. `"logits"`, the
                  return value is `(P, logits)`.
                - If `extra_outputs` is a sequence of strings, the return
                  value is `(P, out_1, out_2, ...)`, where `out_i`
                  corresponds to the i-th name in `extra_outputs`.

            Returns
            -------
            P : numpy.ndarray of shape (n_samples, n_classes)
                Class probabilities of the test samples. Classes are ordered
                according to `self.classes_`.
            *extras : numpy.ndarray, optional
                Additional outputs. Only present if `extra_outputs` is not
                `None`. In that case, the method returns a single tuple whose
                first element is `P` and whose remaining elements
                (`extras`) correspond to the requested forward outputs in the
                order given by `extra_outputs`.
            """
            # Initialize module, if not done yet.
            if not hasattr(self, "neural_net_"):
                self.initialize()

            # Check input parameters.
            X = check_array(X, **self.check_X_dict_)
            check_n_features(
                self, X, reset=not hasattr(self, "n_features_in_")
            )
            self._check_prefit_prediction_ambiguity()

            # Resolve effective forward_outputs (either user-provided or
            # defaulted based on the criterion).
            forward_outputs = self._effective_forward_outputs()

            # Forward propagation whose return values depends on the request
            # ones.
            fw_out = self._forward_with_named_outputs(
                X, forward_outputs=forward_outputs, extra_outputs=extra_outputs
            )

            # First element is expected to be the class probabilities.
            P = fw_out[0] if isinstance(fw_out, tuple) else fw_out
            self._initialize_fallbacks(P)
            return fw_out

        def _effective_forward_outputs(self, y=None):
            """Return the effective `forward_outputs` mapping.

            If the user did not specify `forward_outputs`, choose a reasonable
            default for common criteria (e.g., `nn.CrossEntropyLoss`) and a
            simple single-output module.

            The returned mapping has the form::

                {name: (idx, transform)}

            where `idx` is the index into the tuple returned by
            `module.forward` (0-based) and `transform` is a callable or
            `None`. For the defaults below, a single-output module is assumed,
            i.e., `idx == 0`.
            """
            # User explicitly provided a mapping: trust it.
            if self.forward_outputs is not None:
                return self.forward_outputs

            # No explicit mapping: handle common single-output cases.
            if self.criterion is None:
                crit_cls = (
                    nn.BCEWithLogitsLoss
                    if self._uses_multilabel_target(y=y)
                    else nn.CrossEntropyLoss
                )
            else:
                crit_cls = (
                    self.criterion
                    if isinstance(self.criterion, type)
                    else self.criterion.__class__
                )

            if crit_cls is nn.CrossEntropyLoss:
                # Single-output network returning logits.
                return {"proba": (0, nn.Softmax(dim=-1))}

            if crit_cls is nn.NLLLoss:
                # Module returns log-probabilities.
                return {"proba": (0, torch.exp)}

            if crit_cls is nn.BCEWithLogitsLoss:
                # Multi-label modules return logits per label.
                return {"proba": (0, torch.sigmoid)}

            # Fallback: treat the single forward output as already in
            # probability space. Caller is responsible for making this true.
            return {"proba": (0, None)}

        def _provisional_target_type(self, y=None):
            """Resolve semantics needed while constructing an unfitted net."""
            target_spec = getattr(self, "target_spec_", None)
            if target_spec is not None:
                return target_spec.target_type
            if self.classes is not None:
                y_dummy = (
                    np.empty(
                        (0, len(self.classes)),
                        dtype=np.asarray(self.missing_label).dtype,
                    )
                    if _has_nested_classes(self.classes)
                    else self.classes
                )
                return self._resolve_target_spec(y_dummy).target_type

            target_type = (
                "single-output"
                if self.target_type == "auto"
                else self.target_type
            )
            _check_target_capability(
                type(self).__name__,
                ("classification", target_type, "single-annotator"),
                self._target_capabilities,
            )
            return target_type

        def _uses_multilabel_target(self, y=None):
            return self._provisional_target_type(y) == "multi-label"

        def _net_parts(self, X=None, y=None):
            """Assemble and validate network components.

            Implementations should perform any optional checks or normalization
            of constructor/init parameters (e.g., shape consistency, dtype
            checks, wrapping criteria), then return the ready-to-use pieces for
            `skorch.NeuralNet`.

            Parameters
            ----------
            X : array-like of shape (n_samples, ...), default=None
                Input samples for optional validation.
            y : array-like of shape (n_samples, ...), default=None
                Target values for optional validation.

            Returns
            -------
            module : torch.nn.Module.__class__ or torch.nn.Module
                A PyTorch `torch.nn.Module`. In general, the uninstantiated
                class should be passed, although instantiated modules will also
                work.
            criterion : torch.nn.Module.__class__
                The uninitialized criterion (loss) used to optimize the module.
            params : dict
                Keyword arguments (excluding `predict_non_linearity`) for
                `skorch.NeuralNet` construction. Must be a mapping and may be
                empty.
            """
            if self.criterion is None:
                if self._uses_multilabel_target(y=y):
                    criterion = nn.BCEWithLogitsLoss
                else:
                    criterion = nn.CrossEntropyLoss
            else:
                criterion = self.criterion
            criterion = make_criterion_tuple_aware(
                criterion=criterion,
                criterion_output_keys=self.criterion_output_keys,
                forward_outputs=self._effective_forward_outputs(y=y),
            )
            return (
                self.module,
                criterion,
                self.neural_net_param_dict or {},
            )

        def _validate_data_kwargs(self):
            """Return kwargs forwarded to `_validate_data`.

            Returns
            -------
            kwargs : dict or None
                Keyword arguments consumed by `_validate_data`.
            """
            self.check_X_dict_ = {
                "ensure_min_samples": 0,
                "ensure_min_features": 0,
                "allow_nd": True,
                "dtype": self.sample_dtype,
            }
            check_type(
                self.include_unlabeled_samples,
                "include_unlabeled_samples",
                bool,
            )
            return {
                "check_X_dict": self.check_X_dict_,
            }

        def _return_training_data(self, X, y):
            """Return only samples and labels required for training.

            Parameters
            ----------
            X : array-like of shape (n_samples, ...)
                Input samples.
            y : array-like of shape (n_samples, ...)
                Targets with unlabeled entries following the subclass'
                convention.

            Returns
            -------
            X_train : ndarray or None
                Training samples or `None` if none exist.
            y_train : ndarray or None
                Training labels or `None` if none exist.
            """
            X_train, y_train = None, None
            if self.include_unlabeled_samples:
                is_included = np.full_like(y, fill_value=True, dtype=bool)
            else:
                is_included = is_labeled(
                    y=y,
                    missing_label=-1,
                    target_type=self.target_spec_.target_type,
                )
            if np.sum(is_included) > 0:
                X_train = X[is_included]
                if self.target_dtype is None:
                    y_dtype = self._infer_target_numpy_dtype(
                        self.neural_net_.criterion
                    )
                else:
                    y_dtype = self.target_dtype
                y_train = y[is_included].astype(y_dtype)
            return X_train, y_train

        def _initialize_fallbacks(self, P):
            """Initialize label/cost fallbacks if the classifier was not fitted
            before.

            Parameters
            ----------
            P : array-like of shape (n_samples, n_classes)
                Class-probability array used only to infer `n_classes` when
                `self.classes` is `None`.
            """
            self.random_state_ = check_random_state(self.random_state)
            if not hasattr(self, "_le"):
                if self.classes is not None:
                    y_dummy = self.classes
                else:
                    y_dummy = np.arange(P.shape[-1], dtype=int)
                self._initialize_label_state(y_dummy)
            check_classifier_params(
                self.classes_, self.missing_label, self.cost_matrix_
            )

        def _check_prefit_prediction_ambiguity(self):
            """Reject prefit prediction when the output task type is unknown.

            Without fitted label state or user-provided `classes`, predictions
            cannot be interpreted unambiguously as multiclass versus
            multi-label. `_initialize_fallbacks` can infer
            flat classes from `P.shape[-1]`, but that only yields the single-
            output interpretation.
            """
            if hasattr(self, "_le") or self.classes is not None:
                return
            raise ValueError(
                "`predict_proba` is ambiguous before fitting when "
                "`classes=None`. Call `fit` first or provide `classes` to "
                "disambiguate multiclass versus multilabel behavior."
            )

        def _infer_target_numpy_dtype(self, criterion, *, default=np.int64):
            """Infer the NumPy dtype to use for encoded targets based on a
            PyTorch loss.

            Parameters
            ----------
            criterion : type or torch.nn.modules.loss._Loss
                Loss class or instance. Only a small set of common
                classification losses is handled explicitly.

                - nn.CrossEntropyLoss, nn.NLLLoss -> np.int64 (class indices)
                - nn.BCEWithLogitsLoss, nn.BCELoss -> np.float32
                  (binary/multi-label targets)

            default : np.dtype, default=np.int64
                Fallback dtype if the criterion is not recognized.

            Returns
            -------
            dtype : np.dtype
                Inferred NumPy data type for casting targets before converting
                to torch.
            """
            crit_cls = (
                criterion if isinstance(criterion, type) else type(criterion)
            )
            if issubclass(crit_cls, (nn.BCEWithLogitsLoss, nn.BCELoss)):
                return np.float32
            if issubclass(crit_cls, (nn.CrossEntropyLoss, nn.NLLLoss)):
                return np.int64
            return default


if successful_capymoa_import:

    class CapyMOAClassifier(SkactivemlClassifier):
        """CapyMOA Classifier

        Implementation of a wrapper class for `CapyMOA` [1]_ classifiers such
        that missing labels can be handled and the interfaces are compatible
        with `scikit-activeml`. Therefore, samples with missing labels are
        filtered.

        Parameters
        ----------
        estimator_class : capymoa.base.MOAClassifier.__class__
            The `capymoa` classifier class that is used to initialize the
            `capymoa` classifier.
        estimator_param_dict : dict, default=None
            Additional arguments for `capymoa.base.MOAClassifier`. If
            `estimator_param_dict` is `None`, no additional arguments are
            added. `schema` is not allowed in this dictionary and will be
            created internally.
        classes : array-like of shape (n_classes,), default=None
            Holds the label for each class. If `None`, the classes are
            determined during `fit`.
        missing_label : scalar or string or np.nan or None, default=np.nan
            Value to represent a missing label.
        cost_matrix : array-like of shape (n_classes, n_classes)
            Cost matrix with `cost_matrix[i,j]` indicating cost of predicting
            class `classes[j]` for a sample of class `classes[i]`. Can be only
            set, if `classes` is not `None`.
        random_state : int or RandomState instance or None, default=None
            Determines random number for `predict` method. Pass an int for
            reproducible results across multiple method calls.
        target_type : "auto" or "single-output", default="auto"
            Declared target type. This wrapper supports only single-output
            classification.

        Attributes
        ----------
        classes_ : numpy.ndarray of shape (n_classes,)
            Holds the label for each class after fitting.
        cost_matrix_ : numpy.ndarray of shape (classes, classes)
            Cost matrix with `cost_matrix_[i,j]` indicating cost of predicting
            class `classes_[j]` for a sample of class `classes_[i]`.
        estimator_ : capymoa.base.MOAClassifier
            initialized MOAClassifier whose predictions and training are
            wrapped.

        References
        ----------
        .. [1] Gomes, H.M., Lee, A., Gunasekara, N., Sun, Y., Cassales, G.W.,
           Liu, J., Heyden, M., Cerqueira, V., Bahri, M., Koh, Y.S. and
           Pfahringer, B., 2025. Capymoa: Efficient machine learning for data
           streams in python. arXiv preprint arXiv:2502.07432.
        """

        def __init__(
            self,
            estimator_class,
            estimator_param_dict=None,
            classes=None,
            missing_label=MISSING_LABEL,
            cost_matrix=None,
            random_state=None,
            target_type="auto",
        ):
            super().__init__(
                classes=classes,
                missing_label=missing_label,
                cost_matrix=cost_matrix,
                random_state=random_state,
                target_type=target_type,
            )
            self.estimator_class = estimator_class
            self.estimator_param_dict = estimator_param_dict

        def fit(self, X, y):
            """Fit the module with (re-)initialization using `X` as training
            data and `y` as class labels. The model is reinitialized from
            scratch when using `fit`

            Parameters
            ----------
            X : matrix-like, shape (n_samples, n_features)
                Training data set, usually complete, i.e. including the labeled
                and unlabeled samples
            y : array-like of shape (n_samples, )
                Labels of the training data set (possibly including unlabeled
                ones indicated by self.missing_label)

            Returns
            -------
            self: CapyMOAClassifier,
                `CapyMOAClassifier` object fitted on the training data.
            """
            return self._fit("fit", X, y)

        def partial_fit(self, X, y):
            """Fit the module without re-initialization. If the module was
            already initialized, by calling `partial_fit` or `fit`, the module
            will not be re-initialized again.

            Parameters
            ----------
            X : matrix-like, shape (n_samples, n_features)
                Training data set, usually complete, i.e. including the labeled
                and unlabeled samples
            y : array-like of shape (n_samples, )
                Labels of the training data set (possibly including unlabeled
                ones indicated by `self.missing_label`)

            Returns
            -------
            self: CapyMOAClassifier
                `CapyMOAClassifier` object fitted on the training data.
            """
            return self._fit("partial_fit", X, y)

        def predict_proba(self, X):
            """Return probability estimates for the input data `X`.

            Parameters
            ----------
            X : array-like of shape (n_samples, ...)
                Input samples.

            Returns
            -------
            P : array-like of shape (n_samples, classes)
                The class probabilities of the input samples. Classes are
                ordered according to the attribute `self.classes_`.
            """
            import capymoa
            import capymoa.instance

            check_is_fitted(self)
            predict_dict = {"ensure_min_samples": 1, "ensure_min_features": 1}
            X = check_array(X, **(self.check_X_dict_ | predict_dict))
            check_n_features(self, X, reset=False)
            n_classes = len(self.classes_)
            if self.is_fitted_:
                p_list = []
                for x in X:
                    x_instance = capymoa.instance.Instance(
                        schema=self.schema_, instance=x
                    )
                    p_i = self.estimator_.predict_proba(x_instance)
                    # if estimator_ fails, it returns None. In this case, we
                    # use a uniform distribution as fallback
                    if p_i is None:
                        p_i = np.ones(n_classes) / n_classes
                        if sum(self._label_counts) > 0:
                            p_i = self._label_counts / np.sum(
                                self._label_counts
                            )
                    pad_length = n_classes - len(p_i)
                    if pad_length > 0:
                        p_i = np.pad(p_i, (0, pad_length))
                    p_list.append(p_i)
                P = np.array(p_list)
                if not np.any(np.isnan(P)):
                    return P

            warnings.warn(
                f"Since the 'base_estimator' could not be fitted when"
                f" calling the `fit` method, the class label "
                f"distribution`_label_counts={self._label_counts}` is used to "
                f"make the predictions."
            )
            # fallback if clf could not be fitted (i.e., no labeled data)
            if sum(self._label_counts) == 0:
                n_classes = len(self.classes_)
                return np.ones([len(X), n_classes]) / n_classes
            else:
                return np.tile(
                    self._label_counts / np.sum(self._label_counts),
                    [len(X), 1],
                )

        def _fit(self, fit_function, X, y, sample_weight=None):
            import capymoa
            import capymoa.base
            import capymoa.instance

            target_spec = self._resolve_target_spec_for_fit(
                y, is_incremental=fit_function == "partial_fit"
            )

            # Check input parameters.
            self.check_X_dict_ = {
                "ensure_min_samples": 0,
                "ensure_min_features": 0,
                "allow_nd": True,
                "dtype": None,
            }
            X, y, _ = self._validate_data(
                X=X,
                y=y,
                sample_weight=None,
                check_X_dict=self.check_X_dict_,
                reset=fit_function == "fit"
                or not hasattr(self, "n_features_in_"),
                target_spec=target_spec,
            )

            # Check whether estimator is a valid classifier.
            if not isinstance(self.estimator_class, type) or not issubclass(
                self.estimator_class, capymoa.base.MOAClassifier
            ):
                raise TypeError(
                    "'{}' must be a capymoa "
                    "classifier.".format(self.estimator_class)
                )
            is_included = is_labeled(y, missing_label=-1)
            self._label_counts = [
                np.sum(y[is_included] == c)
                for c in range(len(self._le.classes_))
            ]
            if hasattr(self, "estimator_"):
                if fit_function != "partial_fit":
                    self.estimator_ = self._create_estimator(X)
            else:
                self.estimator_ = self._create_estimator(X)
            if self.estimator_ is None:
                self.is_fitted_ = False
                return self

            try:
                X_train = X[is_included]
                y_train = y[is_included].astype(np.int64)
                if np.sum(is_included) == 0:
                    raise ValueError("There is no labeled data.")

                column_list = [(str(i) for i in range(X.shape[1]))]
                column_list += ["label"]

                for i in range(len(y_train)):
                    x_inst = X_train[i]
                    y_inst = y_train[i].item()
                    instance = capymoa.instance.LabeledInstance.from_array(
                        self.schema_,
                        x=x_inst,
                        y_index=y_inst,
                    )
                    self.estimator_.train(instance)
                self.is_fitted_ = True
            except Exception as e:
                self.is_fitted_ = False
                warnings.warn(
                    "The 'base_estimator' could not be fitted because of"
                    " '{}'. Therefore, the class labels of the samples "
                    "are counted and will be used to make predictions. "
                    "The class label distribution is "
                    "`_label_counts={}`.".format(e, self._label_counts)
                )
            return self

        def _create_estimator(self, X):
            """Initialize the estimator according so `self.classes_` and `X`.
            This function assumes that self.validate_data has been used to
            guarantee that `self.classes_` exists.

            Parameters
            ----------
            X : array-like of shape (n_samples, ...)
                The feature matrix representing the samples.

            Returns
            -------
            estimator: CapyMOAClassifier,
                The initialized but untrained `CapyMOAClassifier`.
            """
            import capymoa
            import capymoa.stream

            estimator_kwargs = {}
            if self.estimator_param_dict is not None:
                if not isinstance(self.estimator_param_dict, dict):
                    raise TypeError(
                        "The 'estimator_param_dict=' must be a dictionary but"
                        f"is {self.estimator_param_dict}."
                    )
                if "schema" in self.estimator_param_dict:
                    raise AttributeError(
                        "The schema must not be set in "
                        "'self.estimator_param_dict' and must only be set"
                        "with 'self.schema'."
                    )
                estimator_kwargs = self.estimator_param_dict
            # features here means all attributes of an instance including their
            # class label
            X_shape = X.shape
            if len(X_shape) != 2:
                return None
            features = [f"f{f}" for f in range(X.shape[1])]
            features.append("label")
            categories = {"label": [str(c) for c in range(len(self.classes_))]}
            self.schema_ = capymoa.stream.Schema.from_custom(
                features=features, target="label", categories=categories
            )
            return self.estimator_class(self.schema_, **estimator_kwargs)


if successful_river_import:

    class RiverClassifier(SkactivemlClassifier, MetaEstimatorMixin):
        """River Classifier

        Implementation of a wrapper class for `river` [1]_ classifiers such
        that they implement the `SkactivemlClassifier` interfaces for
        classifiers. Additionally, filters the samples with missing labels if
        needed.

        Parameters
        ----------
        estimator : river.base.Classifier
            The `river` classifier to be wrapped.
        classes : array-like of shape (n_classes,), default=None
            Holds the label for each class. If `None`, the classes are
            determined during `fit`.
        missing_label : scalar or string or np.nan or None, default=np.nan
            Value to represent a missing label.
        cost_matrix : array-like of shape (n_classes, n_classes)
            Cost matrix with `cost_matrix[i,j]` indicating cost of predicting
            class `classes[j]` for a sample of class `classes[i]`. Can be only
            set, if `classes` is not `None`.
        random_state : int or RandomState instance or None, default=None
            Determines random number for `predict` method. Pass an int for
            reproducible results across multiple method calls.
        target_type : "auto" or "single-output", default="auto"
            Declared target type. This wrapper supports only single-output
            classification.

        Attributes
        ----------
        classes_ : numpy.ndarray of shape (n_classes,)
            Holds the label for each class after fitting.
        cost_matrix_ : numpy.ndarray of shape (classes, classes)
            Cost matrix with `cost_matrix_[i,j]` indicating cost of predicting
            class `classes_[j]` for a sample of class `classes_[i]`.
        estimator_ : river.base.Classifier
            The `river` classifier after calling the `fit` method.

        References
        ----------
        .. [1] Montiel, J., Halford, M., Mastelini, S.M., Bolmier, G.,
           Sourty, R., Vaysse, R., Zouitine, A., Gomes, H.M., Read, J.,
           Abdessalem, T. and Bifet, A., 2021. River: machine learning for
           streaming data in python. Journal of Machine Learning Research,
           22(110), pp.1-8.
        """

        def __init__(
            self,
            estimator,
            classes=None,
            missing_label=MISSING_LABEL,
            cost_matrix=None,
            random_state=None,
            target_type="auto",
        ):
            super().__init__(
                classes=classes,
                missing_label=missing_label,
                cost_matrix=cost_matrix,
                random_state=random_state,
                target_type=target_type,
            )
            self.estimator = estimator

        def fit(self, X, y, sample_weight=None):
            """Fit the model using `X` as training data and `y` as class
            labels.

            Parameters
            ----------
            X : array-like of shape (n_samples, ...)
                The feature matrix representing the samples.
            y : array-like of shape (n_samples,) or (n_samples, n_outputs)
                It contains the class labels of the training samples. Missing
                labels are represented by the attribute `self.missing_label_`.
                In case of multiple labels per sample (i.e., n_outputs > 1),
                the samples are duplicated.
            sample_weight : array-like of shape (n_samples,) or\
                    (n_samples, n_outputs)
                It contains the weights of the training samples' class labels.
                It must have the same shape as `y`.

            Returns
            -------
            self: Riverclassifier,
                The `Riverclassifier` fitted on the training data.
            """
            return self._fit(
                fit_function="fit",
                X=X,
                y=y,
                sample_weight=sample_weight,
            )

        def partial_fit(self, X, y, sample_weight=None):
            """Partially fitting the model using `X` as training data and `y`
            as class labels.

            Parameters
            ----------
            X : array-like of shape (n_samples, ...)
                The feature matrix representing the samples.
            y : array-like of shape (n_samples,) or (n_samples, n_outputs)
                It contains the class labels of the training samples. Missing
                labels are represented the attribute `self.missing_label_`. In
                case of multiple labels per sample (i.e., n_outputs > 1), the
                samples are duplicated.
            sample_weight : array-like of shape (n_samples,) or\
                    (n_samples, n_outputs)
                It contains the weights of the training samples' class labels.
                It must have the same shape as `y`.

            Returns
            -------
            self : Riverclassifier,
                The `Riverclassifier` is fitted on the training data.
            """
            return self._fit(
                fit_function="partial_fit",
                X=X,
                y=y,
                sample_weight=sample_weight,
            )

        def predict_proba(self, X, **predict_proba_kwargs):
            """Return probability estimates for the input data `X`.

            Parameters
            ----------
            X : array-like of shape (n_samples, ...)
                Input samples.
            predict_proba_kwargs : dict-like
                Further parameters as input to the `predict_proba` method of
                the `estimator`.

            Returns
            -------
            P : array-like of shape (n_samples, classes)
                The class probabilities of the input samples. Classes are
                ordered according to the attribute `self.classes_`.
            """
            check_is_fitted(self)
            predict_dict = {"ensure_min_samples": 1, "ensure_min_features": 1}
            X = check_array(X, **(self.check_X_dict_ | predict_dict))
            check_n_features(self, X, reset=False)
            if self.is_fitted_:
                P_list = []
                est_classes = None
                for x in X:
                    x_dict = self.transform_data_to_dict(x)
                    P_i_dict = self.estimator_.predict_proba_one(
                        x_dict, **predict_proba_kwargs
                    )
                    P_i = []
                    if est_classes is None:
                        est_classes = np.sort(list(P_i_dict.keys()))
                    for c in self.classes_:
                        P_i.append(P_i_dict.get(c, 0.0))
                    P_list.append(P_i)
                P = np.array(P_list)
                if not np.any(np.isnan(P)):
                    return P

            warnings.warn(
                f"Since the 'base_estimator' could not be fitted when"
                f" calling the `fit` method, the class label "
                f"distribution`_label_counts={self._label_counts}` is used to "
                f"make the predictions."
            )
            # fallback if clf could not be fitted (i.e., no labeled data)
            if sum(self._label_counts) == 0:
                n_classes = len(self.classes_)
                return np.ones([len(X), n_classes]) / n_classes
            else:
                return np.tile(
                    self._label_counts / np.sum(self._label_counts),
                    [len(X), 1],
                )

        def _fit(self, fit_function, X, y, sample_weight=None):
            target_spec = self._resolve_target_spec_for_fit(
                y, is_incremental=fit_function == "partial_fit"
            )

            # Check input parameters.
            self.check_X_dict_ = {
                "ensure_min_samples": 0,
                "ensure_min_features": 0,
                "allow_nd": True,
                "dtype": None,
            }
            X, y, sample_weight = self._validate_data(
                X=X,
                y=y,
                sample_weight=sample_weight,
                check_X_dict=self.check_X_dict_,
                reset=fit_function == "fit"
                or not hasattr(self, "n_features_in_"),
                target_spec=target_spec,
            )

            # Check whether estimator is a valid classifier.
            if not isinstance(self.estimator, river.base.Classifier):
                raise TypeError(
                    "'{}' must be a river classifier.".format(self.estimator)
                )

            if hasattr(self, "estimator_"):
                if fit_function != "partial_fit":
                    self.estimator_ = deepcopy(self.estimator)
            else:
                self.estimator_ = deepcopy(self.estimator)
            # count labels per class
            is_included = is_labeled(y, missing_label=-1)
            self._label_counts = [
                np.sum(y[is_included] == c)
                for c in range(len(self._le.classes_))
            ]
            try:
                X_train = X[is_included]
                y_train = y[is_included].astype(np.int64)
                if np.sum(is_included) == 0:
                    raise ValueError("There is no labeled data.")

                supports_learn_many = hasattr(self.estimator_, "learn_many")
                if supports_learn_many:
                    params = signature(self.estimator_.learn_many).parameters
                else:
                    params = signature(self.estimator_.learn_one).parameters
                supports_sample_weight = "w" in params or np.any(
                    [p.kind == p.VAR_KEYWORD for p_name, p in params.items()]
                )

                if not supports_sample_weight and sample_weight is not None:
                    raise ValueError(
                        "The estimator does not support training "
                        "with sample_weight."
                    )

                sample_weight_train = None
                if supports_sample_weight and sample_weight is not None:
                    sample_weight_train = sample_weight[is_included]
                if supports_learn_many:
                    fit_args = {}
                    fit_args["X"] = pd.DataFrame(
                        self.transform_data_to_dict(X_train)
                    )
                    fit_args["y"] = pd.Series(y_train)
                    if sample_weight_train is not None:
                        fit_args["w"] = pd.Series(sample_weight_train)
                    self.estimator_.learn_many(**fit_args)
                else:
                    for idx in range(len(X_train)):
                        fit_args = {}
                        fit_args["x"] = self.transform_data_to_dict(
                            X_train[idx]
                        )
                        fit_args["y"] = y_train[idx]
                        if sample_weight_train is not None:
                            fit_args["w"] = sample_weight_train[idx]
                        self.estimator_.learn_one(**fit_args)
                self.is_fitted_ = True
            except Exception as e:
                if (
                    "supports_sample_weight" in locals()
                    and not supports_sample_weight
                    and sample_weight is not None
                ):
                    raise e
                else:
                    self.is_fitted_ = False
                    warnings.warn(
                        "The 'estimator' could not be fitted because of"
                        " '{}'. Therefore, the number of classes  are counted"
                        "and will be used to make predictions. The class"
                        "labels are assumed to be uniformly "
                        "distributed.".format(e)
                    )
            return self

        def transform_data_to_dict(self, data):
            return {str(i): x for i, x in enumerate(data.T)}

        def __sklearn_is_fitted__(self):
            if hasattr(self, "is_fitted_"):
                return True
