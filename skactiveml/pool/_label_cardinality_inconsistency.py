import numpy as np

from ..base import SingleAnnotatorPoolQueryStrategy, SkactivemlClassifier
from ..utils import (
    ExtLabelEncoder,
    MISSING_LABEL,
    is_labeled,
    simple_batch,
)
from ._target import _fit_and_resolve_estimator_target_spec


class LabelCardinalityInconsistency(SingleAnnotatorPoolQueryStrategy):
    """Label Cardinality Inconsistency (LCI)

    This class implements the query strategy Label Cardinality Inconsistency
    (LCI) [1]_ that selects samples based on the difference in label
    cardinality between the labeled pool and the predicted number of positive
    labels in the unlabeled pool. This strategy is multilabel-only: `y` must
    be two-dimensional and each row must be either fully labeled or fully
    unlabeled.

    Parameters
    ----------
    missing_label : scalar or string or np.nan or None, default=np.nan
        Value to represent a missing label.
    random_state : int or RandomState instance or None, default=None
        Controls the randomness of the estimator.
    target_type : "auto" or "multi-label", default="auto"
        Declared target type. A fitted classifier's target specification is
        authoritative when available. This strategy supports only multi-label
        classification with a single annotator.

    References
    ----------
    .. [1] R. Wang and S. Ye (2019). Multi-Label Active Learning Driven by
       Uncertainty and Inconsistency. In 2019 International Conference on
       Machine Learning and Cybernetics.
    """

    def __init__(
        self,
        missing_label=MISSING_LABEL,
        random_state=None,
        target_type="auto",
    ):
        super().__init__(
            missing_label=missing_label, random_state=random_state
        )
        self.target_type = target_type

    @property
    def _target_capabilities(self):
        return frozenset(
            {("classification", "multi-label", "single-annotator")}
        )

    def query(
        self,
        X,
        y,
        clf,
        fit_clf=True,
        sample_weight=None,
        candidates=None,
        batch_size=1,
        return_utilities=False,
    ):
        """Determines for which candidate samples labels are to be queried.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data set, usually complete, i.e., including the labeled
            and unlabeled samples.
        y : array-like of shape (n_samples, n_outputs)
            Labels of the training data set (possibly including unlabeled
            rows indicated by `self.missing_label`). Each row must either
            contain only observed labels or only `missing_label` values, i.e.,
            no mixing within a row. This strategy supports multilabel data
            only.
        clf : skactiveml.base.SkactivemlClassifier
            Classifier implementing the methods `fit` and `predict`.
        fit_clf : bool, default=True
            Defines whether the classifier `clf` should be fitted on `X`, `y`,
            and `sample_weight`.
        sample_weight : array-like of shape (n_samples,) or \
                (n_samples, n_outputs), default=None
            Weights of training samples in `X`. One weight per sample is
            supported. Per-target weights are forwarded to `clf.fit` without
            additional validation and require estimator support.
        candidates : None or array-like of shape (n_candidates), dtype=int or \
                array-like of shape (n_candidates, n_features), default=None
            - If `candidates` is `None`, the unlabeled samples from `(X, y)`
              are considered as candidates.
            - If `candidates` is of shape `(n_candidates,)` and of type
              `int`, `candidates` is considered as the indices of samples in
              `(X, y)`.
            - If `candidates` is of shape `(n_candidates, n_features)`,
              the candidates are directly given in `candidates`.

            A given `candidates` is authoritative, i.e., an index array is
            taken as given, such that labeled samples remain candidates, e.g.,
            to relabel them or to recompute their utilities.
        batch_size : int, default=1
            The number of samples to be selected in one AL cycle.
        return_utilities : bool, default=False
            If `True`, also return the utilities based on the query strategy.

        Returns
        -------
        query_indices : numpy.ndarray of shape (batch_size,)
            The query indices indicate for which candidate sample a label is
            to be queried, e.g., `query_indices[0]` indicates the first
            selected sample.

            - If `candidates` is `None` or of shape `(n_candidates,)`, the
              indexing refers to the samples in `X`.
            - If `candidates` is of shape `(n_candidates, n_features)`, the
              indexing refers to the samples in `candidates`.
        utilities : numpy.ndarray of shape (batch_size, n_samples) or \
                numpy.ndarray of shape (batch_size, n_candidates)
            The utilities of samples after each selected sample of the batch,
            e.g., `utilities[0]` indicates the utilities used for selecting
            the first sample (with index `query_indices[0]`) of the batch.
            Utilities for samples that are no candidates will be set to
            np.nan.

            - If `candidates` is `None` or of shape `(n_candidates,)`, the
              indexing refers to the samples in `X`.
            - If `candidates` is of shape `(n_candidates, n_features)`, the
              indexing refers to the samples in `candidates`.
        """
        # Resolve through the classifier before acquisition state is changed.
        clf, target_spec = _fit_and_resolve_estimator_target_spec(
            self,
            clf,
            X,
            y,
            fit_estimator=fit_clf,
            sample_weight=sample_weight,
            estimator_name="clf",
            fit_name="fit_clf",
            estimator_types=(SkactivemlClassifier,),
        )

        # Validate input parameters
        X, y, candidates, batch_size, return_utilities = self._validate_data(
            X,
            y,
            candidates,
            batch_size,
            return_utilities,
            reset=True,
            target_type=target_spec.target_type,
        )

        X_cand, mapping = self._transform_candidates(
            candidates, X, y, target_type=target_spec.target_type
        )

        lbld_mask = is_labeled(
            y,
            missing_label=self.missing_label_,
            target_type=target_spec.target_type,
        )

        # Encode targets and predictions so that the acquisition function
        # never performs arithmetic on raw class values.
        label_encoder = ExtLabelEncoder(
            classes=target_spec.classes,
            missing_label=self.missing_label_,
            target_type=target_spec.target_type,
        ).fit(y[lbld_mask])
        y_labeled = label_encoder.transform(y[lbld_mask])
        y_pred = label_encoder.transform(clf.predict(X_cand))

        utilities_cand = label_cardinality_inconsistency(y_pred, y_labeled)

        if mapping is None:
            utilities = utilities_cand
        else:
            utilities = np.full(len(X), np.nan)
            utilities[mapping] = utilities_cand

        return simple_batch(
            utilities,
            self.random_state_,
            batch_size=batch_size,
            return_utilities=return_utilities,
        )


def label_cardinality_inconsistency(y_pred, y_labeled):
    """Calculate the label cardinality inconsistency.

    The label cardinality of a sample is its number of positive labels. This
    acquisition function scores each candidate sample by the absolute
    difference between its predicted label cardinality and the mean label
    cardinality of the labeled samples [1]_. An empty labeled pool is treated
    as having a label cardinality of zero.

    Both targets must be encoded, i.e., `0` for the negative and `1` for the
    positive class of each output, so that the acquisition function performs
    no arithmetic on raw class values. Use
    `skactiveml.utils.ExtLabelEncoder` with `target_type="multi-label"` to
    encode raw class vocabularies.

    Parameters
    ----------
    y_pred : array-like of shape (n_candidates, n_outputs)
        Encoded predicted labels of the candidate samples.
    y_labeled : array-like of shape (n_labeled, n_outputs)
        Encoded observed labels of the labeled samples. May be empty, i.e.,
        of shape `(0, n_outputs)`.

    Returns
    -------
    utilities : numpy.ndarray of shape (n_candidates,)
        Absolute difference between each candidate's predicted label
        cardinality and the mean label cardinality of the labeled samples,
        i.e., one finite value in `[0, n_outputs]` per candidate. Larger
        values indicate more useful candidates.

    Raises
    ------
    ValueError
        If `y_pred` or `y_labeled` is not a two-dimensional array with
        `n_outputs` columns, or if either contains values other than `0` and
        `1`, e.g., unlabeled rows or raw class values.

    References
    ----------
    .. [1] R. Wang and S. Ye (2019). Multi-Label Active Learning Driven by
       Uncertainty and Inconsistency. In 2019 International Conference on
       Machine Learning and Cybernetics.
    """
    y_pred = _check_encoded_multilabel_targets(y_pred, "y_pred")
    y_labeled = _check_encoded_multilabel_targets(
        y_labeled, "y_labeled", n_outputs=y_pred.shape[1]
    )

    label_cardinality = 0.0
    if len(y_labeled) > 0:
        label_cardinality = y_labeled.sum() / len(y_labeled)

    return np.abs(y_pred.sum(axis=1) - label_cardinality)


def _check_encoded_multilabel_targets(y, name, n_outputs=None):
    """Check that `y` is a matrix of encoded multi-label targets.

    Parameters
    ----------
    y : array-like of shape (n_samples, n_outputs)
        Encoded multi-label targets, i.e., one `0` or `1` per output.
    name : str
        Name of `y` used in error messages.
    n_outputs : int or None, default=None
        Expected number of outputs. If not `None`, `y` must have this many
        columns.

    Returns
    -------
    y : numpy.ndarray of shape (n_samples, n_outputs)
        Encoded multi-label targets as an integer array.

    Raises
    ------
    ValueError
        If `y` is not a two-dimensional array of the expected width, or if it
        contains values other than `0` and `1`.
    """
    y = np.asarray(y)
    if y.ndim != 2:
        raise ValueError(
            f"`{name}` must have shape `(n_samples, n_outputs)`, got "
            f"{y.shape}."
        )
    if n_outputs is not None and y.shape[1] != n_outputs:
        raise ValueError(
            f"`{name}` has {y.shape[1]} outputs, expected {n_outputs}."
        )
    if not np.isin(y, [0, 1]).all():
        raise ValueError(
            f"`{name}` must contain encoded labels, i.e., `0` or `1` per "
            "output. Unlabeled or raw class values are not supported."
        )
    return y.astype(int)
