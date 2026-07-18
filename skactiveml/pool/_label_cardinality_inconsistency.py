import numpy as np

from ..base import SingleAnnotatorPoolQueryStrategy, SkactivemlClassifier
from ..utils import (
    ExtLabelEncoder,
    MISSING_LABEL,
    is_unlabeled,
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
            Utilities for labeled samples will be set to np.nan.

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

        # Determine candidate samples that are currently unlabeled.
        if mapping is None:
            cand_mask = np.ones(len(X_cand), dtype=bool)
        else:
            cand_mask = is_unlabeled(
                y[mapping],
                missing_label=self.missing_label_,
                target_type=target_spec.target_type,
            )
        X_unlbld = X_cand[cand_mask]

        lbld_mask = is_labeled(
            y,
            missing_label=self.missing_label_,
            target_type=target_spec.target_type,
        )
        n_lbld = int(lbld_mask.sum())

        label_encoder = ExtLabelEncoder(
            classes=target_spec.classes,
            missing_label=self.missing_label_,
            target_type=target_spec.target_type,
        ).fit(y[lbld_mask])
        y_label_cardinality = 0
        if n_lbld != 0:
            y_label_cardinality = (
                label_encoder.transform(y[lbld_mask]).sum() / n_lbld
            )

        Y_pred = clf.predict(X_unlbld)
        pred_mean_cardinality = label_encoder.transform(Y_pred).sum(axis=-1)

        utilities_cand = np.abs(pred_mean_cardinality - y_label_cardinality)

        if mapping is None:
            utilities = utilities_cand
        else:
            utilities = np.full(len(X), np.nan)
            utilities[mapping[cand_mask]] = utilities_cand

        return simple_batch(
            utilities,
            self.random_state_,
            batch_size=batch_size,
            return_utilities=return_utilities,
        )
