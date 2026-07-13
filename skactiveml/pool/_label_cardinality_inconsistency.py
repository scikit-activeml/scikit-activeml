import numpy as np
from sklearn import clone

from ..base import SingleAnnotatorPoolQueryStrategy, SkactivemlClassifier
from ..utils import (
    MISSING_LABEL,
    check_type,
    check_equal_missing_label,
    check_scalar,
    is_unlabeled,
    is_labeled,
    simple_batch,
)


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
    ):
        super().__init__(
            missing_label=missing_label, random_state=random_state
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
        # Validate input parameters
        X, y, candidates, batch_size, return_utilities = self._validate_data(
            X,
            y,
            candidates,
            batch_size,
            return_utilities,
            reset=True,
            allow_multioutput=True,
        )

        is_multioutput = y.ndim == 2
        if not is_multioutput:
            raise ValueError(
                "`y` must be in multi-label format, as the "
                "`LabelCardinalityInconsistency` strategy is multi-label only."
            )
        X_cand, mapping = self._transform_candidates(
            candidates, X, y, is_multioutput=is_multioutput
        )

        # Validate classifier type
        check_type(clf, "clf", SkactivemlClassifier)
        check_equal_missing_label(clf.missing_label, self.missing_label_)
        check_scalar(fit_clf, "fit_clf", bool)

        # Fit the classifier
        if fit_clf:
            if sample_weight is None:
                clf = clone(clf).fit(X, y)
            else:
                clf = clone(clf).fit(X, y, sample_weight=sample_weight)

        # Determine candidate samples that are currently unlabeled.
        if mapping is None:
            cand_mask = np.ones(len(X_cand), dtype=bool)
        else:
            cand_mask = is_unlabeled(
                y[mapping],
                missing_label=self.missing_label_,
                target_type="multi-label",
            )
        X_unlbld = X_cand[cand_mask]

        n_lbld = int(
            is_labeled(
                y,
                missing_label=self.missing_label_,
                target_type="multi-label",
            ).sum()
        )

        y_label_cardinality = 0
        if n_lbld != 0:
            y_label_cardinality = np.nansum(y) / n_lbld

        Y_pred = clf.predict(X_unlbld)
        pred_mean_cardinality = Y_pred.sum(axis=-1)

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
