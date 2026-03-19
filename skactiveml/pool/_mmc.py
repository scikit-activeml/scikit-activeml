"""
Module implementing the multilabel pool-based query strategy MMC.
"""

import numpy as np
from sklearn import clone

from ..base import SingleAnnotatorPoolQueryStrategy, SkactivemlClassifier
from ..utils import (
    MISSING_LABEL,
    is_unlabeled,
    simple_batch,
    check_type,
    check_equal_missing_label,
    check_scalar,
)


class MaxLossReductionMaxConfidence(SingleAnnotatorPoolQueryStrategy):
    """Maximum Loss Reduction with Maximal Confidence (MMC)

    This class implements the query strategy Maximum Loss Reduction with
    Maximal Confidence (MMC) [1]_ that selects samples based on a combination
    of a multilabel classifier and a label-cardinality discriminator. This
    strategy is multilabel-only: `y` must be two-dimensional and each row must
    be either fully labeled or fully unlabeled.

    Parameters
    ----------
    missing_label : scalar or string or np.nan or None, default=np.nan
        Value to represent a missing label.
    random_state : int or np.random.RandomState, default=None
        Random state for candidate selection.

    References
    ----------
    .. [1] Li, X., & Guo, Y. (2013). Active Learning with Multi-Label
       SVM Classification. In IjCAI (Vol. 13, pp. 1479-1485).
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
        discriminator,
        clf,
        fit_clf=True,
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
        discriminator : skactiveml.base.SkactivemlClassifier
            Model implementing the methods `fit` and `predict_proba`.
            The parameters `classes` and `missing_label` will be internally
            redefined.
        clf : skactiveml.base.SkactivemlClassifier
            Classifier implementing the methods `fit` and `predict_proba`.
        fit_clf : bool, default=True
            Defines whether the classifier `clf` should be fitted on `X`
            and `y`.
        candidates : None or array-like of shape (n_candidates), dtype=int or \
                array-like of shape (n_candidates, n_features), default=None
            - If `candidates` is `None`, the unlabeled samples from `(X, y)`
              are considered as candidates.
            - If `candidates` is of shape `(n_candidates,)` and of type
              `int`, `candidates` is considered as the indices of the samples
              in `(X, y)`.
            - If `candidates` is of shape `(n_candidates, n_features)`, the
              candidate samples are directly given in `candidates` (not
              necessarily contained in `X`).
        batch_size : int, default=1
            The number of samples to be selected in one AL cycle.
        return_utilities : bool, default=False
            If `True`, also return the utilities based on the query strategy.

        Returns
        -------
        query_indices : numpy.ndarray of shape (batch_size,)
            The `query_indices` indicate for which candidate sample a label is
            to be queried, e.g., `query_indices[0]` indicates the index of
            the first selected sample.
            If `candidates` is `None` or of shape `(n_candidates,)`, the
            indexing refers to samples in `X`.
            If `candidates` is of shape `(n_candidates, n_features)`, the
            indexing refers to samples in `candidates`.
        utilities : numpy.ndarray of shape (batch_size, n_samples) or
                numpy.ndarray of shape (batch_size, n_candidates)
            The utilities of samples after each selected sample of the batch,
            e.g., `utilities[0]` indicates the utilities used for selecting
            the first sample (with index `query_indices[0]`) of the batch.
            Utilities for labeled samples will be set to np.nan.
            If `candidates` is `None` or of shape `(n_candidates,)`, the
            indexing refers to samples in `X`.
            If `candidates` is of shape `(n_candidates, n_features)`, the
            indexing refers to samples in `candidates`.
        """
        # Validate parameters.
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
                "`MaxLossReductionMaxConfidence` strategy is multi-label "
                "only."
            )
        X_cand, mapping = self._transform_candidates(
            candidates, X, y, is_multioutput=is_multioutput
        )

        check_type(discriminator, "discriminator", SkactivemlClassifier)
        check_type(clf, "clf", SkactivemlClassifier)
        check_equal_missing_label(clf.missing_label, self.missing_label_)
        check_scalar(fit_clf, "fit_clf", bool)

        discriminator = clone(discriminator)
        discriminator.classes = list(range(y.shape[1] + 1))
        discriminator.missing_label = -1

        if fit_clf:
            clf = clone(clf).fit(X, y)

        # Determine unlabeled vs. labeled samples.
        lbld_mask = ~is_unlabeled(
            y, missing_label=self.missing_label_, is_multioutput=True
        )
        if mapping is None:
            cand_mask = np.ones(len(X_cand), dtype=bool)
        else:
            cand_mask = is_unlabeled(
                y[mapping],
                missing_label=self.missing_label_,
                is_multioutput=True,
            )
        X_unlbld = X_cand[cand_mask]

        probas = clf.predict_proba(X)
        lbld_probas = probas[lbld_mask]
        unlbld_probas = clf.predict_proba(X_unlbld)
        f = unlbld_probas * 2 - 1

        lbld_probas = np.flip(np.sort(lbld_probas, axis=1), axis=-1)
        lbld_probas /= lbld_probas.sum(axis=1, keepdims=True)

        unlbld_probas_idx = np.flip(np.argsort(unlbld_probas, axis=1), axis=-1)
        unlbld_probas = np.flip(np.sort(unlbld_probas, axis=1), axis=-1)
        unlbld_probas /= unlbld_probas.sum(axis=1, keepdims=True)

        y_discriminator = y[lbld_mask].sum(axis=1).astype(int)
        discriminator.fit(lbld_probas, y_discriminator)

        unlbld_pred = discriminator.predict(unlbld_probas)

        yhat = -1 * np.ones((len(unlbld_pred), y.shape[1]), dtype=int)
        for i, p in enumerate(unlbld_pred):
            yhat[i, unlbld_probas_idx[i, :p]] = 1

        utilities_cand = ((1 - yhat * f) / 2).sum(axis=1)

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
