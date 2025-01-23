"""
Module implementing discriminative active learning.
"""

# Authors: Marek Herde <marek.herde@uni-kassel.de>

import numpy as np
from sklearn import clone

from ..base import SingleAnnotatorPoolQueryStrategy, SkactivemlClassifier
from ..utils import (
    MISSING_LABEL,
    rand_argmax,
    is_unlabeled,
    simple_batch,
    check_type,
)


class MMC(SingleAnnotatorPoolQueryStrategy):
    def __init__(
        self,
        greedy_selection=False,
        missing_label=MISSING_LABEL,
        random_state=None,
    ):
        super().__init__(
            missing_label=missing_label, random_state=random_state
        )
        self.greedy_selection = greedy_selection

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
        y : array-like of shape (n_samples)
            Labels of the training data set (possibly including unlabeled ones
            indicated by `self.missing_label`).
        discriminator : skactiveml.base.SkactivemlClassifier
            Model implementing the methods `fit` and `predict_proba`.
            The parameters `classes` and `missing_label` will be internally
            redefined.
        candidates : None or array-like of shape (n_candidates), dtype=int or
        array-like of shape (n_candidates, n_features), optional (default=None)
            If `candidates` is `None`, the unlabeled samples from `(X, y)` are
            considered as candidates.
            If `candidates` is of shape `(n_candidates,)` and of type int,
            `candidates` is considered as the indices of the samples in
            `(X, y)`.
            If `candidates` is of shape `(n_candidates, n_features)`, the
            candidates are directly given in candidates (not necessarily
            contained in `X`).
        batch_size : int, optional (default=1)
            The number of samples to be selected in one AL cycle.
        return_utilities : bool, optional (default=False)
            If true, also return the utilities based on the query strategy.

        Returns
        -------
        query_indices : numpy.ndarray of shape (batch_size,)
            The `query_indices` indicate for which candidate sample a label is
            to be queried, e.g., `query_indices[0]` indicates the index of
            the first selected sample.
            If `candidates` is `None` or of shape `(n_candidates,)`, the
            indexing refers to samples in `X`.
            If `candidates` is of shape (n_candidates, n_features), the
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

        is_multilabel = np.array(y).ndim == 2  # here changes

        # Validate parameters.
        X, y, candidates, batch_size, return_utilities = self._validate_data(
            X, y, candidates, batch_size, return_utilities, reset=True, is_multilabel=is_multilabel
        )

        X_cand, mapping = self._transform_candidates(candidates, X, y, is_multilabel=is_multilabel,)

        check_type(discriminator, "discriminator", SkactivemlClassifier)
        check_type(self.greedy_selection, "greedy_selection", bool)


        discriminator = clone(discriminator)
        discriminator.classes = list(range(y.shape[1] + 1))
        discriminator.missing_label = -1

        probas = clf.predict_proba(X)
        probas_sorted = np.flip(np.sort(probas, axis=1))
        # Determine unlabeled vs. labeled samples.
        unlbld_mask = is_unlabeled(y, missing_label=self.missing_label)
        lbld_mask = np.all(~unlbld_mask, axis=1)
        unlbld_mask = np.all(unlbld_mask, axis=1)

        if y[lbld_mask].shape[0] == 0:
            print("no labels, fallback case?") # TODO

        X_discriminator = probas_sorted[lbld_mask]
        y_discriminator = y[lbld_mask].sum(axis=1)
        discriminator.fit(X_discriminator, y_discriminator)

        X_discriminator_pred = probas_sorted[unlbld_mask]
        pred_n_lbl = discriminator.predict(X_discriminator_pred)

        yhat = np.full(probas_sorted[unlbld_mask].shape, -1)
        idx_sorted_unlbld = np.flip(np.argsort(probas[unlbld_mask], axis=1))
        for j, p in enumerate(pred_n_lbl):
            yhat[j, idx_sorted_unlbld[j, :p]] = 1

        utilities_cand = ((1 - yhat * (probas[unlbld_mask])) / 2).sum(axis=1)

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


