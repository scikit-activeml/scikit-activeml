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


class DiscriminativeAL(SingleAnnotatorPoolQueryStrategy):
    """Discriminative Active Learning (DAL)

    This class implements the "Discriminative Active Learning" (DAL) [1]_
    strategy. Its idea is to solve a binary classification task to choose
    samples for labeling such that the labeled set and the unlabeled pool are
    indistinguishable.

    Parameters
    ----------
    greedy_selection : bool, default=False
        This parameter is only relevant for `batch_size>1`.

        - If `greedy_selection=False`, the classifying discriminator is
          refitted after each sample selection within a batch.
        - If `greedy_selection=True`, the discriminator is kept fixed.
    missing_label : scalar or string or np.nan or None, default=np.nan
        Value to represent a missing label.
    random_state : None or int or np.random.RandomState, default=None
        The random state to use.

    References
    ----------
    .. [1] D. Gissin and S. Shalev-Shwartz. Discriminative Active Learning.
       arXiv:1907.06347, 2019.
    """

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
        y : array-like of shape (n_samples,)
            Labels of the training data set (possibly including unlabeled ones
            indicated by `self.missing_label`).
        discriminator : skactiveml.base.SkactivemlClassifier
            Classification model implementing the methods `fit` and
            `predict_proba`. It will be used to solve the binary classification
            problem of separating labeled and unlabeled samples. The parameters
            `classes` and `missing_label` will be internally redefined.
        candidates : None or array-like of shape (n_candidates), dtype=int or \
                array-like of shape (n_candidates, n_features), default=None
            - If `candidates` is `None`, the unlabeled samples from
              `(X,y)` are considered as `candidates`.
            - If `candidates` is of shape `(n_candidates,)` and of type
              `int`, `candidates` is considered as the indices of the
              samples in `(X,y)`.
            - If `candidates` is of shape `(n_candidates, *)`, the
              candidate samples are directly given in `candidates` (not
              necessarily contained in `X`).
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

            - If `candidates` is `None` or of shape
              `(n_candidates,)`, the indexing refers to the samples in
              `X`.
            - If `candidates` is of shape `(n_candidates, n_features)`,
              the indexing refers to the samples in `candidates`.
        utilities : numpy.ndarray of shape (batch_size, n_samples) or \
                numpy.ndarray of shape (batch_size, n_candidates)
            The utilities of samples after each selected sample of the batch,
            e.g., `utilities[0]` indicates the utilities used for selecting
            the first sample (with index `query_indices[0]`) of the batch.
            Utilities for labeled samples will be set to np.nan.

            - If `candidates` is `None` or of shape
              `(n_candidates,)`, the indexing refers to the samples in
              `X`.
            - If `candidates` is of shape `(n_candidates, n_features)`,
              the indexing refers to the samples in `candidates`.
        """
        # Validate parameters.
        X, y, candidates, batch_size, return_utilities = self._validate_data(
            X, y, candidates, batch_size, return_utilities, reset=True
        )
        check_type(discriminator, "discriminator", SkactivemlClassifier)
        check_type(self.greedy_selection, "greedy_selection", bool)

        # Retransform candidates and create a potential mapping to the samples
        # in `X`.
        X_cand, mapping = self._transform_candidates(
            candidates, X, y, enforce_mapping=True
        )

        # Re-define discriminator to fit the setting of classifying
        # labeled (y=0) and unlabeled samples (y=1).
        discriminator = clone(discriminator)
        discriminator.classes = [0, 1]
        discriminator.missing_label = -1

        if self.greedy_selection:
            # Return the top samples with the highest probabilities of
            # being unlabeled, which correspond to their utilities.
            y_discriminator = is_unlabeled(y, missing_label=self.missing_label)
            y_discriminator = y_discriminator.astype(int)
            discriminator.fit(X, y_discriminator)
            utilities_cand = discriminator.predict_proba(X_cand)[:, 1]

            # Remapping of `utilities` and `query_indices` if required.
            utilities = np.full(len(X), np.nan)
            utilities[mapping] = utilities_cand

            # Return `query_indices` and potential `utilities`.
            return simple_batch(
                utilities,
                self.random_state_,
                batch_size=batch_size,
                return_utilities=return_utilities,
            )
        else:
            # Refit the binary classifier, i.e., the discriminator, after each
            # selected sample in a batch.
            X_discriminator = X
            query_indices_cand = []
            utilities_cand = np.empty((batch_size, len(X_cand)), dtype=float)
            for i in range(batch_size):
                # Determine unlabeled vs. labeled samples.
                y_discriminator = is_unlabeled(
                    y, missing_label=self.missing_label
                )
                y_discriminator = y_discriminator.astype(int)

                # Mark already selected samples as labeled.
                y_discriminator[mapping[query_indices_cand]] = 0

                # Fit discriminator to classify unlabeled vs. labeled samples.
                discriminator.fit(X_discriminator, y_discriminator)

                # Compute utilities as probabilities of being unlabeled.
                utilities_cand[i] = discriminator.predict_proba(X_cand)[:, 1]
                utilities_cand[i, query_indices_cand] = np.nan
                query_indices_cand.append(
                    rand_argmax(utilities_cand[i], self.random_state_)[0]
                )

            # Remapping of `utilities` and `query_indices`
            utilities = np.full((batch_size, len(X)), np.nan)
            utilities[:, mapping] = utilities_cand
            query_indices = mapping[query_indices_cand]

            # Check whether `utilities` are to be returned.
            if return_utilities:
                return query_indices, utilities
            else:
                return query_indices
