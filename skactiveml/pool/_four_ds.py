"""
Module implementing 4DS active learning strategy.
"""
# Author: Marek Herde <marek.herde@uni-kassel.de>


import numpy as np
from sklearn.base import clone
from sklearn.utils.validation import check_scalar

from ..base import SingleAnnotatorPoolQueryStrategy
from ..classifier import MixtureModelClassifier
from ..utils import (
    rand_argmax,
    is_labeled,
    check_type,
    MISSING_LABEL,
    check_equal_missing_label,
)


class FourDs(SingleAnnotatorPoolQueryStrategy):
    """FourDs

    Implementation of the pool-based query strategy 4DS for training a
    MixtureModelClassifier [1].

    Parameters
    ----------
    lmbda : float between 0 and 1, optional
    (default=min((batch_size-1)*0.05, 0.5))
        For the selection of more than one sample within each query round, 4DS
        uses a diversity measure to avoid the selection of redundant samples
        whose influence is regulated by the weighting factor 'lmbda'.
    missing_label : scalar or string or np.nan or None, optional
    (default=MISSING_LABEL)
        Value to represent a missing label.
    random_state : numeric or np.random.RandomState, optional (default=None)
        The random state to use.

    References
    ---------
    [1] Reitmaier, T., & Sick, B. (2013). Let us know your decision: Pool-based
        active training of a generative classifier with the selection strategy
        4DS. Information Sciences, 230, 106-131.
    """

    def __init__(
            self, lmbda=None, missing_label=MISSING_LABEL, random_state=None
    ):
        super().__init__(
            missing_label=missing_label, random_state=random_state
        )
        self.lmbda = lmbda

    def query(
            self,
            X,
            y,
            clf,
            fit_clf=True,
            sample_weight=None,
            candidates=None,
            return_utilities=False,
            batch_size=1,
    ):
        """Determines for which candidate samples labels are to be queried.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            Training data set, usually complete, i.e. including the labeled and
            unlabeled samples.
        y: array-like of shape (n_samples)
            Labels of the training data set (possibly including unlabeled ones
            indicated by self.MISSING_LABEL.
        clf : skactiveml.classifier.MixtureModelClassifier
            GMM-based classifier to be trained.
        fit_clf : bool, optional (default=True)
            Defines whether the classifier should be fitted on `X`, `y`, and
            `sample_weight`.
        sample_weight: array-like of shape (n_samples), optional (default=None)
            Weights of training samples in `X`.
        candidates : None or array-like of shape (n_candidates), dtype=int or
            array-like of shape (n_candidates, n_features),
            optional (default=None)
            If candidates is None, the unlabeled samples from (X,y) are
            considered as candidates.
            If candidates is of shape (n_candidates) and of type int,
            candidates is considered as the indices of the samples in (X,y).
            If candidates is of shape (n_candidates, n_features), the
            candidates are directly given in candidates (not necessarily
            contained in X). This is not supported by all query strategies.
        batch_size : int, optional (default=1)
            The number of samples to be selected in one AL cycle.
        return_utilities : bool, optional (default=False)
            If True, also return the utilities based on the query strategy.

        Returns
        -------
        query_indices : numpy.ndarray of shape (batch_size)
            The query_indices indicate for which candidate sample a label is
            to queried, e.g., `query_indices[0]` indicates the first selected
            sample.
            If candidates is None or of shape (n_candidates), the indexing
            refers to samples in X.
            If candidates is of shape (n_candidates, n_features), the indexing
            refers to samples in candidates.
        utilities : numpy.ndarray of shape (batch_size, n_samples) or
            numpy.ndarray of shape (batch_size, n_candidates)
            The utilities of samples after each selected sample of the batch,
            e.g., `utilities[0]` indicates the utilities used for selecting
            the first sample (with index `query_indices[0]`) of the batch.
            Utilities for labeled samples will be set to np.nan.
            If candidates is None or of shape (n_candidates), the indexing
            refers to samples in X.
            If candidates is of shape (n_candidates, n_features), the indexing
            refers to samples in candidates.
        """
        # Check standard parameters.
        (
            X,
            y,
            candidates,
            batch_size,
            return_utilities,
        ) = super()._validate_data(
            X=X,
            y=y,
            candidates=candidates,
            batch_size=batch_size,
            return_utilities=return_utilities,
            reset=True,
        )

        # Check classifier type.
        check_type(clf, "clf", MixtureModelClassifier)
        check_type(fit_clf, "fit_clf", bool)
        check_equal_missing_label(clf.missing_label, self.missing_label_)

        # Check lmbda.
        lmbda = self.lmbda
        if lmbda is None:
            lmbda = np.min(((batch_size - 1) * 0.05, 0.5))
        check_scalar(
            lmbda, target_type=float, name="lmbda", min_val=0, max_val=1
        )

        # Obtain candidates plus mapping.
        X_cand, mapping = self._transform_candidates(candidates, X, y)

        # Storage for query indices.
        query_indices_cand = np.full(batch_size, fill_value=-1, dtype=int)

        # Fit the classifier and get the probabilities.
        if fit_clf:
            clf = clone(clf).fit(X, y, sample_weight)
        P_cand = clf.predict_proba(X_cand)
        R_cand = clf.mixture_model_.predict_proba(X_cand)
        is_lbld = is_labeled(y, missing_label=clf.missing_label)
        if np.sum(is_lbld) >= 1:
            R_lbld = clf.mixture_model_.predict_proba(X[is_lbld])
        else:
            R_lbld = np.array([0])

        # Compute distance according to Eq. 9 in [1].
        P_cand_sorted = np.sort(P_cand, axis=1)
        distance_cand = np.log(
            (P_cand_sorted[:, -1] + 1.0e-5) / (P_cand_sorted[:, -2] + 1.0e-5)
        )
        distance_cand = (distance_cand - np.min(distance_cand) + 1.0e-5) / (
                np.max(distance_cand) - np.min(distance_cand) + 1.0e-5
        )

        # Compute densities according to Eq. 10 in [1].
        density_cand = clf.mixture_model_.score_samples(X_cand)
        density_cand = (density_cand - np.min(density_cand) + 1.0e-5) / (
                np.max(density_cand) - np.min(density_cand) + 1.0e-5
        )

        # Compute distributions according to Eq. 11 in [1].
        R_lbld_sum = np.sum(R_lbld, axis=0, keepdims=True)
        R_sum = R_cand + R_lbld_sum
        R_mean = R_sum / (len(R_lbld) + 1)
        distribution_cand = clf.mixture_model_.weights_ - R_mean
        distribution_cand = np.maximum(
            np.zeros_like(distribution_cand), distribution_cand
        )
        distribution_cand = 1 - np.sum(distribution_cand, axis=1)

        # Compute rho according to Eq. 15  in [1].
        diff = np.sum(
            np.abs(clf.mixture_model_.weights_ - np.mean(R_lbld, axis=0))
        )
        rho = min(1, diff)

        # Compute e_dwus according to Eq. 13  in [1].
        e_dwus = np.mean((1 - P_cand_sorted[:, -1]) * density_cand)

        # Normalization such that alpha, beta, and rho sum up to one.
        alpha = (1 - rho) * e_dwus
        beta = 1 - rho - alpha

        # Compute utilities to select sample.
        utilities_cand = np.empty((batch_size, len(X_cand)), dtype=float)
        utilities_cand[0] = (
                alpha * (1 - distance_cand)
                + beta * density_cand
                + rho * distribution_cand
        )
        query_indices_cand[0] = rand_argmax(
            utilities_cand[0], self.random_state_
        )
        is_selected = np.zeros(len(X_cand), dtype=bool)
        is_selected[query_indices_cand[0]] = True

        if batch_size > 1:
            # Compute e_us according to Eq. 14  in [1].
            e_us = np.mean(1 - P_cand_sorted[:, -1])

            # Normalization of the coefficients alpha, beta, and rho such
            # that these coefficients plus
            # lmbda sum up to one.
            rho = min(rho, 1 - lmbda)
            alpha = (1 - (rho + lmbda)) * (1 - e_us)
            beta = 1 - (rho + lmbda) - alpha

            for i in range(1, batch_size):
                # Update distributions according to Eq. 11 in [1].
                R_sum = (
                        R_cand
                        + np.sum(R_cand[is_selected], axis=0, keepdims=True)
                        + R_lbld_sum
                )
                R_mean = R_sum / (len(R_lbld) + len(query_indices_cand) + 1)
                distribution_cand = clf.mixture_model_.weights_ - R_mean
                distribution_cand = np.maximum(
                    np.zeros_like(distribution_cand), distribution_cand
                )
                distribution_cand = 1 - np.sum(distribution_cand, axis=1)

                # Compute diversity according to Eq. 12 in [1].
                diversity_cand = -np.log(
                    density_cand + np.sum(density_cand[is_selected])
                ) / (len(query_indices_cand) + 1)
                diversity_cand = (diversity_cand - np.min(diversity_cand)) / (
                        np.max(diversity_cand) - np.min(diversity_cand)
                )

                # Compute utilities to select sample.
                utilities_cand[i] = (
                        alpha * (1 - distance_cand)
                        + beta * density_cand
                        + lmbda * diversity_cand
                        + rho * distribution_cand
                )
                utilities_cand[i, is_selected] = np.nan
                query_indices_cand[i] = rand_argmax(
                    utilities_cand[i], self.random_state_
                )
                is_selected[query_indices_cand[i]] = True

        # Remapping of utilities and query indices if required.
        if mapping is None:
            utilities = utilities_cand
            query_indices = query_indices_cand
        if mapping is not None:
            utilities = np.full((batch_size, len(X)), np.nan)
            utilities[:, mapping] = utilities_cand
            query_indices = mapping[query_indices_cand]

        # Check whether utilities are to be returned.
        if return_utilities:
            return query_indices, utilities
        else:
            return query_indices
