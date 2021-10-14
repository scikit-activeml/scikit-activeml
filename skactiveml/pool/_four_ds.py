"""
Module implementing 4DS active learning strategy.
"""
# Author: Marek Herde <marek.herde@uni-kassel.de>


import numpy as np
from sklearn.utils.validation import check_array, check_scalar, column_or_1d

from ..base import SingleAnnotPoolBasedQueryStrategy
from ..classifier import CMM
from ..utils import rand_argmax, is_labeled, fit_if_not_fitted, check_type


class FourDS(SingleAnnotPoolBasedQueryStrategy):
    """FourDS

    Implementation of the pool-based query strategy 4DS for training a CMM [1].

    Parameters
    ----------
    lmbda : float between 0 and 1, optional
    (default=min((batch_size-1)*0.05, 0.5))
        For the selection of more than one sample within each query round, 4DS
        uses a diversity measure to avoid the selection of redundant samples
        whose influence is regulated by the weighting factor 'lmbda'.
    random_state : numeric | np.random.RandomState, optional (default=None)
        The random state to use.

    References
    ---------
    [1] Reitmaier, T., & Sick, B. (2013). Let us know your decision: Pool-based
        active training of a generative classifier with the selection strategy
        4DS. Information Sciences, 230, 106-131.
    """

    def __init__(self, lmbda=None, random_state=None):
        super().__init__(random_state=random_state)
        self.lmbda = lmbda

    def query(self, X_cand, clf, X=None, y=None, sample_weight=None,
              return_utilities=False, batch_size=1):
        """Ask the query strategy which sample in 'X_cand' to query.

        Parameters
        ----------
        X_cand : array-like, shape (n_candidate_samples, n_features)
            Candidate samples from which the strategy can select.
        clf : skactiveml.classifier.CMM
            GMM-based classifier to be trained.
        X: array-like, shape (n_samples, n_features), optional (default=None)
            Complete training data set.
        y: array-like, shape (n_samples), optional (default=None)
            Labels of the training data set.
        sample_weight: array-like, shape (n_samples), optional (default=None)
            Weights of training samples in `X`.
        batch_size : int, optional (default=1)
            The number of samples to be selected in one AL cycle.
        return_utilities : bool, optional (default=False)
            If true, also return the utilities based on the query strategy.

        Returns
        -------
        query_indices : numpy.ndarray, shape (batch_size)
            The query_indices indicate for which candidate sample a label is
            to queried, e.g., `query_indices[0]` indicates the first selected
            sample.
        utilities : numpy.ndarray, shape (batch_size, n_samples)
            The utilities of all candidate samples after each selected
            sample of the batch, e.g., `utilities[0]` indicates the utilities
            used for selecting the first sample (with index `query_indices[0]`)
            of the batch.
        """
        # Validate input.
        X_cand, return_utilities, batch_size, random_state = \
            self._validate_data(X_cand, return_utilities, batch_size,
                                self.random_state, reset=True)

        # Check input training data.
        X = check_array(X, ensure_min_samples=0)
        self._check_n_features(X, reset=False)
        y = column_or_1d(y)

        # Check classifier type.
        check_type(clf, CMM, 'clf')

        # Storage for query indices.
        query_indices = np.full(batch_size, fill_value=-1, dtype=int)

        # Check lmbda.
        lmbda = self.lmbda
        if lmbda is None:
            lmbda = np.min(((batch_size - 1) * 0.05, 0.5))
        check_scalar(lmbda, target_type=float, name='lmbda', min_val=0,
                     max_val=1)

        # Fit the classifier and get the probabilities.
        clf = fit_if_not_fitted(clf, X, y, sample_weight)
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
            (P_cand_sorted[:, -1] + 1.e-5) / (P_cand_sorted[:, -2] + 1.e-5))
        distance_cand = (distance_cand - np.min(distance_cand) + 1.e-5) / (
                np.max(distance_cand) - np.min(distance_cand) + 1.e-5)

        # Compute densities according to Eq. 10 in [1].
        density_cand = clf.mixture_model_.score_samples(X_cand)
        density_cand = (density_cand - np.min(density_cand) + 1.e-5) / (
                np.max(density_cand) - np.min(density_cand) + 1.e-5)

        # Compute distributions according to Eq. 11 in [1].
        R_lbld_sum = np.sum(R_lbld, axis=0, keepdims=True)
        R_sum = R_cand + R_lbld_sum
        R_mean = R_sum / (len(R_lbld) + 1)
        distribution_cand = clf.mixture_model_.weights_ - R_mean
        distribution_cand = np.maximum(np.zeros_like(distribution_cand),
                                       distribution_cand)
        distribution_cand = 1 - np.sum(distribution_cand, axis=1)

        # Compute rho according to Eq. 15  in [1].
        diff = np.sum(
            np.abs(clf.mixture_model_.weights_ - np.mean(R_lbld, axis=0)))
        rho = min(1, diff)

        # Compute e_dwus according to Eq. 13  in [1].
        e_dwus = np.mean((1 - P_cand_sorted[:, -1]) * density_cand)

        # Normalization such that alpha, beta, and rho sum up to one.
        alpha = (1 - rho) * e_dwus
        beta = 1 - rho - alpha

        # Compute utilities to select sample.
        utilities = np.empty((batch_size, len(X_cand)), dtype=float)
        utilities[0] = alpha * (
                1 - distance_cand) + beta * density_cand + \
                          rho * distribution_cand
        query_indices[0] = rand_argmax(utilities[0], random_state)
        is_selected = np.zeros(len(X_cand), dtype=bool)
        is_selected[query_indices[0]] = True

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
                R_sum = R_cand + np.sum(R_cand[is_selected], axis=0,
                                        keepdims=True) + R_lbld_sum
                R_mean = R_sum / (len(R_lbld) + len(query_indices) + 1)
                distribution_cand = clf.mixture_model_.weights_ - R_mean
                distribution_cand = np.maximum(
                    np.zeros_like(distribution_cand), distribution_cand)
                distribution_cand = 1 - np.sum(distribution_cand, axis=1)

                # Compute diversity according to Eq. 12 in [1].
                diversity_cand = - np.log(
                    density_cand + np.sum(density_cand[is_selected])) / (
                                         len(query_indices) + 1)
                diversity_cand = (diversity_cand - np.min(diversity_cand)) / (
                        np.max(diversity_cand) - np.min(diversity_cand))

                # Compute utilities to select sample.
                utilities[i] = alpha * (
                        1 - distance_cand) + beta * density_cand + \
                                  lmbda * diversity_cand \
                                  + rho * distribution_cand
                utilities[i, is_selected] = np.nan
                query_indices[i] = rand_argmax(utilities[i], random_state)
                is_selected[query_indices[i]] = True

        # Check whether utilities are to be returned.
        if return_utilities:
            return query_indices, utilities
        else:
            return query_indices
