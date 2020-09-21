import numpy as np
import warnings

from sklearn.utils import check_array, check_random_state, check_scalar

from ..base import PoolBasedQueryStrategy
from ..utils import rand_argmax, is_labeled
from ..classifier import CMM


class FourDS(PoolBasedQueryStrategy):
    """FourDS

    Implementation of the pool-based query strategy 4DS for training a CMM [1].

    Parameters
    ----------
    clf : skactiveml.classifier.CMM
        GMM-based Classifier to be trained.
    batch_size : int, optional (default=1)
        The number of samples to be selected in one AL cycle.
    lmbda : float between 0 and 1, optional
    (default=min((batch_size-1)*0.05, 0.5))
        For the selection of more than one sample within each query round, 4DS
        uses a diversity measure to avoid
        the selection of redundant samples whose influence is regulated by the
        weighting factor 'lmbda'.
    random_state : numeric | np.random.RandomState, optional (default=None)
        The random state to use.

    Attributes
    ----------
    clf : skactiveml.classifier.CMM
        GMM-based Classifier to be trained.
    batch_size : int, optional (default=1)
        The number of samples to be selected in one AL cycle.
    lmbda : float between 0 and 1, optional
    (default=min((batch_size-1)*0.05, 0.5))
        For the selection of more than one sample within each query round, 4DS
        uses a diversity measure to avoid
        the selection of redundant samples whose influence is regulated by the
        weighting factor 'lmbda'.
    random_state : numeric | np.random.RandomState, optional (default=None)
        The random state to use.

    References
    ---------
    [1] Reitmaier, T., & Sick, B. (2013). Let us know your decision: Pool-based
    active training of a generative classifier with the selection strategy 4DS.
    Information Sciences, 230, 106-131.
    """

    def __init__(self, clf, batch_size=1, lmbda=None, random_state=None):
        super().__init__(random_state=random_state)
        if not isinstance(clf, CMM):
            raise TypeError(
                "'clf' must be a 'CMM' but got {}".format(type(clf)))
        self.clf = clf
        check_scalar(batch_size, target_type=int, name='batch_size', min_val=1)
        self.batch_size = batch_size
        if lmbda is None:
            lmbda = np.min(((self.batch_size - 1) * 0.05, 0.5))
        check_scalar(lmbda, target_type=float, name='lmbda', min_val=0,
                     max_val=1)
        self.lmbda = lmbda
        self.random_state = check_random_state(random_state)

    def query(self, X_cand, X, y, return_utilities=False, **kwargs):
        """Ask the query strategy which sample in 'X_cand' to query.

        Parameters
        ----------
        X_cand : {array-like, sparse matrix},
        shape (n_cand_samples, n_features)
            The samples which may be queried. Sparse matrices are accepted
            only if they are supported by the base query strategy.
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Input samples used to fit the classifier.
        y : {array-like, sparse matrix}, shape (n_samples)
            Labels of the input samples 'X'. There may be missing labels.
        return_utilities : bool, optional (default=False)
            If true, also return the utilities based on the query strategy.

        Returns
        -------
        query_indices : numpy.ndarray, shape (batch_size)
            The indices of samples in `X_cand` whose labels should be queried.
        utilities: numpy.ndarray, shape (n_cand_samples) for batch_size=1 else
        (n_cand_samples, batch_size)
            The utilities of all instances of `X_cand` after each selection
            step (if return_utilities=True).
        """
        # Check X_cand to be a non-empty 2D array.
        X_cand = check_array(X_cand)

        batch_size = self.batch_size
        if len(X_cand) < self.batch_size:
            warnings.warn(
                "'batch_size={}' is larger than number of candidate samples "
                "in 'X_cand'. Set 'batch_size={}'".format(
                    self.batch_size, len(X_cand)))
            batch_size = len(X_cand)

        # Fit the classifier and get the probabilities.
        self.clf.fit(X, y)
        P_cand = self.clf.predict_proba(X_cand)
        R_cand = self.clf.mixture_model_.predict_proba(X_cand)
        is_lbld = is_labeled(y, missing_label=self.clf.missing_label)
        R_lbld = self.clf.mixture_model_.predict_proba(X[is_lbld])

        # Compute distance according to Eq. 9 in [1].
        P_cand_sorted = np.sort(P_cand, axis=1)
        distance_cand = np.log(
            (P_cand_sorted[:, -1] + 1.e-5) / (P_cand_sorted[:, -2] + 1.e-5))
        distance_cand = (distance_cand - np.min(distance_cand)) / (
                np.max(distance_cand) - np.min(distance_cand))

        # Compute densities according to Eq. 10 in [1].
        density_cand = self.clf.mixture_model_.score_samples(X_cand)
        density_cand = (density_cand - np.min(density_cand)) / (
                np.max(density_cand) - np.min(density_cand))

        # Compute distributions according to Eq. 11 in [1].
        R_lbld_sum = np.sum(R_lbld, axis=0, keepdims=True)
        R_sum = R_cand + R_lbld_sum
        R_mean = R_sum / (len(R_lbld) + 1)
        distribution_cand = self.clf.mixture_model_.weights_ - R_mean
        distribution_cand = np.maximum(np.zeros_like(distribution_cand),
                                       distribution_cand)
        distribution_cand = 1 - np.sum(distribution_cand, axis=1)

        # Compute rho according to Eq. 15  in [1].
        diff = np.sum(
            np.abs(self.clf.mixture_model_.weights_ - np.mean(R_lbld, axis=0)))
        rho = min(1, diff)

        # Compute e_dwus according to Eq. 13  in [1].
        e_dwus = np.mean((1 - P_cand_sorted[:, -1]) * density_cand)

        # Normalization such that alpha, beta, and rho sum up to one.
        alpha = (1 - rho) * e_dwus
        beta = 1 - rho - alpha

        # Compute utilities to select sample.
        utilities = np.empty((len(X_cand), batch_size), dtype=float)
        utilities[:, 0] = alpha * (
                1 - distance_cand) + beta * density_cand + \
                          rho * distribution_cand
        query_indices = rand_argmax(utilities[:, 0])
        is_selected = np.zeros(len(X_cand), dtype=bool)
        is_selected[query_indices] = True

        if batch_size > 1:
            # Compute e_us according to Eq. 14  in [1].
            e_us = np.mean(1 - P_cand_sorted[:, -1])

            # Normalization of the coefficients alpha, beta, and rho such
            # that these coefficients plus
            # lmbda sum up to one.
            rho = min(rho, 1 - self.lmbda)
            alpha = (1 - (rho + self.lmbda)) * (1 - e_us)
            beta = 1 - (rho + self.lmbda) - alpha

            for i in range(1, batch_size):
                # Update distributions according to Eq. 11 in [1].
                R_sum = R_cand + np.sum(R_cand[is_selected], axis=0,
                                        keepdims=True) + R_lbld_sum
                R_mean = R_sum / (len(R_lbld) + len(query_indices) + 1)
                distribution_cand = self.clf.mixture_model_.weights_ - R_mean
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
                utilities[:, i] = alpha * (
                        1 - distance_cand) + beta * density_cand + \
                                  self.lmbda * diversity_cand \
                                  + rho * distribution_cand
                utilities[is_selected, i] = - np.inf
                query_indices = np.append(query_indices,
                                          rand_argmax(utilities[:, i]))
                is_selected[query_indices] = True

        # Check whether utilities are to be returned.
        if return_utilities:
            if self.batch_size == 1:
                utilities = utilities.ravel()
            return query_indices, utilities
        else:
            return query_indices
