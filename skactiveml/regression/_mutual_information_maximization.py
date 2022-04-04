from copy import deepcopy

import numpy as np
from sklearn import clone

from skactiveml.base import (
    SingleAnnotatorPoolQueryStrategy,
    SkactivemlConditionalEstimator,
)
from skactiveml.regressor.estimator._nichke import NormalInverseChiKernelEstimator
from skactiveml.utils import check_type, simple_batch
from skactiveml.utils._approximation import conditional_expect
from skactiveml.utils._functions import update_X_y, update_X_y_map


class MutualInformationGainMaximization(SingleAnnotatorPoolQueryStrategy):
    """Regression based Mutual Information Gain Maximization

    This class implements a mutual information based selection strategies, where
    it is assumed that the prediction probability for different samples
    are independent.

    Parameters
    ----------
    random_state: numeric | np.random.RandomState, optional
        Random state for candidate selection.
    integration_dict: dict,
        Dictionary for integration arguments, i.e. `integration method` etc.,
        used for calculating the expected `y` value for the candidate samples.
        For details see method `conditional_expect`.

    References
    ----------
    [1] Elreedy, Dina and F Atiya, Amir and I Shaheen, Samir. A novel active
        learning regression framework for balancing the exploration-exploitation
        trade-off, page 651 and subsequently, 2019.

    """

    def __init__(self, random_state=None, integration_dict=None):
        super().__init__(random_state=random_state)
        if integration_dict is not None:
            self.integration_dict = integration_dict
        else:
            self.integration_dict = {"method": "assume_linear"}

    def query(
        self,
        X,
        y,
        cond_est,
        sample_weight=None,
        candidates=None,
        batch_size=1,
        return_utilities=False,
        fit_cond_est=True,
    ):
        """Determines for which candidate samples labels are to be queried.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data set, usually complete, i.e. including the labeled and
            unlabeled samples.
        y : array-like of shape (n_samples)
            Labels of the training data set (possibly including unlabeled ones
            indicated by self.MISSING_LABEL.
        cond_est: SkactivemlConditionalEstimator
            Estimates the entropy.
        fit_cond_est : bool, optional (default=True)
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
            If true, also return the utilities based on the query strategy.

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
        X, y, candidates, batch_size, return_utilities = self._validate_data(
            X, y, candidates, batch_size, return_utilities, reset=True
        )

        check_type(cond_est, "cond_est", SkactivemlConditionalEstimator)
        check_type(self.integration_dict, "self.integration_dict", dict)

        X_cand, mapping = self._transform_candidates(candidates, X, y)

        if fit_cond_est:
            cond_est = clone(cond_est).fit(X, y, sample_weight)

        if sample_weight is not None and mapping is not None:
            raise ValueError(
                "If `sample_weight` is not `None`, a mapping "
                "between candidates and the training dataset must "
                "exist."
            )

        utilities_cand = self._mutual_information(
            X, X_cand, mapping, cond_est, X, y, sample_weight
        )

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

    def _mutual_information(
        self, X_eval, X_cand, mapping, cond_est, X, y, sample_weight=None
    ):
        """Calculates the mutual information gain over the evaluation set if each
        candidate where to be labeled.

        Parameters
        ----------
        X_eval : array-like of shape (n_samples, n_features)
            The samples where the information gain should be evaluated.
        cond_est: SkactivemlConditionalEstimator
            Estimates the entropy, predicts values.
        X : array-like of shape (n_samples, n_features)
            Training data set, usually complete, i.e. including the labeled and
            unlabeled samples.
        y : array-like of shape (n_samples)
            Labels of the training data set (possibly including unlabeled ones
            indicated by self.MISSING_LABEL.
        sample_weight: array-like of shape (n_samples), optional (default=None)
            Weights of training samples in `X`.

        Returns
        -------
        query_indices : numpy.ndarray of shape (n_candidate_samples)
            The expected information gain for each candidate sample.
        """

        prior_entropy = np.sum(cond_est.predict(X_eval, return_entropy=True)[1])

        def new_entropy(idx, x_cand, y_pot):
            X_new, y_new = update_X_y_map(X, y, y_pot, idx, x_cand, mapping)
            new_cond_est = clone(cond_est).fit(X_new, y_new, sample_weight)
            _, entropy_cand = new_cond_est.predict(X_eval, return_entropy=True)
            potentials_post_entropy = np.sum(entropy_cand)
            return potentials_post_entropy

        cond_entropy = conditional_expect(
            X_cand,
            new_entropy,
            cond_est,
            random_state=self.random_state_,
            include_idx=True,
            include_x=True,
            **self.integration_dict
        )

        mi_gain = prior_entropy - cond_entropy

        return mi_gain
