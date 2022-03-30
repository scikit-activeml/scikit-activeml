import numpy as np
from sklearn import clone

from skactiveml.base import (
    SingleAnnotatorPoolQueryStrategy,
    SkactivemlConditionalEstimator,
)
from skactiveml.utils import check_type, simple_batch, check_random_state
from skactiveml.utils._approximation import conditional_expect
from skactiveml.utils._functions import update_X_y


class KullbackLeiblerDivergenceMaximization(SingleAnnotatorPoolQueryStrategy):
    """Regression based Kullback Leibler Divergence Maximization

    This class implements a Kullback Leibler divergence based selection
    strategies, where it is assumed that the prediction probability for
    different samples are independent.

    Parameters
    ----------
    random_state: numeric | np.random.RandomState, optional
        Random state for candidate selection.
    integration_dict_potential_y_val: dict, optional (default=None)
        Dictionary for integration arguments, i.e. `integration method` etc.,
        used for calculating the expected `y` value for the candidate samples.
        For details see method `conditional_expect`.
    integration_dict_cross_entropy: dict, optional (default=None)
        Dictionary for integration arguments, i.e. `integration method` etc.,
        used for calculating the cross entropy between the updated conditional
        estimator by the `X_cand` value and the old conditional estimator.
        For details see method `conditional_expect`.

    References
    ----------
    [1] Elreedy, Dina and F Atiya, Amir and I Shaheen, Samir. A novel active
        learning regression framework for balancing the exploration-exploitation
        trade-off, page 651 and subsequently, 2019.

    """

    def __init__(
        self,
        random_state=None,
        integration_dict_potential_y_val=None,
        integration_dict_cross_entropy=None,
    ):
        super().__init__(random_state=random_state)

        if integration_dict_potential_y_val is not None:
            self.integration_dict_potential_y_val = integration_dict_potential_y_val
        else:
            self.integration_dict_potential_y_val = {"method": "assume_linear"}

        if integration_dict_cross_entropy is not None:
            self.integration_dict_cross_entropy = integration_dict_cross_entropy
        else:
            self.integration_dict_cross_entropy = {
                "method": "monte_carlo",
                "n_monte_carlo": 10,
            }

    def query(
        self,
        X,
        y,
        cond_est,
        sample_weight=None,
        candidates=None,
        batch_size=1,
        return_utilities=False,
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

        X_cand, mapping = self._transform_candidates(candidates, X, y)

        if sample_weight is not None and mapping is not None:
            raise ValueError(
                "If `sample_weight` is not `None`a mapping "
                "between candidates and the training dataset must "
                "exist."
            )

        utilities_cand = self._kullback_leibler_divergence(
            X_cand, X_cand, cond_est, X, y, sample_weight
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

    def _kullback_leibler_divergence(
        self, X_eval, X_cand, mapping, cond_est, X, y, sample_weight=None
    ):
        """Calculates the mutual information gain over the evaluation set if each
        candidate sample where to be labeled.

        Parameters
        ----------
        X_eval : array-like of shape (n_samples, n_features)
            The samples where the information gain should be evaluated.
        X_cand : array-like of shape (n_candidate_samples, n_features)
            The candidate samples that determine the information gain.
        mapping : array-like of shape (n_candidate_samples,) or None
            A mapping between `X_cand` and `X` if it exists.
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

        random_state = check_random_state(random_state=self.random_state)
        prior_entropy = np.sum(cond_est.predict(X_cand, return_entropy=True)[1])

        def cross_entropy_new_old(idx, x_cand, y_pot):
            if mapping is not None:
                X_new, y_new = update_X_y(X, y, y_pot, idx_update=mapping[idx])
            else:
                X_new, y_new = update_X_y(X, y, y_pot, X_update=x_cand)

            new_cond_est = clone(cond_est).fit(X_new, y_new, sample_weight)
            return cross_entropy(
                X_eval,
                new_cond_est,
                cond_est,
                integration_dict=self.integration_dict_cross_entropy,
                random_state=random_state,
            )

        cross_ent = conditional_expect(
            X_cand,
            cross_entropy_new_old,
            cond_est,
            random_state=random_state,
            include_idx=True,
            include_x=True,
        )

        return cross_ent - prior_entropy


def cross_entropy(
    X_eval, true_cond_est, other_cond_est, integration_dict=None, random_state=None
):
    """Calculates the cross entropy.

    Parameters
    ----------
    X_eval : array-like of shape (n_samples, n_features)
        The samples where the cross entropy should be evaluated.
    true_cond_est: SkactivemlConditionalEstimator
        True distribution for the cross entropy.
    other_cond_est: SkactivemlConditionalEstimator
        Distribution to evaluate by the cross entropy
    integration_dict: dict,
        Dictionary for integration arguments, i.e. `integration method` etc..
        For details see method `conditional_expect`.
    random_state: numeric | np.random.RandomState, optional
        Random state for cross entropy calculation.

    Returns
    -------
    query_indices : numpy.ndarray of shape (n_candidate_samples)
        The conditional entropy for each candidate sample.
    """

    check_type(integration_dict, "integration_dict", dict)
    check_type(true_cond_est, "true_cond_est", SkactivemlConditionalEstimator)
    check_type(other_cond_est, "other_cond_est", SkactivemlConditionalEstimator)
    random_state = check_random_state(random_state)

    dist = other_cond_est.estimate_conditional_distribution(X_eval)
    cross_ent = -conditional_expect(
        X_eval,
        dist.logpdf,
        cond_est=true_cond_est,
        random_state=random_state,
        **integration_dict,
        vector_func=True
    )

    return cross_ent
