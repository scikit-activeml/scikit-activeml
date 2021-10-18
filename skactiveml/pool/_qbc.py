"""
Query-by-committee strategies.
"""
# Author: Pascal Mergard <Pascal.Mergard@student.uni-kassel.de>
#         Marek Herde <marek.herde@uni-kassel.de>

from copy import deepcopy

import numpy as np
from sklearn.utils.validation import check_array, _is_arraylike

from ..base import SingleAnnotPoolBasedQueryStrategy, SkactivemlClassifier
from ..utils import simple_batch, fit_if_not_fitted, check_type, \
    compute_vote_vectors


class QBC(SingleAnnotPoolBasedQueryStrategy):
    """QBC

    The Query-By-Committee (QBC) algorithm minimizes the version space, which
    is the set of hypotheses that are consistent with the current labeled
    training data.

    Parameters
    ----------
    method : {'KL_divergence', 'vote_entropy'}, optional
    (default='KL_divergence')
        The method to calculate the disagreement.
    random_state: numeric | np.random.RandomState, optional (default=None)
        Random state for annotator selection.

    References
    ----------
    [1] H.S. Seung, M. Opper, and H. Sompolinsky. Query by committee.
        In Proceedings of the ACM Workshop on Computational Learning Theory,
        pages 287-294, 1992.
    [2] N. Abe and H. Mamitsuka. Query learning strategies using boosting and
        bagging. In Proceedings of the International Conference on Machine
        Learning (ICML), pages 1-9. Morgan Kaufmann, 1998.
    """

    def __init__(self, method='KL_divergence', random_state=None):
        super().__init__(random_state=random_state)
        self.method = method

    def query(self, X_cand, ensemble, X=None, y=None, sample_weight=None,
              batch_size=1, return_utilities=False):
        """Queries the next instance to be labeled.

        Parameters
        ----------
        X_cand : array-like, shape (n_candidate_samples, n_features)
            Candidate samples from which the strategy can select.
        ensemble : {skactiveml.base.SkactivemlClassifier, array-like}
            If `ensemble` is a `SkactivemlClassifier`, it must have
            `n_estimators` and `estimators_` after fitting as attribute. Then,
            its estimators will be used as committee. If `ensemble` is
            array-like, each element of this list must be
            `SkactivemlClassifier` and will be used as committee member.
        X: array-like, shape (n_samples, n_features), optional (default=None)
            Complete training data set.
        y: array-like, shape (n_samples), optional (default=None)
            Labels of the training data set.
        sample_weight: array-like, shape (n_samples), optional
        (default=None)
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
        # Validate input parameters.
        X_cand, return_utilities, batch_size, random_state = \
            self._validate_data(X_cand, return_utilities, batch_size,
                                self.random_state, reset=True)

        # Check attributed `method`.
        if self.method not in ['KL_divergence', 'vote_entropy']:
            raise ValueError(
                f"The given method {self.method} is not valid. "
                f"Supported methods are 'KL_divergence' and 'vote_entropy'")

        # Check if the parameter `ensemble` is valid.
        if isinstance(ensemble, SkactivemlClassifier) and \
                hasattr(ensemble, 'n_estimators'):
            ensemble = fit_if_not_fitted(
                ensemble, X, y, sample_weight=sample_weight
            )
            classes = ensemble.classes_
            if hasattr(ensemble, 'estimators_'):
                est_arr = ensemble.estimators_
            else:
                est_arr = [ensemble] * ensemble.n_estimators
        elif _is_arraylike(ensemble):
            est_arr = deepcopy(ensemble)
            for i in range(len(est_arr)):
                check_type(est_arr[i], SkactivemlClassifier, f'ensemble[{i}]')
                est_arr[i] = fit_if_not_fitted(
                    est_arr[i], X, y, sample_weight=sample_weight
                )
                if i > 0:
                    np.testing.assert_array_equal(
                        est_arr[i - 1].classes_, est_arr[i].classes_,
                        err_msg=f'The inferred classes of the {i - 1}-th and '
                                f'{i}-th are not equal. Set the `classes` '
                                f'parameter of each ensemble member to avoid '
                                f'this error.'
                    )
            classes = est_arr[0].classes_
        else:
            raise TypeError(
                f'`ensemble` must either be a `{SkactivemlClassifier} '
                f'with the attribute `n_esembles` and `estimators_` after '
                f'fitting or a list of {SkactivemlClassifier} objects.'
            )

        # Compute utilities.
        if self.method == 'KL_divergence':
            probas = np.array([est.predict_proba(X_cand) for est in est_arr])
            utilities = average_kl_divergence(probas)
        elif self.method == 'vote_entropy':
            votes = np.array([est.predict(X_cand) for est in est_arr]).T
            utilities = vote_entropy(votes, classes)

        return simple_batch(utilities, random_state,
                            batch_size=batch_size,
                            return_utilities=return_utilities)


def average_kl_divergence(probas):
    """Calculates the average Kullback-Leibler (KL) divergence for measuring
    the level of disagreement in QBC.

    Parameters
    ----------
    probas : array-like, shape (n_estimators, n_samples, n_classes)
        The probability estimates of all estimators, samples, and classes.

    Returns
    -------
    scores: np.ndarray, shape (n_samples)
        The Kullback-Leibler (KL) divergences.

    References
    ----------
    [1] A. McCallum and K. Nigam. Employing EM in pool-based active learning
        for text classification. In Proceedings of the International Conference
        on Machine Learning (ICML), pages 359-367. Morgan Kaufmann, 1998.
    """
    # Check probabilities.
    probas = check_array(probas, allow_nd=True)
    if probas.ndim != 3:
        raise ValueError(
            f"Expected 3D array, got {probas.ndim}D array instead."
        )

    # Calculate the average KL divergence.
    probas_mean = np.mean(probas, axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        scores = np.nansum(
            np.nansum(probas * np.log(probas / probas_mean), axis=2), axis=0)
    scores = scores / probas.shape[0]

    return scores


def vote_entropy(votes, classes):
    """Calculates the vote entropy for measuring the level of disagreement in
    QBC.

    Parameters
    ----------
    votes : array-like, shape (n_samples, n_estimators)
        The class predicted by the estimators for each sample.
    classes : array-like, shape (n_classes)
        A list of all possible classes.

    Returns
    -------
    vote_entropy : np.ndarray, shape (n_samples)
        The vote entropy of each row in `votes`.

    References
    ----------
    [1] Engelson, Sean P., and Ido Dagan.
        Minimizing manual annotation cost in supervised training from corpora.
        arXiv preprint cmp-lg/9606030 (1996).
    """
    # Check `votes` array.
    votes = check_array(votes)

    # Count the votes.
    vote_count = compute_vote_vectors(y=votes, classes=classes,
                                      missing_label=None)

    # Compute vote entropy.
    v = vote_count / len(votes)
    with np.errstate(divide='ignore', invalid='ignore'):
        scores = -np.nansum(v * np.log(v), axis=1) / np.log(len(votes))
    return scores
