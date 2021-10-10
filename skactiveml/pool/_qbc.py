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
    clf : SkactivemlClassifier
        If clf is an wrapped ensemble, it will be used as committee. If `clf`
        is a classifier, it will be used for ensemble construction with the
        specified ensemble or with `BaggigngClassifier, if ensemble is None. clf must
        implementing the methods 'fit', 'predict'(for vote entropy) and
        'predict_proba'(for KL divergence).
    ensemble : BaseEnsemble, default=None
        sklearn.ensemble used to construct the committee. If None,
        baggingClassifier is used.
    method : string, default='KL_divergence'
        The method to calculate the disagreement.
        'vote_entropy' or 'KL_divergence' are possible.
    random_state : numeric | np.random.RandomState
        Random state to use.
    ensemble_dict : dictionary
        Will be passed on to the ensemble.

    Attributes
    ----------
    ensemble : sklearn.ensemble
        Ensemble used as committee. Implementing the methods 'fit',
        'predict'(for vote entropy) and 'predict_proba'(for KL divergence).
    method : string, default='KL_divergence'
        The method to calculate the disagreement. 'vote_entropy' or
        'KL_divergence' are possible.
    random_state : numeric | np.random.RandomState
        Random state to use.

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
        """
        Queries the next instance to be labeled.

        Parameters
        ----------
        X_cand : array-like
            The unlabeled pool from which to choose.
        X : array-like
            The labeled pool used to fit the classifier.
        y : array-like
            The labels of the labeled pool X.
        sample_weight : array-like of shape (n_samples,) (default=None)
            Sample weights.
        batch_size : int, optional (default=1)
            The number of samples to be selected in one AL cycle.
        return_utilities : bool (default=False)
            If True, the utilities are returned.

        Returns
        -------
        best_indices : np.ndarray, shape (batch_size)
            The index of the queried instance.
        batch_utilities : np.ndarray,  shape (batch_size, len(X_cnad))
            The utilities of all instances of
            X_cand(if return_utilities=True).
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
    """
    Calculate the average Kullback-Leibler (KL) divergence for measuring the
    level of disagreement in QBC.

    Parameters
    ----------
    probas : array-like, shape (n_estimators, n_samples, n_classes)
        The probability estimations of all estimators, instances and classes.

    Returns
    -------
    scores: np.ndarray, shape (n_samples)
        The Kullback-Leibler (KL) divergences.

    References
    ----------
    [1] A. McCallum and K. Nigam. Employing EM in pool-based active learning
    for text classification. In Proceedings of the International Conference on
    Machine Learning (ICML), pages 359-367. Morgan Kaufmann, 1998.
    """

    # validation:
    # check P
    probas = check_array(probas, accept_sparse=False,
                         accept_large_sparse=True, dtype="numeric", order=None,
                         copy=False, force_all_finite=True, ensure_2d=True,
                         allow_nd=True, ensure_min_samples=1,
                         ensure_min_features=1, estimator=None)
    if probas.ndim != 3:
        raise ValueError("Expected 2D array, got 1D array instead:"
                         "\narray={}.".format(probas))

    # calculate the average KL divergence:
    probas = np.array(probas)
    probas_mean = np.mean(probas, axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        scores = np.nansum(
            np.nansum(probas * np.log(probas / probas_mean), axis=2), axis=0)
    scores = scores / probas.shape[0]

    return scores


def vote_entropy(votes, classes):
    """
    Calculate the vote entropy for measuring the level of disagreement in QBC.

    Parameters
    ----------
    votes : array-like, shape (n_samples, n_estimators)
        The class predicted by the estimators for each instance.
    classes : array-like, shape (n_classes)
        A list of all possible classes.

    Returns
    -------
    vote_entropy : np.ndarray, shape (n_samples)
        The vote entropy of each instance in `X_cand`.

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
