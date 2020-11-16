import numpy as np
import warnings

from sklearn import clone

from ..base import SingleAnnotPoolBasedQueryStrategy

from sklearn.ensemble import BaggingClassifier, BaseEnsemble
from sklearn.utils import check_array
from ..utils import is_labeled, MISSING_LABEL, check_X_y, check_scalar, \
    simple_batch


class QBC(SingleAnnotPoolBasedQueryStrategy):
    """QBC

    The Query-By-Committee (QBC) algorithm minimizes the version space, which is the set of hypotheses that are
    consistent with the current labeled training data.
    This class implement the query-by-bagging method, which uses the bagging in sklearn to
    construct the committee. So your model should be a sklearn model.

    Parameters
    ----------
    classes : array-like, shape=(n_classes)
        Holds the label for each class.
    clf : sklearn classifier | ensamble
        If clf is an ensemble, it will used as committee. If clf is a
        classifier, it will used for ensemble construction with the specified
        ensemble or with BaggigngClassifier, if ensemble is None. clf must
        implementing the methods 'fit', 'predict'(for vote entropy) and
        'predict_proba'(for KL divergence).
    ensemble : sklearn.ensemble, default=None
        sklear.ensemble used as committee. If None, baggingClassifier is used.
    method : string, default='KL_divergence'
        The method to calculate the disagreement.
        'vote_entropy' or 'KL_divergence' are possible.
    missing_label : scalar | str | None | np.nan, (default=MISSING_LABEL)
        Specifies the symbol that represents a missing label.
        Important: We do not differ between None and np.nan.
    random_state : numeric | np.random.RandomState
        Random state to use.
    **kwargs :
        will be passed on to the ensemble.

    Attributes
    ----------
    ensemble : sklearn.ensemble
        Ensemble used as committee.
        Implementing the methods 'fit', 'predict'(for vote entropy) and 'predict_proba'(for KL divergence).
    method : string, default='KL_divergence'
        The method to calculate the disagreement. 'vote_entropy' or 'KL_divergence' are possible.
    classes : array-like, shape=(n_classes)
        Holds the label for each class.
    unlabeled_class : scalar | str | None | np.nan, default=np.nan
        Symbol to represent a missing label. Important: We do not differ between None and np.nan.
    random_state : numeric | np.random.RandomState
        Random state to use.

    References
    ----------
    [1] H.S. Seung, M. Opper, and H. Sompolinsky. Query by committee.
        In Proceedings of the ACM Workshop on Computational Learning Theory,
        pages 287-294, 1992.
    [2] N. Abe and H. Mamitsuka. Query learning strategies using boosting and bagging.
        In Proceedings of the International Conference on Machine Learning (ICML),
        pages 1-9. Morgan Kaufmann, 1998.
    """

    def __init__(self, clf, ensemble=None, method='KL_divergence', missing_label=MISSING_LABEL, random_state=None,
                 **kwargs):
        super().__init__(random_state=random_state)

        self.missing_label = missing_label
        self.method = method
        self.ensemble = ensemble
        self.clf = clf
        self.kwargs_init = kwargs

    def query(self, X_cand, X, y, batch_size=1, return_utilities=False, **kwargs):
        """
        Queries the next instance to be labeled.

        Parameters
        ----------
        X_cand : np.ndarray
            The unlabeled pool from which to choose.
        X : np.ndarray
            The labeled pool used to fit the classifier.
        y : np.array
            The labels of the labeled pool X.
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
        # validation:
        # Check batch size.
        check_scalar(batch_size, target_type=int, name='batch_size',
                     min_val=1)
        if len(X_cand) < batch_size:
            warnings.warn(
                "'batch_size={}' is larger than number of candidate samples "
                "in 'X_cand'. Instead, 'batch_size={}' was set ".format(
                    batch_size, len(X_cand)))
            batch_size = len(X_cand)

        # check self.clf and self.method
        if self.method != 'KL_divergence' and self.method != 'vote_entropy':
            raise ValueError(
                'The method {} does not exist.'.format(self.method))
        if self.method == 'vote_entropy' and \
                ((getattr(self.clf, 'fit', None) is None or
                  getattr(self.clf, 'predict', None) is None)):
            raise TypeError(
                "'clf' must implement the methods 'fit' and 'predict'")
        elif self.method == 'KL_divergence' and \
                ((getattr(self.clf, 'fit', None) is None or
                  getattr(self.clf, 'predict_proba', None) is None)):
            raise TypeError(
                "'clf' must implement the methods 'fit' and 'predict_proba'")

        # check self.ensemble and self.clf
        if not isinstance(self.clf, BaseEnsemble):
            if self.ensemble is None:
                warnings.warn('\'ensemble\' is not specified, '
                              '\'BaggingClassifier\' will be used.')
                self.ensemble = BaggingClassifier
            self.clf = self.ensemble(base_estimator=clone(self.clf),
                                     random_state=self.random_state)

        # check X, y and X_cand
        X, y, X_cand = check_X_y(X, y, X_cand, force_all_finite=False)

        # remove unlabeled instances from X and y

        mask_labeled = is_labeled(y, self.missing_label)
        self.clf.fit(X[mask_labeled], y[mask_labeled])

        # choose the disagreement method and calculate the utilities
        if self.method == 'KL_divergence':
            utilities = average_KL_divergence(self.clf, X_cand)
        elif self.method == 'vote_entropy':
            utilities = vote_entropy(self.clf, X_cand, )

        return simple_batch(utilities, self.random_state,
                            batch_size=batch_size,
                            return_utilities=return_utilities)


def average_KL_divergence(ensemble, X_cand):
    """
    Calculate the average Kullback-Leibler (KL) divergence for measuring the
    level of disagreement in QBC.

    Parameters
    ----------
    ensemble : sklearn.ensemble
         fited sklearn.ensemble used as committee.
    X_cand : np.ndarray
        The unlabeled pool for which to calculated the calc_avg_KL_divergence.

    Returns
    -------
    scores: np.ndarray, shape=(len(X_cand)
        The Kullback-Leibler (KL) divergences.

    References
    ----------
    [1] A. McCallum and K. Nigam. Employing EM in pool-based active learning
    for text classification. In Proceedings of the International Conference on
    Machine Learning (ICML), pages 359-367. Morgan Kaufmann, 1998.
    """

    # validation:
    # check X
    X_cand = check_array(X_cand, accept_sparse=False,
                         accept_large_sparse=True, dtype="numeric", order=None,
                         copy=False, force_all_finite=True, ensure_2d=True,
                         allow_nd=False, ensure_min_samples=1,
                         ensure_min_features=1, estimator=None)

    # check ensemble
    if not isinstance(ensemble, BaseEnsemble):
        raise TypeError("'ensemble' most be an instance of 'BaseEnsemble'")

    # calculate the average KL divergence:
    est_arr = ensemble.estimators_
    P = [est_arr[e_idx].predict_proba(X_cand) for e_idx in range(len(est_arr))]
    P = np.array(P)
    P_com = np.mean(P, axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        scores = np.nansum(np.nansum(P * np.log(P / P_com), axis=2), axis=0)
    scores = scores / ensemble.n_classes_

    return scores


def vote_entropy(ensemble, X_cand):
    """
    Calculate the vote entropy for measuring the level of disagreement in QBC.

    Parameters
    ----------
    ensemble : sklearn.ensemble
         fited sklearn BaggingClassifier used as committee.
    X_cand : np.ndarray
        The unlabeled pool for which to calculated the vote entropy.

    Returns
    -------
    vote_entropy : np.ndarray, shape=(len(X_cand))
        The vote entropy of each instance in X_cand.

    References
    ----------
    [1] Engelson, Sean P., and Ido Dagan.
    "Minimizing manual annotation cost in supervised training from corpora."
    arXiv preprint cmp-lg/9606030 (1996).
    """

    # validation:
    # check x
    X_cand = check_array(X_cand, accept_sparse=False,
                         accept_large_sparse=True, dtype="numeric", order=None,
                         copy=False, force_all_finite=True, ensure_2d=True,
                         allow_nd=False, ensure_min_samples=1,
                         ensure_min_features=1, estimator=None)

    # check ensemble
    if not isinstance(ensemble, BaseEnsemble):
        raise TypeError("'ensamble' most be an instance of 'BaseEnsamble'")

    estimators = ensemble.estimators_
    # Let the models vote for unlabeled data
    votes = np.zeros((len(X_cand), len(estimators)))
    for i, model in enumerate(estimators):
        votes[:, i] = model.predict(X_cand)

    # count the votes
    vote_count = np.zeros((len(X_cand), ensemble.n_classes_))
    for i in range(len(X_cand)):
        for c in range(ensemble.n_classes_):
            for m in range(len(estimators)):
                vote_count[i, c] += (votes[i, m] == c)

    # compute vote entropy
    vote_entropy = np.zeros(len(X_cand))
    for i in range(len(X_cand)):
        for c in range(ensemble.n_classes_):
            if vote_count[i, c] != 0:  # definition gap at vote_count[i,c]==0
                a = vote_count[i, c] / len(estimators)
                vote_entropy[i] += a * np.log(a)
    vote_entropy *= -1 / np.log(len(estimators))

    return vote_entropy
