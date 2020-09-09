import numpy as np
import warnings

from ..base import PoolBasedQueryStrategy

from sklearn.ensemble import BaggingClassifier, BaseEnsemble
from sklearn.utils import check_random_state
from ..utils import rand_argmax, is_labeled, MISSING_LABEL


class QBC(PoolBasedQueryStrategy):
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
        If clf is an ensemble, it will used as committee. If clf is a classifier, it will used for ensemble construction with the specified ensemble or with BaggigngClassifier, if ensemble is None.
        clf must implementing the methods 'fit', 'predict'(for vote entropy) and 'predict_proba'(for KL divergence).
    ensemble : sklearn.ensemble, default=None
        sklear.ensemble used as committee. If None, baggingClassifier is used.
    method : string, default='KL_divergence'
        The method to calculate the disagreement. 'vote_entropy' or 'KL_divergence' are possible.
    unlabeled_class : scalar | str | None | np.nan, default=np.nan
        Symbol to represent a missing label. Important: We do not differ between None and np.nan.
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

    def __init__(self, clf, ensemble=None, method='KL_divergence', missing_label=MISSING_LABEL, random_state=None, **kwargs):
        super().__init__(random_state=random_state)

        if method != 'KL_divergence' and method != 'vote_entropy':
            raise ValueError('The method \'' + method + '\' does not exist.')

        if method == 'vote_entropy' and ((getattr(clf, 'fit', None) is None or getattr(clf, 'predict', None) is None)):
            raise TypeError("'clf' must implement the methods 'fit' and 'predict'")
        elif method == 'KL_divergence' and ((getattr(clf, 'fit', None) is None or getattr(clf, 'predict_proba', None) is None)):
            raise TypeError("'clf' must implement the methods 'fit' and 'predict_proba'")

        if not isinstance(clf, BaseEnsemble):
            if ensemble is None:
                warnings.warn('\'ensemble\' is not specified, \'BaggingClassifier\' will be used.')
                ensemble = BaggingClassifier
            ensemble = ensemble(base_estimator=clf, random_state=self.random_state, **kwargs)


        self.missing_label = missing_label
        self.method = method
        self.ensemble = ensemble


    def query(self, X_cand, X, y, return_utilities=False, **kwargs):
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
        return_utilities : bool (default=False)
            If True, the utilities are returned.

        Returns
        -------
        np.ndarray (shape=1)
            The index of the queried instance.
        np.ndarray  (shape=(1xlen(X_cnad))
            The utilities of all instances of X_cand(if return_utilities=True).
        """

        mask_labeled = is_labeled(y,self.missing_label)
        self.ensemble.fit(X[mask_labeled],y[mask_labeled])
        # choose the disagreement method and calculate the utilities
        if self.method == 'KL_divergence':
            utilities = calc_avg_KL_divergence(self.ensemble, X_cand)
        elif self.method == 'vote_entropy':
            utilities = vote_entropy(self.ensemble, X_cand,)

        # best_indices is a np.array (batch_size=1)
        # utilities is a np.array (batch_size=1 x len(X_cand))
        best_indices = rand_argmax([utilities], axis=1, random_state=self.random_state)
        if return_utilities:
            return best_indices, np.array([utilities])
        else:
            return best_indices


def calc_avg_KL_divergence(ensemble, X_cand):
    """
    Calculate the average Kullback-Leibler (KL) divergence for measuring the level of disagreement in QBC.

    Parameters
    ----------
    ensemble: sklearn.ensemble
         fited sklearn.ensemble used as committee.
    X_cand : np.ndarray
        The unlabeled pool from which to choose.

    Returns
    -------
    scores: np.ndarray, shape=(len(X_cand)
        The Kullback-Leibler (KL) divergences.

    References
    ----------
    [1] A. McCallum and K. Nigam. Employing EM in pool-based active learning for
        text classification. In Proceedings of the International Conference on Machine
        Learning (ICML), pages 359-367. Morgan Kaufmann, 1998.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        est_arr = ensemble.estimators_
        P = [est_arr[e_idx].predict_proba(X_cand) for e_idx in range(len(est_arr))]
        P = np.array(P)
        P_com = np.mean(P, axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            scores = np.nansum(np.nansum(P * np.log(P / P_com), axis=2), axis=0)
    scores = scores/ensemble.n_classes_
    return scores


def vote_entropy(ensemble, X_cand):
    """
    Calculate the vote entropy for measuring the level of disagreement in QBC.

    Parameters
    ----------
    ensemble: sklearn.ensemble
         fited sklearn BaggingClassifier used as committee.
    X_cand : np.ndarray
        The unlabeled pool from which to choose.
    classes : list
        All possible classes.

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
                vote_count[i,c] += (votes[i,m] == c)
        
    # cumpute vote entropy
    vote_entropy = np.zeros(len(X_cand))
    for i in range(len(X_cand)):
        for c in range(ensemble.n_classes_):
            if vote_count[i,c]!=0:
                #definition gap at vote_count[i,c]==0:
                a = vote_count[i,c]/len(estimators)
                vote_entropy[i] += a*np.log(a)
    vote_entropy *= -1/np.log(len(estimators))
        
    return vote_entropy