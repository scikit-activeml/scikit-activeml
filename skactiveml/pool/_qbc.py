import numpy as np
import warnings

from ..base import PoolBasedQueryStrategy

from sklearn.ensemble import BaggingClassifier
from sklearn.utils import check_random_state
from ..utils.selection import rand_argmax


class QBC(PoolBasedQueryStrategy):
    """QBC

    The Query-By-Committee (QBC) algorithm minimizes the version space, which is the set of hypotheses that are
    consistent with the current labeled training data.
    This class implement the query-by-bagging method, which uses the bagging in sklearn to
    construct the committee. So your model should be a sklearn model.

    Parameters
    ----------
    model: model used for committee construction
        Model implementing the methods 'fit' and and 'predict_proba'.
    method: string (default='KL_divergence')
        The method to calculate the disagreement. 'vote_entropy' or 'KL_divergence' are possible.
    n_classes : int
        Number of possible classes.
    random_state: numeric | np.random.RandomState
        Random state to use.

    Attributes
    ----------
    model: model used for committee construction
        Model implementing the methods 'fit' and and 'predict_proba'.
    method: string (default='KL_divergence')
        The method to calculate the disagreement. 'vote_entropy' or 'KL_divergence' are possible.
    n_classes : int
        Number of possible classes.
    random_state: numeric | np.random.RandomState
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

    def __init__(self, model, method='KL_divergence', n_classes=0, random_state=None):
        super().__init__(random_state=random_state)

        if method != 'KL_divergence' and method != 'vote_entropy':
            warnings.warn('The method \'' + method + '\' does not exist, \'KL_divergence\' will be used.')
            method = 'KL_divergence'

        if method == 'vote_entropy' and n_classes < 1:
            raise TypeError('n_classes is not specified.')
        if method == 'vote_entropy' and (getattr(model, 'fit', None) is None or getattr(model, 'predict', None) is None):
            raise TypeError("'model' must implement the methods 'fit' and 'predict'")
        elif (getattr(model, 'fit', None) is None or getattr(model, 'predict_proba', None) is None):
            raise TypeError("'model' must implement the methods 'fit' and 'predict_proba'")

        self.method = method
        self.model = model
        self.n_classes = n_classes


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
        n_detected_classes = len(np.unique(y))
        if n_detected_classes < 2:
            warnings.warn('The number of detected classes is less than 2.')
            utilities = np.zeros((len(X_cand)))
        else:
            # create and train the models
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                n_features = X.shape[1]
                max_features = self.random_state.choice(np.arange(np.ceil(n_features / 2), n_features))
                max_features = int(max_features)
                bagging = BaggingClassifier(base_estimator=self.model, n_estimators=25,
                                            max_features=1.0, random_state=self.random_state).fit(X=X, y=y)
            # choose the disagreement method and calculate the utilities
            if self.method == 'KL_divergence':
                utilities = calc_avg_KL_divergence(bagging=bagging, X=X, y=y, X_cand=X_cand, random_state=self.random_state)
            elif self.method == 'vote_entropy':
                utilities = vote_entropy(bagging, X_cand, X, y, n_classes=self.n_classes)

        # best_indices is a np.array (batch_size=1)
        # utilities is a np.array (batch_size=1 x len(X_cand))
        best_indices = rand_argmax([utilities], axis=1, random_state=self.random_state)
        if return_utilities:
            return best_indices, np.array([utilities])
        else:
            return best_indices


def calc_avg_KL_divergence(bagging:BaggingClassifier, X_cand, X, y, random_state):
    """
    Calculate the average Kullback-Leibler (KL) divergence for measuring the level of disagreement in QBC.

    Parameters
    ----------
    bagging: sklearn BaggingClassifier
         fited sklearn BaggingClassifier used as committee.
    X_cand : np.ndarray
        The unlabeled pool from which to choose.
    X : np.ndarray
        The labeled pool used to fit the classifier.
    y : np.array
        The labels of the labeled pool X.
    random_state: numeric | np.random.RandomState
        Random state to use.

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
    random_state = check_random_state(random_state)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        est_arr = bagging.estimators_
        est_features = bagging.estimators_features_
        P = [est_arr[e_idx].predict_proba(X_cand[:, est_features[e_idx]]) for e_idx in range(len(est_arr))]
        P = np.array(P)
        P_com = np.mean(P, axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            scores = np.nansum(np.nansum(P * np.log(P / P_com), axis=2), axis=0)
    return scores


def vote_entropy(bagging, X_cand, X, y, n_classes):
    """
    Calculate the average Kullback-Leibler (KL) divergence for measuring the level of disagreement in QBC.

    Parameters
    ----------
    bagging: sklearn BaggingClassifier
         fited sklearn BaggingClassifier used as committee.
    X_cand : np.ndarray
        The unlabeled pool from which to choose.
    X : np.ndarray
        The labeled pool used to fit the classifier.
    y : np.array
        The labels of the labeled pool X.
    n_classes : int
        Number of possible classes.

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
    models = bagging.estimators_
    # Let the models vote for unlabeled data
    votes = np.zeros((len(X_cand), len(models)))
    for i, model in enumerate(models):
        votes[:, i] = model.predict(X_cand)
 
    # count the votes
    vote_count = np.zeros((len(X_cand), n_classes))
    for i in range(len(X_cand)):
        for c in range(n_classes):
            for m in range(len(models)):
                vote_count[i,c] += (votes[i,m] == c)
        
    # cumpute vote entropy
    vote_entropy = np.zeros(len(X_cand))
    for i in range(len(X_cand)):
        for c in range(n_classes):
            if vote_count[i,c]!=0:
                #definition gap at vote_count[i,c]==0:
                a = vote_count[i,c]/len(models)
                vote_entropy[i] += a*np.log(a)
    vote_entropy *= -1/np.log(len(models))
        
    return vote_entropy