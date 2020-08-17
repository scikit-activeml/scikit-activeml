from sklearn.utils import check_array

from ..base import PoolBasedQueryStrategy
from ..utils import rand_argmax, is_labeled

import numpy as np
import warnings


class UncertaintySampling(PoolBasedQueryStrategy):
    """
    Uncertainty Sampling query stratagy.

    Parameters
    ----------
    clf : sklearn classifier
        A probabilistic sklearn classifier.
    method : string (default='margin_sampling')
        The method to calculate the uncertainty, entropy, least_confident, margin_sampling and expected_average_precision are possible.
    random_state : numeric | np.random.RandomState
        The random state to use.

    Attributes
    ----------
    random_state: numeric | np.random.RandomState
        Random state to use.
    method : string
        The method to calculate the uncertainty. entropy, least_confident, margin_sampling and expected_average_precisionare possible.
    clf : sklearn classifier
        A probabilistic sklearn classifier.
    classes : array-like, shape=(n_classes)
        Holds the label for each class.

    Methods
    -------
    query(X_cand, X, y, return_utilities=False, **kwargs)
        Queries the next instance to be labeled.

    Refereces
    ---------
    [1] Settles, Burr. Active learning literature survey.
        University of Wisconsin-Madison Department of Computer Sciences, 2009.
        http://www.burrsettles.com/pub/settles.activelearning.pdf
    [2] Wang, Hanmo, et al. "Uncertainty sampling for action recognition
        via maximizing expected average precision."
        IJCAI International Joint Conference on Artificial Intelligence. 2018.
    """
    def __init__(self, clf, classes=None, method='margin_sampling', unlabeled_class=np.nan, random_state=None):
        super().__init__(random_state=random_state)

        if method != 'entropy' and method != 'least_confident' and method != 'margin_sampling' and method != 'expected_average_precision':
            warnings.warn('The method \'' + method + '\' does not exist, \'margin_sampling\' will be used.')
            method = 'margin_sampling'

        if method == 'expected_average_precision' and classes is None:
            raise ValueError('\'classes\' has to be specified')

        self.unlabeled_class = unlabeled_class
        self.method = method
        self.classes = classes
        self.clf = clf

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
        return_utilities : bool (default=False
            If True, the utilities are returned.

        Returns
        -------
        np.ndarray (shape=1)
            The index of the queried instance.
        np.ndarray  (shape=(1xlen(X_cnad))
            The utilities of all instances of X_cand(if return_utilities=True).
        """

        # check X_cand to be a non-empty 2D array containing only finite values.
        X_cand = check_array(X_cand, force_all_finite=False)

        # fit the classifier and get the probabilities
        mask_labeled = is_labeled(y)
        self.clf.fit(X[mask_labeled], y[mask_labeled])
        probas = self.clf.predict_proba(X_cand)

        # caculate the utilities
        with np.errstate(divide='ignore'):
            if self.method == 'least_confident':
                utilities = -np.max(probas, axis=1)
            elif self.method == 'margin_sampling':
                sort_probas = np.sort(probas, axis=1)
                utilities = sort_probas[:,-2] - sort_probas[:,-1]
            elif self.method == 'entropy':
                utilities = -np.sum(probas * np.log(probas), axis=1)
            elif self.method == 'expected_average_precision':
                utilities = expected_average_precision(X_cand, self.classes, probas)

        # best_indices is a np.array (batch_size=1)
        # utilities is a np.array (batch_size=1 x len(X_cand))
        best_indices = rand_argmax([utilities], axis=1, random_state=self.random_state)
        if return_utilities:
            return best_indices, np.array([utilities])
        else:
            return best_indices


def expected_average_precision(X_cand, classes, proba):
    """
    Calculate the expected average precision.

    Parameters
    ----------
    X_cand : np.ndarray
        The unlabeled pool from which to choose.
    classes : array-like, shape=(n_classes)
        Holds the label for each class.
    proba : np.ndarray, shape=(n_X_cand, n_classes)
        The probabilities for each classes and all instance in X_cand.

    Returns
    -------
    score : np.ndarray, shape=(n_X_cand)
        The expected average precision score of all instances in X_cand.
    """
    score = np.zeros(len(X_cand))
    for i in range(len(classes)):
        for j, x in enumerate(X_cand):
            # The i-th column of p without p[j,i]
            p = proba[:,i]
            p = np.delete(p,[j])
            # Sort p in descending order
            p = np.flipud(np.sort(p, axis=0))
            
            # calculate g_arr
            g_arr = np.zeros((len(p),len(p)))
            for n in range(len(p)):
                for h in range(n+1):
                    g_arr[n,h] = g(n, h, p, g_arr)
            
            # calculate f_arr
            f_arr = np.zeros((len(p)+1,len(p)+1))
            for a in range(len(p)+1):
                for b in range(a+1):
                    f_arr[a,b] = f(a, b, p, f_arr, g_arr)
            
            # calculate score
            for t in range(len(p)):
                score[j] += f_arr[len(p),t+1]/(t+1)
                
    return score


def g(n,t,p,g_arr):

    if t>n or (t==0 and n>0):
        return 0
    if t==0 and n==0:
        return 1
    return p[n]*g_arr[n-1,t-1] + (1-p[n])*g_arr[n-1,t]


def f(n,t,p,f_arr,g_arr):
    if t>n or (t==0 and n>0):
        return 0
    if t==0 and n==0:
        return 1
    return p[n-1]*f_arr[n-1,t-1] + p[n-1]*t*g_arr[n-1,t-1]/n + (1-p[n-1])*f_arr[n-1,t]
