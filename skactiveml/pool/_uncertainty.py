from sklearn.utils import check_array

from ..base import PoolBasedQueryStrategy
from ..utils import rand_argmax

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
        The method to calculate the uncertainty, entropy, least confident and margin_sampling are possible.
    random_state : numeric | np.random.RandomState
        The random state to use.

    Attributes
    ----------
    random_state: numeric | np.random.RandomState
        Random state to use.
    method : string
        The method to calculate the uncertainty. entropy, least confident and margin_sampling are possible.
    clf : sklearn classifier
        A probabilistic sklearn classifier.
    classes : array-like, shape=(n_classes)
        Holds the label for each class.

    Methods
    -------
    query(X_cand, X, y, return_utilities=False, **kwargs)
        Represent the photo in the given colorspace.

    Refereces
    ---------
    [1] Settles, Burr. Active learning literature survey.
        University of Wisconsin-Madison Department of Computer Sciences, 2009.
        http://www.burrsettles.com/pub/settles.activelearning.pdf
    """
    def __init__(self, clf, classes=None, method='margin_sampling', random_state=None):
        super().__init__(random_state=random_state)

        if method != 'entropy' and method != 'least_confident' and method != 'margin_sampling' and method != 'expected_average_precision':
            warnings.warn('The method \'' + method + '\' does not exist, \'margin_sampling\' will be used.')
            method = 'margin_sampling'

        if method == 'expected_average_precision' and classes is None:
            raise ValueError('\'classes\' has to be specified')

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
        self.clf.fit(X,y)
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
    score = np.zeros(len(X_cand))
    for i, c in enumerate(classes):
        for j, x in enumerate(X_cand):
            # The i-th column of p without p[j,i]
            p = proba[:,i]
            p = np.delete(p,[j])
            # Sort p in descending order
            p = np.flipud(np.sort(p, axis=0))

            g_arr = np.zeros((len(p),len(p)))
            for n in range(len(p)):
                for t in range(n):
                    g_arr[n,t] = g(n,t,p)
                    print(n,t)
            print(g_arr)
            for t in range(n):
                print(t)
                score[j] += f(len(p),t+1,p,g_arr)/(t+1)


def g(n,t,p):
    if t>n or (t==0 and n>0):
        return 0
    if t==0 and n==0:
        return 1
    return p[n]*g(n-1,t-1,p) + (1-p[n])*g(n-1,t,p)


def f(n,t,p,g_arr):
    if t>n or (t==0 and n>0):
        return 0
    if t==0 and n==0:
        return 1
    return p[n]*f(n-1,t-1,p) + p[n]*t*g_arr[n-1,t-1]/n + (1-p[n])*f(n-1,t,p)