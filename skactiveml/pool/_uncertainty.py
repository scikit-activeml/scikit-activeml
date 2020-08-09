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
    method : string (default='entropy')
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
    def __init__(self, clf, method='entropy', random_state=None):
        super().__init__(random_state=random_state)

        if method != 'entropy' and method != 'least_confident' and method != 'margin_sampling':
            warnings.warn('The method \'' + method + '\' does not exist, \'entropy\' will be used.')
            method = 'entropy'

        self.method = method
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

        # best_indices is a np.array (batch_size=1)
        # utilities is a np.array (batch_size=1 x len(X_cand))
        best_indices = rand_argmax([utilities], axis=1, random_state=self.random_state)
        if return_utilities:
            return best_indices, np.array([utilities])
        else:
            return best_indices


