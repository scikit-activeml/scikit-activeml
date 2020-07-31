from sklearn.utils import check_array

from ..base import PoolBasedQueryStrategy

import numpy as np
import warnings


class UncertaintySampling(PoolBasedQueryStrategy):
    # TODO: @PM: add comments and doc_string (incl paper reference) as in sklearn @PM: if you copy functions
    # from modAL, please make clear where they come from (check with licence, if this is okay) hallos

    def __init__(self, clf, method='entropy', random_state=None):
        super().__init__(random_state=random_state)

        if method is not 'entropy' or method is not 'least_confident' or method is not 'margin_sampling':
            warnings.warn('The method \'' + method + '\' does not exist, \'entropy\' will be used.')
            method = 'entropy'

        self.method = method
        self.clf = clf

    def query(self, X_cand, X, y, return_utilities=False, **kwargs):
        X_cand = check_array(X_cand, force_all_finite=False)

        # TODO: @PM: please use functions outside this class if appropriate
        self.clf.fit(X,y)
        probas = self.clf.predict_proba(X_cand)
        with np.errstate(divide='ignore'):
            if self.method == 'least_confident':
                utilities = -np.max(probas, axis=1)
            elif self.method == 'margin_sampling':
                sort_probas = np.sort(probas)
                utilities = sort_probas[:,-2] - sort_probas[:,-1]
            elif self.method == 'entropy':
                utilities = -np.sum(probas * np.log(probas), axis=1)

        # best_indices is a np.array (batch_size=1)
        # utilities is a np.array (batch_size=1 x len(X_cand)
        best_indices = np.ndarray(np.argmax(utilities))
        if return_utilities:
            return best_indices, utilities
        else:
            return best_indices


