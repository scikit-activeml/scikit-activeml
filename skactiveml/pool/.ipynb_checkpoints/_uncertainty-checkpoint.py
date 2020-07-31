from sklearn.utils import check_array

from ..base import PoolBasedQueryStrategy

class UncertaintySampling(PoolBasedQueryStrategy):
    # TODO: @PM: add comments and doc_string (incl paper reference) as in sklearn @PM: if you copy functions
    #  from modAL, please make clear where they come from (check with licence, if this is okay)

    def __init__(self, clf, random_state=None):
        super().__init__(random_state=random_state)

        # TODO: @PM: add all necessary parameters
        self.clf = clf

    def query(self, X_cand, X, y, return_utilities=False, **kwargs):
        X_cand = check_array(X_cand, force_all_finite=False)

        # TODO: @PM: complete
        # TODO: @PM: please use functions outside this class if appropriate

        # best_indices is a np.array (batch_size=1)
        # utilities is a np.array (batch_size=1 x len(X_cand)
        if return_utilities:
            return best_indices, utilities
        else:
            return best_indices
