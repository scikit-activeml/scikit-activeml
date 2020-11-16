from sklearn.utils import check_array

from ..base import SingleAnnotPoolBasedQueryStrategy


class RandomSampler(SingleAnnotPoolBasedQueryStrategy):

    def __init__(self, batch_size=1, random_state=None):
        super().__init__(random_state=random_state)

        self.batch_size = batch_size

    def query(self, X_cand, return_utilities=False, **kwargs):
        X_cand = check_array(X_cand, force_all_finite=False)

        utilities = self.random_state.random_sample(len(X_cand))

        best_indices = utilities.argsort()[-self.batch_size:][::-1]

        if return_utilities:
            return best_indices, utilities
        else:
            return best_indices
