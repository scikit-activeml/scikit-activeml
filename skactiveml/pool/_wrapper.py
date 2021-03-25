import numpy as np
from sklearn.utils import check_random_state

from skactiveml.base import QueryStrategy, MultiAnnotPoolBasedQueryStrategy, \
    SingleAnnotStreamBasedQueryStrategy, SingleAnnotPoolBasedQueryStrategy


class MultiAnnotWrapper(MultiAnnotPoolBasedQueryStrategy):
    """MultiAnnotWrapper

    Implementation of a wrapper class for scikit-learn pool-based active
    learning query strategies with a single annotator such that it transforms
    the query strategy for the single annotator into a query strategy for
    multiple annotators by randomly choosing an annotator and setting the
    labeled matrix to an labeled vector by majority vote.

    Parameters
    ----------
    strategy : SingleAnnotStreamBasedQueryStrategy that is to deal with missing
        labels.
    random_state : int, RandomState instance, default=None
        Controls the randomness of the estimator.
    """

    def __init__(self, strategy, random_state=None):
        super().__init__(random_state=random_state)

        self.strategy = strategy

    def query(self, X_cand, *args, A_cand=None, batch_size=1,
              return_utilities=False, **kwargs):
        """Determines which candidate sample is to be annotated by which
        annotator. The sample is chosen by the given strategy as if one
        unspecified annotator where to annotate the sample.

        Parameters
        ----------
        X_cand : array-like, shape (n_samples, n_features)
            Candidate samples from which the strategy can select.
        A_cand : array-like, shape (n_samples, n_annotators), optional
        (default=None)
            Boolean matrix where `A_cand[i,j] = True` indicates that
            annotator `j` can be selected for annotating sample `X_cand[i]`,
            while `A_cand[i,j] = False` indicates that annotator `j` cannot be
            selected for annotating sample `X_cand[i]`. If A_cand=None, each
            annotator is assumed to be available for labeling each sample.
        batch_size : int, optional (default=1)
            The number of samples to be selected in one AL cycle.
        return_utilities : bool, optional (default=False)
            If true, also return the utilities based on the query strategy.

        Returns
        -------
        query_indices : numpy.ndarray, shape (batch_size, 2)
            The query_indices indicate which candidate sample is to be
            annotated by which annotator, e.g., `query_indices[:, 0]`
            indicates the selected candidate samples and `query_indices[:, 1]`
            indicates the respectively selected annotators.
        utilities: numpy.ndarray, shape (batch_size, n_samples, n_annotators)
            The utilities of all candidate samples w.r.t. to the available
            annotators after each selected sample of the batch, e.g.,
            `utilities[0, :, j]` indicates the utilities used for selecting
            the first sample-annotator pair (with indices `query_indices[0]`).
        """

        # check strategy

        if not isinstance(self.strategy, SingleAnnotPoolBasedQueryStrategy):
            raise ValueError(
                "The given Strategy must be a SingleAnnotStreamBasedQueryStrategy"
            )

        # check

        if not np.all(np.sum(A_cand, axis=1) > 0):
            raise ValueError(
                "Some X-Values are not assignable to an annotator"
            )

        # check random state

        random_state = check_random_state(self.random_state)

        # perform query

        n_samples = X_cand.shape[0]
        n_annotators = A_cand.shape[1]

        if return_utilities:
            single_query_indices, utilities = self.strategy.query(X_cand, *args, batch_size,
                                                                  return_utilities, **kwargs)
        else:
            single_query_indices = self.strategy.query(X_cand, *args, batch_size,
                                                       return_utilities, **kwargs)

        # get available random query annotators

        cand_off_indices = np.repeat(np.arange(1, n_annotators + 1)
                                     .reshape(1, n_annotators), batch_size, axis=0)

        print("cand_off_indices", cand_off_indices)
        print("A_cand", A_cand)
        print("single_query_indices", single_query_indices)

        avail_cand_off_indices = A_cand[single_query_indices] * cand_off_indices
        print("avail_cand_off_indices", avail_cand_off_indices)
        sorted_avail_cand_off_indices = -np.sort(-avail_cand_off_indices)
        print("sorted_avail_cand_off_indices", sorted_avail_cand_off_indices)
        num_avail_cand = np.sum(A_cand[single_query_indices], axis=1)
        print("len_array", num_avail_cand)
        distribution = num_avail_cand * random_state.rand(batch_size)
        print("dist_array", distribution)
        choose_array = distribution.astype(int)
        print("choose_array", choose_array)
        query_cand_off_indices = sorted_avail_cand_off_indices[np.arange(batch_size), choose_array]
        print("query_cand_off_indices", query_cand_off_indices)
        query_cand_indices = query_cand_off_indices - 1

        query_indices = np.concatenate((single_query_indices.reshape(batch_size, 1),
                                        query_cand_indices.reshape(batch_size, 1)), axis=0)

        print("query_indices", query_indices)

        if return_utilities:
            return query_indices, utilities

        else:
            return query_indices

