import warnings

import numpy as np
from sklearn.utils import check_random_state

from skactiveml.base import MultiAnnotPoolBasedQueryStrategy, \
    SingleAnnotPoolBasedQueryStrategy

from skactiveml.utils import compute_vote_vectors


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
        unspecified annotator where to annotate the sample and the
        annotators are chosen at random.

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
        args : if (X, y) is contained by args. y can be a matrix or a vector
            of assigned values. If y is a matrix the entries are interpreted
            as follows: y[i,j] = k indicates that for the i-th sample the
            j-th candidate annotator annotated the value k.
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
            raise TypeError(
                "The given Strategy must be a "
                "SingleAnnotStreamBasedQueryStrategy"
            )

        # check X-Values are assignable

        if not np.all(np.sum(A_cand, axis=1) > 0):
            raise ValueError(
                "Some X-Values are not assignable to an annotator"
            )

        # check if A_cand number of samples equals X_cand number of samples

        if A_cand.shape[0] != X_cand.shape[0]:
            raise ValueError(
                "A_cand.shape[0] has to equal X_cand.shape[0]"
                "A_cand.shape[0] equals: " + A_cand.shape[0] +
                "X_cand.shpae[0] equals: " + X_cand.shape[0]
            )

        # check random state

        random_state = check_random_state(self.random_state)

        # perform query

        n_samples = X_cand.shape[0]

        # check if batch size is smaller than candidate matrix

        if n_samples < batch_size:
            warnings.warn(
                "batch_size was greater than number of samples."
                "batch_size: " + batch_size + "samples: " + n_samples +
                "batch_size is reduced to number of samples."
            )
            batch_size = n_samples

        # check args

        if len(args) > 1:
            X, y = args[0], args[1]

            if isinstance(y, np.ndarray) and isinstance(y, np.ndarray):
                if y.ndim > 2:
                    raise ValueError(
                        "The entries of args[0] and args[1] are interpreted as "
                        "follows: args[0] is the labeling pool and args[1] is "
                        "either the labels-vector of the labeling pool or the "
                        "labels-annotator-matrix, where the entry at the i-th "
                        "row and the j-th column, says that, the j-th "
                        "annotator labeled the i-th data point with the given "
                        "entry. Thereby the dimension of y has to be smaller "
                        "or equal than 2."
                    )
                if y.ndim == 2:
                    if y.shape[1] != A_cand.shape[1]:
                        warnings.warn(
                            "The number of input annotators does not match"
                            "the number of output annotators."
                        )
                    else:
                        args_list = list(args)
                        vote_matrix = compute_vote_vectors(y)
                        vote_vector = vote_matrix.argmax(axis=1)
                        args_list[1] = vote_vector
                        args = tuple(args_list)

        re_val = self.strategy.query(X_cand, *args, batch_size=batch_size,
                                     return_utilities=True, **kwargs)
        single_query_indices, utilities = re_val

        return self._query_available_annotators(A_cand, batch_size, utilities,
                                                single_query_indices,
                                                return_utilities)

    def _query_available_annotators(self, A_cand, batch_size,
                                    candidate_utilities, single_query_indices,
                                    return_utilities):

        random_state = check_random_state(self.random_state)

        n_annotators = A_cand.shape[1]
        n_samples = A_cand.shape[0]

        # prepare candidate_utilities
        candidate_utilities = candidate_utilities\
            .reshape((batch_size, n_samples, 1)).repeat(n_annotators, axis=2)

        # prepare annotator_utilities and get annotator indices
        annotator_utilities = random_state.rand(batch_size, n_annotators)
        u_a_indices = np.where(np.logical_not(A_cand[single_query_indices]))
        annotator_utilities[u_a_indices] = -1

        query_annotator_indices = annotator_utilities.argmax(axis=1)

        annotator_utilities[np.where(annotator_utilities < 0)] = np.nan
        annotator_utilities = annotator_utilities\
            .reshape(batch_size, 1, n_annotators).repeat(n_samples, axis=1)

        # combine utilities by multiplication
        utilties = candidate_utilities * annotator_utilities

        # get candidate annotator pair indices
        query_indices = np.concatenate((single_query_indices
                                        .reshape(batch_size, 1),
                                        query_annotator_indices
                                        .reshape(batch_size, 1)),
                                       axis=1)

        if return_utilities:
            return query_indices, utilties
        else:
            return query_indices
