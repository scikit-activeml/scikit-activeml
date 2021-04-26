import warnings

import numpy as np
from sklearn.utils import check_random_state

from skactiveml.base import MultiAnnotPoolBasedQueryStrategy, \
    SingleAnnotPoolBasedQueryStrategy

from skactiveml.utils import compute_vote_vectors, simple_batch

from scipy.stats import rankdata

from ..utils import rand_argmax


class MultiAnnotWrapper(MultiAnnotPoolBasedQueryStrategy):
    """MultiAnnotWrapper

    Implementation of a wrapper class for scikit-learn pool-based active
    learning query strategies with a single annotator such that it transforms
    the query strategy for the single annotator into a query strategy for
    multiple annotators by randomly choosing an annotator and setting the
    labeled matrix to a labeled vector by majority vote.

    Parameters
    ----------
    strategy : SingleAnnotStreamBasedQueryStrategy
        An active learning strategy for a single annotator.
    random_state : int, RandomState instance, default=None
        Controls the randomness of the estimator.
    """

    def __init__(self, strategy, random_state=None):
        super().__init__(random_state=random_state)
        self.strategy = strategy

    def query(self, X_cand, *args, A_cand=None, batch_size=1,
              return_utilities=False, method='one_per_sample', **kwargs):
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
        args : list
            If (X, y) is contained by args. y can be a matrix or a vector
            of assigned values. If y is a matrix the entries are interpreted
            as follows: y[i,j] = k indicates that for the i-th sample the
            j-th candidate annotator annotated the value k.
        batch_size : int, optional (default=1)
            The number of samples to be selected in one AL cycle.
        return_utilities : bool, optional (default=False)
            If true, also return the utilities based on the query strategy.
        method : {'one_per_sample', 'ra', 'entropy'},
            optional (default='one_per_sample')

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
            `utilities[i, j, k]` indicates the utilities of
            the i-th batch regarding the j-th sample and the k-th annotator.
        """
        super()._validate_data(X_cand, A_cand, return_utilities, batch_size,
                               self.random_state, reset=True)

        # check strategy
        if not isinstance(self.strategy, SingleAnnotPoolBasedQueryStrategy):
            raise TypeError(
                f"The given strategy is of the type `{type(self.strategy)}`, "
                "but it must be a `SingleAnnotStreamBasedQueryStrategy`."
            )

        # check X-Values are assignable
        # if np.all(np.sum(A_cand, axis=1) == 0):
        #     raise ValueError(
        #         "Some X-Values are not assignable to an annotator"
        #     )

        # perform query

        n_samples = X_cand.shape[0]

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

        return self._query_one_annotator_per_sample(A_cand, batch_size,
                                                    utilities, return_utilities)

    def _query_max_annotators_per_sample(self, A_cand, batch_size,
                                         candidate_utilities, return_utilities):

        random_state = check_random_state(self.random_state)

        n_annotators = A_cand.shape[1]
        n_samples = A_cand.shape[0]

        n_av_annotatos = np.sum(A_cand, axis=1)

        s_utilities = self._get_order_preserving_utilities(A_cand,
                                                           candidate_utilities,
                                                           batch_size)

        utilities = np.zeros((batch_size, n_samples, n_annotators))
        query_indices = np.zeros((batch_size, 2), dtype=int)

        b = 0  # actual batch index
        v = 0  # current annotators per sample
        a = 0  # sample batch index
        while b < batch_size:
            utilities[b] = s_utilities[a]
            query_indices[b] = rand_argmax(utilities[b],
                                           random_state=random_state)
            s_utilities[a, query_indices[b, 0], query_indices[b, 1]] = np.nan
            b += 1
            v += 1
            if v >= n_av_annotatos[a]:
                a += 1
                v = 0

        if return_utilities:
            return query_indices, utilities
        else:
            return query_indices

    def _query_annotator_sample_pairs(self, A_cand, batch_size,
                                      candidate_utilities, return_utilities):

        random_state = check_random_state(self.random_state)

        n_annotators = A_cand.shape[1]
        n_samples = A_cand.shape[0]

        candidate_utilities = candidate_utilities[0, :] \
            .reshape((n_samples, 1)) \
            .repeat(n_annotators, axis=1)
        annotator_utilities = random_state.rand(n_annotators) \
            .reshape((n_samples, n_annotators))

        utilities = candidate_utilities * annotator_utilities
        utilities[np.where(np.logical_not(A_cand))] = np.nan

        flat_utilities = utilities.flatten()

        flat_indices, flat_utilities = simple_batch(flat_utilities,
                                                    random_state,
                                                    batch_size=batch_size,
                                                    return_utilities=True)
        flat_indices_0 = flat_indices // n_annotators
        flat_indices_1 = flat_indices % n_annotators

        query_indices = np.concatenate((flat_indices_0.reshape(batch_size, 1),
                                        flat_indices_1.reshape(batch_size, 1)),
                                       axis=1)

        query_utilities = flat_utilities.reshape((batch_size, n_samples,
                                                  n_annotators))

        if return_utilities:
            return query_indices, query_utilities
        else:
            return query_indices

    def _query_one_annotator_per_sample(self, A_cand, batch_size,
                                        candidate_utilities, return_utilities):

        random_state = check_random_state(self.random_state)

        utilities = self._get_order_preserving_utilities(A_cand,
                                                         candidate_utilities,
                                                         batch_size)

        # get candidate annotator pair indices
        query_indices = np.zeros((batch_size, 2), dtype=int)
        for b in range(batch_size):
            query_indices[b] = rand_argmax(utilities[b],
                                           random_state=random_state)

        if return_utilities:
            return query_indices, utilities
        else:
            return query_indices

    def _get_order_preserving_utilities(self, A_cand, candidate_utilities,
                                        batch_size):

        random_state = check_random_state(self.random_state)

        n_annotators = A_cand.shape[1]
        n_samples = A_cand.shape[0]

        nan_indices = np.argwhere(np.isnan(candidate_utilities))

        candidate_utilities[nan_indices[:, 0], nan_indices[:, 1]] = -np.inf

        # prepare candidate_utilities
        candidate_utilities = rankdata(candidate_utilities, method='dense',
                                       axis=1).astype(float)

        candidate_utilities[nan_indices[:, 0], nan_indices[:, 1]] = np.nan

        # prepare annotator_utilities and get annotator indices
        annotator_utilities = random_state.rand(batch_size, n_samples,
                                                n_annotators)

        annotator_utilities[:, A_cand == 0] = np.nan

        # combine utilities by addition
        utilities = candidate_utilities[:, :, np.newaxis] + annotator_utilities

        return utilities

