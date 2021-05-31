import warnings

import numpy as np
from sklearn.utils import check_random_state, check_array

from skactiveml.base import MultiAnnotPoolBasedQueryStrategy, \
    SingleAnnotPoolBasedQueryStrategy

from skactiveml.utils import compute_vote_vectors

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
              return_utilities=False, pref_annotators_per_sample=1, **kwargs):
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
            If true, also returns the utilities based on the query strategy.
        pref_annotators_per_sample : int, array-like, optional (default=1)
                                     array-like, shape (k), k <= n_samples
            If pref_annotators_per_sample is an int, the value indicates
            the number of annotators that are preferably assigned to a candidate
            sample, if annotators can still be assigned to the given candidate
            sample.
            If pref_annotators_per_sample is an int array, the values of the
            array are interpreted as follows. The value at the i-th index
            determines the preferred number of annotators for the candidate
            sample at the i-th index in the ranking of the batch. The last index
            of the pref_annotators_per_sample array (k-1) indicates the
            preferred number of annotators for all candidate sample at an index
            greater of equal than k-1.

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

        batch_size_sq = min(batch_size, X_cand.shape[0])

        if type(pref_annotators_per_sample) in [int, np.int_]:
            pref_n_annotators = pref_annotators_per_sample * \
                                np.ones(batch_size_sq)
        else:
            pref_n_annotators = check_array(pref_annotators_per_sample,
                                            ensure_2d=False)

            if pref_n_annotators.ndim != 1:
                raise ValueError(
                    "pref_annotators_per_sample, if an array, must be of dim"
                    f"1 but, it is of dim {pref_n_annotators.dim}"
                )
            else:
                pref_length = pref_n_annotators.shape[0]
                if pref_length > batch_size_sq:
                    pref_n_annotators = pref_n_annotators[:batch_size_sq]

                if pref_length < batch_size_sq:
                    appended = pref_n_annotators[-1] * \
                        np.ones(batch_size_sq - pref_length)

                    pref_n_annotators = np.append(pref_n_annotators, appended)

        val = self.strategy.query(X_cand, *args, batch_size=batch_size_sq,
                                  return_utilities=True, **kwargs)

        single_query_indices, utilities = val

        return self._query_annotators(A_cand, batch_size, utilities,
                                      return_utilities, pref_n_annotators,
                                      batch_size_sq)

    def _query_annotators(self, A_cand, batch_size, candidate_utilities,
                          return_utilities, pref_n_annotators,
                          batch_size_sq):

        random_state = check_random_state(self.random_state)

        n_annotators = A_cand.shape[1]
        n_samples = A_cand.shape[0]

        re_val = self._get_order_preserving_s_query(A_cand, candidate_utilities,
                                                    batch_size_sq)

        s_indices, s_utilities = re_val

        n_as_annotators = self._n_to_assign_annotators(batch_size, A_cand,
                                                       s_indices,
                                                       pref_n_annotators)

        utilities = np.zeros((batch_size, n_samples, n_annotators))
        query_indices = np.zeros((batch_size, 2), dtype=int)

        batch_index = 0  # actual batch index
        annotator_ps = 0  # current annotators per sample
        sample_index = 0  # sample batch index

        while batch_index < batch_size:
            utilities[batch_index] = s_utilities[sample_index]
            query_indices[batch_index] = rand_argmax(utilities[batch_index],
                                                     random_state=random_state)

            s_utilities[sample_index,
                        query_indices[batch_index, 0],
                        query_indices[batch_index, 1]] = np.nan

            batch_index += 1
            annotator_ps += 1
            if annotator_ps >= n_as_annotators[sample_index]:
                sample_index += 1
                annotator_ps = 0

        if return_utilities:
            return query_indices, utilities
        else:
            return query_indices

    def _get_order_preserving_s_query(self, A_cand, candidate_utilities,
                                      batch_size_sq):

        random_state = check_random_state(self.random_state)

        n_annotators = A_cand.shape[1]
        n_samples = A_cand.shape[0]

        nan_indices = np.argwhere(np.isnan(candidate_utilities))

        candidate_utilities[nan_indices[:, 0], nan_indices[:, 1]] = -np.inf

        # prepare candidate_utilities
        candidate_utilities = rankdata(candidate_utilities, method='dense',
                                       axis=1).astype(float)

        # calculate indices of maximum sample
        indices = np.argmax(candidate_utilities, axis=1)

        candidate_utilities[nan_indices[:, 0], nan_indices[:, 1]] = np.nan

        # prepare annotator_utilities and get annotator indices
        annotator_utilities = random_state.rand(batch_size_sq, n_samples,
                                                n_annotators)

        annotator_utilities[:, A_cand == 0] = np.nan

        # combine utilities by addition
        utilities = candidate_utilities[:, :, np.newaxis] + annotator_utilities

        return indices, utilities

    @staticmethod
    def _n_to_assign_annotators(batch_size, A_cand, s_indices,
                                pref_n_annotators):

        n_max_annotators = np.sum(A_cand, axis=1)
        n_max_chosen_annotators = n_max_annotators[s_indices]
        annot_per_sample = np.minimum(n_max_chosen_annotators,
                                      pref_n_annotators)

        total_annotators = np.sum(annot_per_sample)

        while total_annotators < batch_size:
            annot_per_sample = np.minimum(n_max_chosen_annotators,
                                          annot_per_sample + 1)

            total_annotators = np.sum(annot_per_sample)

        return annot_per_sample
