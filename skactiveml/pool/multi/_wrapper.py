import warnings

import numpy as np
from sklearn.utils import check_random_state, check_array, check_scalar
import sklearn.utils.validation
from scipy.stats import rankdata

from ...base import MultiAnnotPoolBasedQueryStrategy, \
    SingleAnnotPoolBasedQueryStrategy

from ...utils import compute_vote_vectors, rand_argmax


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
    n_annotators : int,
        Sets the number of annotators if no A_cand is None
    """

    def __init__(self, strategy, random_state=None, n_annotators=None):
        super().__init__(random_state=random_state)
        self.strategy = strategy
        self.n_annotators = n_annotators

    def query(self, X_cand, *args, A_cand=None, batch_size=1,
              return_utilities=False, pref_annotators_per_sample=1,
              A_perfs=None, **kwargs):

        """Determines which candidate sample is to be annotated by which
        annotator. The samples are first and primarily ranked by the given
        strategy as if one unspecified annotator where to annotate the sample.
        Then for each sample the sample-annotator pairs are ranked based either
        on previously set preferences or at random.

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
            as follows: `y[i,j] = k` indicates that for the i-th sample the
            j-th candidate annotator annotated the value k.
        batch_size : 'adaptive'|int, optional (default=1)
            The number of samples to be selected in one AL cycle. If 'adaptive'
            is set, the `batch_size` is set to 1.
        A_perfs : array-like, shape (n_samples, n_annotators) or
                  (n_annotators,) optional (default=None)
            The preferred ranking of each annotator.
            1.) If `A_perfs` is of shape (n_samples, n_annotators) for each
             sample `i` the value-annotators pair `(i, j)` is preferably picked
             over the pair `(i, k)` if `A_perfs[i, j]` is greater or
             equal to `A_perfs[i, k]`.
            2.) If `A_perfs` is of shape (n_annotators,) for each sample
            `i` the value-annotators pair `(i, j)` is preferentially picked
             over the pair `(i, k)` if `A_perfs[j]` is greater or
             equal to `A_perfs[k]`.
            3.) If `A_perfs` is None, the annotators are chosen at random, with
             a different distribution for each sample.
        return_utilities : bool, optional (default=False)
            If true, also returns the utilities based on the query strategy.
        pref_annotators_per_sample : int, array-like, optional (default=1)
        array-like, shape (k), k <= n_samples
            If `pref_annotators_per_sample` is an int, the value indicates
            the number of annotators that are preferably assigned to a candidate
            sample, if annotators can still be assigned to the given candidate
            sample.
            If `pref_annotators_per_sample` is an int array, the values of the
            array are interpreted as follows. The value at the i-th index
            determines the preferred number of annotators for the candidate
            sample at the i-th index in the ranking of the batch. The last index
            of the pref_annotators_per_sample array (k-1) indicates the
            preferred number of annotators for all candidate sample at an index
            greater of equal to k-1.

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

        X_cand, A_cand, return_utilities, batch_size, random_state = \
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

            if isinstance(X, np.ndarray) and isinstance(y, np.ndarray):
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
                        warnings.warn(f"y.shape[1] = {y.shape[1]} != "
                                      f"A_cand.shape[1] = {A_cand.shape[1]}")
                    args_list = list(args)
                    vote_matrix = compute_vote_vectors(y)
                    vote_vector = vote_matrix.argmax(axis=1)
                    vote_vector = np.array(vote_vector, dtype=float)
                    vote_vector[vote_matrix.sum(axis=1) == 0] = np.nan
                    args_list[1] = vote_vector
                    args = tuple(args_list)

        # set up X_cand and batch_size for the single annotator query
        a_indices = np.argwhere(np.any(A_cand, axis=1)).flatten()

        n_samples = X_cand.shape[0]
        n_annotators = A_cand.shape[1]

        X_cand_sq = X_cand[a_indices]
        batch_size_sq = min(batch_size, X_cand_sq.shape[0])

        # check pref_annotators_per_sample and set pref_n_annotators
        if isinstance(pref_annotators_per_sample, (int, np.int_)):
            check_scalar(pref_annotators_per_sample,
                         name='pref_annotators_per_sample',
                         target_type=int, min_val=1)
            pref_n_annotators = pref_annotators_per_sample * \
                                np.ones(batch_size_sq)
        elif sklearn.utils.validation._is_arraylike(pref_annotators_per_sample):
            pref_n_annotators = check_array(pref_annotators_per_sample,
                                            ensure_2d=False)

            if pref_n_annotators.ndim != 1:
                raise ValueError(
                    "pref_annotators_per_sample, if an array, must be of dim"
                    f"1 but, it is of dim {pref_n_annotators.ndim}"
                )
            else:
                pref_length = pref_n_annotators.shape[0]
                if pref_length > batch_size_sq:
                    pref_n_annotators = pref_n_annotators[:batch_size_sq]

                if pref_length < batch_size_sq:
                    appended = pref_n_annotators[-1] * \
                               np.ones(batch_size_sq - pref_length)

                    pref_n_annotators = np.append(pref_n_annotators, appended)
        else:
            raise TypeError("pref_annotators_per_sample must be array like"
                            "or an integer")

        # check A_perfs and set annotator_utilities
        if A_perfs is None:
            annotator_utilities = random_state.rand(1, n_samples, n_annotators)\
                .repeat(batch_size_sq, axis=0)
        elif sklearn.utils.validation._is_arraylike(A_perfs):
            A_perfs = check_array(A_perfs, ensure_2d=False)
            # ensure A_perfs lies in [0, 1)
            if A_perfs.min() != A_perfs.max():
                A_perfs = 1/(A_perfs.max()-A_perfs.min()+1)\
                          *(A_perfs - A_perfs.min())
            else:
                A_perfs = np.zeros_like(A_perfs, dtype=float)

            if A_perfs.shape == (n_samples, n_annotators):
                annotator_utilities = A_perfs[np.newaxis, :, :]\
                    .repeat(batch_size_sq, axis=0)
            elif A_perfs.shape == (n_annotators,):
                annotator_utilities = A_perfs[np.newaxis, np.newaxis, :]\
                    .repeat(n_samples, axis=1)\
                    .repeat(batch_size_sq, axis=0)
            else:
                raise ValueError(
                    f"`A_perfs` is of shape {A_perfs.shape}, but must be of "
                    f"shape ({n_samples}, {n_annotators}) or of shape "
                    f"({n_annotators},)."
                )
        else:
            raise TypeError(
                f"`A_perfs` is of type {type(A_perfs)}, but must be array like"
                f"of type None"
            )

        val = self.strategy.query(X_cand_sq, *args, batch_size=batch_size_sq,
                                  return_utilities=True, **kwargs)

        single_query_indices, w_utilities = val
        sample_utilities = np.nan * np.ones((batch_size_sq, n_samples))
        sample_utilities[:, a_indices] = w_utilities
        single_query_indices = a_indices[single_query_indices]

        return self._query_annotators(A_cand, batch_size, sample_utilities,
                                      annotator_utilities, return_utilities,
                                      pref_n_annotators, single_query_indices)

    def _query_annotators(self, A_cand, batch_size, sample_utilities,
                          annotator_utilities, return_utilities,
                          pref_n_annotators, single_query_indices):

        random_state = check_random_state(self.random_state)

        n_annotators = A_cand.shape[1]
        n_samples = A_cand.shape[0]

        re_val = self._get_order_preserving_s_query(A_cand, sample_utilities,
                                                    annotator_utilities,
                                                    single_query_indices)

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

    @staticmethod
    def _get_order_preserving_s_query(A_cand, candidate_utilities,
                                      annotator_utilities,
                                      single_query_indices):

        nan_indices = np.argwhere(np.isnan(candidate_utilities))

        candidate_utilities[nan_indices[:, 0], nan_indices[:, 1]] = -np.inf

        # prepare candidate_utilities
        candidate_utilities = rankdata(candidate_utilities, method='dense',
                                       axis=1).astype(float)
        batch_size_sq = len(single_query_indices)

        # prefer selected sample of the single query strategy
        candidate_utilities[np.arange(batch_size_sq), single_query_indices] += 1

        # calculate indices of maximum sample
        indices = np.argmax(candidate_utilities, axis=1)

        candidate_utilities[nan_indices[:, 0], nan_indices[:, 1]] = np.nan

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
