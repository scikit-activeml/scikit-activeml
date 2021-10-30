from inspect import signature, Parameter

import numpy as np
import sklearn.utils.validation
from scipy.stats import rankdata

from ...base import MultiAnnotPoolBasedQueryStrategy, \
    SingleAnnotPoolBasedQueryStrategy

from ...utils import rand_argmax, check_type
from ...utils._aggregation import majority_vote
from ...utils._validation import check_random_state, check_array, check_scalar


class MultiAnnotWrapper(MultiAnnotPoolBasedQueryStrategy):
    """MultiAnnotWrapper

    Implementation of a wrapper class for scikit-learn pool-based active
    learning query strategies with a single annotator such that it transforms
    the query strategy for the single annotator into a query strategy for
    multiple annotators by randomly choosing an annotator and setting the
    labeled matrix to a labeled vector by an aggregation function.

    Parameters
    ----------
    strategy : SingleAnnotPoolBasedQueryStrategy
        An active learning strategy for a single annotator.
    n_annotators : int,
        Sets the number of annotators if `A_cand is None`.
    y_aggregate : callable, default=None
        `y_aggregate` is used, if the given `strategy` depends on y-values as
        labels for samples. These y-values are passed to `query_params_dict`
        when calling `query`.
        `y_aggregate` is in this case used to transform `y` as a matrix of shape
        (n_samples, n_annotators) into a vector of shape (n_samples) during
        the querying process and then passed to the given `strategy`.
        If `y_aggregate is None` and `y` is used in the strategy,
        majority_vote is used as `y_aggregate`.
    random_state : int, RandomState instance, default=None
        Controls the randomness of the estimator.
    n_annotators : int, default=None
        Sets the number of annotators if `A_cand` (see query method) is None.
        If `n_annotators` is None, `A_cand` has to be passed as an argument.
    """

    def __init__(self, strategy, n_annotators=None, y_aggregate=None,
                 random_state=None):
        super().__init__(random_state=random_state)
        self.strategy = strategy
        self.n_annotators = n_annotators
        self.y_aggregate = y_aggregate

    def query(self, X_cand, query_params_dict=None, A_cand=None, batch_size=1,
              n_annotators_per_sample=1, A_perf=None, return_utilities=False):

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
            selected for annotating sample `X_cand[i]`. If `A_cand is None`, each
            annotator is assumed to be available for labeling each sample.
            `A_cand` must only be None, if `n_annotators is non None`.
        query_params_dict : dict, default=None
            Dictionary for the parameters of the query method.
            If `dict` contains a value for `y`, it is transformed by
            `y_aggregate`. If `query_params_dict is None` an empty
            dictionary is used.
        batch_size : 'adaptive'|int, optional (default=1)
            The number of samples to be selected in one AL cycle. If 'adaptive'
            is set, the `batch_size` is set to 1.
        A_perf : array-like, shape (n_samples, n_annotators) or
                  (n_annotators,) optional (default=None)
            The performance based ranking of each annotator.
            1.) If `A_perf` is of shape (n_samples, n_annotators) for each
             sample `i` the value-annotators pair `(i, j)` is chosen over
             over the pair `(i, k)` if `A_perf[i, j]` is greater or
             equal to `A_perf[i, k]`.
            2.) If `A_perf` is of shape (n_annotators,) for each sample
            `i` the value-annotators pair `(i, j)` is chosen over
             over the pair `(i, k)` if `A_perf[j]` is greater or
             equal to `A_perf[k]`.
            3.) If `A_perf` is None, the annotators are chosen at random, with
             a different distribution for each sample.
        return_utilities : bool, optional (default=False)
            If true, also returns the utilities based on the query strategy.
        n_annotators_per_sample : int, array-like, optional (default=1)
        array-like, shape (k,), k <= n_samples
            If `n_annotators_per_sample` is an int, the value indicates
            the number of annotators that are preferably assigned to a candidate
            sample, selected by the query_strategy.
            `Preferably` in this case means depending on how many annotators
            can be assigned to a given candidate sample and how many
            annotator-sample pairs should be assigned considering the
            `batch_size`.
            If `n_annotators_per_sample` is an int array, the values of the
            array are interpreted as follows. The value at the i-th index
            determines the preferred number of annotators for the candidate
            sample at the i-th index in the ranking of the batch.
            The ranking of the batch is given by the `strategy`
            (SingleAnnotPoolBasedQueryStrategy). The last index
            of the n_annotators_per_sample array (k-1) indicates the
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
        check_type(self.strategy, SingleAnnotPoolBasedQueryStrategy,
                   'self.strategy')

        # check query_params_dict
        if query_params_dict is None:
            query_params_dict = {}

        check_type(query_params_dict, dict, 'query_params_dict')

        # aggregate y
        if 'y' in query_params_dict:
            y_aggregate = majority_vote if self.y_aggregate is None \
                else self.y_aggregate

            if not callable(y_aggregate):
                raise TypeError(
                    f"`self.y_aggregate` must be callable. "
                    f"`self.y_aggregate` is of type {type(y_aggregate)}"
                )

            # count the number of arguments that have no default value
            n_free_params = len(list(
                filter(lambda x: x.default == Parameter.empty,
                       signature(y_aggregate).parameters.values())
            ))

            if n_free_params != 1:
                raise TypeError(
                    f"The number of free parameters of the callable has to "
                    f"equal one. "
                    f"The number of free parameters is {n_free_params}."
                )

            query_params_dict['y'] = y_aggregate(query_params_dict['y'])

        # set up X_cand and batch_size for the single annotator query
        a_indices = np.argwhere(np.any(A_cand, axis=1)).flatten()

        n_samples = X_cand.shape[0]
        n_annotators = A_cand.shape[1]

        X_cand_sq = X_cand[a_indices]
        batch_size_sq = min(batch_size, X_cand_sq.shape[0])

        # check n_annotators_per_sample and set pref_n_annotators
        if isinstance(n_annotators_per_sample, (int, np.int_)):
            check_scalar(n_annotators_per_sample,
                         name='n_annotators_per_sample',
                         target_type=int, min_val=1)
            pref_n_annotators = n_annotators_per_sample * \
                                np.ones(batch_size_sq)
        elif sklearn.utils.validation._is_arraylike(n_annotators_per_sample):
            pref_n_annotators = check_array(n_annotators_per_sample,
                                            ensure_2d=False)

            if pref_n_annotators.ndim != 1:
                raise ValueError(
                    "n_annotators_per_sample, if an array, must be of dim "
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
            raise TypeError("n_annotators_per_sample must be array like "
                            "or an integer")

        # check A_perf and set annotator_utilities
        if A_perf is None:
            annotator_utilities = random_state.rand(1, n_samples, n_annotators)\
                .repeat(batch_size_sq, axis=0)
        elif sklearn.utils.validation._is_arraylike(A_perf):
            A_perf = check_array(A_perf, ensure_2d=False)
            # ensure A_perf lies in [0, 1)
            if A_perf.min() != A_perf.max():
                A_perf = 1 / (A_perf.max() - A_perf.min() + 1) \
                         * (A_perf - A_perf.min())
            else:
                A_perf = np.zeros_like(A_perf, dtype=float)

            if A_perf.shape == (n_samples, n_annotators):
                annotator_utilities = A_perf[np.newaxis, :, :]\
                    .repeat(batch_size_sq, axis=0)
            elif A_perf.shape == (n_annotators,):
                annotator_utilities = A_perf[np.newaxis, np.newaxis, :]\
                    .repeat(n_samples, axis=1)\
                    .repeat(batch_size_sq, axis=0)
            else:
                raise ValueError(
                    f"`A_perf` is of shape {A_perf.shape}, but must be of "
                    f"shape ({n_samples}, {n_annotators}) or of shape "
                    f"({n_annotators},)."
                )
        else:
            raise TypeError(
                f"`A_perf` is of type {type(A_perf)}, but must be array like "
                f"or of type None."
            )

        re_val = self.strategy.query(X_cand_sq, **query_params_dict,
                                     batch_size=batch_size_sq,
                                     return_utilities=True)

        single_query_indices, w_utilities = re_val
        sample_utilities = np.nan * np.ones((batch_size_sq, n_samples))
        sample_utilities[:, a_indices] = w_utilities

        return self._query_annotators(A_cand, batch_size, sample_utilities,
                                      annotator_utilities, return_utilities,
                                      pref_n_annotators)

    def _query_annotators(self, A_cand, batch_size, sample_utilities,
                          annotator_utilities, return_utilities,
                          pref_n_annotators):

        random_state = check_random_state(self.random_state)

        n_annotators = A_cand.shape[1]
        n_samples = A_cand.shape[0]

        re_val = self._get_order_preserving_s_query(A_cand, sample_utilities,
                                                    annotator_utilities)

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
                                      annotator_utilities):

        nan_indices = np.argwhere(np.isnan(candidate_utilities))

        candidate_utilities[nan_indices[:, 0], nan_indices[:, 1]] = -np.inf

        # prepare candidate_utilities
        candidate_utilities = rankdata(candidate_utilities, method='ordinal',
                                       axis=1).astype(float)

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

        n_annotator_sample_pairs = np.sum(annot_per_sample)

        while n_annotator_sample_pairs < batch_size:
            annot_per_sample = np.minimum(n_max_chosen_annotators,
                                          annot_per_sample + 1)

            n_annotator_sample_pairs = np.sum(annot_per_sample)

        return annot_per_sample
