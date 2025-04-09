from inspect import signature, Parameter

import numpy as np
from scipy.stats import rankdata
from sklearn.utils.validation import check_array, _is_arraylike

from ...base import (
    MultiAnnotatorPoolQueryStrategy,
    SingleAnnotatorPoolQueryStrategy,
)
from ...utils import (
    rand_argmax,
    check_type,
    MISSING_LABEL,
    majority_vote,
    check_random_state,
    check_scalar,
)


class SingleAnnotatorWrapper(MultiAnnotatorPoolQueryStrategy):
    """Single Annotator Wrapper

    Implementation of a wrapper class for pool-based active
    learning query strategies with a single annotator such that it transforms
    the query strategy for the single annotator into a query strategy for
    multiple annotators by choosing an annotator randomly or according to the
    parameter `A_perf` and setting the labeled matrix to a labeled vector by an
    aggregation function, e.g., majority voting.

    Parameters
    ----------
    strategy : SingleAnnotatorPoolQueryStrategy
        An active learning strategy for a single annotator.
    y_aggregate : callable, default=None
        `y_aggregate` is used to transform `y` as a matrix of shape
        `(n_samples, n_annotators)` into a vector of shape `(n_samples,)`
        during the querying process and is then passed to the given
        `strategy`. If `y_aggregate is None` and `y` is used in the strategy,
        `majority_vote` is used as `y_aggregate`.
    missing_label : scalar or string or np.nan or None, default=np.nan
        Value to represent a missing label.
    random_state : int or RandomState instance, default=None
        Controls the randomness of the estimator.
    """

    def __init__(
        self,
        strategy,
        y_aggregate=None,
        missing_label=MISSING_LABEL,
        random_state=None,
    ):
        super().__init__(
            random_state=random_state, missing_label=missing_label
        )
        self.strategy = strategy
        self.y_aggregate = y_aggregate

    def query(
        self,
        X,
        y,
        candidates=None,
        annotators=None,
        batch_size=1,
        n_annotators_per_sample=1,
        A_perf=None,
        return_utilities=False,
        **query_kwargs,
    ):
        """Determines which candidate sample is to be annotated by which
        annotator. The samples are first and primarily ranked by the given
        strategy as if one unspecified annotator where to annotate the sample.
        Then for each sample the sample-annotator pairs are ranked based either
        on previously set preferences or at random.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data set, usually complete, i.e., including the labeled
            and unlabeled samples.
        y : array-like of shape (n_samples, n_annotators)
            Labels of the training data set for each annotator (possibly
            including unlabeled ones indicated by self.MISSING_LABEL), meaning
            that `y[i, j]` contains the label annotated by annotator `i` for
            sample `j`.
        candidates : None or array-like of shape (n_candidates), dtype=int or\
                array-like of shape (n_candidates, n_features), default=None
            See parameter `annotators`.
        annotators : None or array-like of shape (n_avl_annotators), dtype=int\
                or array-like of shape (n_candidates, n_annotators),\
                default=None
            - If candidate samples and annotators are not specified, i.e.,
              `candidates=None`, `annotators=None` the unlabeled target values,
              `y`, are the candidates annotator-sample-pairs.
            - If candidate samples and available annotators are specified:
              The annotator-sample-pairs, for which the sample is a candidate
              sample and the annotator is an available annotator are considered
              as candidate annotator-sample-pairs.
            - If `candidates` is None, all samples of `X` are considered as
              candidate samples. In this case `n_candidates` equals `len(X)`.
            - If `candidates` is of shape `(n_candidates,)` and of type int,
              `candidates` is considered as the indices of the sample
              candidates in `(X, y)`.
            - If `candidates` is of shape (n_candidates, n_features), the
              sample candidates are directly given in `candidates` (not
              necessarily contained in `X`). This is not supported by all query
              strategies.
            - If `annotators` is `None`, all annotators are considered as
              available annotators.
            - If `annotators` is of shape (n_avl_annotators), and of type int,
              `annotators` is considered as the indices of the available
              annotators.
            - If `annotators` is a boolean array of shape `(n_candidates,
              n_annotators)` the annotator-sample-pairs, for which the sample
              is a candidate sample and the boolean matrix has entry `True` are
              considered as candidate annotator-sample pairs.
        batch_size : int, default=1
            The number of annotators sample pairs to be selected in one AL
            cycle.
        A_perf : array-like, shape (n_annotators,) or\
        (n_candidates, n_annotators), default=None
            The performance based ranking of each annotator.

            - 1.) If `A_perf` is of shape (n_candidates, n_annotators) for each
              sample `i` the value-annotators pair `(i, j)` is chosen
              over the pair `(i, k)` if `A_perf[i, j]` is greater or
              equal to `A_perf[i, k]`.
            - 2.) If `A_perf` is of shape (n_annotators,) for each sample
              `i` the value-annotators pair `(i, j)` is chosen over
              the pair `(i, k)` if `A_perf[j]` is greater or
              equal to `A_perf[k]`.
            - 3.) If `A_perf` is None, the annotators are chosen at random,
              with a different distribution for each sample.
        return_utilities : bool, default=False
            If `True`, also returns the utilities based on the query strategy.
        n_annotators_per_sample : int or array-like, default=1
            - If `n_annotators_per_sample` is an int, the value indicates
              the number of annotators that are preferably assigned to a
              candidate sample, selected by the query_strategy.
              `Preferably` in this case means depending on how many annotators
              can be assigned to a given candidate sample and how many
              annotator-sample-pairs should be assigned considering the
              `batch_size`.
            - If `n_annotators_per_sample` is an int array, the values of the
              array are interpreted as follows. The value at the i-th index
              determines the preferred number of annotators for the candidate
              sample at the i-th index in the ranking of the batch.
              The ranking of the batch is given by the `strategy`
              (`SingleAnnotatorPoolQueryStrategy`). The last index
              of the n_annotators_per_sample array (k-1) indicates the
              preferred number of annotators for all candidate sample at an
              index greater of equal to k-1.
        query_kwargs : dict, optional
            Dictionary for the parameters of the query method besides `X` and
            the transformed `y`.

        Returns
        -------
        query_indices : np.ndarray of shape (batch_size, 2)
            The `query_indices` indicate which candidate sample pairs are to be
            queried is, i.e., which candidate sample is to be annotated by
            which annotator, e.g., `query_indices[:, 0]` indicates the selected
            candidate samples and `query_indices[:, 1]` indicates the
            respectively selected annotators.

            - If `candidates` is `None` or of shape `(n_candidates,)`, the
              indexing of refers to samples in `X`.
            - If `candidates` is of shape `(n_candidates, n_features)`, the
              indexing refers to samples in `candidates`.
        utilities: numpy.ndarray of shape (batch_size, n_samples,\
                n_annotators) or numpy.ndarray of shape (batch_size,\
                n_candidates, n_annotators)
            The utilities of all candidate samples w.r.t. to the available
            annotators after each selected sample of the batch, e.g.,
            `utilities[0, :, j]` indicates the utilities used for selecting
            the first sample-annotator-pair (with indices `query_indices[0]`).

            - If `candidates` is `None` or of shape `(n_candidates,)`, the
              indexing refers to samples in `X`.
            - If `candidates` is of shape `(n_candidates, n_features)`, the
              indexing refers to samples in `candidates`.
        """

        (
            X,
            y,
            candidates,
            annotators,
            batch_size,
            return_utilities,
        ) = super()._validate_data(
            X,
            y,
            candidates,
            annotators,
            batch_size,
            return_utilities,
            reset=True,
        )

        X_cand, mapping, A_cand = self._transform_cand_annot(
            candidates, annotators, X, y
        )

        random_state = self.random_state_

        # check strategy
        check_type(
            self.strategy, "self.strategy", SingleAnnotatorPoolQueryStrategy
        )
        if self.strategy.missing_label != self.missing_label and not (
            np.isnan(self.strategy.missing_label)
            & np.isnan(self.missing_label)
        ):
            raise ValueError(
                f"`self.missing_label` must equal "
                f"`self.strategy.missing_label`, but "
                f"`self.missing_label` equals {self.missing_label} and"
                f"`self.strategy.missing_label` equals "
                f"{self.strategy.missing_label}."
            )

        # aggregate y
        if self.y_aggregate is None:

            def y_aggregate(y):
                return majority_vote(y, random_state=random_state)

        else:
            y_aggregate = self.y_aggregate

        if not callable(y_aggregate):
            raise TypeError(
                f"`self.y_aggregate` must be callable. "
                f"`self.y_aggregate` is of type {type(y_aggregate)}"
            )

        # count the number of arguments that have no default value
        n_free_params = len(
            list(
                filter(
                    lambda x: x.default == Parameter.empty,
                    signature(y_aggregate).parameters.values(),
                )
            )
        )

        if n_free_params != 1:
            raise TypeError(
                f"The number of free parameters of the callable has to "
                f"equal one. "
                f"The number of free parameters is {n_free_params}."
            )

        y_sq = y_aggregate(y)

        n_selectable_candidates = len(X_cand)
        n_candidates = len(candidates) if candidates is not None else len(X)
        n_annotators = y.shape[1]
        n_samples = X.shape[0]

        batch_size_sq = min(batch_size, X_cand.shape[0])

        # check n_annotators_per_sample and set pref_n_annotators
        if isinstance(n_annotators_per_sample, (int, np.int_)):
            check_scalar(
                n_annotators_per_sample,
                name="n_annotators_per_sample",
                target_type=int,
                min_val=1,
            )
            pref_n_annotators = n_annotators_per_sample * np.ones(
                batch_size_sq
            )
        elif _is_arraylike(n_annotators_per_sample):
            pref_n_annotators = check_array(
                n_annotators_per_sample, ensure_2d=False
            )

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
                    appended = pref_n_annotators[-1] * np.ones(
                        batch_size_sq - pref_length
                    )

                    pref_n_annotators = np.append(pref_n_annotators, appended)
        else:
            raise TypeError(
                "n_annotators_per_sample must be array like " "or an integer"
            )

        # check A_perf and set annotator_utilities
        if A_perf is None:
            annotator_utilities = random_state.rand(
                1, n_selectable_candidates, n_annotators
            ).repeat(batch_size_sq, axis=0)
        elif _is_arraylike(A_perf):
            A_perf = check_array(A_perf, ensure_2d=False)
            # ensure A_perf lies in [0, 1)
            if A_perf.min() != A_perf.max():
                A_perf = (
                    1
                    / (A_perf.max() - A_perf.min() + 1)
                    * (A_perf - A_perf.min())
                )
            else:
                A_perf = np.zeros_like(A_perf, dtype=float)

            if A_perf.shape == (n_candidates, n_annotators):
                annotator_utilities = A_perf[np.newaxis, :, :].repeat(
                    batch_size_sq, axis=0
                )
                if candidates is None:
                    annotator_utilities = annotator_utilities[:, mapping, :]
            elif A_perf.shape == (n_annotators,):
                annotator_utilities = (
                    A_perf[np.newaxis, np.newaxis, :]
                    .repeat(n_selectable_candidates, axis=1)
                    .repeat(batch_size_sq, axis=0)
                )
            else:
                raise ValueError(
                    f"`A_perf` is of shape {A_perf.shape}, but must be of "
                    f"shape ({n_selectable_candidates}, {n_annotators}) or of "
                    f"shape ({n_annotators},)."
                )
        else:
            raise TypeError(
                f"`A_perf` is of type {type(A_perf)}, but must be array like "
                f"or of type None."
            )

        candidates_sq = mapping if mapping is not None else X_cand
        qs_indices, w_utilities = self.strategy.query(
            X=X,
            y=y_sq,
            candidates=candidates_sq,
            **query_kwargs,
            batch_size=batch_size_sq,
            return_utilities=True,
        )

        if mapping is None:
            sample_utilities = w_utilities
            sample_indices = qs_indices
        else:
            sample_utilities = w_utilities[:, mapping]
            sample_indices = np.array(
                [np.argwhere(mapping == i)[0, 0] for i in qs_indices]
            )

        re_val = self._query_annotators(
            A_cand,
            batch_size,
            sample_utilities,
            annotator_utilities,
            return_utilities,
            pref_n_annotators,
            sample_indices,
        )

        if mapping is None:
            return re_val
        elif return_utilities:
            w_indices, w_utilities = re_val
            utilities = np.full((batch_size, n_samples, n_annotators), np.nan)
            utilities[:, mapping, :] = w_utilities
            indices = np.zeros_like(w_indices)
            indices[:, 0] = mapping[w_indices[:, 0]]
            indices[:, 1] = w_indices[:, 1]
            return indices, utilities
        else:
            w_indices = re_val
            indices = np.zeros_like(w_indices)
            indices[:, 0] = mapping[w_indices[:, 0]]
            indices[:, 1] = w_indices[:, 1]
            return indices

    def _query_annotators(
        self,
        A_cand,
        batch_size,
        sample_utilities,
        annotator_utilities,
        return_utilities,
        pref_n_annotators,
        qs_indices,
    ):
        random_state = check_random_state(self.random_state)

        n_annotators = A_cand.shape[1]
        n_samples = A_cand.shape[0]

        s_indices, s_utilities = self._get_order_preserving_s_query(
            A_cand, sample_utilities, annotator_utilities, qs_indices
        )

        n_as_annotators = self._n_to_assign_annotators(
            batch_size, A_cand, s_indices, pref_n_annotators
        )

        utilities = np.zeros((batch_size, n_samples, n_annotators))
        query_indices = np.zeros((batch_size, 2), dtype=int)

        annotator_ps = 0  # current annotators per sample
        sample_index = 0  # sample batch index

        for batch_index in range(batch_size):  # actual batch index
            utilities[batch_index] = s_utilities[sample_index]
            query_indices[batch_index] = rand_argmax(
                utilities[batch_index], random_state=random_state
            )

            s_utilities[
                :, query_indices[batch_index, 0], query_indices[batch_index, 1]
            ] = np.nan

            annotator_ps += 1
            if annotator_ps >= n_as_annotators[sample_index]:
                sample_index += 1
                annotator_ps = 0

        if return_utilities:
            return query_indices, utilities
        else:
            return query_indices

    @staticmethod
    def _get_order_preserving_s_query(
        A, candidate_utilities, annotator_utilities, sample_indices
    ):
        nan_indices = np.argwhere(np.isnan(candidate_utilities))

        candidate_utilities[nan_indices[:, 0], nan_indices[:, 1]] = -np.inf

        # force selected sample indices to have the maximum utility
        for i in range(len(sample_indices)):
            max_utility_i = np.nanmax(candidate_utilities[i]) + 1
            candidate_utilities[i, sample_indices[i]] = max_utility_i

        # prepare candidate_utilities
        candidate_utilities = rankdata(
            candidate_utilities, method="ordinal", axis=1
        ).astype(float)

        candidate_utilities[nan_indices[:, 0], nan_indices[:, 1]] = np.nan

        annotator_utilities[:, ~A] = np.nan

        # combine utilities by addition
        utilities = candidate_utilities[:, :, np.newaxis] + annotator_utilities

        return sample_indices, utilities

    @staticmethod
    def _n_to_assign_annotators(batch_size, A, s_indices, pref_n_annotators):
        n_max_annotators = np.sum(A, axis=1)
        n_max_chosen_annotators = n_max_annotators[s_indices]
        annot_per_sample = np.minimum(
            n_max_chosen_annotators, pref_n_annotators
        )

        n_annotator_sample_pairs = np.sum(annot_per_sample)

        while n_annotator_sample_pairs < batch_size:
            annot_per_sample = np.minimum(
                n_max_chosen_annotators, annot_per_sample + 1
            )

            n_annotator_sample_pairs = np.sum(annot_per_sample)
            if n_annotator_sample_pairs >= batch_size:
                break

        return annot_per_sample
