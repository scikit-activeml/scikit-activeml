from ..base import SingleAnnotatorPoolQueryStrategy
from ..utils import (
    MISSING_LABEL,
    check_random_state,
    is_labeled,
    labeled_indices,
    unlabeled_indices,
    check_scalar,
    simple_batch,
    match_signature,
)
from math import ceil
import numpy as np
from joblib import Parallel, delayed, cpu_count
import warnings


class SubSamplingWrapper(SingleAnnotatorPoolQueryStrategy):
    """Sub-sampling Wrapper

    This class implements a wrapper for single-annotator pool-based strategies
    that randomly sub-samples a set of candidates before computing their
    utilities.

    Parameters
    ----------
    query_strategy : skactiveml.base.SingleAnnotatorPoolQueryStrategy
        The strategy used for computing the utilities of the candidate
        sub-sample.
    max_candidates : int or float, default=0.1
        Determines the number of candidates. If `max_candidates` is an
        integer, `max_candidates` is the maximum number of candidates whose
        utilities are computed. If `max_candidates` is a float,
        `max_candidates` is the fraction of the original number of candidates.
    exclude_non_subsample : bool, default=False
        - If `True`, unlabeled candidates in `X` and `y` are excluded which
          are not part of the subsample.  If `candidates` is an array-like of
          shape `(n_candidates, n_features)`, all unlabeled data will be
          removed from `X` and `y`.
        - If `False`, `X` and `y` stay the same.
    missing_label : scalar or string or np.nan or None, default=np.nan
        Value to represent a missing label.
    random_state : int or np.random.RandomState, default=None
        The random state to use.
    """

    def __init__(
        self,
        query_strategy=None,
        max_candidates=0.1,
        exclude_non_subsample=False,
        missing_label=MISSING_LABEL,
        random_state=None,
    ):
        super().__init__(
            missing_label=missing_label, random_state=random_state
        )
        self.query_strategy = query_strategy
        self.max_candidates = max_candidates
        self.exclude_non_subsample = exclude_non_subsample

    @match_signature("query_strategy", "query")
    def query(
        self,
        X,
        y,
        candidates=None,
        batch_size=1,
        return_utilities=False,
        **query_kwargs,
    ):
        """Determines for which candidate samples labels are to be queried.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data set, usually complete, i.e., including the labeled
            and unlabeled samples.
        y : array-like of shape (n_samples)
            Labels of the training data set (possibly including unlabeled ones
            indicated by self.MISSING_LABEL).
        candidates : None or array-like of shape (n_candidates), dtype=int or
            array-like of shape (n_candidates, n_features), default=None

            - If `candidates` is `None`, the unlabeled samples from `(X,y)` are
              considered as `candidates`.
            - If `candidates` is of shape `(n_candidates,)` and of type
              `int`, `candidates` is considered as the indices of the
              samples in `(X,y)`.
            - If `candidates` is of shape `(n_candidates, *)`, the
              candidate samples are directly given in `candidates` (not
              necessarily contained in `X`). This is not supported by all
              query strategies.
        batch_size : int, default=1
            The number of samples to be selected in one AL cycle.
        return_utilities : bool, default=False
            If `True`, also return the utilities based on the query strategy.
        **query_kwargs : dict-like
            Further keyword arguments are passed to the `query` method of the
            `query_strategy` object.

        Returns
        -------
        query_indices : numpy.ndarray of shape (batch_size,)
            The query indices indicate for which candidate sample a label is
            to be queried, e.g., `query_indices[0]` indicates the first
            selected sample.

            - If `candidates` is `None` or of shape
              `(n_candidates,)`, the indexing refers to the samples in
              `X`.
            - If `candidates` is of shape `(n_candidates, n_features)`,
              the indexing refers to the samples in `candidates`.
        utilities : numpy.ndarray of shape (batch_size, n_samples) or \
                numpy.ndarray of shape (batch_size, n_candidates)
            The utilities of samples after each selected sample of the batch,
            e.g., `utilities[0]` indicates the utilities used for selecting
            the first sample (with index `query_indices[0]`) of the batch.
            Utilities for labeled samples will be set to np.nan.

            - If `candidates` is `None` or of shape
              `(n_candidates,)`, the indexing refers to the samples in
              `X`.
            - If `candidates` is of shape `(n_candidates, n_features)`,
              the indexing refers to the samples in `candidates`.
        """

        X, y, candidates, batch_size, return_utilities = self._validate_data(
            X, y, candidates, batch_size, return_utilities, reset=True
        )
        if not isinstance(
            self.query_strategy, SingleAnnotatorPoolQueryStrategy
        ):
            raise TypeError(
                f"`query_strategy` is of type `{type(self.query_strategy)}` "
                f"but must be of type `SingleAnnotatorPoolQueryStrategy`."
            )
        check_scalar(self.exclude_non_subsample, "exclude_non_subsample", bool)
        seed_multiplier = (
            int(is_labeled(y, missing_label=self.missing_label_).sum()) + 1
        )
        max_candidates = self.max_candidates
        if isinstance(self.max_candidates, int):
            check_scalar(
                self.max_candidates,
                name="max_candidates",
                target_type=int,
                min_inclusive=True,
                min_val=1,
            )
        elif isinstance(self.max_candidates, float):
            check_scalar(
                self.max_candidates,
                name="max_candidates",
                target_type=float,
                min_inclusive=False,
                max_inclusive=True,
                min_val=0.0,
                max_val=1.0,
            )
        else:
            raise TypeError(
                f"`max_candidates` is of type `{type(self.max_candidates)}`"
                f" but must be in `[int, float]`."
            )
        random_state = check_random_state(self.random_state, seed_multiplier)

        # subsampling with no explicit provided candidates
        if candidates is None:
            candidate_indices = unlabeled_indices(
                y=y, missing_label=self.missing_label_
            )
            # transform max_candidates to int if a ratio is given
            if isinstance(max_candidates, float):
                max_candidates = ceil(
                    len(candidate_indices) * self.max_candidates
                )
            max_candidates = min(max_candidates, len(candidate_indices))
            # subsample new candidates
            new_candidates = random_state.choice(
                a=candidate_indices, size=max_candidates, replace=False
            )
        # subsampling with privided explicit candidates
        else:
            # transform max_candidates to int if a ratio is given
            if isinstance(max_candidates, float):
                max_candidates = ceil(len(candidates) * self.max_candidates)
            max_candidates = min(max_candidates, len(candidates))
            if candidates.ndim == 1:
                candidate_indices = candidates
                # subsample new candidates
                new_candidates = random_state.choice(
                    a=candidates, size=max_candidates, replace=False
                )
            else:
                candidate_indices = range(len(candidates))
                # subsample new candidates
                new_candidate_indices = random_state.choice(
                    a=candidate_indices, size=max_candidates, replace=False
                )
                new_candidates = candidates[new_candidate_indices]

        # check if to exclude unlabeled non-candidate training data
        if self.exclude_non_subsample:
            all_labeled = labeled_indices(
                y=y, missing_label=self.missing_label_
            )
            if candidates is not None and candidates.ndim > 1:
                subset_and_labeled_indices = all_labeled
            else:
                subset_and_labeled_indices = np.concatenate(
                    [all_labeled, new_candidates]
                )
            subset_and_labeled_indices = np.sort(subset_and_labeled_indices)

            new_X = X[subset_and_labeled_indices]
            new_y = y[subset_and_labeled_indices]
            # for explicitely provided candidates recalculate candidate indices
            # that are passed to the wrapped query strategy
            if candidates is None or candidates.ndim == 1:
                new_candidates = unlabeled_indices(
                    y=new_y, missing_label=self.missing_label_
                )
        else:
            new_X = X
            new_y = y

        qs_output = self.query_strategy.query(
            X=new_X,
            y=new_y,
            candidates=new_candidates,
            batch_size=batch_size,
            return_utilities=return_utilities,
            **query_kwargs,
        )

        # unpack result of query strategy if needed
        queried_indices = qs_output
        utilities = None
        if return_utilities:
            queried_indices, utilities = qs_output

        # retransform queried indices and utilities as if no training data was
        # removed
        if self.exclude_non_subsample and (
            candidates is None or candidates.ndim == 1
        ):
            # transform to original candidate indices
            queried_indices = subset_and_labeled_indices[queried_indices]
            # transform to original utilities shape
            if utilities is not None:
                new_utilities = np.full(
                    shape=(batch_size, len(X)), fill_value=np.nan
                )
                transformed_new_candidates = subset_and_labeled_indices[
                    new_candidates
                ]
                new_utilities[:, transformed_new_candidates] = utilities[
                    :, new_candidates
                ]
                utilities = new_utilities
                new_candidates = transformed_new_candidates

        # transform indices if candidates was provided in the shape of
        # (n_candidates, n_features)
        if candidates is not None and candidates.ndim > 1:
            new_queried_indices = new_candidate_indices[queried_indices]
        else:
            new_queried_indices = queried_indices

        # transform utilities from subsampled shape to original utilities shape
        if return_utilities:
            if candidates is None or candidates.ndim == 1:
                new_utilities = np.full(
                    shape=(batch_size, len(X)), fill_value=np.nan
                )
                new_utilities[:, candidate_indices] = -np.inf
                new_utilities[:, new_candidates] = utilities[:, new_candidates]
            else:
                new_utilities = np.full(
                    shape=(batch_size, len(candidates)), fill_value=np.nan
                )
                new_utilities[:, candidate_indices] = -np.inf
                new_utilities[:, new_candidate_indices] = utilities

        if return_utilities:
            return new_queried_indices, new_utilities
        else:
            return new_queried_indices


class ParallelUtilityEstimationWrapper(SingleAnnotatorPoolQueryStrategy):
    """Parallel Utility Estimation Wrapper

    This class implements a wrapper for single-annotator pool-based strategies
    such that utilities for candidates can be calculated in parallel. The main
    assumption for this is that the utility computations are independent from
    another. Therefore, only `batch_size=1` is supported.

    Parameters
    ----------
    query_strategy : skactiveml.base.SingleAnnotatorPoolQueryStrategy
        The strategy used for computing the utilities of the candidates.
    n_jobs : int, default=-1
        Determines the number of maximum number of parallel utility
        computations. If `n_jobs` is set to -1 (default), the number of
        parallel computations is set to the number of available CPU cores are.
        For further details refer to `n_jobs` in `joblib.Parallel`.
    parallel_dict : dict-like, default=None
        Further arguments that will be passed to `joblib.Parallel`. Note that,
        `n_jobs` should not be set in `parallel_dict`.
    missing_label : scalar or string or np.nan or None, default=np.nan
        Value to represent a missing label.
    random_state : int or np.random.RandomState, default=None
        The random state to use.

    """

    def __init__(
        self,
        query_strategy=None,
        n_jobs=-1,
        parallel_dict=None,
        missing_label=MISSING_LABEL,
        random_state=None,
    ):
        super().__init__(
            missing_label=missing_label, random_state=random_state
        )
        self.query_strategy = query_strategy
        self.n_jobs = n_jobs
        self.parallel_dict = parallel_dict

    @match_signature("query_strategy", "query")
    def query(
        self,
        X,
        y,
        candidates=None,
        batch_size=1,
        return_utilities=False,
        **query_kwargs,
    ):
        """Determines for which candidate samples labels are to be queried.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data set, usually complete, i.e., including the labeled
            and unlabeled samples.
        y : array-like of shape (n_samples)
            Labels of the training data set (possibly including unlabeled ones
            indicated by self.MISSING_LABEL).
        candidates : None or array-like of shape (n_candidates), dtype=int or
            array-like of shape (n_candidates, n_features), (default=None)

            - If `candidates` is `None`, the unlabeled samples from `(X,y)` are
              considered as `candidates`.
            - If `candidates` is of shape `(n_candidates,)` and of type
              `int`, `candidates` is considered as the indices of the
              samples in `(X,y)`.
            - If `candidates` is of shape `(n_candidates, *)`, the
              candidate samples are directly given in `candidates` (not
              necessarily contained in `X`). This is not supported by all
              query strategies.
        batch_size : int, default=1
            The number of samples to be selected in one AL cycle. For this
            wrapper, only `batch_size=1` is supported.
        return_utilities : bool, default=False
            If `True`, also return the utilities based on the query strategy.
        **query_kwargs : dict-like
            Further keyword arguments are passed to the `query` method of the
            `query_strategy` object.

        Returns
        -------
        query_indices : numpy.ndarray of shape (batch_size,)
            The query indices indicate for which candidate sample a label is
            to be queried, e.g., `query_indices[0]` indicates the first
            selected sample.

            - If `candidates` is `None` or of shape
              `(n_candidates,)`, the indexing refers to the samples in
              `X`.
            - If `candidates` is of shape `(n_candidates, n_features)`,
              the indexing refers to the samples in `candidates`.
        utilities : numpy.ndarray of shape (batch_size, n_samples) or \
                numpy.ndarray of shape (batch_size, n_candidates)
            The utilities of samples after each selected sample of the batch,
            e.g., `utilities[0]` indicates the utilities used for selecting
            the first sample (with index `query_indices[0]`) of the batch.
            Utilities for labeled samples will be set to np.nan.

            - If `candidates` is `None` or of shape
              `(n_candidates,)`, the indexing refers to the samples in
              `X`.
            - If `candidates` is of shape `(n_candidates, n_features)`,
              the indexing refers to the samples in `candidates`.
        """

        X, y, candidates, batch_size, return_utilities = self._validate_data(
            X, y, candidates, batch_size, return_utilities, reset=True
        )

        if batch_size != 1:
            raise ValueError("`batch_size` must be set to 1.")

        if not isinstance(
            self.query_strategy, SingleAnnotatorPoolQueryStrategy
        ):
            raise TypeError(
                f"`query_strategy` is of type `{type(self.query_strategy)}` "
                f"but must be of type `SingleAnnotatorPoolQueryStrategy`."
            )

        X_cand, mapping = self._transform_candidates(candidates, X, y)

        if self.parallel_dict is None:
            parallel_dict = {}
        elif isinstance(self.parallel_dict, dict):
            parallel_dict = self.parallel_dict.copy()
            if "n_jobs" in parallel_dict.keys():
                warnings.warn(
                    f"`n_jobs` ({parallel_dict['n_jobs']}) "
                    "is specified in `parallel_dict`. "
                    f"This will be replaced with `n_jobs={self.n_jobs}`."
                )
        else:
            raise TypeError(
                f"`parallel_dict` is of type `{type(self.parallel_dict)}` "
                f"but must be a dictionary or None."
            )

        parallel_dict["n_jobs"] = min(self.n_jobs, len(X_cand))
        parallel_pool = Parallel(**parallel_dict)

        def query_lambda_func(candidate):
            return self.query_strategy.query(
                X=X,
                y=y,
                candidates=np.array(candidate),
                batch_size=1,
                return_utilities=True,
                **query_kwargs,
            )

        if parallel_dict["n_jobs"] < 0:
            chunks = np.array_split(X_cand, cpu_count())
        else:
            chunks = np.array_split(X_cand, parallel_dict["n_jobs"])
        qs_outputs = parallel_pool(
            delayed(query_lambda_func)(c) for c in chunks
        )

        utilities_cand = np.concatenate(
            [qs_output[1][0] for qs_output in qs_outputs], axis=0
        )

        if mapping is None:
            utilities = utilities_cand
        else:
            utilities = np.full(len(X), np.nan)
            utilities[mapping] = utilities_cand

        return simple_batch(
            utilities,
            self.random_state_,
            batch_size=batch_size,
            return_utilities=return_utilities,
        )
