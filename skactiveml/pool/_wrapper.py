from ..base import SingleAnnotatorPoolQueryStrategy
from ..utils import (
    MISSING_LABEL,
    check_random_state,
    check_equal_missing_label,
    is_labeled,
    is_unlabeled,
    labeled_indices,
    unlabeled_indices,
    check_scalar,
    simple_batch,
    match_signature,
)
from math import ceil
import numpy as np
from joblib import Parallel, delayed, cpu_count
from sklearn import clone
import warnings
from ._target import (
    _check_resolved_target_capability,
    _collect_declared_authorities,
    _reconcile_target_declarations,
)


class _TargetPreservingWrapper(SingleAnnotatorPoolQueryStrategy):
    """Base class for wrappers preserving the wrapped target semantics.

    A wrapper owns the reconciliation of its own target declaration with the
    declarations of the strategies it wraps and with the estimators passed as
    query arguments. It therefore keeps every step that depends on wrapper
    internals, i.e., traversing the wrapped strategy chain, discovering
    declared authorities among the query arguments, and validating the wrapped
    capabilities. The wrapper-agnostic steps, i.e., comparing target
    declarations, resolving target specifications, and checking capabilities,
    are shared through :mod:`skactiveml.pool._target`.
    """

    @property
    def _target_capabilities(self):
        return getattr(
            self.query_strategy, "_target_capabilities", frozenset()
        )

    @property
    def _target_authority_params(self):
        """Delegate the declared authority roles to the wrapped strategy."""
        return getattr(self.query_strategy, "_target_authority_params", ())

    def _resolve_wrapped_target_type(self, y, query_kwargs):
        """Resolve the target type this wrapper must preserve.

        This is the entry point of the wrapper's target reconciliation. It
        fails before any query state is committed.

        Parameters
        ----------
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Labels of the training data set (possibly including unlabeled
            ones indicated by `self.missing_label`).
        query_kwargs : dict
            The keyword arguments forwarded to the wrapped query strategy.

        Returns
        -------
        target_type : str
            The resolved target type.
        """
        self._check_wrapped_strategy()
        target_type, target_spec = _reconcile_target_declarations(
            self._collect_target_declarations(),
            self._collect_target_authorities(query_kwargs),
            y,
            missing_label=self.missing_label,
            owner_name=type(self).__name__,
        )
        self._check_wrapped_target_capability(target_type, target_spec)
        is_unlabeled(
            y,
            missing_label=self.missing_label,
            target_type=target_type,
        )
        return target_type

    def _check_wrapped_strategy(self):
        """Check the wrapped strategy's type and missing label."""
        if not isinstance(
            self.query_strategy, SingleAnnotatorPoolQueryStrategy
        ):
            raise TypeError(
                f"`query_strategy` is of type `{type(self.query_strategy)}` "
                f"but must be of type `SingleAnnotatorPoolQueryStrategy`."
            )
        check_equal_missing_label(
            self.query_strategy.missing_label, self.missing_label
        )

    def _collect_target_declarations(self):
        """Collect the target declarations along the wrapped chain.

        Returns
        -------
        declarations : list of (str, str)
            The declared target types with their declaring component's name,
            ordered from this wrapper to the innermost wrapped strategy. A
            cyclic chain is traversed only once per strategy.
        """
        declarations = [(self.target_type, type(self).__name__)]
        seen_strategies = {id(self)}
        strategy = self.query_strategy
        while strategy is not None and id(strategy) not in seen_strategies:
            seen_strategies.add(id(strategy))
            declarations.append(
                (
                    getattr(strategy, "target_type", "auto"),
                    type(strategy).__name__,
                )
            )
            strategy = getattr(strategy, "query_strategy", None)
        return declarations

    def _collect_target_authorities(self, query_kwargs):
        """Discover the target authorities among the query arguments."""
        return _collect_declared_authorities(
            self._target_authority_params, query_kwargs
        )

    def _check_wrapped_target_capability(self, target_type, target_spec):
        """Check the resolved target against the wrapped capabilities."""
        capabilities = self._target_capabilities
        if capabilities:
            _check_resolved_target_capability(
                type(self.query_strategy).__name__,
                target_type,
                target_spec,
                capabilities,
            )

    def _query_strategy_for_target_type(self, target_type):
        query_strategy = self.query_strategy
        if getattr(query_strategy, "target_type", None) == "auto":
            query_strategy = clone(query_strategy).set_params(
                target_type=target_type
            )
        return query_strategy


class SubSamplingWrapper(_TargetPreservingWrapper):
    """Sub-sampling Wrapper

    This class implements a wrapper for single-annotator pool-based strategies
    that randomly sub-samples a set of candidates before computing their
    utilities. This is useful when the number of available candidates is too
    large and a small subset of candidates is sufficient to select a good batch
    for labeling. The number of candidates can be controlled using
    `max_candidates` which supports an absolute number or a fraction of the
    available candidates. Additionally, `exclude_non_subsample` provides an
    option to mask all candidates that were not included in the subsample. This
    can further improve the runtime for query strategies that utilize all
    available unlabeled data in their selection.

    Resolved multi-label targets preserve sample-level masks, so each row must
    be either fully labeled or fully unlabeled.

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
    embed_samples_func : Callable or None, default=None
        - If `embed_samples_func` is a `Callable`, it must accept the samples
          `X` as input and return the sample-wise embeddings.
        - If `embed_samples_func` is None, no action is performed.
    missing_label : scalar or string or np.nan or None, default=np.nan
        Value to represent a missing label.
    random_state : int or np.random.RandomState, default=None
        The random state to use.
    target_type : "auto" or "single-output" or "multi-label", default="auto"
        Declared target type. The selected target type must be supported by
        `query_strategy`. Automatic resolution preserves target semantics
        declared by the wrapped strategy or a supplied estimator.
    """

    def __init__(
        self,
        query_strategy=None,
        max_candidates=0.1,
        exclude_non_subsample=False,
        embed_samples_func=None,
        missing_label=MISSING_LABEL,
        random_state=None,
        target_type="auto",
    ):
        super().__init__(
            missing_label=missing_label, random_state=random_state
        )
        self.query_strategy = query_strategy
        self.max_candidates = max_candidates
        self.exclude_non_subsample = exclude_non_subsample
        self.embed_samples_func = embed_samples_func
        self.target_type = target_type

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
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Labels of the training data set (possibly including unlabeled ones
            indicated by `self.missing_label`). For multi-label targets, a row
            `y[i]` must either contain only observed labels or only
            `missing_label` values, i.e., no mixing within a row.
        candidates : None or array-like of shape (n_candidates), dtype=int or\
                array-like of shape (n_candidates, n_features), default=None
            - If `candidates` is `None`, the unlabeled samples from `(X,y)` are
              considered as `candidates`.
            - If `candidates` is of shape `(n_candidates,)` and of type
              `int`, `candidates` is considered as the indices of the
              samples in `(X,y)`.
            - If `candidates` is of shape `(n_candidates, ...)`, the
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
        target_type = self._resolve_wrapped_target_type(y, query_kwargs)
        query_strategy = self._query_strategy_for_target_type(target_type)

        X, y, candidates, batch_size, return_utilities = self._validate_data(
            X,
            y,
            candidates,
            batch_size,
            return_utilities,
            reset=True,
            target_type=target_type,
        )

        check_scalar(self.exclude_non_subsample, "exclude_non_subsample", bool)
        is_lbld = is_labeled(
            y=y,
            missing_label=self.missing_label_,
            target_type=target_type,
        )
        seed_multiplier = int(is_lbld.sum() + 1)
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
        if self.embed_samples_func is not None and not callable(
            self.embed_samples_func
        ):
            raise TypeError(
                "`embed_samples_func` must be either a `Callable` or `None`."
            )
        random_state = check_random_state(self.random_state, seed_multiplier)

        # subsampling with no explicit provided candidates
        if candidates is None:
            candidate_indices = unlabeled_indices(
                y=y,
                missing_label=self.missing_label_,
                target_type=target_type,
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
        # subsampling with provided explicit candidates
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
                y=y,
                missing_label=self.missing_label_,
                target_type=target_type,
            )
            if candidates is not None and candidates.ndim > 1:
                subset_and_labeled_indices = all_labeled
            else:
                # ignore labeled candidates to avoid duplicate samples
                all_labeled = np.setdiff1d(all_labeled, new_candidates)
                subset_and_labeled_indices = np.concatenate(
                    [all_labeled, new_candidates]
                )
            sorted_idx = np.argsort(subset_and_labeled_indices)
            subset_and_labeled_indices = subset_and_labeled_indices[sorted_idx]

            new_X = X[subset_and_labeled_indices]
            new_y = y[subset_and_labeled_indices]
            # for explicitly provided candidates recalculate candidate indices
            # that are passed to the wrapped query strategy
            if candidates is None or candidates.ndim == 1:
                new_candidates = np.flatnonzero(sorted_idx >= len(all_labeled))
        else:
            new_X = X
            new_y = y

        if self.embed_samples_func:
            new_X = self.embed_samples_func(new_X)

        qs_output = query_strategy.query(
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
        effective_batch_size = len(np.atleast_1d(queried_indices))

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
                    shape=(effective_batch_size, len(X)), fill_value=np.nan
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
                    shape=(effective_batch_size, len(X)), fill_value=np.nan
                )
                new_utilities[:, candidate_indices] = -np.inf
                new_utilities[:, new_candidates] = utilities[:, new_candidates]
            else:
                new_utilities = np.full(
                    shape=(effective_batch_size, len(candidates)),
                    fill_value=np.nan,
                )
                new_utilities[:, candidate_indices] = -np.inf
                new_utilities[:, new_candidate_indices] = utilities

        if return_utilities:
            return new_queried_indices, new_utilities
        else:
            return new_queried_indices


class ParallelUtilityEstimationWrapper(_TargetPreservingWrapper):
    """Parallel Utility Estimation Wrapper

    This class implements a wrapper for single-annotator pool-based strategies
    such that utilities for candidates can be calculated in parallel. The main
    assumption for this is that the utility computations are independent from
    another. Therefore, only `batch_size=1` is supported.

    Resolved multi-label targets preserve sample-level masks, so each row must
    be either fully labeled or fully unlabeled.

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
    target_type : "auto" or "single-output" or "multi-label", default="auto"
        Declared target type. The selected target type must be supported by
        `query_strategy`. Automatic resolution preserves target semantics
        declared by the wrapped strategy or a supplied estimator.

    """

    def __init__(
        self,
        query_strategy=None,
        n_jobs=-1,
        parallel_dict=None,
        missing_label=MISSING_LABEL,
        random_state=None,
        target_type="auto",
    ):
        super().__init__(
            missing_label=missing_label, random_state=random_state
        )
        self.query_strategy = query_strategy
        self.n_jobs = n_jobs
        self.parallel_dict = parallel_dict
        self.target_type = target_type

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
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Labels of the training data set (possibly including unlabeled ones
            indicated by `self.missing_label`). For multi-label targets, a row
            `y[i]` must either contain only observed labels or only
            `missing_label` values, i.e., no mixing within a row.
        candidates : None or array-like of shape (n_candidates), dtype=int or
            array-like of shape (n_candidates, n_features), (default=None)

            - If `candidates` is `None`, the unlabeled samples from `(X,y)` are
              considered as `candidates`.
            - If `candidates` is of shape `(n_candidates,)` and of type
              `int`, `candidates` is considered as the indices of the
              samples in `(X,y)`.
            - If `candidates` is of shape `(n_candidates, ...)`, the
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
        target_type = self._resolve_wrapped_target_type(y, query_kwargs)
        query_strategy = self._query_strategy_for_target_type(target_type)

        # Validate parameters.
        X, y, candidates, batch_size, return_utilities = self._validate_data(
            X,
            y,
            candidates,
            batch_size,
            return_utilities,
            reset=True,
            target_type=target_type,
        )
        if batch_size != 1:
            raise ValueError("`batch_size` must be set to 1.")
        # Determine candidate samples for selection.
        X_cand, mapping = self._transform_candidates(
            candidates=candidates, X=X, y=y, target_type=target_type
        )

        # Determine number of parallel jobs.
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
            return query_strategy.query(
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
