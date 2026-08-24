"""
The :mod:`skactiveml.base` package implements the base classes for
:mod:`skactiveml`.
"""

import numpy as np
import warnings

from abc import ABC, abstractmethod
from copy import deepcopy
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.metrics import accuracy_score
from sklearn.utils.validation import (
    check_array,
    check_consistent_length,
    column_or_1d,
)

from .exceptions import MappingError, _ExhaustedCandidatePool
from .utils._functions import _guard_own_query
from .utils._selection import (
    _answer_exhausted_candidate_pool,
    _maps_to_samples,
)
from .utils._target import (
    _check_target_capability,
    _has_no_class_evidence,
    _resolve_task_agnostic_target_type,
    _check_target_spec_capability,
)
from .utils import (
    MISSING_LABEL,
    is_labeled,
    is_unlabeled,
    unlabeled_indices,
    ExtLabelEncoder,
    rand_argmin,
    resolve_target_spec,
    check_classifier_params,
    check_random_state,
    check_cost_matrix,
    check_scalar,
    check_class_prior,
    check_missing_label,
    check_indices,
    check_n_features,
    check_type,
    compute_vote_vectors,
)

__all__ = [
    "QueryStrategy",
    "PoolQueryStrategy",
    "SingleAnnotatorPoolQueryStrategy",
    "MultiAnnotatorPoolQueryStrategy",
    "BudgetManager",
    "SingleAnnotatorStreamQueryStrategy",
    "SkactivemlClassifier",
    "ClassFrequencyEstimator",
    "SkactivemlRegressor",
    "ProbabilisticRegressor",
]

successful_skorch_torch_import = False
try:
    from collections.abc import Sequence
    from skorch import NeuralNet
    from skorch.utils import to_numpy
    from .utils import _check_forward_outputs

    successful_skorch_torch_import = True
except ImportError:  # pragma: no cover
    pass


_TARGET_SPEC_NOT_PROVIDED = object()

# Canonical names of `query` parameters carrying an estimator whose target
# semantics describe the queried targets `y`. Estimators passed under any other
# name (e.g. the labeled-vs-unlabeled `discriminator` of `DiscriminativeAL`)
# solve an auxiliary problem and are therefore no semantic authority for `y`.
_TARGET_AUTHORITY_QUERY_PARAMS = ("clf", "reg", "ensemble", "estimator")


def _reuse_established_target_spec(resolved_spec, established_spec=None):
    if established_spec is None:
        return resolved_spec
    if resolved_spec != established_spec:
        raise ValueError(
            "Incremental fitting cannot change the established target "
            f"specification {established_spec!r}; received "
            f"{resolved_spec!r}."
        )
    return established_spec


class QueryStrategy(ABC, BaseEstimator):
    """Base class for all query strategies in scikit-activeml.

    Parameters
    ----------
    random_state : int or RandomState instance, optional (default=None)
        Controls the randomness of the estimator.
    """

    def __init__(self, random_state=None):
        self.random_state = random_state

    @abstractmethod
    def query(self, *args, **kwargs):
        """
        Determines the query for active learning based on input arguments.
        """
        raise NotImplementedError


class PoolQueryStrategy(QueryStrategy):
    """Base class for all pool-based active learning query strategies in
    scikit-activeml.

    Parameters
    ----------
    missing_label : scalar or string or np.nan or None, default=np.nan
        Value to represent a missing label.
    random_state : int or RandomState instance or None, default=None
        Controls the randomness of the estimator.
    """

    def __init_subclass__(cls, **kwargs):
        # Every `query` is guarded, so that the shared validation can answer an
        # exhausted candidate pool for all pool strategies at once.
        super().__init_subclass__(**kwargs)
        _guard_own_query(cls)

    def __init__(self, missing_label=MISSING_LABEL, random_state=None):
        super().__init__(random_state=random_state)
        self.missing_label = missing_label

    def _validate_data(
        self,
        X,
        y,
        candidates,
        batch_size,
        return_utilities,
        reset=True,
        check_X_dict=None,
        target_type="single-output",
    ):
        """Validate input data, all attributes and set or check the
        `n_features_in_` attribute.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data set, usually complete, i.e. including the labeled and
            unlabeled samples.
        y : array-like of shape (n_samples, ...)
            Labels of the training data set (possibly including unlabeled ones
            indicated by self.MISSING_LABEL.
        candidates : None or array-like of shape (n_candidates), dtype=int or \
                array-like of shape (n_candidates, n_features), default=None
            - If `candidates` is `None`, the unlabeled samples from
              `(X,y)` are considered as `candidates`.
            - If `candidates` is of shape `(n_candidates,)` and of type
              `int`, `candidates` is considered as the indices of the
              samples in `(X,y)`.
            - If `candidates` is of shape `(n_candidates, ...)`, the
              candidate samples are directly given in `candidates` (not
              necessarily contained in `X`). This is not supported by all
              query strategies.
        batch_size : int
            The number of samples to be selected in one AL cycle.
        return_utilities : bool
            If true, also return the utilities based on the query strategy.
        reset : bool, default=True
            Whether to reset the `n_features_in_` attribute.
            If False, the input will be checked for consistency with data
            provided when reset was last True.
        **check_X_dict : kwargs
            Parameters passed to :func:`sklearn.utils.check_array`.

        Returns
        -------
        X : np.ndarray of shape (n_samples, n_features)
            Checked training data set.
        y : np.ndarray of shape (n_samples, ...)
            Checked labels of the training data set.
        candidates : None or np.ndarray of shape (n_candidates), dtype=int or\
                np.ndarray of shape (n_candidates, n_features)
            Checked candidate samples.
        batch_size : int
            Checked number of samples to be selected in one AL cycle.
        return_utilities : bool
            Checked boolean value of `return_utilities`.
        """
        # Check samples.
        if check_X_dict is None:
            check_X_dict = {"allow_nd": True}
        X = check_array(X, **check_X_dict)

        # Check number of features.
        check_n_features(self, X, reset=reset)

        # Check labels
        y = check_array(
            y, ensure_2d=False, ensure_all_finite="allow-nan", dtype=None
        )
        check_consistent_length(X, y)

        # Check missing_label
        check_missing_label(self.missing_label, target_type=y.dtype)
        self.missing_label_ = self.missing_label

        # Check candidates (+1 to avoid zero multiplier).
        seed_mult = (
            int(
                np.sum(
                    is_unlabeled(
                        y,
                        self.missing_label_,
                        target_type=target_type,
                    )
                )
            )
            + 1
        )
        if candidates is not None:
            candidates = np.array(candidates)
            if candidates.ndim == 1:
                candidates = check_indices(candidates, y, dim=0)
            else:
                check_candidates_dict = deepcopy(check_X_dict)
                check_candidates_dict["ensure_2d"] = False
                # An empty candidate matrix is an exhausted candidate pool,
                # i.e., a valid acquisition state the guarded `query` answers
                # with an empty batch. Anything without a sample axis at all
                # stays a rejected input.
                check_candidates_dict["ensure_min_samples"] = (
                    0 if np.ndim(candidates) > 0 else 1
                )
                candidates = check_array(candidates, **check_candidates_dict)
                check_n_features(self, candidates, reset=False)

        # Check return_utilities.
        check_scalar(return_utilities, "return_utilities", bool)

        # Check batch size.
        check_scalar(batch_size, target_type=int, name="batch_size", min_val=1)

        # Check random state.
        self.random_state_ = check_random_state(self.random_state, seed_mult)

        return X, y, candidates, batch_size, return_utilities


class SingleAnnotatorPoolQueryStrategy(PoolQueryStrategy):
    """Base class for all pool-based active learning query strategies with a
    single annotator in scikit-activeml.

    Parameters
    ----------
    missing_label : scalar or str or np.nan or None, default=np.nan
        Value to represent a missing label.
    random_state : int or RandomState instance, default=None
        Controls the randomness of the estimator. If None, the RandomState
        singleton used by `np.random` is used.
    target_type : "auto" or "single-output" or "multi-label" or \
            "multi-output", default="auto"
        Declared target type. Subclasses reject resolved specifications
        outside their exact capabilities.
    """

    def __init__(
        self,
        missing_label=MISSING_LABEL,
        random_state=None,
        target_type="auto",
    ):
        super().__init__(
            missing_label=missing_label,
            random_state=random_state,
        )
        self.target_type = target_type

    @property
    def _target_capabilities(self):
        """Conservative base capability for single-annotator strategies."""
        return frozenset(
            {("classification", "single-output", "single-annotator")}
        )

    @property
    def _target_authority_params(self):
        """Names of `query` parameters carrying a target authority.

        This is the narrow interface through which a strategy declares which
        of its `query` arguments hold an estimator whose target semantics are
        authoritative for `y`. Wrappers use it instead of inspecting arbitrary
        query arguments. Strategies naming such an estimator differently, or
        accepting estimators that describe an auxiliary problem rather than
        `y`, override this property.

        Returns
        -------
        authority_params : tuple of str
            The declared parameter names in deterministic resolution order.
        """
        return _TARGET_AUTHORITY_QUERY_PARAMS

    def _resolve_query_target_type(self, y):
        tasks = {capability[0] for capability in self._target_capabilities}
        if len(tasks) == 1:
            task = next(iter(tasks))
            if task == "classification":
                classes = getattr(self, "classes", None)
                if classes is None:
                    target_type = _resolve_task_agnostic_target_type(
                        y,
                        target_type=self.target_type,
                        missing_label=self.missing_label,
                    )
                    _check_target_capability(
                        type(self).__name__,
                        (task, target_type, "single-annotator"),
                        self._target_capabilities,
                    )
                    return target_type
                else:
                    target_spec = resolve_target_spec(
                        y,
                        task=task,
                        target_type=self.target_type,
                        annotation_type="single-annotator",
                        classes=classes,
                        missing_label=self.missing_label,
                    )
            else:
                target_spec = resolve_target_spec(
                    y,
                    task=task,
                    target_type=self.target_type,
                    annotation_type="single-annotator",
                    classes=None,
                    missing_label=self.missing_label,
                )
            _check_target_spec_capability(
                type(self).__name__, target_spec, self._target_capabilities
            )
            return target_spec.target_type
        target_type = _resolve_task_agnostic_target_type(
            y,
            target_type=self.target_type,
            missing_label=self.missing_label,
        )
        applicable_tasks = (
            {"classification"} if target_type == "multi-label" else tasks
        )
        for task in applicable_tasks:
            _check_target_capability(
                type(self).__name__,
                (task, target_type, "single-annotator"),
                self._target_capabilities,
            )
        return target_type

    @abstractmethod
    def query(
        self,
        X,
        y,
        *args,
        candidates=None,
        batch_size=1,
        return_utilities=False,
        **kwargs,
    ):
        """Determines for which candidate samples labels are to be queried.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data set, usually complete, i.e. including the labeled and
            unlabeled samples.
        y : array-like of shape (n_samples,)
            Labels of the training data set (possibly including unlabeled ones
            indicated by self.missing_label).
        candidates : None or array-like of shape (n_candidates), dtype=int or \
                array-like of shape (n_candidates, n_features), default=None
            - If `candidates` is `None`, the unlabeled samples from
              `(X,y)` are considered as `candidates`.
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
            If true, also return the utilities based on the query strategy.

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

        Notes
        -----
        An exhausted candidate pool, i.e., a fully labeled `(X, y)` queried
        with `candidates=None` or an empty `candidates`, is a valid
        acquisition state. It is answered with an empty batch of `batch_size`
        zero and a warning naming the exhaustion, so that a budget loop
        running one cycle past exhaustion needs no special case.
        """
        raise NotImplementedError

    def _validate_data(
        self,
        X,
        y,
        candidates,
        batch_size,
        return_utilities,
        reset=True,
        check_X_dict=None,
        target_type=None,
    ):
        """Validate input data, all attributes and set or check the
        `n_features_in_` attribute.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data set, usually complete, i.e. including the labeled and
            unlabeled samples.
        y : array-like of shape (n_samples)
            Labels of the training data set (possibly including unlabeled ones
            indicated by self.MISSING_LABEL.
        candidates : None or array-like of shape (n_candidates), dtype=int or \
                array-like of shape (n_candidates, n_features), default=None
            - If `candidates` is `None`, the unlabeled samples from
              `(X,y)` are considered as `candidates`.
            - If `candidates` is of shape `(n_candidates,)` and of type
              `int`, `candidates` is considered as the indices of the
              samples in `(X,y)`.
            - If `candidates` is of shape `(n_candidates, ...)`, the
              candidate samples are directly given in `candidates` (not
              necessarily contained in `X`). This is not supported by all
              query strategies.
        batch_size : int
            The number of samples to be selected in one AL cycle.
        return_utilities : bool
            If true, also return the utilities based on the query strategy.
        reset : bool, default=True
            Whether to reset the `n_features_in_` attribute.
            If False, the input will be checked for consistency with data
            provided when reset was last True.
        **check_X_dict : kwargs
            Parameters passed to :func:`sklearn.utils.check_array`.
        Returns
        -------
        X : np.ndarray of shape (n_samples, n_features)
            Checked training data set.
        y : np.ndarray of shape (n_samples,)
            Checked labels of the training data set.
        candidates :  None or np.ndarray of shape (n_candidates), dtype=int or
            np.ndarray of shape (n_candidates, n_features)
            Checked candidate samples.
        batch_size : int
            Checked number of samples to be selected in one AL cycle.
        return_utilities : bool
            Checked boolean value of `return_utilities`.
        """

        if target_type is None:
            target_type = self._resolve_query_target_type(y)

        (
            X,
            y,
            candidates,
            batch_size,
            return_utilities,
        ) = super()._validate_data(
            X,
            y,
            candidates,
            batch_size,
            return_utilities,
            reset,
            check_X_dict,
            target_type=target_type,
        )

        if target_type == "multi-label":
            y = check_array(
                y, ensure_2d=False, ensure_all_finite="allow-nan", dtype=None
            )
        elif target_type == "single-output":
            y = column_or_1d(y, warn=True)
        else:
            raise ValueError(
                "`target_type` must be either 'single-output' or "
                "'multi-label'."
            )

        if candidates is None:
            is_ulbld = is_unlabeled(
                y,
                missing_label=self.missing_label_,
                target_type=target_type,
            )
            n_candidates = int(is_ulbld.sum())
        else:
            n_candidates = len(candidates)

        if n_candidates == 0:
            # Abort before any strategy code sees the empty candidate slice.
            raise _ExhaustedCandidatePool(
                _answer_exhausted_candidate_pool(
                    (0,),
                    (0, len(X) if _maps_to_samples(candidates) else 0),
                    return_utilities,
                )
            )

        if n_candidates < batch_size:
            warnings.warn(
                f"'batch_size={batch_size}' is larger than number of "
                f"candidates. Instead, 'batch_size={n_candidates}' was set."
            )
            batch_size = n_candidates

        return X, y, candidates, batch_size, return_utilities

    def _transform_candidates(
        self,
        candidates,
        X,
        y,
        enforce_mapping=False,
        allow_only_unlabeled=False,
        target_type="single-output",
    ):
        """Transforms the `candidates` parameter into a sample array and the
        corresponding index array `mapping` such that
        `candidates = X[mapping]`.

        Parameters
        ----------
        candidates : None or array-like of shape (n_candidates), dtype=int or \
                array-like of shape (n_candidates, n_features), default=None
            - If `candidates` is `None`, the unlabeled samples from
              `(X,y)` are considered as `candidates`.
            - If `candidates` is of shape `(n_candidates,)` and of type
              `int`, `candidates` is considered as the indices of the
              samples in `(X,y)`.
            - If `candidates` is of shape `(n_candidates, ...)`, the
              candidate samples are directly given in `candidates` (not
              necessarily contained in `X`).
        X : np.ndarray of shape (n_samples, n_features)
            Checked training data set.
        y : np.ndarray of shape (n_samples,)
            Checked labels of the training data set.
        enforce_mapping : bool, default=False
            If True, an exception is raised when no exact mapping can be
            determined (i.e., `mapping` is None).
        allow_only_unlabeled : bool, default=False
            If True, an exception is raised when indices of candidates contain
            labeled samples.
        Returns
        -------
        candidates : np.ndarray of shape (n_candidates, n_features)
            Candidate samples from which the strategy can query the label.
        mapping : np.ndarray of shape (n_candidates) or None
            Index array that maps `candidates` to `X`.
            (`candidates = X[mapping]`)
        """

        if candidates is None:
            ulbd_idx = unlabeled_indices(
                y,
                self.missing_label_,
                target_type=target_type,
            )
            return X[ulbd_idx], ulbd_idx
        elif candidates.ndim == 1:
            if allow_only_unlabeled:
                if is_labeled(
                    y[candidates],
                    self.missing_label_,
                    target_type=target_type,
                ).any():
                    raise ValueError(
                        "Candidates must not contain labeled " "samples."
                    )
            return X[candidates], candidates
        else:
            if enforce_mapping:
                raise MappingError(
                    "Mapping `candidates` to `X` is not "
                    "possible but `enforce_mapping` is True. "
                    "Use index array for `candidates` instead."
                )
            else:
                return candidates, None


class MultiAnnotatorPoolQueryStrategy(PoolQueryStrategy):
    """Base class for all pool-based active learning query strategies with
    multiple annotators in scikit-activeml.

    Parameters
    ----------
    missing_label : scalar or str or np.nan or None, default=np.nan
        Value to represent a missing label.
    random_state : int or RandomState instance, default=None
        Controls the randomness of the estimator. If None, the RandomState
        singleton used by `np.random` is used.
    target_type : "auto" or "single-output" or "multi-label" or \
            "multi-output", default="auto"
        Declared target type. Multi-annotator strategies support only
        single-output classification in version 1.1.
    """

    def __init__(
        self,
        missing_label=MISSING_LABEL,
        random_state=None,
        target_type="auto",
    ):
        super().__init__(
            missing_label=missing_label,
            random_state=random_state,
        )
        self.target_type = target_type

    @property
    def _target_capabilities(self):
        """Exact target semantics supported by multi-annotator strategies."""
        return frozenset(
            {("classification", "single-output", "multi-annotator")}
        )

    def _resolve_target_spec(self, y, classes=None):
        try:
            target_spec = resolve_target_spec(
                y,
                task="classification",
                target_type=self.target_type,
                annotation_type="multi-annotator",
                classes=classes,
                missing_label=self.missing_label,
            )
        except ValueError:
            # A class-agnostic strategy can still acquire the first
            # sample-annotator pair before a class vocabulary is observable.
            # The matrix structure remains unambiguously multi-annotator.
            lacks_class_evidence = classes is None and _has_no_class_evidence(
                y,
                self.target_type,
                "multi-annotator",
                self.missing_label,
            )
            if lacks_class_evidence:
                return None
            raise
        _check_target_spec_capability(
            type(self).__name__, target_spec, self._target_capabilities
        )
        return target_spec

    @abstractmethod
    def query(
        self,
        X,
        y,
        *args,
        candidates=None,
        annotators=None,
        batch_size=1,
        return_utilities=False,
        **kwargs,
    ):
        """Determines which candidate sample is to be annotated by which
        annotator.

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
        batch_size : int or str, default=1
            The number of annotators-sample pairs to be selected in one AL
            cycle. If `adaptive=True`, `batch_size='adaptive'` is allowed.
        return_utilities : bool, default=False
            If True, also return the utilities based on the query strategy.

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

        Notes
        -----
        An exhausted candidate pool, i.e., no candidate annotator-sample pair
        left to be queried, is a valid acquisition state. It is answered with
        an empty batch of `batch_size` zero and a warning naming the
        exhaustion, so that a budget loop running one cycle past exhaustion
        needs no special case.
        """
        raise NotImplementedError

    def _validate_data(
        self,
        X,
        y,
        candidates,
        annotators,
        batch_size,
        return_utilities,
        reset=True,
        check_X_dict=None,
        classes=None,
        target_spec=_TARGET_SPEC_NOT_PROVIDED,
    ):
        """Validate input data, all attributes and set or check the
        `n_features_in_` attribute.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data set, usually complete, i.e., including the labeled
            and unlabeled samples.
        y : array-like of shape (n_samples, n_annotators)
            Labels of the training data set for each annotator (possibly
            including unlabeled ones indicated by `self.missing_label`),
            meaning that `y[i, j]` contains the label annotated by annotator
            `i` for sample `j`.
        candidates : None or array-like of shape (n_candidates), dtype=int or\
                array-like of shape (n_candidates, n_features),
            See annotators.
        annotators : None or array-like of shape (n_avl_annotators), dtype=int\
                or array-like of shape (n_candidates, n_annotators),
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
        batch_size : int or string,
            The number of annotators sample pairs to be selected in one AL
            cycle. If `adaptive=True`, `batch_size='adaptive'` is allowed.
        return_utilities : bool
            If true, also return the utilities based on the query strategy.
        reset : bool, default=True
            Whether to reset the `n_features_in_` attribute.
            If False, the input will be checked for consistency with data
            provided when reset was last True.
        target_spec : TargetSpec or None, optional
            The already resolved local target specification. If omitted, it
            is resolved from `y`. `None` represents a cycle with no observable
            class evidence.
        **check_X_dict : kwargs
            Parameters passed to :func:`sklearn.utils.check_array`.

        Returns
        -------
        X : np.ndarray of shape (n_samples, n_features)
            Checked training data set.
        y : np.ndarray of shape (n_samples, n_annotators)
            Checked labels of the training data set.
        candidates :  None or np.ndarray of shape (n_candidates), dtype=int or\
                np.ndarray of shape (n_candidates, n_features)
            Checked candidate samples.
        annotators : None or np.ndarray of shape (n_avl_annotators), dtype=int\
                or np.ndarray of shape (n_candidates, n_annotators)
            Checked annotator boolean array
        batch_size : int
            Checked number of samples to be selected in one AL cycle.
        return_utilities : bool,
            Checked boolean value of `return_utilities`.
        """
        if target_spec is _TARGET_SPEC_NOT_PROVIDED:
            self._resolve_target_spec(y, classes=classes)

        (
            X,
            y,
            candidates,
            batch_size,
            return_utilities,
        ) = super()._validate_data(
            X, y, candidates, batch_size, return_utilities, reset, check_X_dict
        )

        check_array(y, ensure_2d=True, ensure_all_finite="allow-nan")
        unlabeled_pairs = is_unlabeled(y, missing_label=self.missing_label_)

        if annotators is not None:
            # An empty set of available annotators exhausts the candidate
            # pairs, i.e., a valid acquisition state the guarded `query`
            # answers with an empty batch. Anything without a sample axis at
            # all stays a rejected input.
            annotators = check_array(
                annotators,
                ensure_2d=False,
                allow_nd=True,
                ensure_min_samples=0 if np.ndim(annotators) > 0 else 1,
            )

            if annotators.ndim == 1:
                annotators = check_indices(annotators, y, dim=1)
            elif annotators.ndim == 2:
                annotators = check_array(
                    annotators, dtype=bool, ensure_min_samples=0
                )
                if candidates is None:
                    check_consistent_length(X, annotators)
                else:
                    check_consistent_length(candidates, annotators)
                check_consistent_length(y.T, annotators.T)
            else:
                raise ValueError(
                    "`annotators` must be either None, 1d or 2d array-like."
                )

        if annotators is None:
            if candidates is None:
                n_candidate_pairs = int(np.sum(unlabeled_pairs))
            else:
                n_candidate_pairs = len(candidates) * len(y.T)
        elif annotators.ndim == 1:
            if candidates is None:
                n_candidate_pairs = len(X) * len(annotators)
            else:
                n_candidate_pairs = len(candidates) * len(annotators)
        else:
            n_candidate_pairs = int(np.sum(annotators))

        if n_candidate_pairs == 0:
            # Abort before any strategy code sees the empty candidate slice.
            n_rows = len(X) if _maps_to_samples(candidates) else 0
            raise _ExhaustedCandidatePool(
                _answer_exhausted_candidate_pool(
                    (0, 2),
                    (0, n_rows, len(y.T)),
                    return_utilities,
                )
            )

        if n_candidate_pairs < batch_size:
            warnings.warn(
                f"'batch_size={batch_size}' is larger than number of "
                f"candidates pairs. Instead, 'batch_size={n_candidate_pairs}'"
                f" was set."
            )
            batch_size = n_candidate_pairs

        return X, y, candidates, annotators, batch_size, return_utilities

    def _transform_cand_annot(
        self, candidates, annotators, X, y, enforce_mapping=False
    ):
        """
        Transforms the `candidates` parameter into a sample array and the
        corresponding index array `mapping` such that
        `candidates = X[mapping]`, and transforms `annotators` into a boolean
        array such that `A_cand` represents the available annotator sample
        pairs for the samples of candidates.

        Parameters
        ----------
        candidates : None or array-like of shape (n_candidates), dtype=int or\
                array-like of shape (n_candidates, n_features),
            See annotators.
        annotators : None or array-like of shape (n_avl_annotators), dtype=int\
                or array-like of shape (n_candidates, n_annotators),
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
        X : np.ndarray of shape (n_samples, n_features)
            Checked training data set.
        y : np.ndarray of shape (n_samples,)
            Checked labels of the training data set.
        enforce_mapping : bool, default=False
            If `True`, an exception is raised when no exact mapping can be
            determined (i.e., `mapping` is `None`).

        Returns
        -------
        candidates : np.ndarray of shape (n_selectable_candidates, n_features)
            Candidate samples from which the strategy can query the label.
        mapping : np.ndarray of shape (n_selectable_candidates) or None
            Index array that maps `candidates` to `X`
            (`candidates = X[mapping]`).
        A_cand : np.ndarray of shape(n_selectable_candidates, n_annotators)
            Available annotator-sample-pairs with respect to `candidates`.
        """
        unlbd_pairs = is_unlabeled(y, self.missing_label_)
        unlbd_sample_indices = np.argwhere(
            np.any(unlbd_pairs, axis=1)
        ).flatten()
        n_annotators = y.shape[1]

        # if mapping does not exist
        if candidates is not None and candidates.ndim == 2:
            n_candidates = len(candidates)
            if annotators is None:
                A_cand = np.full((n_candidates, n_annotators), True)
            elif annotators.ndim == 1:
                A_cand = np.full((n_candidates, n_annotators), False)
                A_cand[:, annotators] = True
            else:
                A_cand = annotators

            if enforce_mapping:
                raise ValueError(
                    "Mapping `candidates` to `X` is not posssible"
                    "but `enforce_mapping` is True. Use index"
                    "array for `candidates` instead."
                )
            else:
                return candidates, None, A_cand
        # mapping exists
        if candidates is None:
            if annotators is None:
                candidates = unlbd_sample_indices
                A_cand = unlbd_pairs[unlbd_sample_indices]
            elif annotators.ndim == 1:
                candidates = np.arange(len(X), dtype=int)
                A_cand = np.full_like(y, False)
                A_cand[:, annotators] = True
            else:
                candidates = np.arange(len(X), dtype=int)
                A_cand = annotators
        else:  # candidates indices array
            if annotators is None:
                A_cand = np.full((len(candidates), y.shape[1]), True)
            elif annotators.ndim == 1:
                A_cand = np.full((len(candidates), y.shape[1]), False)
                A_cand[:, annotators] = True
            else:
                candidates = candidates
                A_cand = annotators
        return X[candidates], candidates, A_cand


class BudgetManager(ABC, BaseEstimator):
    """Base class for all budget managers for stream-based active learning
    to model budgeting constraints.

    Parameters
    ----------
    budget : float, default=None
        Specifies the ratio of samples which are allowed to be sampled, with
        `0 <= budget <= 1`. If `budget` is `None`, it is replaced with the
        default budget 0.1.
    """

    def __init__(self, budget=None):
        self.budget = budget

    @abstractmethod
    def query_by_utility(self, utilities, *args, **kwargs):
        """Ask the budget manager which `utilities` are sufficient to query the
        corresponding labels.

        Parameters
        ----------
        utilities : array-like of shape (n_samples,)
            The utilities provided by the stream-based active learning
            strategy, which are used to determine whether querying a sample
            is worth it given the budgeting constraint.

        Returns
        -------
        queried_indices : np.ndarray of shape (n_queried_indices,)
            The indices of samples in candidates whose labels are queried,
            with `0 <= queried_indices <= n_candidates`.
        """
        raise NotImplementedError

    @abstractmethod
    def update(self, candidates, queried_indices, *args, **kwargs):
        """Updates the budget manager.

        Parameters
        ----------
        candidates : {array-like, sparse matrix} of shape\
                (n_candidates, n_features)
            The samples which may be queried. Sparse matrices are accepted
            only if they are supported by the base query strategy.
        queried_indices : np.ndarray of shape (n_queried_indices,)
            The indices of samples in candidates whose labels are queried,
            with `0 <= queried_indices <= n_candidates`.

        Returns
        -------
        self : BudgetManager
            The budget manager returns itself, after it is updated.
        """
        raise NotImplementedError

    def _validate_budget(self):
        """check the assigned `budget` and set the default value 0.1 if
        `budget` is set to `None`.
        """
        if self.budget is not None:
            self.budget_ = self.budget
        else:
            self.budget_ = 0.1
        check_scalar(
            self.budget_,
            "budget",
            float,
            min_val=0.0,
            max_val=1.0,
            min_inclusive=False,
        )

    def _validate_data(self, utilities, *args, **kwargs):
        """Validate input data.

        Parameters
        ----------
        utilities: array-like of shape (n_samples,)
            The `utilities` provided by the stream-based active learning
            strategy.

        Returns
        -------
        utilities: ndarray of shape (n_samples,)
            Checked `utilities`.
        """
        # Check if utilities is set
        if not isinstance(utilities, np.ndarray):
            raise TypeError(
                "{} is not a valid type for utilities".format(type(utilities))
            )
        # Check budget
        self._validate_budget()
        return utilities


class SingleAnnotatorStreamQueryStrategy(QueryStrategy):
    """Base class for all stream-based active learning query strategies.

    Parameters
    ----------
    budget : float
        Specifies the ratio of labels which are allowed to be queried, with
        `0 <= budget <= 1`.
    random_state : int or RandomState instance or None, default=None
        Controls the randomness of the estimator.
    """

    def __init__(self, budget, random_state=None):
        super().__init__(random_state=random_state)
        self.budget = budget

    @abstractmethod
    def query(self, candidates, *args, return_utilities=False, **kwargs):
        """Determines for which candidate samples labels are to be queried.

        The query startegy determines the most useful samples in candidates,
        which can be acquired within the budgeting constraint specified by
        `budget`. Please note that, this method does not change the internal
        state of the query strategy. To adapt the query strategy to the
        selected candidates, use `update(...)`.

        Parameters
        ----------
        candidates : {array-like, sparse matrix} of shape\
                (n_candidates, n_features)
            The samples which may be queried. Sparse matrices are accepted
            only if they are supported by the base query strategy.
        return_utilities : bool, default=False
            If `True`, also return the utilities based on the query strategy.

        Returns
        -------
        queried_indices : np.ndarray of shape (n_queried_indices,)
            The indices of samples in candidates whose labels are queried,
            with `0 <= queried_indices <= n_candidates`.
        utilities: np.ndarray of shape (n_candidates,),
            The utilities based on the query strategy. Only provided if
            `return_utilities` is `True`.
        """
        raise NotImplementedError

    @abstractmethod
    def update(
        self,
        candidates,
        queried_indices,
        *args,
        budget_manager_param_dict=None,
        **kwargs,
    ):
        """Updates the budget manager and the count for seen and queried
        labels. This function should be used in conjunction with the `query`
        function.

        Parameters
        ----------
        candidates : {array-like, sparse matrix} of shape\
                (n_candidates, n_features)
            The samples which may be queried. Sparse matrices are accepted
            only if they are supported by the base query strategy.
        queried_indices : np.ndarray of shape (n_queried_indices,)
            The indices of samples in candidates whose labels are queried,
            with `0 <= queried_indices <= n_candidates`.
        budget_manager_param_dict : dict, default=None
            Optional kwargs for budget_manager.

        Returns
        -------
        self : SingleAnnotatorStreamQueryStrategy
            The query strategy returns itself, after it is updated.
        """
        raise NotImplementedError

    def _validate_random_state(self):
        """Creates a copy 'random_state_' if random_state is an instance of
        np.random_state. If not create a new random state. See also
        :func:`~sklearn.utils.check_random_state`
        """
        if not hasattr(self, "random_state_"):
            self.random_state_ = deepcopy(self.random_state)
        self.random_state_ = check_random_state(self.random_state_)

    def _validate_budget(self):
        """Creates a copy "budget_" if budget is a float between 0 and 1. If it
        is `None`, `budget_` is set to 0.1.
        """
        if self.budget is not None:
            self.budget_ = self.budget
        else:
            self.budget_ = 0.1
        check_scalar(
            self.budget_,
            "budget",
            float,
            min_val=0.0,
            max_val=1.0,
            min_inclusive=False,
        )

    def _validate_data(
        self,
        candidates,
        return_utilities,
        *args,
        reset=True,
        **check_candidates_params,
    ):
        """Validate input data and set or check the `n_features_in_` attribute.

        Parameters
        ----------
        candidates: array-like of shape (n_candidates, n_features)
            The samples which may be queried. Sparse matrices are accepted
            only if they are supported by the base query strategy.
        return_utilities : bool,
            If `True`, also return the utilities based on the query strategy.
        reset : bool, default=True
            Whether to reset the `n_features_in_` attribute.
            If False, the input will be checked for consistency with data
            provided when reset was last True.
        **check_candidates_params : kwargs
            Parameters passed to :func:`sklearn.utils.check_array`.

        Returns
        -------
        candidates: np.ndarray, shape (n_candidates, n_features)
            Checked candidate samples.
        return_utilities : bool,
            Checked boolean value of `return_utilities`.
        """
        # Check candidate samples.
        candidates = check_array(candidates, **check_candidates_params)

        # Check number of features.
        check_n_features(self, candidates, reset=reset)

        # Check return_utilities.
        check_scalar(return_utilities, "return_utilities", bool)

        # Check random state.
        self._validate_random_state()

        # Check budgetmanager.
        self._validate_budget()

        return candidates, return_utilities


class SkactivemlClassifier(ClassifierMixin, BaseEstimator, ABC):
    """Skactiveml Classifier

    Base class for `scikit-activeml` classifiers such that missing labels,
    user-defined classes, and cost-sensitive classification (i.e., cost matrix)
    can be handled.

    Parameters
    ----------
    classes : array-like of shape (n_classes,) or a list of such array-likes, \
            default=None
        - A flat vocabulary describes single-output classification and is
          applied to every annotator entry for multi-annotator components.
        - Nested binary vocabularies describe multi-label classification, one
          class vocabulary per label output. Nested non-binary vocabularies
          describe recognized multi-output classification semantics.
    missing_label : scalar, string, np.nan, or None, default=np.nan
        Value to represent a missing label.
    cost_matrix : array-like of shape (n_classes, n_classes)
        Cost matrix with `cost_matrix[i,j]` indicating cost of predicting class
        `classes[j]`  for a sample of class `classes[i]`. Can be only set, if
        `classes` is not `None` and one-dimensional, which corresponds to
        single output classification.
    random_state : int or RandomState instance or None, default=None
        Determines random number for `predict` method. Pass an int for
        reproducible results across multiple method calls.
    target_type : "auto" or "single-output" or "multi-label" or \
            "multi-output", default="auto"
        Declared target type. Components reject resolved target specifications
        outside their exact capabilities.

    Attributes
    ----------
    target_spec_ : skactiveml.utils.TargetSpec
        Immutable target specification established by a successful fit. Its
        class vocabularies use the canonical ordering of `classes_`.
    """

    def __init__(
        self,
        classes=None,
        missing_label=MISSING_LABEL,
        cost_matrix=None,
        random_state=None,
        target_type="auto",
    ):
        self.classes = classes
        self.missing_label = missing_label
        self.cost_matrix = cost_matrix
        self.random_state = random_state
        self.target_type = target_type

    @property
    def _target_capabilities(self):
        """Exact target semantics supported by a conservative classifier."""
        return frozenset(
            {("classification", "single-output", "single-annotator")}
        )

    def _resolve_target_spec(self, y, classes=None):
        annotation_type = getattr(self, "_annotation_type", "single-annotator")
        resolution_y = y
        if annotation_type == "multi-annotator":
            y_array = np.asarray(y)
            if y_array.ndim == 1 and y_array.size == 0:
                resolution_y = y_array.reshape(0, 1)
        target_spec = resolve_target_spec(
            resolution_y,
            task="classification",
            target_type=getattr(self, "target_type", "auto"),
            annotation_type=annotation_type,
            classes=self.classes if classes is None else classes,
            missing_label=self.missing_label,
        )
        _check_target_spec_capability(
            type(self).__name__, target_spec, self._target_capabilities
        )
        return target_spec

    def _resolve_fitting_target_spec(
        self, y, established_spec=None, classes=None
    ):
        classes = self.classes if classes is None else classes
        if classes is None and established_spec is not None:
            classes = established_spec.classes
        resolved_spec = self._resolve_target_spec(y, classes=classes)
        return _reuse_established_target_spec(resolved_spec, established_spec)

    def _resolve_target_spec_for_fit(self, y, *, is_incremental, classes=None):
        established_spec = (
            getattr(self, "target_spec_", None) if is_incremental else None
        )
        return self._resolve_fitting_target_spec(
            y,
            established_spec=established_spec,
            classes=classes,
        )

    def _initialize_label_state(self, y, classes=None):
        """Initialize resolved class metadata without fitting model state."""
        effective_classes = self.classes if classes is None else classes
        annotation_type = getattr(self, "_annotation_type", "single-annotator")
        resolution_y = np.asarray(y)
        if annotation_type == "multi-annotator" and resolution_y.ndim == 1:
            resolution_y = resolution_y.reshape(-1, 1)
        target_spec = self._resolve_target_spec(
            resolution_y, classes=effective_classes
        )
        self.target_spec_ = target_spec
        self._le = ExtLabelEncoder(
            classes=target_spec.classes,
            missing_label=self.missing_label,
            target_type=target_spec.target_type,
        ).fit(resolution_y)
        self.classes_ = self._le.classes_
        if target_spec.target_type == "multi-label":
            self.cost_matrix_ = None
        else:
            self.cost_matrix_ = (
                1 - np.eye(len(self.classes_))
                if self.cost_matrix is None
                else self.cost_matrix
            )
            self.cost_matrix_ = check_cost_matrix(
                self.cost_matrix_, len(self.classes_)
            )

    @abstractmethod
    def fit(self, X, y, sample_weight=None):
        """Fit the model using `X` as training data and `y` as class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, ...)
            The samples `X` whose shape depends on the respective classifier.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs) or \
                (n_samples, n_annotators)
            Labels of the training data set (possibly including unlabeled
            ones indicated by `missing_label`). For multioutput
            problems, a row `y[i]` must either contain only observed
            labels or only `missing_label` values, i.e., no mixing
            within a row. For multi-annotator classification, a row can contain
            labeled and unlabeled entries, where `y[i, j]` indicates the
            potential class label for sample `X[i]` from annotator `j`.
        sample_weight : array-like of shape (n_samples,) or \
                (n_samples, n_outputs), default=None
            It contains the weights of the training samples. For two-
            dimensional targets, either one weight per sample or one weight
            per target entry can be provided.

        Returns
        -------
        self: skactiveml.base.SkactivemlClassifier
            The `skactiveml.base.SkactivemlClassifier` object fitted on the
            training data.
        """
        raise NotImplementedError

    def predict_proba(self, X, **kwargs):
        """Return probability estimates for the test data `X`.

        Parameters
        ----------
        X : array-like of shape (n_samples, ...)
            Test samples.

        Returns
        -------
        P : numpy.ndarray of shape (n_samples, classes)
            The class probabilities of the test samples. Classes are ordered
            according to `self.classes_`.
        """
        raise NotImplementedError

    def predict(self, X, **kwargs):
        """Return class label predictions for the test samples `X`.

        Parameters
        ----------
        X :  array-like of shape (n_samples, ...)
            Input samples.

        Returns
        -------
        y : numpy.ndarray of shape (n_samples,)
            Predicted class labels of the test samples `X`.
        """
        # Extract primary output.
        out = self.predict_proba(X, **kwargs)
        P = out[0] if isinstance(out, tuple) else out

        if self.target_spec_.target_type == "single-output":
            costs = np.dot(P, self.cost_matrix_)
            y_pred = rand_argmin(
                costs, random_state=self.random_state_, axis=1
            )
        elif self.target_spec_.target_type == "multi-label":
            y_pred = (P >= 0.5).astype(int, copy=False)

        # Transform labels and append extra outputs.
        y_pred = self._le.inverse_transform(y_pred)
        if isinstance(out, tuple):
            return (y_pred,) + out[1:]
        else:
            return y_pred

    def score(self, X, y, sample_weight=None):
        """Return the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, ...)
            Test samples.
        y : array-like of shape (n_samples,)
            True class labels of the test samples `X`.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights of the test sample `X`.

        Returns
        -------
        score : float
            Mean accuracy of `self.predict(X)` regarding `y`.
        """
        y_pred = self.predict(X)
        y_pred = self._le.transform(y_pred)
        y_true = self._le.transform(y)
        return accuracy_score(
            y_pred=y_pred, y_true=y_true, sample_weight=sample_weight
        )

    def _validate_data(
        self,
        X,
        y,
        sample_weight=None,
        check_X_dict=None,
        check_y_dict=None,
        reset=True,
        target_spec=None,
    ):
        target_spec = self._resolve_fitting_target_spec(
            y, established_spec=target_spec
        )
        if check_X_dict is None:
            check_X_dict = {"ensure_min_samples": 0, "ensure_min_features": 0}
        if check_y_dict is None:
            check_y_dict = {
                "ensure_min_samples": 0,
                "ensure_min_features": 0,
                "ensure_2d": False,
                "ensure_all_finite": False,
                "dtype": None,
            }

        # Check common classifier parameters.
        check_classifier_params(
            self.classes, self.missing_label, self.cost_matrix
        )

        # Store and check random state.
        self.random_state_ = check_random_state(self.random_state)

        # Create label encoder.
        self._le = ExtLabelEncoder(
            classes=target_spec.classes,
            missing_label=self.missing_label,
            target_type=target_spec.target_type,
        )

        # Check input parameters.
        y = check_array(y, **check_y_dict)
        error_msg = (
            "No class label is known because `y` contains no actual "
            "class labels and `classes` is not defined. Change at "
            "least on of both parameters to overcome this error."
        )
        structured_target = (
            target_spec.target_type == "multi-label"
            or target_spec.annotation_type == "multi-annotator"
        )
        if len(y) > 0:
            y = y if structured_target else column_or_1d(y, warn=True)
            y = self._le.fit_transform(y)
            if target_spec.target_type == "multi-label":
                is_unlabeled(y, missing_label=-1, target_type="multi-label")
        else:
            if self.classes is None:
                raise ValueError(error_msg)
            self._le.fit(self.classes)
            check_X_dict["ensure_2d"] = False
        X = check_array(X, **check_X_dict)
        check_consistent_length(X, y)
        check_n_features(self, X, reset=reset)

        # Update detected classes.
        self.classes_ = self._le.classes_

        # Check classes.
        if sample_weight is not None:
            sample_weight = check_array(sample_weight, **check_y_dict)
            if sample_weight.ndim == 1:
                if len(y) != len(sample_weight):
                    raise ValueError(
                        f"`y` has the length {len(y)} and `sample_weight` has "
                        f"the shape {sample_weight.shape}. Both need to have "
                        f"the same one-dimensional shape."
                    )
            elif sample_weight.ndim == 2 and structured_target:
                if not np.array_equal(y.shape, sample_weight.shape):
                    raise ValueError(
                        f"`y` has the shape {y.shape} and `sample_weight` has "
                        f"the shape {sample_weight.shape}. Both need to have "
                        f"identical shapes."
                    )
            else:
                raise ValueError(
                    "`sample_weight` must have shape `(n_samples,)` or, for "
                    "two-dimensional targets, the same shape as `y`."
                )

        # Update cost matrix.
        if target_spec.target_type == "multi-label":
            self.cost_matrix_ = None
        else:
            self.cost_matrix_ = (
                1 - np.eye(len(self.classes_))
                if self.cost_matrix is None
                else self.cost_matrix
            )
            self.cost_matrix_ = check_cost_matrix(
                self.cost_matrix_, len(self.classes_)
            )
            if self.classes is not None:
                class_indices = np.argsort(self.classes)
                self.cost_matrix_ = self.cost_matrix_[class_indices]
                self.cost_matrix_ = self.cost_matrix_[:, class_indices]

        self.target_spec_ = target_spec

        return X, y, sample_weight


class ClassFrequencyEstimator(SkactivemlClassifier):
    """Class Frequency Estimator

    Extends `scikit-activeml` classifiers to estimators that are able to
    estimate class frequencies for given samples (by calling `predict_freq`).

    Parameters
    ----------
    classes : array-like of shape (n_classes,) or a list of array-like of \
            shape (2,), default=None
        Holds the label for each class. Nested binary vocabularies describe
        one vocabulary per output for multi-label classification. If `None`,
        the classes are determined during the fit.
    missing_label : scalar or str or np.nan or None, default=np.nan
        Value to represent a missing label.
    cost_matrix : array-like of shape (n_classes, n_classes)
        Cost matrix with `cost_matrix[i,j]` indicating cost of predicting class
        `classes[j]`  for a sample of class `classes[i]`. Can be only set, if
        classes is not `None`.
    class_prior : float or array-like of shape (n_classes,) or \
            (n_outputs, 2), default=0
        Prior observations of the class frequency estimates. If `class_prior`
        is an array for single-output classification, the entry
        `class_prior[i]` indicates the non-negative prior number of samples
        belonging to class `classes_[i]`. For multi-label classification, an
        array must contain one binary prior per output. If `class_prior` is a
        float, it indicates the non-negative prior number of samples per class
        for every output.
    random_state : int or np.RandomState or None, default=None
        Determines random number for `predict` method. Pass an int for
        reproducible results across multiple method calls.
    target_type : "auto" or "single-output" or "multi-label" or \
            "multi-output", default="auto"
        Declared target type. Concrete estimators reject resolved
        specifications outside their exact capabilities.

    Attributes
    ----------
    class_prior_ : np.ndarray of shape (n_classes,) or (n_outputs, 2)
        Validated prior observations. The two-dimensional representation is
        used only for multi-label targets and follows each output's canonical
        binary class vocabulary.
    """

    def __init__(
        self,
        class_prior=0,
        classes=None,
        missing_label=MISSING_LABEL,
        cost_matrix=None,
        random_state=None,
        target_type="auto",
    ):
        super().__init__(
            classes=classes,
            missing_label=missing_label,
            cost_matrix=cost_matrix,
            random_state=random_state,
            target_type=target_type,
        )
        self.class_prior = class_prior

    @abstractmethod
    def predict_freq(self, X, **kwargs):
        """Return class frequency estimates for the test samples `X`.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            Test samples whose class frequencies are to be estimated.

        Returns
        -------
        F: array-like of shape (n_samples, n_classes) or \
                (n_samples, n_outputs, 2)
            The class frequency estimates of the test samples `X`. For
            multi-label targets, the final axis follows each output's
            canonical binary class vocabulary.
        """
        raise NotImplementedError

    def predict_proba(self, X, **kwargs):
        """Return probability estimates for the test data `X`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        P : array-like of shape (n_samples, n_classes) or \
                (n_samples, n_outputs)
            The class probabilities of the test samples. For multi-label
            targets, each entry is the probability of the second class in the
            corresponding canonical binary class vocabulary. An output with
            zero estimated frequencies and zero prior has probability `0.5`.
        """
        out = self.predict_freq(X, **kwargs)
        F = out[0] if isinstance(out, tuple) else out
        P = F + self.class_prior_
        target_type = getattr(
            getattr(self, "target_spec_", None),
            "target_type",
            "single-output",
        )
        if target_type == "multi-label":
            normalizer = np.sum(P, axis=-1)
            nonzero = normalizer > 0
            P[nonzero] /= normalizer[nonzero, np.newaxis]
            P[~nonzero] = 0.5
            return P[..., 1]

        normalizer = np.sum(P, axis=1)
        P[normalizer > 0] /= normalizer[normalizer > 0, np.newaxis]
        P[normalizer == 0, :] = [1 / len(self.classes_)] * len(self.classes_)
        return P

    def sample_proba(self, X, n_samples=10, random_state=None):
        """Samples probability vectors from Dirichlet distributions whose
        parameters `alphas` are defined as the sum of the frequency estimates
        returned by `predict_freq` and the `class_prior`.

        Parameters
        ----------
        X : array-like of shape (n_test_samples, n_features)
            Test samples for which `n_samples` probability vectors are to be
            sampled.
        n_samples : int, default=10
            Number of probability vectors to sample for each `X[i]`.
        random_state : int or numpy.random.RandomState or None, default=None
            Ensure reproducibility when sampling probability vectors from the
            Dirichlet distributions.

        Returns
        -------
        P : array-like of shape (n_samples, n_test_samples, n_classes) or \
                (n_samples, n_test_samples, n_outputs, 2)
            There are `n_samples` class probability vectors for each test
            sample in `X`. For multi-label targets, the final axis follows
            each output's canonical binary class vocabulary.

        Raises
        ------
        ValueError
            If any class has zero frequency observations after adding the
            prior. Set a positive `class_prior` to make every Dirichlet
            parameter positive.
        """
        random_state = check_random_state(random_state)
        alphas = self.predict_freq(X) + self.class_prior_
        target_type = getattr(
            getattr(self, "target_spec_", None),
            "target_type",
            "single-output",
        )
        if target_type == "multi-label":
            alphas = np.repeat(alphas[np.newaxis], n_samples, axis=0)
            if (alphas == 0).any():
                raise ValueError(
                    "There are zero frequency observations. "
                    "Set `class_prior > 0` to avoid this error."
                )
            R = random_state.standard_gamma(alphas)
            R_flat = R.reshape(-1, R.shape[-1])
            is_zero = R_flat.sum(axis=-1) == 0.0
            sampled_class_indices = random_state.choice(
                np.array(R.shape[-1]), size=is_zero.sum()
            )
            R_flat[np.flatnonzero(is_zero), sampled_class_indices] = 1.0
            return R / R.sum(axis=-1, keepdims=True)

        alphas = alphas.repeat(repeats=n_samples, axis=0)
        if (alphas == 0).any():
            raise ValueError(
                "There are zero frequency observations. "
                "Set `class_prior > 0` to avoid this error."
            )
        R = random_state.standard_gamma(alphas)
        R_sums = R.sum(axis=-1)
        is_zero = (R_sums == 0.0).ravel()
        sampled_class_indices = random_state.choice(
            np.array(R.shape[-1]), size=is_zero.sum()
        )
        R[is_zero, sampled_class_indices] = 1.0
        P = R / R.sum(axis=-1, keepdims=True)
        P = P.reshape(n_samples, len(X), P.shape[-1], order="F")
        return P

    def _validate_data(
        self,
        X,
        y,
        sample_weight=None,
        check_X_dict=None,
        check_y_dict=None,
        reset=True,
        target_spec=None,
    ):
        X, y, sample_weight = super()._validate_data(
            X=X,
            y=y,
            sample_weight=sample_weight,
            check_X_dict=check_X_dict,
            check_y_dict=check_y_dict,
            reset=reset,
            target_spec=target_spec,
        )

        # Check class prior.
        if self.target_spec_.target_type == "multi-label":
            n_outputs = len(self.classes_)
            if np.isscalar(self.class_prior):
                check_scalar(
                    self.class_prior,
                    name="class_prior",
                    target_type=(int, float),
                    min_val=0,
                )
                self.class_prior_ = np.full((n_outputs, 2), self.class_prior)
            else:
                class_prior = check_array(self.class_prior, ensure_2d=False)
                if class_prior.shape != (n_outputs, 2) or np.any(
                    class_prior < 0
                ):
                    raise ValueError(
                        "`class_prior` must be either a non-negative float or "
                        "an array of shape `(n_outputs, 2)` containing "
                        "non-negative values."
                    )
                self.class_prior_ = class_prior
        else:
            self.class_prior_ = check_class_prior(
                self.class_prior, len(self.classes_)
            )

        return X, y, sample_weight

    def _compute_class_frequency_vectors(self, y, sample_weight):
        """Convert encoded targets to per-sample class-frequency vectors."""
        if self.target_spec_.target_type == "single-output":
            return compute_vote_vectors(
                y=y,
                w=sample_weight,
                classes=np.arange(len(self.classes_)),
                missing_label=-1,
            )

        weights = (
            np.ones_like(y, dtype=float)
            if sample_weight is None
            else np.asarray(sample_weight, dtype=float).copy()
        )
        if weights.ndim == 1:
            weights = np.repeat(weights[:, np.newaxis], y.shape[1], axis=1)
        is_missing = y == -1
        weights[np.isnan(weights) | is_missing] = 0
        encoded_y = np.where(is_missing, 0, y).astype(int, copy=False)
        return np.eye(2)[encoded_y] * weights[..., np.newaxis]


class SkactivemlRegressor(RegressorMixin, BaseEstimator, ABC):
    """Skactiveml Regressor

    Base class for `scikit-activeml` regressors.

    Parameters
    ----------
    missing_label : scalar, string, np.nan, or None, default=np.nan
        Value to represent a missing label.
    random_state : int, RandomState or None, default=None
        Determines random number for `fit` and `predict` method. Pass an int
        for reproducible results across multiple method calls.
    target_type : "auto" or "single-output" or "multi-output", default="auto"
        Declared target type. Multi-output regression is recognized but not
        supported for execution in version 1.1.

    Attributes
    ----------
    target_spec_ : skactiveml.utils.TargetSpec
        Immutable target specification established by a successful fit. For
        regression, its `classes` field is `None`.
    """

    def __init__(
        self,
        missing_label=MISSING_LABEL,
        random_state=None,
        target_type="auto",
    ):
        self.missing_label = missing_label
        self.random_state = random_state
        self.target_type = target_type

    @property
    def _target_capabilities(self):
        """Exact target semantics supported by regressors in version 1.1."""
        return frozenset({("regression", "single-output", "single-annotator")})

    def _resolve_target_spec(self, y):
        target_spec = resolve_target_spec(
            y,
            task="regression",
            target_type=self.target_type,
            annotation_type="single-annotator",
            classes=None,
            missing_label=self.missing_label,
        )
        _check_target_spec_capability(
            type(self).__name__, target_spec, self._target_capabilities
        )
        return target_spec

    def _resolve_fitting_target_spec(self, y, established_spec=None):
        resolved_spec = self._resolve_target_spec(y)
        return _reuse_established_target_spec(resolved_spec, established_spec)

    @abstractmethod
    def fit(self, X, y, sample_weight=None):
        """Fit the model using `X` as training data and y as numerical labels.

        Parameters
        ----------
        X : matrix-like of shape (n_samples, n_features)
            The sample matrix X is the feature matrix representing the samples.
        y : array-like, shape (n_samples) or (n_samples, n_targets)
            It contains the labels of the training samples.
            The number of numerical labels may be variable for the samples,
            where missing labels are represented as `missing_label_`.
        sample_weight : array-like, shape (n_samples)
            It contains the weights of the training samples' values.

        Returns
        -------
        self: skactiveml.base.SkactivemlRegressor,
            The `skactiveml.base.SkactivemlRegressor` object fitted on the
            training data.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, X):
        """Return value predictions for the test samples `X`.

        Parameters
        ----------
        X :  array-like of shape (n_samples, n_features)
            Input samples.
        Returns
        -------
        y : numpy.ndarray of shape (n_samples,)
            Predicted values of the test samples `X`.
        """
        raise NotImplementedError

    def _validate_data(
        self,
        X,
        y,
        sample_weight=None,
        check_X_dict=None,
        check_y_dict=None,
        y_ensure_1d=True,
        reset=True,
        target_spec=None,
    ):
        target_spec = self._resolve_fitting_target_spec(
            y, established_spec=target_spec
        )

        if check_X_dict is None:
            check_X_dict = {"ensure_min_samples": 0, "ensure_min_features": 0}
        if check_y_dict is None:
            check_y_dict = {
                "ensure_min_samples": 0,
                "ensure_min_features": 0,
                "ensure_2d": False,
                "ensure_all_finite": False,
                "dtype": None,
            }

        check_missing_label(self.missing_label)
        self.missing_label_ = self.missing_label

        # Store and check random state.
        self.random_state_ = check_random_state(self.random_state)

        y = check_array(y, **check_y_dict)
        if len(y) > 0:
            y = column_or_1d(y) if y_ensure_1d else y
        else:
            check_X_dict["ensure_2d"] = False

        if sample_weight is not None:
            sample_weight = check_array(sample_weight, **check_y_dict)
            if not np.array_equal(y.shape, sample_weight.shape):
                raise ValueError(
                    f"`y` has the shape {y.shape} and `sample_weight` has the "
                    f"shape {sample_weight.shape}. Both need to have "
                    f"identical shapes."
                )
        X = check_array(X, **check_X_dict)
        check_consistent_length(X, y)
        check_n_features(self, X, reset=reset)

        self.target_spec_ = target_spec

        return X, y, sample_weight


class ProbabilisticRegressor(SkactivemlRegressor):
    """ProbabilisticRegressor

    Base class for `scikit-activeml` probabilistic regressors.

    """

    @abstractmethod
    def predict_target_distribution(self, X):
        """Returns the predicted target distribution conditioned on the test
        samples `X`.

        Parameters
        ----------
        X :  array-like, shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        dist : scipy.stats._distn_infrastructure.rv_frozen
            The distribution of the targets at the test samples.

        """
        raise NotImplementedError

    def predict(self, X, return_std=False, return_entropy=False):
        """Returns the mean, std (optional) and differential entropy (optional)
        of the predicted target distribution conditioned on the test samples
        `X`.

        Parameters
        ----------
        X :  array-like of shape (n_samples, n_features)
            Input samples.
        return_std : bool, default=False
            Whether to return the standard deviation.
        return_entropy : bool, default=False
            Whether to return the differential entropy.

        Returns
        -------
        mu : numpy.ndarray, shape (n_samples,)
            Predicted mean conditioned on `X`.
        std : numpy.ndarray, shape (n_samples,), optional
            Predicted standard deviation conditioned on `X`.
        entropy : numpy.ndarray, optional
            Predicted differential entropy conditioned on `X`.
        """
        check_scalar(return_std, "return_std", bool)
        check_scalar(return_entropy, "return_entropy", bool)
        rv = self.predict_target_distribution(X)
        result = (rv.mean(),)
        if return_std:
            result += (rv.std(),)
        if return_entropy:
            result += (rv.entropy(),)
        if len(result) == 1:
            result = result[0]
        return result

    def sample_y(self, X, n_samples=1, random_state=None):
        """Returns random samples from the predicted target distribution
        conditioned on the test samples `X`.

        Parameters
        ----------
        X :  array-like of shape (n_samples_X, n_features)
            Input samples, where the target values are drawn from.
        n_samples: int, default=1
            Number of random samples to be drawn.
        random_state : int or RandomState instance or None, default=None
            Determines random number generation to randomly draw samples. Pass
            an int for reproducible results across multiple method calls.

        Returns
        -------
        y_samples : numpy.ndarray of shape (n_samples_X, n_samples)
            Drawn random target samples.
        """
        rv = self.predict_target_distribution(X)
        rv_samples = rv.rvs(
            size=(n_samples, len(X)), random_state=random_state
        )
        return rv_samples.T


if successful_skorch_torch_import:

    __all__ += ["SkorchMixin"]

    class SkorchMixin(ABC):
        """
        Minimal mixin to build and train a `skorch.NeuralNet`.

        Subclasses must implement the abstract methods to provide the module,
        criterion, validation kwargs, and training data. This mixin always
        rebuilds and initializes `self.neural_net_` on `initialize` and
        fits only on training data in `_fit`.
        """

        def initialize(self, X=None, y=None, enforce_check_X_y=False):
            """
            Initialize the wrapper and (optionally) validate inputs.

            If any data is provided or `enforce_check_X_y` is True, inputs
            are validated via `_validate_data`. A new `skorch.NeuralNet`
            is then created and assigned to `self.neural_net_`.

            Parameters
            ----------
            X : array-like of shape (n_samples, ...), default=None
                Input samples for optional validation.
            y : array-like of shape (n_samples, ...), default=None
                Target values for optional validation.
            enforce_check_X_y : bool, default=False
                Whether to validate even if both `X` and `y` are `None`.

            Returns
            -------
            self : SkorchMixin
                Returned when no input data was supplied
                (both `X` and `y` are `None`).
            X_out, y_out : tuple of nd.array, optional
                Validated `X` and `y` as a tuple, returned when
                `enforce_check_X_y=True`.
            """
            has_data = (X is not None) or (y is not None)
            vd_kwargs = self._validate_data_kwargs()
            if enforce_check_X_y or has_data:
                X, y, _ = self._validate_data(X=X, y=y, **vd_kwargs)

            module, criterion, nn_params = self._net_parts(X=X, y=y)
            check_type(nn_params, "neural_net_param_dict", dict)
            nn_params = dict(nn_params)
            invalid_keys = ["module", "criterion", "predict_nonlinearity"]
            for k in invalid_keys:
                if k in nn_params:
                    raise ValueError(
                        f"{k} must not be a key in `neural_net_param_dict`."
                    )
            self.neural_net_ = NeuralNet(
                module=module,
                criterion=criterion,
                predict_nonlinearity=None,
                **nn_params,
            ).initialize()

            return (self, X, y) if enforce_check_X_y else self

        def _fit(self, fit_function, X, y, **fit_params):
            """
            Initialize and fit the internal `skorch` model on training
            data.

            If the model is uninitialized, or `fit_function == 'fit'` and
            `self.neural_net_.warm_start` is `False`, the network is
            re-initialized.

            Parameters
            ----------
            fit_function : {'fit', 'partial_fit'}
                Name of the caller, used to decide whether to reinitialize when
                warm start is off.
            X : array-like of shape (n_samples, ...)
                Training inputs (may include unlabeled samples).
            y : array-like of shape (n_samples, ...)
                Training targets; unlabeled entries must follow the subclass'
                convention (e.g., `self.missing_label`).
            **fit_params : dict
                Extra keyword arguments forwarded to
                `self.neural_net_.partial_fit`.

            Returns
            -------
            self : SkorchMixin
                The fitted estimator.
            """
            need_reinit = (not hasattr(self, "neural_net_")) or (
                fit_function == "fit"
                and not getattr(self.neural_net_, "warm_start", False)
            )
            if need_reinit:
                _, X, y = self.initialize(X=X, y=y, enforce_check_X_y=True)
            else:
                vd_kwargs = self._validate_data_kwargs()
                if hasattr(self, "target_spec_"):
                    vd_kwargs["target_spec"] = self.target_spec_
                X, y, _ = self._validate_data(X=X, y=y, **vd_kwargs)

            X_train, y_train = self._return_training_data(X=X, y=y)
            if X_train is not None and y_train is not None:
                self.neural_net_.partial_fit(X_train, y_train, **fit_params)
            return self

        def _forward_with_named_outputs(
            self,
            X,
            forward_outputs,
            extra_outputs=None,
        ):
            """Run `module.forward(X)` once and return the primary output plus
            optionally requested extra outputs as NumPy arrays.

            The primary output is defined as the first entry of
            `forward_outputs` (after applying its transform, if any), or the
            sole output of `module.forward` if `forward_outputs` is `None`.
            Primary and extra outputs are always returned after applying their
            configured transforms.

            Parameters
            ----------
            X : array-like of shape (n_samples, ...)
                Input samples. It is assumed that `X` has already been
                validated and that `self.neural_net_` is initialized.
            forward_outputs : dict[str, tuple[int, Callable | None]]
                `dict` that describes how to obtain and post-process the
                outputs of `module.forward` for prediction.

                Given `raw_outputs = module.forward(X)`, each entry
                `name -> (idx, transform)` is interpreted as:

                - `idx`: integer index of `raw_outputs` (0-based).
                - `transform`: callable `f(tensor) -> tensor` or `None`.
                  If `transform` is not `None`, it is applied to the selected
                  raw tensor; otherwise the raw tensor is used.
            extra_outputs : None or str or sequence of str, default=None
                Names of additional outputs to return next to the primary
                output. Must be a subset of `forward_outputs.keys()` if
                `forward_outputs` is not `None`. The first key in
                `forward_outputs` (the primary output) is not allowed here.
                Duplicate entries are not allowed.

            Returns
            -------
            output : numpy.ndarray or tuple of numpy.ndarray
                If `extra_outputs is None`, returns the primary output as a
                single NumPy array. Otherwise, returns a tuple whose first
                element is the primary output and whose remaining elements are
                the requested extra outputs in the order specified by
                `extra_outputs`.
            """
            # Check forward_outputs configured.
            _check_forward_outputs(forward_outputs=forward_outputs)

            # Primary output = first configured output
            # (dicts preserve insertion order).
            primary_name = next(iter(forward_outputs))

            # Normalize and validate extra_outputs:
            # - None / str / sequence of str,
            # - subset of forward_outputs.keys(),
            # - no duplicates,
            # - no primary_name.
            extra_names = self._normalize_extra_outputs(
                extra_outputs,
                allowed_names=forward_outputs.keys(),
                primary_name=primary_name,
            )

            # Run module forward once.
            fw_out = self.neural_net_.forward(X)

            # Normalize to tuple of raw outputs.
            if isinstance(fw_out, tuple):
                raw_outputs = fw_out
            else:
                raw_outputs = (fw_out,)

            # Check that all indices are within range of raw_outputs.
            if forward_outputs:
                max_idx = max(idx for idx, _ in forward_outputs.values())
                if max_idx >= len(raw_outputs):
                    raise ValueError(
                        f"`forward_outputs` references raw output index "
                        f"{max_idx}, but module.forward returned only "
                        f"{len(raw_outputs)} object(s)."
                    )

            # Helper to extract and transform a single named output lazily.
            def _get_named(name: str):
                idx, transform = forward_outputs[name]
                value = raw_outputs[idx]
                if transform is not None:
                    value = transform(value)
                return to_numpy(value)

            # Primary output (transform applied here).
            primary_np = _get_named(primary_name)

            # No extra outputs.
            if not extra_names:
                return primary_np

            extras_np = tuple(_get_named(name) for name in extra_names)
            return (primary_np,) + extras_np

        @staticmethod
        def _normalize_extra_outputs(
            extra_outputs, allowed_names, primary_name=None
        ):
            """Validate `extra_outputs` and return a list of names.

            Parameters
            ----------
            extra_outputs : None or str or sequence of str
                User-specified extra outputs.
            allowed_names : Collection[str]
                Set or iterable of allowed names, e.g.,
                `forward_outputs.keys()`.
            primary_name : str or None, default=None
                Name of the primary output which must not be requested
                as extra.

            Returns
            -------
            list[str]
                Validated list of extra output names.
            """
            if extra_outputs is None:
                return []

            # Normalize to list of strings
            if isinstance(extra_outputs, str):
                names = [extra_outputs]
            elif isinstance(extra_outputs, Sequence) and not isinstance(
                extra_outputs, bytes
            ):
                names = list(extra_outputs)
            else:
                raise TypeError(
                    "`extra_outputs` must be None, a string, or a sequence "
                    f"of strings, got {type(extra_outputs)}."
                )

            if not all(isinstance(n, str) for n in names):
                raise TypeError(
                    "All entries in `extra_outputs` must be strings."
                )

            # No duplicates
            if len(set(names)) != len(names):
                raise ValueError(
                    "`extra_outputs` must not contain duplicate names."
                )

            allowed_names = set(allowed_names)
            unknown = [n for n in names if n not in allowed_names]
            if unknown:
                raise ValueError(
                    f"Requested extra output(s) {unknown!r} are not defined; "
                    f"allowed names are {sorted(allowed_names)!r}."
                )

            if primary_name is not None and primary_name in names:
                raise ValueError(
                    f"Primary output {primary_name!r} (first key in "
                    f"`forward_outputs`) cannot be requested again as an "
                    f"`extra_output`."
                )

            return names

        @abstractmethod
        def _net_parts(self, X=None, y=None):
            """Assemble and validate network components.

            Implementations should perform any optional checks or normalization
            of constructor/init parameters (e.g., shape consistency, dtype
            checks, wrapping criteria), then return the ready-to-use pieces for
            `skorch.NeuralNet`.

            Parameters
            ----------
            X : array-like of shape (n_samples, ...), default=None
                Input samples for optional validation.
            y : array-like of shape (n_samples, ...), default=None
                Target values for optional validation.

            Returns
            -------
            module : torch.nn.Module.__class__ or torch.nn.Module
                A PyTorch `torch.nn.Module`. In general, the uninstantiated
                class should be passed, although instantiated modules will also
                work.
            criterion : torch.nn.Module.__class__
                The criterion (loss) used to optimize the module.
            params : dict
                Keyword arguments (excluding `predict_non_linearity`) for
                `skorch.NeuralNet` construction. Must be a mapping and may be
                empty.
            """
            raise NotImplementedError

        @abstractmethod
        def _validate_data_kwargs(self):
            """Return kwargs forwarded to `_validate_data`.

            Returns
            -------
            kwargs : dict or None
                Keyword arguments consumed by `_validate_data`.
            """
            raise NotImplementedError

        @abstractmethod
        def _validate_data(self, X, y, **kwargs):
            """Validate inputs and return cleaned arrays.

            Parameters
            ----------
            X : array-like of shape (n_samples, ...)
                Input samples.
            y : array-like of shape (n_samples, ...)
                Target values.
            **kwargs
                Additional arguments controlling validation.

            Returns
            -------
            X_out : np.ndarray
                Validated `X`.
            y_out : np.ndarray
                Validated `y`.
            sample_weight_or_dummy : Any
                Third return to maintain compatibility with callers expecting
                sample weights.
            """
            raise NotImplementedError

        @abstractmethod
        def _return_training_data(self, X, y):
            """Return only samples and labels required for training.

            Parameters
            ----------
            X : array-like of shape (n_samples, ...)
                Input samples.
            y : array-like of shape (n_samples, ...)
                Targets with unlabeled entries following the subclass'
                convention.

            Returns
            -------
            X_train : np.ndarray or None
                Training samples or `None` if none exist.
            y_train : np.ndarray or None
                Training labels or `None` if none exist.
            """
            raise NotImplementedError
