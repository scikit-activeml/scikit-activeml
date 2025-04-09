"""
The :mod:`skactiveml.base` package implements the base classes for
:mod:`skactiveml`.
"""

import warnings
from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.metrics import accuracy_score
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import (
    check_array,
    check_consistent_length,
    column_or_1d,
)

from .exceptions import MappingError
from .utils import (
    MISSING_LABEL,
    is_labeled,
    is_unlabeled,
    unlabeled_indices,
    ExtLabelEncoder,
    rand_argmin,
    check_classifier_params,
    check_random_state,
    check_cost_matrix,
    check_scalar,
    check_class_prior,
    check_missing_label,
    check_indices,
    check_n_features,
)

# '__all__' is necessary to create the sphinx docs.
__all__ = [
    "QueryStrategy",
    "SingleAnnotatorPoolQueryStrategy",
    "MultiAnnotatorPoolQueryStrategy",
    "BudgetManager",
    "SingleAnnotatorStreamQueryStrategy",
    "SkactivemlClassifier",
    "ClassFrequencyEstimator",
    "AnnotatorModelMixin",
    "SkactivemlRegressor",
    "ProbabilisticRegressor",
]


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
    ):
        """Validate input data, all attributes and set or check the
        `n_features_in_` attribute.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data set, usually complete, i.e. including the labeled and
            unlabeled samples.
        y : array-like of shape (n_samples, *)
            Labels of the training data set (possibly including unlabeled ones
            indicated by self.MISSING_LABEL.
        candidates : None or array-like of shape (n_candidates), dtype=int or \
                array-like of shape (n_candidates, n_features), default=None
            - If `candidates` is `None`, the unlabeled samples from
              `(X,y)` are considered as `candidates`.
            - If `candidates` is of shape `(n_candidates,)` and of type
              `int`, `candidates` is considered as the indices of the
              samples in `(X,y)`.
            - If `candidates` is of shape `(n_candidates, *)`, the
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
        y : np.ndarray of shape (n_samples, *)
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
        seed_mult = int(np.sum(is_unlabeled(y, self.missing_label_))) + 1
        if candidates is not None:
            candidates = np.array(candidates)
            if candidates.ndim == 1:
                candidates = check_indices(candidates, y, dim=0)
            else:
                check_candidates_dict = deepcopy(check_X_dict)
                check_candidates_dict["ensure_2d"] = False
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
    """

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
            - If `candidates` is of shape `(n_candidates, *)`, the
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
            - If `candidates` is of shape `(n_candidates, *)`, the
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

        (
            X,
            y,
            candidates,
            batch_size,
            return_utilities,
        ) = super()._validate_data(
            X, y, candidates, batch_size, return_utilities, reset, check_X_dict
        )
        y = column_or_1d(y, warn=True)

        if candidates is None:
            n_candidates = int(
                np.sum(is_unlabeled(y, missing_label=self.missing_label_))
            )
        else:
            n_candidates = len(candidates)

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
            - If `candidates` is of shape `(n_candidates, *)`, the
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
            ulbd_idx = unlabeled_indices(y, self.missing_label_)
            return X[ulbd_idx], ulbd_idx
        elif candidates.ndim == 1:
            if allow_only_unlabeled:
                if is_labeled(y[candidates], self.missing_label_).any():
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
    missing_label : scalar or string or np.nan or None, default=np.nan
        Value to represent a missing label.
    random_state : int or RandomState instance, default=None
        Controls the randomness of the estimator.
    """

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
            annotators = check_array(
                annotators, ensure_2d=False, allow_nd=True
            )

            if annotators.ndim == 1:
                annotators = check_indices(annotators, y, dim=1)
            elif annotators.ndim == 2:
                annotators = check_array(annotators, dtype=bool)
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
    classes : array-like of shape (n_classes), default=None
        Holds the label for each class. If `None`, the classes are determined
        during the fit.
    missing_label : scalar, string, np.nan, or None, default=np.nan
        Value to represent a missing label.
    cost_matrix : array-like of shape (n_classes, n_classes)
        Cost matrix with `cost_matrix[i,j]` indicating cost of predicting class
        `classes[j]`  for a sample of class `classes[i]`. Can be only set, if
        `classes` is not `None`.
    random_state : int or RandomState instance or None, default=None
        Determines random number for `predict` method. Pass an int for
        reproducible results across multiple method calls.

    Attributes
    ----------
    classes_ : array-like of shape (n_classes,)
        Holds the label for each class after fitting.
    cost_matrix_ : array-like,of shape (classes, classes)
        Cost matrix after fitting with `cost_matrix_[i,j]` indicating cost of
        predicting class `classes_[j]`  for a sample of class `classes_[i]`.
    """

    def __init__(
        self,
        classes=None,
        missing_label=MISSING_LABEL,
        cost_matrix=None,
        random_state=None,
    ):
        self.classes = classes
        self.missing_label = missing_label
        self.cost_matrix = cost_matrix
        self.random_state = random_state

    @abstractmethod
    def fit(self, X, y, sample_weight=None):
        """Fit the model using X as training data and y as class labels.

        Parameters
        ----------
        X : matrix-like, shape (n_samples, n_features)
            The sample matrix `X` is the feature matrix representing the
            samples.
        y : array-like, shape (n_samples) or (n_samples, n_outputs)
            It contains the class labels of the training samples.
            The number of class labels may be variable for the samples, where
            missing labels are represented the attribute `missing_label`.
        sample_weight : array-like, shape (n_samples) or (n_samples, n_outputs)
            It contains the weights of the training samples' class labels.
            It must have the same shape as `y`.

        Returns
        -------
        self: skactiveml.base.SkactivemlClassifier,
            The `skactiveml.base.SkactivemlClassifier` object fitted on the
            training data.
        """
        raise NotImplementedError

    def predict_proba(self, X):
        """Return probability estimates for the test data X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        Returns
        -------
        P : numpy.ndarray of shape (n_samples, classes)
            The class probabilities of the test samples. Classes are ordered
            according to `self.classes_`.
        """
        raise NotImplementedError

    def predict(self, X):
        """Return class label predictions for the test samples `X`.

        Parameters
        ----------
        X :  array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        y : numpy.ndarray of shape (n_samples,)
            Predicted class labels of the test samples `X`.
        """
        P = self.predict_proba(X)
        costs = np.dot(P, self.cost_matrix_)
        y_pred = rand_argmin(costs, random_state=self.random_state_, axis=1)
        y_pred = self._le.inverse_transform(y_pred)
        y_pred = np.asarray(y_pred, dtype=self.classes_.dtype)
        return y_pred

    def score(self, X, y, sample_weight=None):
        """Return the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        y : array-like of shape (n_samples,)
            True labels for `X`.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        score : float
            Mean accuracy of `self.predict(X)` regarding `y`.
        """
        y = self._le.transform(y)
        y_pred = self._le.transform(self.predict(X))
        return accuracy_score(y, y_pred, sample_weight=sample_weight)

    def _validate_data(
        self,
        X,
        y,
        sample_weight=None,
        check_X_dict=None,
        check_y_dict=None,
        y_ensure_1d=True,
        reset=True,
    ):
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
            classes=self.classes, missing_label=self.missing_label
        )

        # Check input parameters.
        y = check_array(y, **check_y_dict)
        error_msg = (
            "No class label is known because 'y' contains no actual "
            "class labels and 'classes' is not defined. Change at "
            "least on of both to overcome this error."
        )
        if len(y) > 0:
            y = column_or_1d(y) if y_ensure_1d else y
            y = self._le.fit_transform(y)
            is_lbdl = is_labeled(y, missing_label=-1)
            if len(y[is_lbdl]) > 0:
                check_classification_targets(y[is_lbdl])
            if len(self._le.classes_) == 0:
                raise ValueError(error_msg)
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
            if not np.array_equal(y.shape, sample_weight.shape):
                raise ValueError(
                    f"`y` has the shape {y.shape} and `sample_weight` has the "
                    f"shape {sample_weight.shape}. Both need to have "
                    f"identical shapes."
                )

        # Update cost matrix.
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

        return X, y, sample_weight


class ClassFrequencyEstimator(SkactivemlClassifier):
    """Class Frequency Estimator

    Extends `scikit-activeml` classifiers to estimators that are able to
    estimate class frequencies for given samples (by calling `predict_freq`).

    Parameters
    ----------
    classes : array-like, shape (n_classes), default=None
        Holds the label for each class. If `None`, the classes are determined
        during the fit.
    missing_label : scalar or str or np.nan or None, default=np.nan
        Value to represent a missing label.
    cost_matrix : array-like of shape (n_classes, n_classes)
        Cost matrix with `cost_matrix[i,j]` indicating cost of predicting class
        `classes[j]`  for a sample of class `classes[i]`. Can be only set, if
        classes is not `None`.
    class_prior : float or array-like, shape (n_classes), default=0
        Prior observations of the class frequency estimates. If `class_prior`
        is an array, the entry `class_prior[i]` indicates the non-negative
        prior number of samples belonging to class `classes_[i]`. If
        `class_prior` is a float, `class_prior` indicates the non-negative
        prior number of samples per class.
    random_state : int or np.RandomState or None, default=None
        Determines random number for `predict` method. Pass an int for
        reproducible results across multiple method calls.

    Attributes
    ----------
    classes_ : np.ndarray of shape (n_classes)
        Holds the label for each class after fitting.
    class_prior_ : np.ndarray of shape (n_classes)
        Prior observations of the class frequency estimates. The entry
        `class_prior_[i]` indicates the non-negative prior number of samples
        belonging to class `classes_[i]`.
    cost_matrix_ : np.ndarray of shape (classes, classes)
        Cost matrix with `cost_matrix_[i,j]` indicating cost of predicting
        class `classes_[j]` for a sample of class `classes_[i]`.
    """

    def __init__(
        self,
        class_prior=0,
        classes=None,
        missing_label=MISSING_LABEL,
        cost_matrix=None,
        random_state=None,
    ):
        super().__init__(
            classes=classes,
            missing_label=missing_label,
            cost_matrix=cost_matrix,
            random_state=random_state,
        )
        self.class_prior = class_prior

    @abstractmethod
    def predict_freq(self, X):
        """Return class frequency estimates for the test samples `X`.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            Test samples whose class frequencies are to be estimated.

        Returns
        -------
        F: array-like of shape (n_samples, classes)
            The class frequency estimates of the test samples `X`. Classes are
            ordered according to attribute `classes_`.
        """
        raise NotImplementedError

    def predict_proba(self, X):
        """Return probability estimates for the test data `X`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        P : array-like of shape (n_samples, classes)
            The class probabilities of the test samples. Classes are ordered
            according to `self.classes_`.
        """
        # Normalize probabilities of each sample.
        P = self.predict_freq(X) + self.class_prior_
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
        P : array-like of shape (n_samples, n_test_samples, n_classes)
            There are `n_samples` class probability vectors for each test
            sample in `X`. Classes are ordered according to `self.classes_`.
        """
        random_state = check_random_state(random_state)
        alphas = self.predict_freq(X) + self.class_prior_
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
        y_ensure_1d=True,
    ):
        X, y, sample_weight = super()._validate_data(
            X=X,
            y=y,
            sample_weight=sample_weight,
            check_X_dict=check_X_dict,
            check_y_dict=check_y_dict,
            y_ensure_1d=y_ensure_1d,
        )

        # Check class prior.
        self.class_prior_ = check_class_prior(
            self.class_prior, len(self.classes_)
        )

        return X, y, sample_weight


class SkactivemlRegressor(RegressorMixin, BaseEstimator, ABC):
    """Skactiveml Regressor

    Base class for `scikit-activeml` regressors.

    Parameters
    __________
    missing_label : scalar, string, np.nan, or None, default=np.nan
        Value to represent a missing label.
    random_state : int, RandomState or None, default=None
        Determines random number for `fit` and `predict` method. Pass an int
        for reproducible results across multiple method calls.
    """

    def __init__(self, missing_label=MISSING_LABEL, random_state=None):
        self.missing_label = missing_label
        self.random_state = random_state

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
    ):
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


class AnnotatorModelMixin(ABC):
    """Annotator Model

    Base class of all annotator models estimating the performances of
    annotators for given samples.
    """

    @abstractmethod
    def predict_annotator_perf(self, X):
        """Calculates the performance of an annotator to provide the true label
        for a given sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        Returns
        -------
        P_annot : numpy.ndarray of shape (n_samples, n_annotators)
            `P_annot[i,l]` is the performance of annotator `l` regarding the
             annotation of sample `X[i]`.
        """
        raise NotImplementedError
