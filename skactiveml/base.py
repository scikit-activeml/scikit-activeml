"""
The :mod:`skactiveml.base` package implements the base classes for
:mod:`skactiveml`.
"""

import warnings
from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
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
    missing_label : scalar or string or np.nan or None, optional
    (default=np.nan)
        Value to represent a missing label.
    random_state : int or RandomState instance, optional (default=None)
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
        candidates : None or array-like of shape (n_candidates), dtype=int or
            array-like of shape (n_candidates, n_features),
            optional (default=None)
            If candidates is None, the unlabeled samples from (X,y) are
            considered as candidates.
            If candidates is of shape (n_candidates) and of type int,
            candidates is considered as the indices of the samples in (X,y).
            If candidates is of shape (n_candidates, n_features), the
            candidates are directly given in candidates (not necessarily
            contained in X). This is not supported by all query strategies.
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
        candidates : None or np.ndarray of shape (n_candidates), dtype=int or
            np.ndarray of shape (n_candidates, n_features)
            Checked candidate samples.
        batch_size : int
            Checked number of samples to be selected in one AL cycle.
        return_utilities : bool,
            Checked boolean value of `return_utilities`.
        """
        # Check samples.
        if check_X_dict is None:
            check_X_dict = {"allow_nd": True}
        X = check_array(X, **check_X_dict)

        # Check number of features.
        self._check_n_features(X, reset=reset)

        # Check labels
        y = check_array(y, ensure_2d=False, force_all_finite="allow-nan")
        check_consistent_length(X, y)

        # Check missing_label
        check_missing_label(self.missing_label)
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
                self._check_n_features(candidates, reset=False)

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
        y : array-like of shape (n_samples)
            Labels of the training data set (possibly including unlabeled ones
            indicated by self.MISSING_LABEL.
        candidates : None or array-like of shape (n_candidates), dtype=int or
            array-like of shape (n_candidates, n_features),
            optional (default=None)
            If candidates is None, the unlabeled samples from (X,y) are
            considered as candidates.
            If candidates is of shape (n_candidates) and of type int,
            candidates is considered as the indices of the samples in (X,y).
            If candidates is of shape (n_candidates, n_features), the
            candidates are directly given in candidates (not necessarily
            contained in X). This is not supported by all query strategies.
        batch_size : int, optional (default=1)
            The number of samples to be selected in one AL cycle.
        return_utilities : bool, optional (default=False)
            If true, also return the utilities based on the query strategy.

        Returns
        -------
        query_indices : numpy.ndarray of shape (batch_size)
            The query_indices indicate for which candidate sample a label is
            to queried, e.g., `query_indices[0]` indicates the first selected
            sample.
            If candidates is None or of shape (n_candidates), the indexing
            refers to samples in X.
            If candidates is of shape (n_candidates, n_features), the indexing
            refers to samples in candidates.
        utilities : numpy.ndarray of shape (batch_size, n_samples) or
            numpy.ndarray of shape (batch_size, n_candidates)
            The utilities of samples after each selected sample of the batch,
            e.g., `utilities[0]` indicates the utilities used for selecting
            the first sample (with index `query_indices[0]`) of the batch.
            Utilities for labeled samples will be set to np.nan.
            If candidates is None or of shape (n_candidates), the indexing
            refers to samples in X.
            If candidates is of shape (n_candidates, n_features), the indexing
            refers to samples in candidates.
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
        candidates : None or array-like of shape (n_candidates,), dtype=int or
            array-like of shape (n_candidates, n_features),
            optional (default=None)
            If candidates is None, the unlabeled samples from (X,y) are
            considered as candidates.
            If candidates is of shape (n_candidates,) and of type int,
            candidates is considered as the indices of the samples in (X,y).
            If candidates is of shape (n_candidates, n_features), the
            candidates are directly given in candidates (not necessarily
            contained in X). This is not supported by all query strategies.
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
        y : np.ndarray of shape (n_samples)
            Checked labels of the training data set.
        candidates :  None or np.ndarray of shape (n_candidates), dtype=int or
            np.ndarray of shape (n_candidates, n_features)
            Checked candidate samples.
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
        """
        Transforms the `candidates` parameter into a sample array and the
        corresponding index array `mapping` such that
        `candidates = X[mapping]`.

        Parameters
        ----------
        candidates :  None or np.ndarray of shape (n_candidates), dtype=int or
            np.ndarray of shape (n_candidates, n_features)
            Checked candidate samples.
            If candidates is None, the unlabeled samples from (X,y) are
            considered as candidates.
            If candidates is of shape (n_candidates) and of type int,
            candidates is considered as the indices of the samples in (X,y).
            If candidates is of shape (n_candidates, n_features), the
            candidates are directly given in candidates (not necessarily
            contained in X). This is not supported by all query strategies.
        X : np.ndarray of shape (n_samples, n_features)
            Checked training data set.
        y : np.ndarray of shape (n_samples)
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
    missing_label : scalar or string or np.nan or None, optional
    (default=np.nan)
        Value to represent a missing label.
    random_state : int or RandomState instance, optional (default=None)
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
        candidates : None or array-like of shape (n_candidates), dtype=int or
            array-like of shape (n_candidates, n_features),
            optional (default=None)
            If `candidates` is None, the samples from (X,y), for which an
            annotator exists such that the annotator sample pair is
            unlabeled are considered as sample candidates.
            If `candidates` is of shape (n_candidates,) and of type int,
            `candidates` is considered as the indices of the sample candidates
            in (X,y).
            If `candidates` is of shape (n_candidates, n_features), the
            sample candidates are directly given in `candidates` (not
            necessarily contained in `X`). This is not supported by all query
            strategies.
        annotators : array-like of shape (n_candidates, n_annotators), optional
        (default=None)
            If `annotators` is None, all annotators are considered as available
            annotators.
            If `annotators` is of shape (n_avl_annotators), and of type int,
            `annotators` is considered as the indices of the available
            annotators.
            If candidate samples and available annotators are specified:
            The annotator-sample-pairs, for which the sample is a candidate
            sample and the annotator is an available annotator are considered
            as candidate annotator-sample-pairs.
            If `annotators` is a boolean array of shape (n_candidates,
            n_avl_annotators) the annotator-sample-pairs, for which the sample
            is a candidate sample and the boolean matrix has entry `True` are
            considered as candidate sample pairs.
        batch_size : int, optional (default=1)
            The number of annotators sample pairs to be selected in one AL
            cycle.
        return_utilities : bool, optional (default=False)
            If true, also return the utilities based on the query strategy.

        Returns
        -------
        query_indices : np.ndarray of shape (batchsize, 2)
            The query_indices indicate which candidate sample pairs are to be
            queried is, i.e., which candidate sample is to be annotated by
            which annotator, e.g., `query_indices[:, 0]` indicates the selected
            candidate samples and `query_indices[:, 1]` indicates the
            respectively selected annotators.
        utilities: numpy.ndarray of shape (batch_size, n_samples, n_annotators)
         or numpy.ndarray of shape (batch_size, n_candidates, n_annotators)
            The utilities of all candidate samples w.r.t. to the available
            annotators after each selected sample of the batch, e.g.,
            `utilities[0, :, j]` indicates the utilities used for selecting
            the first sample-annotator-pair (with indices `query_indices[0]`).
            If `candidates is None` or of shape (n_candidates), the indexing
            refers to samples in `X`.
            If `candidates` is of shape (n_candidates, n_features), the
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
            including unlabeled ones indicated by self.MISSING_LABEL), meaning
            that `y[i, j]` contains the label annotated by annotator `i` for
            sample `j`.
        candidates : None or array-like of shape (n_candidates), dtype=int or
            array-like of shape (n_candidates, n_features),
            optional (default=None)
            If `candidates` is None, the samples from (X,y), for which an
            annotator exists such that the annotator sample pairs is
            unlabeled are considered as sample candidates.
            If `candidates` is of shape (n_candidates,) and of type int,
            `candidates` is considered as the indices of the sample candidates
            in (X,y).
            If `candidates` is of shape (n_candidates, n_features), the
            sample candidates are directly given in `candidates` (not
            necessarily contained in `X`). This is not supported by all query
            strategies.
        annotators : array-like of shape (n_candidates, n_annotators), optional
        (default=None)
            If `annotators` is None, all annotators are considered as available
            annotators.
            If `annotators` is of shape (n_avl_annotators), and of type int,
            `annotators` is considered as the indices of the available
            annotators.
            If candidate samples and available annotators are specified:
            The annotator-sample-pairs, for which the sample is a candidate
            sample and the annotator is an available annotator are considered
            as candidate annotator-sample-pairs.
            If `annotators` is a boolean array of shape (n_candidates,
            n_avl_annotators) the annotator-sample-pairs, for which the sample
            is a candidate sample and the boolean matrix has entry `True` are
            considered as candidate sample pairs.
        batch_size : int or string, optional (default=1)
            The number of annotators sample pairs to be selected in one AL
            cycle. If `adaptive = True` `batch_size = 'adaptive'` is allowed.
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
        candidates :  None or np.ndarray of shape (n_candidates), dtype=int or
            np.ndarray of shape (n_candidates, n_features)
            Checked candidate samples.
        annotators : None or np.ndarray of shape (n_avl_annotators), dtype=int
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

        check_array(y, ensure_2d=True, force_all_finite="allow-nan")
        unlabeled_pairs = is_unlabeled(y, missing_label=self.missing_label_)

        if annotators is not None:
            annotators = check_array(
                annotators, ensure_2d=False, allow_nd=True
            )

            if annotators.ndim == 1:
                annotators = check_indices(annotators, y, dim=1)
            elif annotators.ndim == 2:
                annotators = check_array(annotators, dtype=bool)
                if candidates is None or candidates.ndim == 1:
                    check_consistent_length(X, annotators)
                else:
                    check_consistent_length(candidates, annotators)
                check_consistent_length(y.T, annotators.T)
            else:
                raise ValueError(
                    "`annotators` must be either None, 1d or 2d " "array-like."
                )

        if annotators is None:
            if candidates is None:
                n_candidate_pairs = int(np.sum(unlabeled_pairs))
            elif candidates.ndim == 1:
                n_candidate_pairs = len(candidates) * len(y.T)
            else:
                n_candidate_pairs = len(candidates) * len(y.T)
        elif annotators.ndim == 1:
            if candidates is None:
                n_candidate_pairs = int(np.sum(unlabeled_pairs[:, annotators]))
            elif candidates.ndim == 1:
                n_candidate_pairs = int(
                    np.sum(unlabeled_pairs[candidates][:, annotators])
                )
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
        candidates : None or array-like of shape (n_candidates), dtype=int or
            array-like of shape (n_candidates, n_features),
            optional (default=None)
            If `candidates` is None, the samples from (X,y), for which an
            annotator exists such that the annotator sample pairs is
            unlabeled are considered as sample candidates.
            If `candidates` is of shape (n_candidates,) and of type int,
            `candidates` is considered as the indices of the sample candidates
            in (X,y).
            If `candidates` is of shape (n_candidates, n_features), the
            sample candidates are directly given in `candidates` (not
            necessarily contained in `X`). This is not supported by all query
            strategies.
        annotators : array-like of shape (n_candidates, n_annotators), optional
        (default=None)
            If `annotators` is None, all annotators are considered as available
            annotators.
            If `annotators` is of shape (n_avl_annotators), and of type int,
            `annotators` is considered as the indices of the available
            annotators.
            If candidate samples and available annotators are specified:
            The annotator-sample-pairs, for which the sample is a candidate
            sample and the annotator is an available annotator are considered
            as candidate annotator-sample-pairs.
            If `annotators` is a boolean array of shape (n_candidates,
            n_avl_annotators) the annotator-sample-pairs, for which the sample
            is a candidate sample and the boolean matrix has entry `True` are
            considered as candidate sample pairs.
        X : np.ndarray of shape (n_samples, n_features)
            Checked training data set.
        y : np.ndarray of shape (n_samples,)
            Checked labels of the training data set.
        enforce_mapping : bool, optional (default=False)
            If `True`, an exception is raised when no exact mapping can be
            determined (i.e., `mapping` is None).

        Returns
        -------
        candidates : np.ndarray of shape (n_candidates, n_features)
            Candidate samples from which the strategy can query the label.
        mapping : np.ndarray of shape (n_candidates) or None
            Index array that maps `candidates` to `X`
            (`candidates = X[mapping]`).
        A_cand : np.ndarray of shape(n_candidates, n_annotators)
            Available annotator sample pair with respect to `candidates`.
        """
        unlbd_pairs = is_unlabeled(y, self.missing_label_)
        unlbd_sample_indices = np.argwhere(
            np.any(unlbd_pairs, axis=1)
        ).flatten()
        n_annotators = y.shape[1]

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

        if candidates is None:
            candidates = unlbd_sample_indices
            only_candidates = False
        elif annotators is not None:
            candidates = np.intersect1d(candidates, unlbd_sample_indices)
            only_candidates = False
        else:
            only_candidates = True

        if only_candidates:
            A_cand = np.full((len(candidates), n_annotators), True)
        elif annotators is None:
            A_cand = unlbd_pairs[candidates, :]
        elif annotators.ndim == 1:
            available_pairs = np.full_like(y, False, dtype=bool)
            available_pairs[:, annotators] = True
            A_cand = (unlbd_pairs & available_pairs)[candidates, :]
        else:
            A_cand = annotators

        return X[candidates], candidates, A_cand


class BudgetManager(ABC, BaseEstimator):
    """Base class for all budget managers for stream-based active learning
    in scikit-activeml to model budgeting constraints.

    Parameters
    ----------
    budget : float (default=None)
        Specifies the ratio of instances which are allowed to be sampled, with
        0 <= budget <= 1. If budget is None, it is replaced with the default
        budget 0.1.
    """

    def __init__(self, budget=None):
        self.budget = budget

    @abstractmethod
    def query_by_utility(self, utilities, *args, **kwargs):
        """Ask the budget manager which utilities are sufficient to query the
        corresponding instance.

        Parameters
        ----------
        utilities : ndarray of shape (n_samples,)
            The utilities provided by the stream-based active learning
            strategy, which are used to determine whether sampling an instance
            is worth it given the budgeting constraint.

        Returns
        -------
        queried_indices : ndarray of shape (n_queried_instances,)
            The indices of instances represented by utilities which should be
            queried, with 0 <= n_queried_instances <= n_samples.
        """
        raise NotImplementedError

    @abstractmethod
    def update(self, candidates, queried_indices, *args, **kwargs):
        """Updates the BudgetManager.

        Parameters
        ----------
        candidates : {array-like, sparse matrix} of shape
        (n_samples, n_features)
            The instances which may be queried. Sparse matrices are accepted
            only if they are supported by the base query strategy.
        queried_indices : array-like
            Indicates which instances from candidates have been queried.

        Returns
        -------
        self : BudgetManager
            The BudgetManager returns itself, after it is updated.
        """
        raise NotImplementedError

    def _validate_budget(self):
        """check the assigned budget and set the default value 0.1 if budget is
        set to None.
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
        utilities: ndarray of shape (n_samples,)
            The utilities provided by the stream-based active learning
            strategy.

        Returns
        -------
        utilities: ndarray of shape (n_samples,)
            Checked utilities
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
    """Base class for all stream-based active learning query strategies in
       scikit-activeml.

    Parameters
    ----------
    budget : float, default=None
        The budget which models the budgeting constraint used in
        the stream-based active learning setting.
    random_state : int, RandomState instance, default=None
        Controls the randomness of the estimator.
    """

    def __init__(self, budget, random_state=None):
        super().__init__(random_state=random_state)
        self.budget = budget

    @abstractmethod
    def query(self, candidates, *args, return_utilities=False, **kwargs):
        """Ask the query strategy which instances in candidates to acquire.

        The query startegy determines the most useful instances in candidates,
        which can be acquired within the budgeting constraint specified by the
        budgetmanager.
        Please note that, when the decisions from this function
        may differ from the final sampling, simulate=True can set, so that the
        query strategy can be updated later with update(...) with the final
        sampling. This is especially helpful, when developing wrapper query
        strategies.

        Parameters
        ----------
        candidates : {array-like, sparse matrix} of shape
        (n_samples, n_features)
            The instances which may be queried. Sparse matrices are accepted
            only if they are supported by the base query strategy.

        return_utilities : bool, optional
            If true, also return the utilities based on the query strategy.
            The default is False.

        Returns
        -------
        queried_indices : ndarray of shape (n_sampled_instances,)
            The indices of instances in candidates which should be sampled,
            with 0 <= n_sampled_instances <= n_samples.

        utilities: ndarray of shape (n_samples,), optional
            The utilities based on the query strategy. Only provided if
            return_utilities is True.
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
        """Update the query strategy with the decisions taken.

        This function should be used in conjunction with the query function,
        when the instances queried from query(...) may differ from the
        instances queried in the end. In this case use query(...) with
        simulate=true and provide the final decisions via update(...).
        This is especially helpful, when developing wrapper query strategies.

        Parameters
        ----------
        candidates : {array-like, sparse matrix} of shape
        (n_samples, n_features)
            The instances which could be queried. Sparse matrices are accepted
            only if they are supported by the base query strategy.

        queried_indices : array-like
            Indicates which instances from candidates have been queried.

        budget_manager_param_dict : kwargs, optional
            Optional kwargs for budgetmanager.
        Returns
        -------
        self : StreamBasedQueryStrategy
            The StreamBasedQueryStrategy returns itself, after it is updated.
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
            The instances which may be queried. Sparse matrices are accepted
            only if they are supported by the base query strategy.
        return_utilities : bool,
            If true, also return the utilities based on the query strategy.
        reset : bool, default=True
            Whether to reset the `n_features_in_` attribute.
            If False, the input will be checked for consistency with data
            provided when reset was last True.
        **check_candidates_params : kwargs
            Parameters passed to :func:`sklearn.utils.check_array`.

        Returns
        -------
        candidates: np.ndarray, shape (n_candidates, n_features)
            Checked candidate samples
        return_utilities : bool,
            Checked boolean value of `return_utilities`.
        """
        # Check candidate instances.
        candidates = check_array(candidates, **check_candidates_params)

        # Check number of features.
        self._check_n_features(candidates, reset=reset)

        # Check return_utilities.
        check_scalar(return_utilities, "return_utilities", bool)

        # Check random state.
        self._validate_random_state()

        # Check budgetmanager.
        self._validate_budget()

        return candidates, return_utilities


class SkactivemlClassifier(BaseEstimator, ClassifierMixin, ABC):
    """SkactivemlClassifier

    Base class for scikit-activeml classifiers such that missing labels,
    user-defined classes, and cost-sensitive classification (i.e., cost matrix)
    can be handled.

    Parameters
    ----------
    classes : array-like of shape (n_classes), default=None
        Holds the label for each class. If none, the classes are determined
        during the fit.
    missing_label : scalar, string, np.nan, or None, default=np.nan
        Value to represent a missing label.
    cost_matrix : array-like of shape (n_classes, n_classes)
        Cost matrix with `cost_matrix[i,j]` indicating cost of predicting class
        `classes[j]`  for a sample of class `classes[i]`. Can be only set, if
        classes is not none.
    random_state : int or RandomState instance or None, default=None
        Determines random number for `predict` method. Pass an int for
        reproducible results across multiple method calls.

    Attributes
    ----------
    classes_ : array-like, shape (n_classes)
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
            The sample matrix X is the feature matrix representing the samples.
        y : array-like, shape (n_samples) or (n_samples, n_outputs)
            It contains the class labels of the training samples.
            The number of class labels may be variable for the samples, where
            missing labels are represented the attribute 'missing_label'.
        sample_weight : array-like, shape (n_samples) or (n_samples, n_outputs)
            It contains the weights of the training samples' class labels.
            It must have the same shape as y.

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
        X : array-like, shape (n_samples, n_features)
            Test samples.

        Returns
        -------
        P : numpy.ndarray, shape (n_samples, classes)
            The class probabilities of the test samples. Classes are ordered
            according to 'classes_'.
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
        y : numpy.ndarray of shape (n_samples)
            Predicted class labels of the test samples `X`. Classes are ordered
            according to `classes_`.
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
    ):
        if check_X_dict is None:
            check_X_dict = {"ensure_min_samples": 0, "ensure_min_features": 0}
        if check_y_dict is None:
            check_y_dict = {
                "ensure_min_samples": 0,
                "ensure_min_features": 0,
                "ensure_2d": False,
                "force_all_finite": False,
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
        if len(y) > 0:
            y = column_or_1d(y) if y_ensure_1d else y
            y = self._le.fit_transform(y)
            is_lbdl = is_labeled(y)
            if len(y[is_lbdl]) > 0:
                check_classification_targets(y[is_lbdl])
            if len(self._le.classes_) == 0:
                raise ValueError(
                    "No class label is known because 'y' contains no actual "
                    "class labels and 'classes' is not defined. Change at "
                    "least on of both to overcome this error."
                )
        else:
            self._le.fit_transform(self.classes)
            check_X_dict["ensure_2d"] = False
        X = check_array(X, **check_X_dict)
        check_consistent_length(X, y)

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

    def _check_n_features(self, X, reset):
        if reset:
            self.n_features_in_ = X.shape[1] if len(X) > 0 else None
        elif not reset:
            if self.n_features_in_ is not None:
                super()._check_n_features(X, reset=reset)


class ClassFrequencyEstimator(SkactivemlClassifier):
    """ClassFrequencyEstimator

    Extends scikit-activeml classifiers to estimators that are able to estimate
    class frequencies for given samples (by calling 'predict_freq').

    Parameters
    ----------
    classes : array-like, shape (n_classes), default=None
        Holds the label for each class. If none, the classes are determined
        during the fit.
    missing_label : scalar or str or np.nan or None, default=np.nan
        Value to represent a missing label.
    cost_matrix : array-like of shape (n_classes, n_classes)
        Cost matrix with `cost_matrix[i,j]` indicating cost of predicting class
        `classes[j]`  for a sample of class `classes[i]`. Can be only set, if
        classes is not none.
    class_prior : float or array-like, shape (n_classes), default=0
        Prior observations of the class frequency estimates. If `class_prior`
        is an array, the entry `class_prior[i]` indicates the non-negative
        prior number of samples belonging to class `classes_[i]`. If
        `class_prior` is a float, `class_prior` indicates the non-negative
        prior number of samples per class.
    random_state : int or np.RandomState or None, default=None
        Determines random number for 'predict' method. Pass an int for
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
            The class frequency estimates of the test samples 'X'. Classes are
            ordered according to attribute 'classes_'.
        """
        raise NotImplementedError

    def predict_proba(self, X):
        """Return probability estimates for the test data `X`.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features) or
        shape (n_samples, m_samples) if metric == 'precomputed'
            Input samples.

        Returns
        -------
        P : array-like of shape (n_samples, classes)
            The class probabilities of the test samples. Classes are ordered
            according to classes_.
        """
        # Normalize probabilities of each sample.
        P = self.predict_freq(X) + self.class_prior_
        normalizer = np.sum(P, axis=1)
        P[normalizer > 0] /= normalizer[normalizer > 0, np.newaxis]
        P[normalizer == 0, :] = [1 / len(self.classes_)] * len(self.classes_)
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


class AnnotatorModelMixin(ABC):
    """AnnotatorModelMixin

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
