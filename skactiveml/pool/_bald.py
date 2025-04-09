"""
Code is based on https://blackhc.github.io/batchbald_redux/ distributed under
the Apache-2.0 license and the associated query strategy is presented:

A. Kirsch, J. Van Amersfoort, and Y. Gal. BatchBALD: Efficient and Diverse
Batch Acquisition for Deep Bayesian Active Learning. In Adv. Neural Inf.
Process. Syst., 2019.
"""

import numpy as np
from sklearn.utils import check_array

from ..base import SkactivemlClassifier
from ..pool._query_by_committee import _check_ensemble, QueryByCommittee
from ..utils import (
    MISSING_LABEL,
    rand_argmax,
    check_type,
    check_scalar,
    check_random_state,
    simple_batch,
)


class _GeneralBALD(QueryByCommittee):
    """General Bayesian Active Learning by Disagreement (_GeneralBALD)

    The Bayesian Active Learning by Disagreement (BatchBALD) [1]_ strategy
    reduces the number of possible hypotheses maximally fast to minimize the
    uncertainty about the parameters using Shannon's entropy. It seeks the data
    point that maximises the decrease in expected posterior entropy. For the
    batch case, by default the advanced strategy BatchBALD [2]_ is used.
    If desired, a greedy (top-k) selection can be applied by setting
    `greedy_selection=True`.

    Parameters
    ----------
    n_MC_samples : int > 0, default=n_estimators
        The number of monte carlo samples used for label estimation.
    greedy_selection : bool, default=False
        Flag to either use BatchBALD (`greedy_selection=False`) or a greedy
        (top-k) selection (`greedy_selection=True`) if `batch_size>1`.
    eps : float > 0, default=1e-7
        Minimum probability threshold to compute log-probabilities.
    sample_predictions_method_name : str, default=None
        Certain estimators may offer methods enabling to construct a committee
        by sampling predictions of committee members. This parameter is to
        indicate the name of such a method.

        - If `sample_predictions_method_name=None` no sampling is
          performed.
        - If `sample_predictions_method_name` is not `None` and in the
          case of classification, the method is expected to take samples of
          the shape `(n_samples, *)` as input and to output probabilities
          of the shape `(n_members, n_samples, n_classes)`, e.g.,
          `sample_proba` in `skactiveml.base.ClassFrequencyEstimator`.
    sample_predictions_dict : dict, default=None
        Parameters (excluding the samples) that are passed to the method with
        the name `sample_predictions_method_name`.

        - This parameter must be `None`, if
          `sample_predictions_method_name` is `None`.
        - Otherwise, it may be used to define the number of sampled
          members, e.g., by defining `n_samples` as parameter to the method
          `sample_proba` of `skactiveml.base.ClassFrequencyEstimator`.
    missing_label : scalar or string or np.nan or None, default=np.nan
        Value to represent a missing label.
    random_state : int or None or np.random.RandomState, default=None
        The random state to use.

    References
    ----------
    .. [1] N. Houlsby, F. Huszár, Z. Ghahramani, and M. Lengyel. Bayesian
       Active Learning for Classification and Preference Learning.
       arXiv:1112.5745, 2011.

    .. [2] A. Kirsch, J. Van Amersfoort, and Y. Gal. BatchBALD: Efficient and
       Diverse Batch Acquisition for Deep Bayesian Active Learning. In Adv.
       Neural Inf. Process. Syst., 2019.
    """

    def __init__(
        self,
        n_MC_samples=None,
        greedy_selection=False,
        eps=1e-7,
        sample_predictions_method_name=None,
        sample_predictions_dict=None,
        missing_label=MISSING_LABEL,
        random_state=None,
    ):
        super().__init__(
            eps=eps,
            sample_predictions_method_name=sample_predictions_method_name,
            sample_predictions_dict=sample_predictions_dict,
            missing_label=missing_label,
            random_state=random_state,
        )
        self.n_MC_samples = n_MC_samples
        self.greedy_selection = greedy_selection

    def query(
        self,
        X,
        y,
        ensemble,
        fit_ensemble=True,
        sample_weight=None,
        candidates=None,
        batch_size=1,
        return_utilities=False,
    ):
        """Determines for which candidate samples labels are to be queried.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data set, usually complete, i.e., including the labeled
            and unlabeled samples.
        y : array-like of shape (n_samples,)
            Labels of the training data set (possibly including unlabeled ones
            indicated by `self.missing_label`).
        ensemble : array-like of SkactivemlClassifier or SkactivemlClassifier
            - If `ensemble` is a `SkactivemlClassifier` and has
              `n_estimators` plus `estimators_` after fitting as
              attributes, its estimators will be used as committee.
            - If `ensemble` is array-like, each element of this list must be
              `SkactivemlClassifier` and will be
              used as committee member.
            - If `ensemble` is a `SkactivemlClassifier` and implements a
              method with the name `sample_predictions_method_name`, this
              method is used to sample predictions of committee members.
        fit_ensemble : bool, default=True
            Defines whether the ensemble should be fitted on `X`, `y`, and
            `sample_weight`.
        sample_weight: array-like of shape (n_samples), default=None
            Weights of training samples in `X`.
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
            If `True`, also return the utilities based on the query strategy.

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
        # Validate input parameters.
        X, y, candidates, batch_size, return_utilities = self._validate_data(
            X, y, candidates, batch_size, return_utilities, reset=True
        )
        check_scalar(
            self.greedy_selection, "greedy_selection", target_type=bool
        )
        X_cand, mapping = self._transform_candidates(candidates, X, y)

        # Validate classifier type.
        check_type(fit_ensemble, "fit_ensemble", bool)

        ensemble, est_arr, _, sample_func, sample_dict = _check_ensemble(
            ensemble=ensemble,
            X=X,
            y=y,
            sample_weight=sample_weight,
            fit_ensemble=fit_ensemble,
            missing_label=self.missing_label_,
            estimator_types=[SkactivemlClassifier],
            sample_predictions_method_name=self.sample_predictions_method_name,
            sample_predictions_dict=self.sample_predictions_dict,
        )

        if sample_func is None:
            probas = self._aggregate_predict_probas(X_cand, ensemble, est_arr)
        else:
            probas = sample_func(X_cand, **sample_dict)

        if self.n_MC_samples is None:
            n_MC_samples_ = len(probas)
        else:
            n_MC_samples_ = self.n_MC_samples
        check_scalar(n_MC_samples_, "n_MC_samples", int, min_val=1)

        utils_batch_size = 1 if self.greedy_selection else batch_size
        batch_utilities_cand = batch_bald(
            probas=probas,
            batch_size=utils_batch_size,
            n_MC_samples=n_MC_samples_,
            eps=self.eps,
            random_state=self.random_state_,
        )

        if mapping is None:
            batch_utilities = batch_utilities_cand
        else:
            batch_utilities = np.full((utils_batch_size, len(X)), np.nan)
            batch_utilities[:, mapping] = batch_utilities_cand

        if self.greedy_selection:
            return simple_batch(
                batch_utilities[0],
                self.random_state_,
                batch_size=batch_size,
                return_utilities=return_utilities,
            )
        else:
            best_indices = rand_argmax(
                batch_utilities, axis=1, random_state=self.random_state_
            )
            if return_utilities:
                return best_indices, batch_utilities
            else:
                return best_indices


class BatchBALD(_GeneralBALD):
    """Batch Bayesian Active Learning by Disagreement (BatchBALD)

    Batch Bayesian Active Learning by Disagreement (BatchBALD) [1]_
    reduces the number of possible hypotheses maximally fast to minimize the
    uncertainty about the parameters using Shannon's entropy. It seeks the data
    point that maximises the decrease in expected posterior entropy.

    Parameters
    ----------
    n_MC_samples : int > 0, default=n_estimators
        The number of monte carlo samples used for label estimation.
    eps : float > 0, default=1e-7
        Minimum probability threshold to compute log-probabilities.
    sample_predictions_method_name : str, default=None
        Certain estimators may offer methods enabling to construct a committee
        by sampling predictions of committee members. This parameter is to
        indicate the name of such a method.

        - If `sample_predictions_method_name=None` no sampling is
          performed.
        - If `sample_predictions_method_name` is not `None` and in the
          case of classification, the method is expected to take samples of
          the shape `(n_samples, *)` as input and to output probabilities
          of the shape `(n_members, n_samples, n_classes)`, e.g.,
          `sample_proba` in `skactiveml.base.ClassFrequencyEstimator`.
    sample_predictions_dict : dict, default=None
        Parameters (excluding the samples) that are passed to the method with
        the name `sample_predictions_method_name`.

        - This parameter must be `None`, if
          `sample_predictions_method_name` is `None`.
        - Otherwise, it may be used to define the number of sampled
          members, e.g., by defining `n_samples` as parameter to the method
          `sample_proba` of `skactiveml.base.ClassFrequencyEstimator`.
    missing_label : scalar or string or np.nan or None, default=np.nan
        Value to represent a missing label.
    random_state : int or None or np.random.RandomState, default=None
        The random state to use.

    References
    ----------
    .. [1] A. Kirsch, J. Van Amersfoort, and Y. Gal. BatchBALD: Efficient and
       Diverse Batch Acquisition for Deep Bayesian Active Learning. In Adv.
       Neural Inf. Process. Syst., 2019.
    """

    def __init__(
        self,
        n_MC_samples=None,
        eps=1e-7,
        sample_predictions_method_name=None,
        sample_predictions_dict=None,
        missing_label=MISSING_LABEL,
        random_state=None,
    ):
        super().__init__(
            n_MC_samples=n_MC_samples,
            greedy_selection=False,
            eps=eps,
            sample_predictions_method_name=sample_predictions_method_name,
            sample_predictions_dict=sample_predictions_dict,
            missing_label=missing_label,
            random_state=random_state,
        )


class GreedyBALD(_GeneralBALD):
    """Greedy Bayesian Active Learning by Disagreement (GreedyBALD)

    The Bayesian Active Learning by Disagreement (BALD) [1]_ strategy
    reduces the number of possible hypotheses maximally fast to minimize the
    uncertainty about the parameters using Shannon's entropy. It seeks the data
    point that maximises the decrease in expected posterior entropy. For the
    batch case, a greedy (top-k) selection is applied.

    Parameters
    ----------
    n_MC_samples : int > 0, default=n_estimators
        The number of monte carlo samples used for label estimation.
    eps : float > 0, default=1e-7
        Minimum probability threshold to compute log-probabilities.
    sample_predictions_method_name : str, default=None
        Certain estimators may offer methods enabling to construct a committee
        by sampling predictions of committee members. This parameter is to
        indicate the name of such a method.

        - If `sample_predictions_method_name=None` no sampling is
          performed.
        - If `sample_predictions_method_name` is not `None` and in the
          case of classification, the method is expected to take samples of
          the shape `(n_samples, *)` as input and to output probabilities
          of the shape `(n_members, n_samples, n_classes)`, e.g.,
          `sample_proba` in `skactiveml.base.ClassFrequencyEstimator`.
    sample_predictions_dict : dict, default=None
        Parameters (excluding the samples) that are passed to the method with
        the name `sample_predictions_method_name`.

        - This parameter must be `None`, if
          `sample_predictions_method_name` is `None`.
        - Otherwise, it may be used to define the number of sampled
          members, e.g., by defining `n_samples` as parameter to the method
          `sample_proba` of `skactiveml.base.ClassFrequencyEstimator`.
    missing_label : scalar or string or np.nan or None, default=np.nan
        Value to represent a missing label.
    random_state : int or None or np.random.RandomState, default=None
        The random state to use.

    References
    ----------
    .. [1] N. Houlsby, F. Huszár, Z. Ghahramani, and M. Lengyel. Bayesian
       Active Learning for Classification and Preference Learning.
       arXiv:1112.5745, 2011.
    """

    def __init__(
        self,
        n_MC_samples=None,
        eps=1e-7,
        sample_predictions_method_name=None,
        sample_predictions_dict=None,
        missing_label=MISSING_LABEL,
        random_state=None,
    ):
        super().__init__(
            n_MC_samples=n_MC_samples,
            greedy_selection=True,
            eps=eps,
            sample_predictions_method_name=sample_predictions_method_name,
            sample_predictions_dict=sample_predictions_dict,
            missing_label=missing_label,
            random_state=random_state,
        )


def batch_bald(
    probas,
    batch_size,
    n_MC_samples=None,
    random_state=None,
    eps=1e-7,
):
    """BatchBALD: Efficient and Diverse Batch Acquisition for Deep Bayesian
    Active Learning

    BatchBALD [1]_ is an extension of BALD  [2]_ (Bayesian Active Learning by
    Disagreement) whereby points are jointly scored by estimating the
    mutual information between a joint of multiple data points and the model
    parameters.

    Parameters
    ----------
    probas : array-like of shape (n_estimators, n_samples, n_classes)
        The probability estimates of all estimators, samples, and classes.
    batch_size : int, default=1
        The number of samples to be selected in one AL cycle.
    n_MC_samples : int > 0, default=n_estimators
        The number of monte carlo samples used for label estimation.
    eps : float  > 0, default=1e-7
        Minimum probability threshold to compute log-probabilities.
    random_state : int or np.random.RandomState, default=None
        The random state to use.

    Returns
    -------
    utilities: numpy.ndarray of shape (batch_size, n_samples)
        Sample utilities computed according to BatchBALD [2].

    References
    ----------
    .. [1] N. Houlsby, F. Huszár, Z. Ghahramani, and M. Lengyel. Bayesian
       Active Learning for Classification and Preference Learning.
       arXiv:1112.5745, 2011.
    .. [2] A. Kirsch, J. Van Amersfoort, and Y. Gal. BatchBALD: Efficient and
       Diverse Batch Acquisition for Deep Bayesian Active Learning. In Adv.
       Neural Inf. Process. Syst., 2019.
    """
    # Validate input parameters.
    if probas.ndim != 3:
        raise ValueError(
            f"'probas' should be of shape 3, but {probas.ndim}" f" were given."
        )
    probs_K_N_C = check_array(probas, ensure_2d=False, allow_nd=True)
    check_scalar(batch_size, "batch_size", int, min_val=1)
    check_scalar(
        eps,
        "eps",
        min_val=0,
        max_val=0.1,
        target_type=(float, int),
        min_inclusive=False,
    )
    if n_MC_samples is None:
        n_MC_samples = len(probas)
    check_scalar(n_MC_samples, "n_MC_samples", int, min_val=1)
    random_state = check_random_state(random_state)

    probs_N_K_C = probs_K_N_C.swapaxes(0, 1)
    np.clip(probs_N_K_C, a_min=eps, a_max=1, out=probs_N_K_C)
    log_probs_N_K_C = np.log(probs_N_K_C)
    N, K, C = log_probs_N_K_C.shape

    batch_size = min(batch_size, N)

    conditional_entropies_N = _compute_conditional_entropy(log_probs_N_K_C)

    batch_joint_entropy = _DynamicJointEntropy(
        n_MC_samples, batch_size - 1, K, C, random_state
    )

    utilities = np.zeros((batch_size, N))
    query_indices = []

    for i in range(batch_size):
        if i > 0:
            latest_index = query_indices[-1]
            batch_joint_entropy.add_variables(
                log_probs_N_K_C[latest_index : latest_index + 1]
            )

        shared_conditinal_entropies = conditional_entropies_N[
            query_indices
        ].sum()

        utilities[i] = batch_joint_entropy.compute_batch(log_probs_N_K_C)

        utilities[i] -= conditional_entropies_N + shared_conditinal_entropies
        utilities[i, query_indices] = np.nan

        query_idx = rand_argmax(utilities[i], random_state=0)[0]

        query_indices.append(query_idx)

    return utilities


class _ExactJointEntropy:
    def __init__(self, joint_probs_M_K):
        self.joint_probs_M_K = joint_probs_M_K

    @staticmethod
    def empty(K):
        return _ExactJointEntropy(np.ones((1, K)))

    def add_variables(self, log_probs_N_K_C):
        N, K, C = log_probs_N_K_C.shape
        joint_probs_K_M_1 = self.joint_probs_M_K.T[:, :, None]

        probs_N_K_C = np.exp(log_probs_N_K_C)

        # Using lots of memory.
        for i in range(N):
            probs_i__K_1_C = probs_N_K_C[i][:, None, :]
            joint_probs_K_M_C = joint_probs_K_M_1 * probs_i__K_1_C
            joint_probs_K_M_1 = joint_probs_K_M_C.reshape((K, -1, 1))

        self.joint_probs_M_K = joint_probs_K_M_1.squeeze(2).T
        return self

    def compute_batch(self, log_probs_B_K_C):
        B, K, C = log_probs_B_K_C.shape
        M = self.joint_probs_M_K.shape[0]

        probs_b_K_C = np.exp(log_probs_B_K_C)
        b = probs_b_K_C.shape[0]
        probs_b_M_C = np.empty((b, M, C))
        for i in range(b):
            np.matmul(
                self.joint_probs_M_K,
                probs_b_K_C[i],
                out=probs_b_M_C[i],
            )
        probs_b_M_C /= K

        output_entropies_B = np.sum(
            -np.log(probs_b_M_C) * probs_b_M_C, axis=(1, 2)
        )

        return output_entropies_B


def _batch_multi_choices(probs_b_C, M, random_state):
    """
    probs_b_C: Ni... x C

    Returns:
        choices: Ni... x M
    """
    probs_B_C = probs_b_C.reshape((-1, probs_b_C.shape[-1]))
    B = probs_B_C.shape[0]
    C = probs_B_C.shape[1]

    # samples: Ni... x draw_per_xx
    choices = [
        random_state.choice(
            C, size=M, p=probs_B_C[b] / np.sum(probs_B_C[b]), replace=True
        )
        for b in range(B)
    ]
    choices = np.array(choices, dtype=int)

    choices_b_M = choices.reshape(list(probs_b_C.shape[:-1]) + [M])
    return choices_b_M


def _gather_expand(data, axis, index):
    max_shape = [max(dr, ir) for dr, ir in zip(data.shape, index.shape)]
    new_data_shape = list(max_shape)
    new_data_shape[axis] = data.shape[axis]

    new_index_shape = list(max_shape)
    new_index_shape[axis] = index.shape[axis]

    data = np.broadcast_to(data, new_data_shape)
    index = np.broadcast_to(index, new_index_shape)

    return np.take_along_axis(data, index, axis=axis)


class _SampledJointEntropy:
    """
    Random variables (all with the same # of categories $C$) can be added
    via `_SampledJointEntropy.add_variables`.

    `_SampledJointEntropy.compute` computes the joint entropy.

    `_SampledJointEntropy.compute_batch` computes the joint entropy of the
    added variables with each of the variables in the provided batch
    probabilities in turn.
    """

    def __init__(self, sampled_joint_probs_M_K):
        self.sampled_joint_probs_M_K = sampled_joint_probs_M_K

    @staticmethod
    def sample(probs_N_K_C, M, random_state):
        K = probs_N_K_C.shape[1]

        # S: num of samples per w
        S = M // K

        choices_N_K_S = _batch_multi_choices(probs_N_K_C, S, random_state)

        expanded_choices_N_1_K_S = choices_N_K_S[:, None, :, :]
        expanded_probs_N_K_1_C = probs_N_K_C[:, :, None, :]

        probs_N_K_K_S = _gather_expand(
            expanded_probs_N_K_1_C, axis=-1, index=expanded_choices_N_1_K_S
        )
        # exp sum log seems necessary to avoid 0s?
        probs_K_K_S = np.exp(
            np.sum(np.log(probs_N_K_K_S), axis=0, keepdims=False)
        )
        samples_K_M = probs_K_K_S.reshape((K, -1))

        samples_M_K = samples_K_M.T
        return _SampledJointEntropy(samples_M_K)

    def compute_batch(self, log_probs_B_K_C):
        B, K, C = log_probs_B_K_C.shape
        M = self.sampled_joint_probs_M_K.shape[0]

        b = log_probs_B_K_C.shape[0]

        probs_b_M_C = np.empty(
            (b, M, C),
        )
        for i in range(b):
            np.matmul(
                self.sampled_joint_probs_M_K,
                np.exp(log_probs_B_K_C[i]),
                out=probs_b_M_C[i],
            )
        probs_b_M_C /= K

        q_1_M_1 = self.sampled_joint_probs_M_K.mean(axis=1, keepdims=True)[
            None
        ]

        output_entropies_B = (
            np.sum(-np.log(probs_b_M_C) * probs_b_M_C / q_1_M_1, axis=(1, 2))
            / M
        )

        return output_entropies_B


class _DynamicJointEntropy:
    def __init__(self, M, max_N, K, C, random_state):
        self.M = M
        self.N = 0
        self.max_N = max_N

        self.inner = _ExactJointEntropy.empty(K)
        self.log_probs_max_N_K_C = np.empty((max_N, K, C))

        self.random_state = random_state

    def add_variables(self, log_probs_N_K_C):
        C = self.log_probs_max_N_K_C.shape[2]
        add_N = log_probs_N_K_C.shape[0]

        self.log_probs_max_N_K_C[self.N : self.N + add_N] = log_probs_N_K_C
        self.N += add_N

        num_exact_samples = C**self.N
        if num_exact_samples > self.M:
            self.inner = _SampledJointEntropy.sample(
                np.exp(self.log_probs_max_N_K_C[: self.N]),
                self.M,
                self.random_state,
            )
        else:
            self.inner.add_variables(log_probs_N_K_C)

        return self

    def compute_batch(self, log_probs_B_K_C):
        """
        Computes the joint entropy of the added variables together with the
        batch (one by one).
        """
        return self.inner.compute_batch(log_probs_B_K_C)


def _compute_conditional_entropy(log_probs_N_K_C):
    N, K, C = log_probs_N_K_C.shape

    nats_N_K_C = log_probs_N_K_C * np.exp(log_probs_N_K_C)
    nats_N_K_C[np.isnan(nats_N_K_C)] = 0
    entropies_N = -np.sum(nats_N_K_C, axis=(1, 2)) / K
    return entropies_N
