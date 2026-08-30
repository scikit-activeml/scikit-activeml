"""
Module implementing `Falcun`, which is a deep active learning strategy jointly
selecting uncertain and diverse samples.
"""

import numpy as np

from ..base import SingleAnnotatorPoolQueryStrategy, SkactivemlClassifier
from ..utils import (
    MISSING_LABEL,
    check_scalar,
)
from ..utils._validation import _canonicalize_multilabel_probas
from ._uncertainty_sampling import uncertainty_scores
from ._target import _fit_and_resolve_estimator_target_spec


class Falcun(SingleAnnotatorPoolQueryStrategy):
    """Fast Active Learning by Contrastive UNcertainty (FALCUN)

    This class implements the "Fast Active Learning by Contrastive UNcertainty"
    (FALCUN) query strategy [1]_, which selects a batch directly in probability
    space using a self-adjusting mix of uncertainty and diversity. By operating
    only on low-dimensional class-probability outputs rather than deep
    embeddings, it achieves fast acquisitions while retaining strong label
    efficiency.

    The distances in probability space are initialized with the uncertainty
    scores themselves (cf. Eq. (3) in [1]_), so the first sample of a batch is
    sampled with a probability proportional to `(2 * uncertainty) ** gamma`
    and thus carries no diversity information. At `batch_size=1`, the
    acquisition is therefore gamma-tempered probabilistic margin sampling.

    FALCUN was proposed for single-output classification. Multi-label support
    in this implementation is an extension and not part of the original
    proposal in [1]_. For resolved multi-label targets, the paper's top-two
    margin is applied to each label output independently, i.e., the per-label
    uncertainty of the label output `j` is the binary margin
    `1 - |2 * p_j - 1|` of its positive-class probability `p_j`, and
    `multilabel_aggregation_fn` reduces these per-label margins along the
    label axis to the uncertainty of one sample. The diversity term stays the
    L1 distance in probability space, which for multi-label targets is taken
    between the independent per-output positive-class probabilities, i.e.,
    `sum_j |p_j(x) - p_j(x_query)|`. Correlations between label outputs
    therefore influence neither term.

    Parameters
    ----------
    gamma : float > 0, default=10
        Controls the randomness in the selection. A value of 0 corresponds to
        random sampling, while a value going to infinity corresponds to
        selecting the sample with the highest utility (relevance).
    missing_label : scalar or string or np.nan or None, default=np.nan
        Value to represent a missing label.
    random_state : None or int or np.random.RandomState, default=None
        The random state to use.
    multilabel_aggregation_fn : callable, default=np.mean
        Callable reducing the per-label uncertainty scores of one sample to
        one uncertainty score. It is only used for resolved multi-label
        classification targets. It is called with the per-label scores of
        shape `(n_samples, n_outputs)` and the label axis passed as the `axis`
        keyword argument, and must return one score per sample within the
        range of that sample's per-label scores, e.g. `np.mean`, `np.average`,
        `np.median`, `np.min`, `np.max`, or a quantile. `np.sum` is not
        supported, because its result grows with the number of label outputs.
        Only the callability of the reduction is validated at runtime, so a
        violating reduction silently changes the acquisition scale. Here, an
        inflated uncertainty would dominate the diversity term, which is
        min-max normalized to `[0, 1]` from the second selection of a batch
        onward.
    target_type : "auto" or "single-output" or "multi-label", default="auto"
        Declared target type. The strategy supports single-output and
        multi-label classification. A fitted classifier's target specification
        is authoritative when available.

    References
    ----------
    .. [1] S. Gilhuber, A. Beer, Y. Ma, and T. Seidl. FALCUN: A Simple and
       Efficient Deep Active Learning Strategy. In Joint Eur. Conf. Mach.
       Learn. Knowl. Discov. Databases, pages 421–439, 2024.
    """

    def __init__(
        self,
        gamma=10,
        missing_label=MISSING_LABEL,
        random_state=None,
        multilabel_aggregation_fn=np.mean,
        target_type="auto",
    ):
        super().__init__(
            missing_label=missing_label,
            random_state=random_state,
            target_type=target_type,
        )
        self.gamma = gamma
        self.multilabel_aggregation_fn = multilabel_aggregation_fn

    @property
    def _target_capabilities(self):
        return frozenset(
            {
                ("classification", "single-output", "single-annotator"),
                ("classification", "multi-label", "single-annotator"),
            }
        )

    def query(
        self,
        X,
        y,
        clf,
        fit_clf=True,
        sample_weight=None,
        candidates=None,
        batch_size=1,
        return_utilities=False,
    ):
        """Query the next samples to be labeled.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data set, usually complete, i.e., including the labeled
            and unlabeled samples.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Labels of the training data set (possibly including unlabeled ones
            indicated by `self.missing_label`). For multi-label targets, a row
            `y[i]` must either contain only observed labels or only
            `missing_label` values, i.e., no mixing within a row. In this
            case, only multilabel classification problems, i.e. multiple
            binary classification tasks, are supported. `predict_proba` must
            then return either shape `(n_samples, n_outputs)` or a list of
            binary probability matrices with shape `(n_samples, 2)` per
            output.
        clf : skactiveml.base.SkactivemlClassifier
            Classifier implementing the methods `fit` and `predict_proba`.
        fit_clf : bool, default=True
            Defines whether the classifier `clf` should be fitted on `X`, `y`,
            and `sample_weight`.
        sample_weight : array-like of shape (n_samples,) or \
                (n_samples, n_outputs), default=None
            Weights of training samples in `X`. For two-dimensional `y`, one
            weight per sample is supported. Per-target weights are forwarded
            to `clf.fit` without additional validation and require estimator
            support.
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
        batch_size : int, default=1
            The number of samples to be selected in one AL cycle.
        return_utilities : bool, default=False
            If true, also return the utilities based on the query strategy.

        Returns
        -------
        query_indices : numpy.ndarray of shape (batch_size)
            The query indices indicate for which candidate sample a label is to
            be queried, e.g., `query_indices[0]` indicates the first selected
            sample.

            - If `candidates` is `None` or of shape
              `(n_candidates,)`, the indexing refers to the samples in
              `X`.
            - If `candidates` is of shape `(n_candidates, n_features)`,
              the indexing refers to the samples in `candidates`.
        utilities : numpy.ndarray of shape (batch_size, n_samples)
            The utilities of samples after each selected sample of the batch,
            e.g., `utilities[0]` indicates the utilities used for selecting
            the first sample (with index `query_indices[0]`) of the batch.
            Utilities for labeled samples will be set to np.nan.

            - If `candidates` is `None`, the indexing refers to the samples
              in `X`.
            - If `candidates` is of shape `(n_candidates,)` and of type
              `int`, `utilities` refers to the samples in `X`.
            - If `candidates` is of shape `(n_candidates, ...)`, `utilities`
              refers to the indexing in `candidates`.
        """
        # Resolve through the classifier before acquisition state is changed.
        clf, target_spec = _fit_and_resolve_estimator_target_spec(
            self,
            clf,
            X,
            y,
            fit_estimator=fit_clf,
            sample_weight=sample_weight,
            estimator_name="clf",
            fit_name="fit_clf",
            estimator_types=(SkactivemlClassifier,),
        )

        # Check parameters.
        X, y, candidates, batch_size, return_utilities = self._validate_data(
            X,
            y,
            candidates,
            batch_size,
            return_utilities,
            reset=True,
            target_type=target_spec.target_type,
        )

        # Determine candidate samples for selection.
        X_cand, mapping = self._transform_candidates(
            candidates=candidates,
            X=X,
            y=y,
            target_type=target_spec.target_type,
        )

        check_scalar(
            self.gamma,
            "gamma",
            min_val=0,
            target_type=(float, int),
            min_inclusive=True,
        )
        if not callable(self.multilabel_aggregation_fn):
            raise TypeError("`multilabel_aggregation_fn` must be callable.")
        # Compute uncertainties via margin sampling (cf. Eq. (1) in [1]).
        probas_cand = clf.predict_proba(X_cand)
        is_multilabel = target_spec.target_type == "multi-label"
        if is_multilabel:
            probas_cand = _canonicalize_multilabel_probas(
                probas_cand,
                n_samples=len(X_cand),
                n_outputs=y.shape[1],
            )
        unc_cand = uncertainty_scores(
            probas=probas_cand,
            method="margin_sampling",
            is_multilabel=is_multilabel,
            multilabel_aggregation_fn=self.multilabel_aggregation_fn,
        )

        # Initialize distances in probability space (cf. Eq. (3) in [1]).
        dist_cand = unc_cand.copy()

        query_indices = []
        utilities_cand = np.full((batch_size, len(X_cand)), np.nan)
        cand_indices = np.arange(len(X_cand))
        for b in range(batch_size):
            if b > 0:
                # Update distances (diversity) values in the class probability
                # space (cf. Eqs. (2) and (4) in [1]).
                probas_q = probas_cand[[query_indices[int(b - 1)]]]
                dist_new = np.abs(probas_cand - probas_q).sum(axis=1)
                dist_cand = np.minimum(dist_new, dist_cand)
                dist_min = dist_cand.min()
                dist_range = dist_cand.max() - dist_min
                dist_cand -= dist_min
                if dist_range > 0:
                    dist_cand /= dist_range

            # Compute relevance scores for candidates (cf. Eq. (5) and
            # (6) in [1]).
            rel_cand = (unc_cand + dist_cand) ** self.gamma
            rel_cand[query_indices] = 0
            rel_cand_sum = np.sum(rel_cand)
            if rel_cand_sum == 0:
                rel_cand = np.ones_like(rel_cand)
                rel_cand[query_indices] = 0
            rel_cand = rel_cand / np.sum(rel_cand)

            # Sample instance to be labeled (cf. Eq. (6) in [1]).
            query_idx = self.random_state_.choice(
                cand_indices, p=rel_cand, size=1
            )
            rel_cand[query_indices] = np.nan
            utilities_cand[b] = rel_cand
            query_indices.append(query_idx[0])

        if mapping is not None:
            query_indices = mapping[query_indices]
            utilities = np.full((batch_size, len(X)), np.nan)
            utilities[:, mapping] = utilities_cand
        else:
            utilities = utilities_cand
        query_indices = np.asarray(query_indices, dtype=int)

        if return_utilities:
            return query_indices, utilities
        else:
            return query_indices
