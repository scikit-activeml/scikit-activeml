"""Implementation of Maximum Loss Reduction with Maximal Confidence."""

import numpy as np
from sklearn import clone

from ..base import SingleAnnotatorPoolQueryStrategy, SkactivemlClassifier
from ..utils import (
    ExtLabelEncoder,
    is_unlabeled,
    simple_batch,
    check_type,
)
from ..utils._validation import _canonicalize_multilabel_probas
from ._target import _fit_and_resolve_estimator_target_spec


class MaxLossReductionMaxConfidence(SingleAnnotatorPoolQueryStrategy):
    """Maximum Loss Reduction with Maximal Confidence (MMC)

    This class implements the query strategy Maximum Loss Reduction with
    Maximal Confidence (MMC) [1]_ that selects the samples with the largest
    loss reduction under their most confident label assignment. That label
    assignment combines a multi-label classifier's label predictions with the
    number of positive labels predicted by a label-cardinality discriminator.
    This strategy is multi-label-only: `y` must be two-dimensional and each row
    must be either fully labeled or fully unlabeled.

    Parameters
    ----------
    missing_label : scalar or string or np.nan or None, default=np.nan
        Value to represent a missing label.
    random_state : int or np.random.RandomState, default=None
        Random state for candidate selection.
    target_type : "auto" or "multi-label", default="auto"
        Declared target type. A fitted classifier's target specification is
        authoritative when available. This strategy supports only multi-label
        classification with a single annotator.

    References
    ----------
    .. [1] Yang, B., Sun, J.-T., Wang, T., & Chen, Z. (2009). Effective
       Multi-Label Active Learning for Text Classification. In Proceedings of
       the 15th ACM SIGKDD International Conference on Knowledge Discovery and
       Data Mining (pp. 917-926).
    """

    @property
    def _target_capabilities(self):
        return frozenset(
            {("classification", "multi-label", "single-annotator")}
        )

    def query(
        self,
        X,
        y,
        discriminator,
        clf,
        fit_clf=True,
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
        y : array-like of shape (n_samples, n_outputs)
            Labels of the training data set (possibly including unlabeled
            rows indicated by `self.missing_label`). Each row must either
            contain only observed labels or only `missing_label` values, i.e.,
            no mixing within a row. This strategy supports multi-label data
            only. `predict_proba` must return either shape
            `(n_samples, n_outputs)` or a list of binary probability matrices
            with shape `(n_samples, 2)` per output.
        discriminator : skactiveml.base.SkactivemlClassifier
            Model implementing the methods `fit` and `predict`. It must support
            single-output classification with a single annotator and declare
            `target_type="auto"` or `target_type="single-output"`. It predicts
            a candidate sample's number of positive labels, i.e., its label
            cardinality. The parameters `classes` and `missing_label` will be
            internally redefined.
        clf : skactiveml.base.SkactivemlClassifier
            Classifier implementing the methods `fit` and `predict_proba`.
        fit_clf : bool, default=True
            Defines whether the classifier `clf` should be fitted on `X`
            and `y`.
        candidates : None or array-like of shape (n_candidates), dtype=int or \
                array-like of shape (n_candidates, n_features), default=None
            - If `candidates` is `None`, the unlabeled samples from `(X, y)`
              are considered as candidates.
            - If `candidates` is of shape `(n_candidates,)` and of type
              `int`, `candidates` is considered as the indices of the samples
              in `(X, y)`.
            - If `candidates` is of shape `(n_candidates, n_features)`, the
              candidate samples are directly given in `candidates` (not
              necessarily contained in `X`).

            A given `candidates` is authoritative, i.e., an index array is
            taken as given, such that labeled samples remain candidates, e.g.,
            to relabel them or to recompute their utilities.
        batch_size : int, default=1
            The number of samples to be selected in one AL cycle.
        return_utilities : bool, default=False
            If `True`, also return the utilities based on the query strategy.

        Returns
        -------
        query_indices : numpy.ndarray of shape (batch_size,)
            The `query_indices` indicate for which candidate sample a label is
            to be queried, e.g., `query_indices[0]` indicates the index of
            the first selected sample.
            If `candidates` is `None` or of shape `(n_candidates,)`, the
            indexing refers to samples in `X`.
            If `candidates` is of shape `(n_candidates, n_features)`, the
            indexing refers to samples in `candidates`.
        utilities : numpy.ndarray of shape (batch_size, n_samples) or \
                numpy.ndarray of shape (batch_size, n_candidates)
            The utilities of samples after each selected sample of the batch,
            e.g., `utilities[0]` indicates the utilities used for selecting
            the first sample (with index `query_indices[0]`) of the batch.
            Utilities for samples that are no candidates will be set to
            np.nan.
            If `candidates` is `None` or of shape `(n_candidates,)`, the
            indexing refers to samples in `X`.
            If `candidates` is of shape `(n_candidates, n_features)`, the
            indexing refers to samples in `candidates`.

        Notes
        -----
        An exhausted candidate pool, i.e., a fully labeled `(X, y)` queried
        with `candidates=None` or an empty `candidates`, is a valid
        acquisition state. It is answered with an empty batch of `batch_size`
        zero and a warning naming the exhaustion.
        """
        # Resolve through the classifier before acquisition state is changed.
        clf, target_spec = _fit_and_resolve_estimator_target_spec(
            self,
            clf,
            X,
            y,
            fit_estimator=fit_clf,
            sample_weight=None,
            estimator_name="clf",
            fit_name="fit_clf",
            estimator_types=(SkactivemlClassifier,),
        )

        # Validate parameters.
        X, y, candidates, batch_size, return_utilities = self._validate_data(
            X,
            y,
            candidates,
            batch_size,
            return_utilities,
            reset=True,
            target_type=target_spec.target_type,
        )
        X_cand, mapping = self._transform_candidates(
            candidates, X, y, target_type=target_spec.target_type
        )

        check_type(discriminator, "discriminator", SkactivemlClassifier)
        discriminator_capability = (
            "classification",
            "single-output",
            "single-annotator",
        )
        if (
            getattr(discriminator, "target_type", "auto")
            not in ("auto", "single-output")
            or discriminator_capability
            not in discriminator._target_capabilities
        ):
            raise ValueError(
                "`discriminator` must support single-output classification "
                "with a single annotator and declare `target_type='auto'` or "
                "`target_type='single-output'`."
            )

        discriminator = clone(discriminator)
        discriminator.classes = list(range(y.shape[1] + 1))
        discriminator.missing_label = -1

        # Determine the labeled samples, which train the discriminator.
        lbld_mask = ~is_unlabeled(
            y,
            missing_label=self.missing_label_,
            target_type=target_spec.target_type,
        )

        # Canonicalize both public multi-label probability formats before any
        # masking or arithmetic is applied.
        n_outputs = y.shape[1]
        probas = _canonicalize_multilabel_probas(
            clf.predict_proba(X), n_samples=len(X), n_outputs=n_outputs
        )
        cand_probas = _canonicalize_multilabel_probas(
            clf.predict_proba(X_cand),
            n_samples=len(X_cand),
            n_outputs=n_outputs,
        )

        # Train the label-cardinality discriminator on the confidence profiles
        # of the labeled samples and predict the candidates' label
        # cardinalities.
        label_encoder = ExtLabelEncoder(
            classes=target_spec.classes,
            missing_label=self.missing_label_,
            target_type=target_spec.target_type,
        )
        y_discriminator = label_encoder.fit_transform(y[lbld_mask]).sum(axis=1)
        discriminator.fit(
            _label_cardinality_features(probas[lbld_mask]), y_discriminator
        )
        n_positive_labels = discriminator.predict(
            _label_cardinality_features(cand_probas)
        )

        utilities_cand = max_loss_reduction_max_confidence(
            cand_probas, n_positive_labels
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


def max_loss_reduction_max_confidence(probas, n_positive_labels):
    """Calculate the maximum loss reduction with maximal confidence.

    For each candidate sample, the `n_positive_labels` most probable labels
    are predicted positive and the remaining ones negative [1]_. The loss
    reduction of this most confident labeling is the sum of the hinge-style
    losses `(1 - yhat * (2 * probas - 1)) / 2`, i.e., it sums `1 - probas` for
    the labels predicted positive and `probas` for the labels predicted
    negative.

    Parameters
    ----------
    probas : array-like of shape (n_candidates, n_outputs)
        Canonical positive-class probabilities of the candidate samples, i.e.,
        one probability per output. The equivalent list of `(n_candidates, 2)`
        binary probability matrices is canonicalized as well, although query
        strategies are expected to canonicalize at their own boundary.
    n_positive_labels : array-like of shape (n_candidates,)
        Predicted number of positive labels per candidate sample, e.g., as
        predicted by a label-cardinality discriminator. Each entry must be an
        integer in `[0, n_outputs]`.

    Returns
    -------
    utilities : numpy.ndarray of shape (n_candidates,)
        Loss reduction of each candidate sample under its most confident
        labeling, i.e., one finite value in `[0, n_outputs]` per candidate.
        Larger values indicate more useful candidates.

    Raises
    ------
    ValueError
        If `n_positive_labels` is not a one-dimensional array of integers
        within `[0, n_outputs]`, if `probas` is not a multi-label probability
        matrix with one row per entry of `n_positive_labels`, or if `probas`
        contains values outside of `[0, 1]`.

    References
    ----------
    .. [1] Yang, B., Sun, J.-T., Wang, T., & Chen, Z. (2009). Effective
       Multi-Label Active Learning for Text Classification. In Proceedings of
       the 15th ACM SIGKDD International Conference on Knowledge Discovery and
       Data Mining (pp. 917-926).
    """
    n_positive_labels = np.asarray(n_positive_labels)
    if n_positive_labels.ndim != 1:
        raise ValueError(
            "`n_positive_labels` must have shape `(n_candidates,)`, got "
            f"{n_positive_labels.shape}."
        )

    # Validating the candidate count here covers the array representation as
    # well as each per-output matrix of the list representation.
    probas = _canonicalize_multilabel_probas(
        probas, n_samples=len(n_positive_labels)
    )
    if not ((probas >= 0) & (probas <= 1)).all():
        raise ValueError(
            "`probas` must contain probabilities within `[0, 1]`."
        )
    n_outputs = probas.shape[1]
    if not np.isin(n_positive_labels, np.arange(n_outputs + 1)).all():
        raise ValueError(
            "`n_positive_labels` must contain integers within "
            f"`[0, {n_outputs}]`."
        )

    # Predict the `n_positive_labels` most probable labels as positive.
    ranking = np.flip(np.argsort(probas, axis=1), axis=-1)
    ranks = np.argsort(ranking, axis=1)
    yhat = np.where(ranks < n_positive_labels[:, None], 1, -1)

    margins = probas * 2 - 1
    return ((1 - yhat * margins) / 2).sum(axis=1)


def _label_cardinality_features(probas):
    """Sort probabilities per sample in decreasing order and normalize them.

    Parameters
    ----------
    probas : numpy.ndarray of shape (n_samples, n_outputs)
        Canonical positive-class probabilities.

    Returns
    -------
    features : numpy.ndarray of shape (n_samples, n_outputs)
        Label-order-independent input representation of the label-cardinality
        discriminator. An all-zero confidence profile remains all zero.
    """
    features = np.flip(np.sort(probas, axis=1), axis=-1)
    feature_sums = features.sum(axis=1, keepdims=True)
    return np.divide(
        features,
        feature_sums,
        out=np.zeros_like(features),
        where=feature_sums != 0,
    )
