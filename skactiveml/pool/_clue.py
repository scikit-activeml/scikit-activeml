"""
Module implementing Clustering Uncertainty-weighted Embeddings (CLUE).

CLUE is a deep active learning strategy, which performs a clustering with
uncertainties as sample weights.
"""

import numpy as np

from ..base import (
    SingleAnnotatorPoolQueryStrategy,
    SkactivemlClassifier,
    SkactivemlRegressor,
)
from ..pool import uncertainty_scores
from ..utils import (
    MISSING_LABEL,
    rand_argmax,
    check_type,
)
from sklearn.cluster import KMeans
from ._target import _fit_and_resolve_estimator_target_spec


class Clue(SingleAnnotatorPoolQueryStrategy):
    """Clustering Uncertainty-weighted Embeddings (CLUE)

    This class implements the Clustering Uncertainty-weighted Embeddings (CLUE)
    query strategy [1]_ clusters latent embeddings while weighting samples by
    predictive uncertainty, then picks samples near the cluster centers. The
    result is a diverse set biased toward uncertain regions of representation
    space.

    The original `Clue` query strategy was proposed for classification tasks
    only and did not include regression or multilabel variants. Support for
    multilabel classification in this implementation is therefore an
    extension of the original formulation. In the multilabel case, predictive
    uncertainties are computed independently per label and then reduced across
    labels via `multilabel_aggregation_fn`. Support for regression is likewise
    an extension and relies on user-provided sample-wise uncertainty
    estimates.

    Parameters
    ----------
    predict_dict : dict or None, default=None
        Optional keyword arguments passed to the estimator's prediction
        method in order to obtain sample embeddings and/or uncertainties as
        additional outputs.

        * For classification, `Clue` calls::

            out = estimator.predict_proba(X, **predict_dict)

        * For regression, `Clue` calls::

            out = estimator.predict(X, **predict_dict)

        If `out` is a tuple, its additional elements are inferred by shape:
        sample-wise uncertainties must be a 1D `numpy.ndarray`, and
        sample embeddings must be a 2D `numpy.ndarray`.

        In the classification case, returning uncertainties is optional,
        because they can be derived from the predicted class probabilities
        (see the documentation of the `method` parameter). In the regression
        case, providing uncertainties as an additional output is mandatory.
    method : 'least_confident' or 'margin_sampling' or 'entropy', \
            default="entropy"
        Fallback uncertainty measure used in the classification case when
        the classifier does not provide explicit uncertainties.

        - `method='least_confident'` queries the sample whose maximal posterior
          probability is minimal.
        - `method='margin_sampling'` queries the sample whose posterior
          probability gap between the most and the second most probable class
          label is minimal.
        - `method='entropy'` queries the sample whose posterior's have the
          maximal entropy.
    cluster_algo : ClusterMixin.__class__, default=KMeans
        The cluster algorithm to be used. It must implement a `fit_transform`
        method, which takes samples `X` and `sample_weight` as inputs, e.g.,
        `sklearn.clustering.KMeans` and `sklearn.clustering.MiniBatchKMeans`.
    cluster_algo_dict : dict, default=None
        The parameters passed to the clustering algorithm `cluster_algo`,
        excluding the parameter for the number of clusters.
    n_cluster_param_name : string, default="n_clusters"
        The name of the parameter for the number of clusters.
    missing_label : scalar or string or np.nan or None, default=np.nan
        Value to represent a missing label.
    random_state : None or int or np.random.RandomState, default=None
        The random state to use.
    multilabel_aggregation_fn: callable, default=np.mean
        Callable that takes axis as kwarg and reduces along that axis. Common
        choices are `np.mean`, `np.min`, `np.max`, or any quantiles, while
        `np.sum` is not allowed. This is only used for the multilabel
        extension of CLUE and is not part of the original method proposed in
        [1]_.
    target_type : {"auto", "single-output", "multi-label", "multi-output"}, \
            default="auto"
        Declared target type. A fitted estimator's target specification is
        authoritative when available.

    References
    ----------
    .. [1] V. Prabhu, A. Chandrasekaran, K. Saenko, and J. Hoffman. Active
       domain adaptation via clustering uncertainty-weighted embeddings. In
       IEEE/CVF Int. Conf. Comput. Vis., pages 8505–8514, 2021.
    """

    def __init__(
        self,
        predict_dict=None,
        method="entropy",
        cluster_algo=KMeans,
        cluster_algo_dict=None,
        n_cluster_param_name="n_clusters",
        missing_label=MISSING_LABEL,
        random_state=None,
        multilabel_aggregation_fn=np.mean,
        target_type="auto",
    ):
        super().__init__(
            missing_label=missing_label, random_state=random_state
        )
        self.cluster_algo = cluster_algo
        self.cluster_algo_dict = cluster_algo_dict
        self.n_cluster_param_name = n_cluster_param_name
        self.method = method
        self.predict_dict = predict_dict
        self.multilabel_aggregation_fn = multilabel_aggregation_fn
        self.target_type = target_type

    @property
    def _target_capabilities(self):
        return frozenset(
            {
                ("classification", "single-output", "single-annotator"),
                ("classification", "multi-label", "single-annotator"),
                ("regression", "single-output", "single-annotator"),
            }
        )

    def query(
        self,
        X,
        y,
        estimator,
        fit_estimator=True,
        sample_weight=None,
        candidates=None,
        batch_size=1,
        return_utilities=False,
    ):
        """Determines for which candidate samples labels are to be queried.

        Parameters
        ----------
        X : array-like of shape (n_samples, ...)
            Training data set, usually complete, i.e., including the labeled
            and unlabeled samples.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Labels of the training data set (possibly including unlabeled ones
            indicated by `self.missing_label`). For multi-label targets, a row
            `y[i]` must either contain only observed labels or only
            `missing_label` values, i.e., no mixing within a row.
        estimator : skactiveml.base.SkactivemlClassifier\
                or skactiveml.base.SkactivemlRegressor
            Estimator implementing the methods `fit` and
            `predict_proba` (classification) or `predict` (regression).
            For multilabel classification, `predict_proba` may either return
            one probability per label with shape `(n_samples, n_outputs)` or a
            list of binary probability matrices with shape
            `(n_samples, 2)` per output as commonly returned by
            multioutput scikit-learn estimators.
        fit_estimator : bool, default=True
            Defines whether the `estimator` should be fitted on
            `X`, `y`, and `sample_weight`.
        sample_weight: array-like of shape (n_samples,) or \
                (n_samples, n_outputs), default=None
            Weights of training samples in `X`. For two-dimensional `y`, one
            weight per sample is supported. Per-target weights are forwarded
            to `estimator.fit` without additional validation and require
            estimator support.
        candidates : None or array-like of shape (n_candidates,), dtype=int or\
                array-like of shape (n_candidates, ...), default=None
            - If `candidates` is `None`, the unlabeled samples from
              `(X,y)` are considered as `candidates`.
            - If `candidates` is of shape `(n_candidates,)` and of type
              `int`, `candidates` is considered as the indices of the
              samples in `(X,y)`.
            - Candidate samples passed directly with shape
              `(n_candidates, ...)` are not supported because `Clue`
              requires a mapping to the samples in `X`.
        batch_size : int, default=1
            The number of samples to be selected in one AL cycle.
        return_utilities : bool, default=False
            If `True`, also return the utilities based on the query strategy.

        Returns
        -------
        query_indices : numpy.ndarray of shape (batch_size,)
            The query indices indicate for which candidate sample a label is
            to be queried, e.g., `query_indices[0]` indicates the first
            selected sample. The indexing refers to the samples in `X`.
        utilities : numpy.ndarray of shape (batch_size, n_samples) or \
                numpy.ndarray of shape (batch_size, n_candidates)
            The utilities of samples after each selected sample of the batch,
            e.g., `utilities[0]` indicates the utilities used for selecting
            the first sample (with index `query_indices[0]`) of the batch.
            Utilities for labeled samples will be set to np.nan. The indexing
            refers to the samples in `X`.
        """
        # Resolve through the estimator before acquisition state is changed.
        estimator, target_spec = _fit_and_resolve_estimator_target_spec(
            self,
            estimator,
            X,
            y,
            fit_estimator=fit_estimator,
            sample_weight=sample_weight,
            estimator_name="estimator",
            fit_name="fit_estimator",
            estimator_types=(SkactivemlClassifier, SkactivemlRegressor),
        )

        # Check `__init__` and `query` parameters.
        X, y, candidates, batch_size, return_utilities = self._validate_data(
            X=X,
            y=y,
            candidates=candidates,
            batch_size=batch_size,
            return_utilities=return_utilities,
            reset=True,
            allow_multioutput=True,
            target_type=target_spec.target_type,
        )
        X_cand, mapping = self._transform_candidates(
            candidates=candidates,
            X=X,
            y=y,
            enforce_mapping=True,
            target_type=target_spec.target_type,
        )
        if not callable(self.multilabel_aggregation_fn):
            raise TypeError("`multilabel_aggregation_fn` must be callable.")
        check_type(
            self.cluster_algo_dict, "cluster_algo_dict", (dict, type(None))
        )
        cluster_algo_dict = (
            {}
            if self.cluster_algo_dict is None
            else self.cluster_algo_dict.copy()
        )
        check_type(self.n_cluster_param_name, "n_cluster_param_name", str)
        predict_dict = {} if self.predict_dict is None else self.predict_dict
        check_type(predict_dict, "predict_dict", dict)
        if self.method not in [
            "least_confident",
            "margin_sampling",
            "entropy",
        ]:
            raise ValueError(
                f"`method` must be 'least_confident' or 'margin_sampling'"
                f"or 'entropy'. Got {self.method} instead."
            )

        # Compute predictions plus optional embeddings and/or uncertainties.
        is_clf = target_spec.task == "classification"
        if is_clf:
            out = estimator.predict_proba(X_cand, **predict_dict)
        else:
            out = estimator.predict(X_cand, **predict_dict)
        if not isinstance(out, tuple):
            out = (out,)
        main = out[0]
        emb = None
        uncertainties = None
        for out_element in out[1:]:
            if out_element.ndim == 1 and uncertainties is None:
                uncertainties = out_element
            elif out_element.ndim == 2 and emb is None:
                emb = out_element
            else:
                raise ValueError(
                    "The optional outputs when calling `predict_proba` or"
                    "`predict` must either be a 1D `np.ndarray` for the "
                    "uncertainties or a 2D `np.ndarray` for the sample "
                    "embeddings."
                )

        # Use original samples as a fallback.
        X_cand = X_cand if emb is None else emb

        if is_clf and uncertainties is None:
            # Compute uncertainties as a fallback in the classification case.
            uncertainties = uncertainty_scores(
                probas=main,
                method=self.method,
                is_multilabel=target_spec.target_type == "multi-label",
                multilabel_aggregation_fn=self.multilabel_aggregation_fn,
            )
        elif not is_clf and uncertainties is None:
            raise ValueError(
                "For regression, `predict` must return uncertainties."
            )

        # Implement a fallback, if all uncertainties are zero.
        if np.nansum(uncertainties) == 0:
            uncertainties = np.ones_like(uncertainties)

        # Perform clustering to get centroids.
        cluster_algo_dict[self.n_cluster_param_name] = batch_size
        cluster_obj = self.cluster_algo(**cluster_algo_dict)
        dist = cluster_obj.fit_transform(
            X_cand, y=None, sample_weight=uncertainties
        )

        # Determine `query_indices` of the samples being closest to the
        # respective centroids.
        query_indices = []
        utilities = np.full((batch_size, len(X)), fill_value=np.nan)
        for b in range(batch_size):
            utilities[b][mapping] = -dist[:, b]
            utilities[b][query_indices] = np.nan
            idx_b = rand_argmax(utilities[b], random_state=self.random_state_)
            query_indices.append(idx_b[0])
        query_indices = np.array(query_indices, dtype=int)

        if return_utilities:
            return query_indices, utilities
        else:
            return query_indices
