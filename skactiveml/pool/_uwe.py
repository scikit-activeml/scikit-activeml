"""
Module implementing the pool-based query strategy Uncertainty-Weighted
Embeddings (UWE).
"""

import numpy as np
from sklearn.metrics import pairwise_distances_argmin_min

from ..base import (
    SingleAnnotatorPoolQueryStrategy,
    SkactivemlClassifier,
    SkactivemlRegressor,
)
from ..utils import (
    MISSING_LABEL,
    check_scalar,
    check_type,
    labeled_indices,
    rand_argmax,
)
from ._target import _fit_and_resolve_estimator_target_spec
from ._uncertainty_sampling import uncertainty_scores


class UWE(SingleAnnotatorPoolQueryStrategy):
    """Uncertainty-Weighted Embeddings (UWE)

    This class implements the hybrid active learning strategy UWE [1]_, which
    generalizes the gradient embeddings of BADGE [2]_. Each sample `x` is
    represented by its uncertainty-weighted embedding
    `uwe(x) = u(x)**tau * f(x)`, where `u(x) >= 0` is an uncertainty measure
    of the current model and `f(x)` is a (learned) sample representation.
    Since the magnitude of `uwe(x)` grows with the uncertainty, a
    distance-based selection in this space yields batches of uncertain and
    diverse samples. Following Algorithm 1 in [1]_, the batch is selected via
    the K-Center-Greedy algorithm: Each round selects the candidate whose
    uncertainty-weighted embedding is farthest from its nearest neighbor among
    the labeled samples and the samples selected so far.

    In contrast to BADGE, the embedding is not tied to the cross-entropy loss,
    i.e., any non-negative uncertainty measure can be used. For
    classification, the uncertainty is computed from the predicted class
    probabilities by default (see `method`). Alternatively, and mandatory for
    regression, the estimator can return sample-wise uncertainties as an
    additional output (see `predict_dict`).

    If there are no labeled samples, the first selected sample is the one with
    the largest embedding norm, i.e., the candidate farthest from the origin,
    as in the authors' reference implementation. The origin is no center for
    the subsequent rounds.

    Parameters
    ----------
    predict_dict : dict or None, default=None
        Optional keyword arguments passed to the estimator's prediction
        method in order to obtain sample embeddings and/or uncertainties as
        additional outputs.

        * For classification, `UWE` calls::

            out = estimator.predict_proba(X, **predict_dict)

        * For regression, `UWE` calls::

            out = estimator.predict(X, **predict_dict)

        If `out` is a tuple, its additional elements are inferred by shape:
        sample-wise uncertainties must be a 1D `numpy.ndarray`, and sample
        embeddings must be a 2D `numpy.ndarray`. If no embeddings are
        returned, the input samples `X` are used as sample representations.

        In the classification case, returning uncertainties is optional,
        because they can be derived from the predicted class probabilities
        (see the documentation of the `method` parameter). In the regression
        case, providing uncertainties as an additional output is mandatory.
    method : 'least_confident' or 'margin_sampling' or 'entropy', \
            default='entropy'
        Uncertainty measure used in the classification case when the
        classifier does not provide explicit uncertainties. The default
        `'entropy'` is the measure used for classification in [1]_.

        - `method='least_confident'` uses one minus the maximal posterior
          probability.
        - `method='margin_sampling'` uses one minus the posterior probability
          gap between the most and the second most probable class label.
        - `method='entropy'` uses the entropy of the posterior probabilities.
    tau : int or float, default=1
        Non-negative exponent applied to the uncertainties, i.e.,
        `uwe(x) = u(x)**tau * f(x)` (see Section 5.2 in [1]_). Increasing
        `tau` emphasizes uncertainty, whereas decreasing `tau` emphasizes
        diversity. For `tau=0`, the uncertainties are ignored, such that the
        selection reduces to the K-Center-Greedy algorithm on the sample
        representations.
    missing_label : scalar or string or np.nan or None, default=np.nan
        Value to represent a missing label.
    random_state : None or int or np.random.RandomState, default=None
        The random state to use. It breaks ties between candidates with equal
        distances.
    target_type : "auto" or "single-output", default="auto"
        Declared target type. The strategy supports single-output
        classification and regression. A fitted estimator's target
        specification is authoritative when available.

    References
    ----------
    .. [1] Y. He, L. Cai, J. Liao, and C.-S. Foo. Hybrid Active Learning with
       Uncertainty-Weighted Embeddings. Trans. Mach. Learn. Res., 2024.
    .. [2] J. T. Ash, C. Zhang, A. Krishnamurthy, J. Langford, and A. Agarwal.
       Deep Batch Active Learning by Diverse, Uncertain Gradient Lower Bounds.
       In Int. Conf. Learn. Represent., 2020.
    """

    def __init__(
        self,
        predict_dict=None,
        method="entropy",
        tau=1,
        missing_label=MISSING_LABEL,
        random_state=None,
        target_type="auto",
    ):
        super().__init__(
            missing_label=missing_label,
            random_state=random_state,
            target_type=target_type,
        )
        self.predict_dict = predict_dict
        self.method = method
        self.tau = tau

    @property
    def _target_capabilities(self):
        return frozenset(
            {
                ("classification", "single-output", "single-annotator"),
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
        X : array-like of shape (n_samples, n_features)
            Training data set, usually complete, i.e., including the labeled
            and unlabeled samples.
        y : array-like of shape (n_samples,)
            Labels of the training data set (possibly including unlabeled ones
            indicated by `self.missing_label`).
        estimator : skactiveml.base.SkactivemlClassifier\
                or skactiveml.base.SkactivemlRegressor
            Estimator implementing the methods `fit` and `predict_proba`
            (classification) or `predict` (regression).
        fit_estimator : bool, default=True
            Defines whether the `estimator` should be fitted on `X`, `y`, and
            `sample_weight`.
        sample_weight : array-like of shape (n_samples,), default=None
            Weights of training samples in `X`.
        candidates : None or array-like of shape (n_candidates,), dtype=int or\
                array-like of shape (n_candidates, n_features), default=None
            - If `candidates` is `None`, the unlabeled samples from
              `(X,y)` are considered as `candidates`.
            - If `candidates` is of shape `(n_candidates,)` and of type
              `int`, `candidates` is considered as the indices of the
              samples in `(X,y)`.
            - If `candidates` is of shape `(n_candidates, ...)`, the
              candidate samples are directly given in `candidates` (not
              necessarily contained in `X`).

            A given index array is taken as given, i.e., labeled samples
            remain candidates. Their distance to the labeled samples is zero,
            so they are only selected once no other candidate is farther from
            the labeled and selected samples.
        batch_size : int, default=1
            The number of samples to be selected in one AL cycle. If it
            exceeds the number of candidates, it is reduced to that number and
            a warning is raised.
        return_utilities : bool, default=False
            If `True`, also return the utilities based on the query strategy.

        Returns
        -------
        query_indices : numpy.ndarray of shape (batch_size,)
            The query indices indicate for which candidate sample a label is
            to be queried, e.g., `query_indices[0]` indicates the first
            selected sample. A sample is selected at most once per batch.

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
            A utility is the Euclidean distance of a candidate's
            uncertainty-weighted embedding to its nearest neighbor among the
            labeled samples and the samples selected in earlier rounds. If
            there are no labeled samples, the first row contains the
            embedding norms instead. Utilities for samples that are no
            candidates and for candidates that have already been selected in
            an earlier round will be set to np.nan.

            - If `candidates` is `None` or of shape
              `(n_candidates,)`, the indexing refers to the samples in
              `X`.
            - If `candidates` is of shape `(n_candidates, n_features)`,
              the indexing refers to the samples in `candidates`.
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
            target_type=target_spec.target_type,
        )
        X_cand, mapping = self._transform_candidates(
            candidates=candidates,
            X=X,
            y=y,
            target_type=target_spec.target_type,
        )
        predict_dict = {} if self.predict_dict is None else self.predict_dict
        check_type(predict_dict, "predict_dict", dict)
        if self.method not in [
            "least_confident",
            "margin_sampling",
            "entropy",
        ]:
            raise ValueError(
                f"`method` must be 'least_confident', 'margin_sampling', or "
                f"'entropy'. Got {self.method} instead."
            )
        if isinstance(self.tau, bool):
            raise TypeError("`tau` must be an `int` or `float`, not `bool`.")
        check_scalar(self.tau, "tau", (int, float), min_val=0)
        if not np.isfinite(self.tau):
            raise ValueError(f"`tau` must be finite. Got {self.tau} instead.")

        # Algorithm 1 in [1] computes the embeddings of the candidates and of
        # the labeled samples, which are the initial centers.
        lbld_indices = labeled_indices(y, missing_label=self.missing_label_)
        n_cand = len(X_cand)
        X_all = np.concatenate([X_cand, X[lbld_indices]], axis=0)

        # Compute predictions plus optional embeddings and/or uncertainties.
        is_clf = target_spec.task == "classification"
        if is_clf:
            out = estimator.predict_proba(X_all, **predict_dict)
        else:
            out = estimator.predict(X_all, **predict_dict)
        if not isinstance(out, tuple):
            out = (out,)
        embeddings = None
        uncertainties = None
        for out_element in out[1:]:
            out_element = np.asarray(out_element)
            if out_element.ndim == 1 and uncertainties is None:
                uncertainties = out_element
            elif out_element.ndim == 2 and embeddings is None:
                embeddings = out_element
            else:
                raise ValueError(
                    "The optional outputs when calling `predict_proba` or "
                    "`predict` must either be a 1D `np.ndarray` for the "
                    "uncertainties or a 2D `np.ndarray` for the sample "
                    "embeddings."
                )
        if embeddings is None:
            embeddings = X_all.reshape(len(X_all), -1)
        if uncertainties is None:
            if not is_clf:
                raise ValueError(
                    "For regression, `predict` must return uncertainties."
                )
            uncertainties = uncertainty_scores(
                probas=out[0], method=self.method
            )
        embeddings = np.asarray(embeddings, dtype=np.float64)
        uncertainties = np.asarray(uncertainties, dtype=np.float64)
        if len(uncertainties) != len(X_all) or len(embeddings) != len(X_all):
            raise ValueError(
                "The uncertainties and embeddings must contain one entry per "
                "sample passed to the estimator."
            )
        # If all uncertainties are zero, all embeddings collapse onto the
        # origin. If all uncertainties are undefined, e.g., for a regressor
        # fitted without labeled samples, they carry no information. In both
        # cases, fall back to the K-Center-Greedy algorithm on the sample
        # representations.
        if np.isnan(uncertainties).all() or not uncertainties.any():
            uncertainties = np.ones_like(uncertainties)
        elif not np.isfinite(uncertainties).all() or (uncertainties < 0).any():
            raise ValueError("The uncertainties must be finite and >= 0.")

        # Uncertainty-weighted embeddings according to Eq. (4) in [1] with the
        # exponent `tau` from Section 5.2 in [1].
        uwe = (uncertainties**self.tau)[:, None] * embeddings
        uwe_cand, uwe_lbld = uwe[:n_cand], uwe[n_cand:]

        # Distance of each candidate to its nearest labeled sample. Without
        # labeled samples, the norms only serve to determine the first center.
        has_centers = len(uwe_lbld) > 0
        if has_centers:
            _, min_dist = pairwise_distances_argmin_min(uwe_cand, uwe_lbld)
        else:
            min_dist = np.linalg.norm(uwe_cand, axis=1)

        if mapping is not None:
            cand_mapping = mapping
            utilities = np.full((batch_size, len(X)), fill_value=np.nan)
        else:
            cand_mapping = np.arange(n_cand)
            utilities = np.full((batch_size, n_cand), fill_value=np.nan)

        # K-Center-Greedy selection, i.e., lines 7-11 of Algorithm 1 in [1].
        query_indices_in_cand = []
        for i in range(batch_size):
            utilities_cand = min_dist.copy()
            utilities_cand[query_indices_in_cand] = np.nan
            utilities[i, cand_mapping] = utilities_cand
            idx_in_cand = int(
                rand_argmax(utilities_cand, random_state=self.random_state_)[0]
            )
            query_indices_in_cand.append(idx_in_cand)
            dist_new = np.linalg.norm(uwe_cand - uwe_cand[idx_in_cand], axis=1)
            if has_centers:
                min_dist = np.minimum(min_dist, dist_new)
            else:
                min_dist = dist_new
                has_centers = True

        query_indices = cand_mapping[query_indices_in_cand].astype(int)
        if return_utilities:
            return query_indices, utilities
        else:
            return query_indices
