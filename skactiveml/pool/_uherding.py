"""
Module implementing `UHerding`, a deep active learning strategy combining
uncertainty and coverage.
"""

import numpy as np

from scipy.special import softmax
from sklearn import clone
from sklearn.metrics import pairwise_distances, pairwise_kernels
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.utils.validation import column_or_1d

from ..base import SingleAnnotatorPoolQueryStrategy, SkactivemlClassifier
from ..utils import (
    MISSING_LABEL,
    check_equal_missing_label,
    check_scalar,
    check_type,
    labeled_indices,
    rand_argmax,
)
from ._uncertainty_sampling import uncertainty_scores


class UHerding(SingleAnnotatorPoolQueryStrategy):
    """Uncertainty Herding (UHerding)

    "Uncertainty Herding" (UHering) is a query strategy [1]_ that
    greedily maximizes an uncertainty-weighted coverage objective in feature
    space. In addition to the greedy selection itself, the implementation
    follows the paper-faithful parameter adaptation scheme:

    - select a temperature based on calibration via train/validation splits of
      the currently labeled set,
    - adapt the Gaussian kernel radius to the current labeled feature space.

    Parameters
    ----------
    method : 'least_confident' or 'margin_sampling' or 'entropy', \
            default='margin_sampling'
        Uncertainty definition applied to temperature-scaled probabilities.
    predict_proba_dict : dict or None, default=None
        Optional keyword arguments forwarded to `clf.predict_proba` to request
        additional outputs such as logits and embeddings.

        If `predict_proba_parser is None`, optional outputs are interpreted by
        the default convention `(probas, logits, embeddings)`.
        Typical usage with `SkorchClassifier` is therefore::

            predict_proba_dict={"extra_outputs": ["logits", "emb"]}

        If logits are not returned by `predict_proba`, `decision_function`
        is used as a fallback when available, e.g. for scikit-learn logistic
        regression models wrapped by `SklearnClassifier`.
    predict_proba_parser : callable or None, default=None
        Optional parser applied to the raw return value of
        `clf.predict_proba(X, **predict_proba_dict)`.

        The parser must return either `(probas, logits)` or
        `(probas, logits, embeddings)`. `probas` may be `None`, in which case
        they are computed from `logits` via softmax. `embeddings` may be
        `None`, in which case the original samples are used.

        If `None`, the default convention is used:

        - array output: treated as `probas`,
        - tuple output: treated as `(probas, logits, embeddings)`.

    temperatures : float or array-like of shape (n_temperatures,) or None, \
            default=None
        Candidate temperatures used during the calibration search. If a single
        positive float or a length-one array is provided, that temperature is
        used directly without internal calibration refits. If `None`,
        `temperatures=np.logspace(-1, 1, 49)` is used.
    validation_size : float or int, default=0.2
        Validation size passed to the calibration train/validation split.
    n_ece_bins : int, default=15
        Number of bins used for the expected calibration error.
    normalize_samples : bool, default=True
        Flag whether to normalize feature vectors to unit length before
        computing pairwise distances and kernels.
    adaptive_sigma : bool, default=True
        Flag whether to adapt the radius according to the minimum non-zero
        labeled pairwise distance. This option requires `metric='rbf'`.
    metric : str or callable, default='rbf'
        Kernel used for the coverage objective.
    metric_dict : dict or None, default=None
        Optional keyword arguments passed to `pairwise_kernels`.
    missing_label : scalar or string or np.nan or None, default=np.nan
        Value to represent a missing label.
    random_state : None or int or np.random.RandomState, default=None
        The random state to use.

    References
    ----------
    .. [1] W. Bae, G. Oliveira, and D. J. Sutherland.
       "Uncertainty Herding: One Active Learning Method for All Label
       Budgets." In Int. Conf. Learn. Represent., 2025.
    """

    def __init__(
        self,
        method="margin_sampling",
        predict_proba_dict=None,
        predict_proba_parser=None,
        temperatures=None,
        validation_size=0.2,
        n_ece_bins=15,
        normalize_samples=True,
        adaptive_sigma=True,
        metric="rbf",
        metric_dict=None,
        missing_label=MISSING_LABEL,
        random_state=None,
    ):
        super().__init__(
            missing_label=missing_label, random_state=random_state
        )
        self.method = method
        self.predict_proba_dict = predict_proba_dict
        self.predict_proba_parser = predict_proba_parser
        self.temperatures = temperatures
        self.validation_size = validation_size
        self.n_ece_bins = n_ece_bins
        self.normalize_samples = normalize_samples
        self.adaptive_sigma = adaptive_sigma
        self.metric = metric
        self.metric_dict = metric_dict

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
        """Determines for which candidate samples labels are to be queried.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data set, usually complete, i.e., including the labeled
            and unlabeled samples.
        y : array-like of shape (n_samples,)
            Labels of the training data set (possibly including unlabeled ones
            indicated by `self.missing_label`).
        clf : skactiveml.base.SkactivemlClassifier
            Classifier implementing `fit` and `predict_proba`. For
            temperature-scaled uncertainty estimation, the classifier should
            either provide logits via `predict_proba` extras or implement
            `decision_function`. Otherwise, the non-calibrated probabilities
            are used as fallback.
        fit_clf : bool, default=True
            Defines whether the classifier `clf` should be fitted on `X`, `y`,
            and `sample_weight` before evaluating the acquisition function.
            Independent of this flag, temporary cloned classifiers may still be
            fitted internally to select the temperature parameter.
        sample_weight : array-like of shape (n_samples,), default=None
            Weights of training samples in `X`.
        candidates : None or array-like of shape (n_candidates,), dtype=int \
                or array-like of shape (n_candidates, n_features), default=None
            - If `candidates` is `None`, the unlabeled samples from `(X, y)`
              are considered as candidates.
            - If `candidates` is of shape `(n_candidates,)` and of type
              `int`, `candidates` is considered as the indices of the samples
              in `(X, y)`.
            - If `candidates` is of shape `(n_candidates, n_features)`, the
              candidate samples are directly given in `candidates` (not
              necessarily contained in `X`).
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

            - If `candidates` is `None` or of shape `(n_candidates,)`, the
              indexing refers to the samples in `X`.
            - If `candidates` is of shape `(n_candidates, n_features)`, the
              indexing refers to the samples in `candidates`.
        utilities : numpy.ndarray of shape (batch_size, n_samples) or \
                numpy.ndarray of shape (batch_size, n_candidates)
            The utilities of samples after each selected sample of the batch,
            e.g., `utilities[0]` indicates the utilities used for selecting
            the first sample (with index `query_indices[0]`) of the batch.
            Utilities for labeled samples or already selected candidates are
            set to `np.nan`.

            - If `candidates` is `None`, the indexing refers to the samples
              in `X`.
            - If `candidates` is of shape `(n_candidates,)` and of type
              `int`, `utilities` refers to the samples in `X`.
            - If `candidates` is of shape `(n_candidates, n_features)`,
              `utilities` refers to the indexing in `candidates`.
        """
        # Determine candidate samples and validate parameters.
        X, y, candidates, batch_size, return_utilities = self._validate_data(
            X, y, candidates, batch_size, return_utilities, reset=True
        )
        X_cand, mapping = self._transform_candidates(candidates, X, y)
        check_type(clf, "clf", SkactivemlClassifier)
        check_equal_missing_label(clf.missing_label, self.missing_label_)
        check_scalar(fit_clf, "fit_clf", bool)
        check_scalar(self.normalize_samples, "normalize_samples", bool)
        check_scalar(self.adaptive_sigma, "adaptive_sigma", bool)
        check_scalar(self.n_ece_bins, "n_ece_bins", int, min_val=1)
        check_type(
            self.predict_proba_dict, "predict_proba_dict", (dict, type(None))
        )
        check_type(
            self.predict_proba_parser,
            "predict_proba_parser",
            type(None),
            indicator_funcs=[callable],
        )
        check_type(self.metric_dict, "metric_dict", (dict, type(None)))
        metric_dict = (
            {} if self.metric_dict is None else self.metric_dict.copy()
        )
        if self.adaptive_sigma and self.metric != "rbf":
            raise ValueError(
                "`adaptive_sigma=True` is only supported with `metric='rbf'`."
            )
        if isinstance(self.validation_size, int):
            check_scalar(
                self.validation_size, "validation_size", int, min_val=1
            )
        else:
            check_scalar(
                self.validation_size,
                "validation_size",
                (float, np.floating),
                min_inclusive=False,
                max_inclusive=False,
                min_val=0.0,
                max_val=1.0,
            )
        if self.temperatures is None:
            temperatures = np.logspace(-1, 1, 49)
        elif np.isscalar(self.temperatures):
            temperatures = float(self.temperatures)
            if temperatures <= 0 or np.isnan(temperatures):
                raise ValueError(
                    "`temperatures` must contain only positive values."
                )
        else:
            temperatures = column_or_1d(self.temperatures, dtype=float)
            if len(temperatures) == 0:
                raise ValueError(
                    "`temperatures` must contain at least one entry."
                )
            if np.any(temperatures <= 0) or np.isnan(temperatures).any():
                raise ValueError(
                    "`temperatures` must contain only positive values."
                )

        # Calibrate classifier by selecting a corresponding temperature.
        tau = self._select_temperature(
            X=X,
            y=y,
            clf=clf,
            temperatures=temperatures,
            sample_weight=sample_weight,
        )

        # (Re-)fit classifier on full labeled data if requested.
        if fit_clf:
            if sample_weight is None:
                clf_eval = clone(clf).fit(X, y)
            else:
                clf_eval = clone(clf).fit(X, y, sample_weight)
        else:
            clf_eval = clf

        # Infere probabilities and if available logits as well as embeddings.
        probas_cand, logits_cand, X_cand_repr = self._predict_with_extras(
            clf_eval, X_cand
        )
        if X_cand_repr is None:
            X_cand_repr = X_cand
        if logits_cand is not None:
            probas_cand = softmax(logits_cand / tau, axis=1)

        # Compute uncertatiny scores by either using the original probability
        # scores or the calibrated ones, if logits were available.
        unc_cand = uncertainty_scores(probas=probas_cand, method=self.method)
        if not np.all(np.isfinite(unc_cand)) or np.allclose(unc_cand, 0.0):
            # Fall back to pure coverage if the uncertainty model carries no
            # information, e.g. when only one class has been observed so far.
            unc_cand = np.ones_like(unc_cand)

        # Get embeddings for the labeled samples.
        labeled_idx = labeled_indices(y=y, missing_label=self.missing_label_)
        X_labeled_repr = None
        if len(labeled_idx) > 0:
            _, _, X_labeled_repr = self._predict_with_extras(
                clf_eval, X[labeled_idx]
            )
            if X_labeled_repr is None:
                X_labeled_repr = X[labeled_idx]

        # Normalize candidate and labeled samples to unit length.
        if self.normalize_samples:
            X_cand_repr = normalize(X_cand_repr, copy=True)
            if X_labeled_repr is not None:
                X_labeled_repr = normalize(X_labeled_repr, copy=True)

        # Compute kernel similarities, where the bandwidth is automatically
        # tune if an RBF kernel is employed.
        metric_dict = self._resolve_metric_dict(
            X_cand_repr=X_cand_repr,
            X_labeled_repr=X_labeled_repr,
            metric_dict=metric_dict,
        )
        K_cand = pairwise_kernels(
            X_cand_repr, metric=self.metric, **metric_dict
        )
        if X_labeled_repr is not None and len(X_labeled_repr) > 0:
            K_cand_labeled = pairwise_kernels(
                X_cand_repr, X_labeled_repr, metric=self.metric, **metric_dict
            )
            k_max = K_cand_labeled.max(axis=1)
        else:
            k_max = np.zeros(len(X_cand_repr), dtype=float)

        # Perform sequential batch selection.
        query_indices_cand = np.empty(batch_size, dtype=int)
        utilities_cand = np.empty((batch_size, len(X_cand_repr)), dtype=float)
        for b in range(batch_size):
            gains = np.maximum(K_cand - k_max[:, None], 0.0)
            utilities_cand[b] = np.mean(unc_cand[:, None] * gains, axis=0)
            utilities_cand[b][query_indices_cand[:b]] = np.nan
            query_indices_cand[b] = rand_argmax(
                utilities_cand[b], random_state=self.random_state_
            )[0]
            k_max = np.maximum(k_max, K_cand[:, query_indices_cand[b]])

        # Map queried indices and utilities back to the expected output.
        if mapping is None:
            query_indices = query_indices_cand
            utilities = utilities_cand
        else:
            query_indices = mapping[query_indices_cand]
            utilities = np.full((batch_size, len(X)), np.nan)
            utilities[:, mapping] = utilities_cand

        if return_utilities:
            return query_indices, utilities
        return query_indices

    def _select_temperature(self, X, y, clf, temperatures, sample_weight=None):
        # Fallback if there is only one temperature candidate.
        if np.isscalar(temperatures):
            return float(temperatures)
        if len(temperatures) == 1:
            return float(temperatures[0])

        # Try to perform train-test split. If it not possilbe, return 1.0 as
        # temperature.
        labeled_idx = labeled_indices(y=y, missing_label=self.missing_label_)
        if len(labeled_idx) < 2:
            return 1.0
        y_labeled = y[labeled_idx]
        split_kwargs = {
            "test_size": self.validation_size,
            "random_state": self.random_state_,
            "shuffle": True,
        }
        if len(np.unique(y_labeled)) > 1:
            split_kwargs["stratify"] = y_labeled
        try:
            train_idx, val_idx = train_test_split(labeled_idx, **split_kwargs)
        except ValueError:
            split_kwargs.pop("stratify", None)
            try:
                train_idx, val_idx = train_test_split(
                    labeled_idx, **split_kwargs
                )
            except ValueError:
                return 1.0

        if len(train_idx) == 0 or len(val_idx) == 0:
            return 1.0
        X_train = X[train_idx]
        y_train = y[train_idx]
        X_val = X[val_idx]
        y_val = y[val_idx]
        sw_train = None if sample_weight is None else sample_weight[train_idx]
        try:
            if sw_train is None:
                clf_cal = clone(clf).fit(X_train, y_train)
            else:
                clf_cal = clone(clf).fit(X_train, y_train, sw_train)
        except Exception:
            return 1.0
        _, logits_val, _ = self._predict_with_extras(clf_cal, X_val)
        if logits_val is None:
            return 1.0

        # Select temperature by iterating over all candidates and selecting
        # the one with the lowest expected calibration error.
        best_tau = float(temperatures[0])
        best_ece = np.inf
        for tau in temperatures:
            probas = softmax(logits_val / tau, axis=1)
            ece = self._expected_calibration_error(
                probas=probas, y_true=y_val, classes=clf_cal.classes_
            )
            if ece < best_ece:
                best_tau = float(tau)
                best_ece = ece
        return best_tau

    def _resolve_metric_dict(self, X_cand_repr, X_labeled_repr, metric_dict):
        """
        Computes adaptive sigma if required.
        """
        # Keep the metric paramters unchanged if no adaptive sigma is requried.
        metric_dict = metric_dict.copy()
        if not self.adaptive_sigma:
            return metric_dict

        if X_labeled_repr is not None:
            # If there are labeled samples compute minimum distance as sigma.
            distances = self._nonzero_distances(X_labeled_repr)
            sigma = np.min(distances)
        else:
            # If there are labeled samples compute median distance between
            # candidate samples as sigma.
            distances = self._nonzero_distances(X_cand_repr)
            sigma = np.median(distances)

        if sigma is None or sigma <= 0 or np.isnan(sigma):
            # Fallback if no valid sigma could be computed.
            sigma = 1.0

        # Transform sigma to the gamma parameter expected by the RBF kernel
        # implementation in sklearn.
        metric_dict["gamma"] = 1.0 / (sigma**2)
        return metric_dict

    def _predict_with_extras(self, clf, X):
        """
        Helper function to streamline required predictions.
        """
        predict_proba_dict = (
            {}
            if self.predict_proba_dict is None
            else self.predict_proba_dict.copy()
        )
        try:
            out = clf.predict_proba(X, **predict_proba_dict)
        except TypeError:
            if predict_proba_dict:
                out = clf.predict_proba(X)
            else:
                raise
        probas, logits, emb = self._parse_predict_output(out)

        if logits is None:
            logits = self._decision_function_logits(clf, X)
        if probas is None and logits is not None:
            probas = softmax(logits, axis=1)

        return probas, logits, emb

    def _parse_predict_output(self, out):
        """
        Helper function to streamline required predictions according to
        user information.
        """
        if self.predict_proba_parser is not None:
            parsed = self.predict_proba_parser(out)
            if not isinstance(parsed, (tuple, list)):
                raise TypeError(
                    "`predict_proba_parser` must return a tuple or list."
                )
            if len(parsed) == 2:
                probas, logits = parsed
                emb = None
            elif len(parsed) == 3:
                probas, logits, emb = parsed
            else:
                raise ValueError(
                    "`predict_proba_parser` must return "
                    "`(probas, logits)` or `(probas, logits, embeddings)`."
                )
            return probas, logits, emb

        if not isinstance(out, tuple):
            return out, None, None

        if len(out) == 0:
            raise ValueError("`predict_proba` returned an empty tuple.")
        if len(out) > 3:
            raise ValueError(
                "`predict_proba` returned more than three outputs. Pass "
                "`predict_proba_parser` to disambiguate them."
            )

        probas = out[0]
        logits = out[1] if len(out) >= 2 else None
        emb = out[2] if len(out) >= 3 else None
        return probas, logits, emb

    def _expected_calibration_error(self, probas, y_true, classes):
        """
        Computes expected calibration error for determining the temperature.
        """
        confidences = np.max(probas, axis=1)
        pred_labels = classes[np.argmax(probas, axis=1)]
        accuracies = pred_labels == y_true
        bins = np.linspace(0.0, 1.0, self.n_ece_bins + 1)
        ece = 0.0
        for left, right in zip(bins[:-1], bins[1:]):
            if right == 1.0:
                mask = (confidences >= left) & (confidences <= right)
            else:
                mask = (confidences >= left) & (confidences < right)
            if not np.any(mask):
                continue
            bin_weight = np.mean(mask)
            bin_acc = np.mean(accuracies[mask])
            bin_conf = np.mean(confidences[mask])
            ece += bin_weight * np.abs(bin_acc - bin_conf)
        return ece

    @staticmethod
    def _decision_function_logits(clf, X):
        """
        Helper function to compute logits from the decision function as a
        common method in sklearn.
        """
        if not hasattr(clf, "decision_function"):
            return None
        try:
            logits = clf.decision_function(X)
        except Exception:
            return None
        logits = np.asarray(logits)
        if logits.ndim == 1:
            logits = np.column_stack([np.zeros_like(logits), logits])
        return logits

    @staticmethod
    def _nonzero_distances(X):
        """
        Helper function for computing non-zero distances.
        """
        if X is None or len(X) < 2:
            return None
        distances = pairwise_distances(X)
        distances = distances[np.triu_indices_from(distances, k=1)]
        distances = distances[distances > 0]
        return distances
