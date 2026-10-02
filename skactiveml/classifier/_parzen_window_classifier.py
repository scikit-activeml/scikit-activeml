"""
Parzen Window Classifier
"""

# Author: Marek Herde <marek.herde@uni-kassel.de>

import numpy as np
import warnings
from sklearn.metrics.pairwise import pairwise_kernels, KERNEL_PARAMS
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted, check_scalar

from ..base import ClassFrequencyEstimator
from ..utils import (
    MISSING_LABEL,
    is_labeled,
    check_n_features,
)

from copy import deepcopy


class ParzenWindowClassifier(ClassFrequencyEstimator):
    """Parzen Window Classifier (PWC)

    The "Parzen Window Classifier" (PWC) [1]_ is a simple and
    probabilistic classifier. This classifier is based on a non-parametric
    density estimation obtained by applying a kernel function.

    Parameters
    ----------
    classes : array-like of shape (n_classes,) or a list of array-like of \
            shape (2,), default=None
        Holds the label for each class. Nested binary vocabularies describe
        one vocabulary per output for multi-label classification. If `None`,
        the classes are determined during the fit.
    missing_label : scalar or string or np.nan or None, default=np.nan
        Value to represent a missing label.
    cost_matrix : array-like of shape (n_classes, n_classes), default=None
        Cost matrix with `cost_matrix[i,j]` indicating cost of predicting class
        `classes[j]` for a sample of class `classes[i]`. Can be only set, if
        `classes` is not `None`.
    class_prior : float or array-like of shape (n_classes,) or \
            (n_outputs, 2), default=0
        Prior observations of the class frequency estimates. If `class_prior`
        is an array for single-output classification, `class_prior[i]`
        indicates the non-negative prior number of samples belonging to class
        `classes_[i]`. For multi-label classification, an array contains one
        binary prior per output. If `class_prior` is a float, it indicates the
        non-negative prior number of samples per class for every output.
    metric : str or callable, default='rbf'
        The metric must be a valid kernel defined by the function
        `sklearn.metrics.pairwise.pairwise_kernels`.
    n_neighbors : int or None, default=None
        Number of nearest neighbours. Default is `None`, which means all
        available samples are considered.
    metric_dict : dict, default=None
        Any further parameters are passed directly to the kernel function.
        For the kernel 'rbf' we allow the use of mean bandwidth criterion [2]_
        and use it when gamma is set to 'mean' (i.e., {'gamma': 'mean'})..
    random_state : int or RandomState instance or None, default=None
        Determines random number for `predict` method. Pass an int for
        reproducible results across multiple method calls.
    target_type : "auto" or "single-output" or "multi-label", default="auto"
        Declared target type. This estimator supports single-output and
        multi-label classification.

    Attributes
    ----------
    classes_ : numpy.ndarray of shape (n_classes,) or list of numpy.ndarray
        Holds the label for each class after fitting.
    class_prior_ : np.ndarray of shape (n_classes,) or (n_outputs, 2)
        Prior observations of the class frequency estimates, ordered like
        `classes_` for single-output targets and like each output's canonical
        binary vocabulary for multi-label targets.
    cost_matrix_ : np.ndarray of shape (classes, classes)
        Cost matrix with `cost_matrix_[i,j]` indicating cost of predicting
        class `classes_[j]` for a sample of class `classes_[i]`.
    X_ : np.ndarray of shape (n_samples, n_features)
        The sample matrix `X` is the feature matrix representing the samples.
    V_ : np.ndarray of shape (n_samples, n_classes) or \
            (n_samples, n_outputs, 2)
        The class labels are represented by counting vectors. For multi-label
        targets, `V_[i, j, c]` contains the weighted count for class `c` of
        output `j` at training sample `X_[i]`.

    References
    ----------
    .. [1] O. Chapelle, "Active Learning for Parzen Window Classifier",
       Proceedings of the Tenth International Workshop Artificial Intelligence
       and Statistics, 2005.
    .. [2] Chaudhuri, A., Kakde, D., Sadek, C., Gonzalez, L., & Kong, S.,
       "The Mean and Median Criteria for Kernel Bandwidth Selection for Support
       Vector Data Description" IEEE International Conference on Data
       Mining Workshops (ICDMW), 2017.
    """

    METRICS = list(KERNEL_PARAMS.keys()) + ["precomputed"]

    @property
    def _target_capabilities(self):
        return super()._target_capabilities | frozenset(
            {("classification", "multi-label", "single-annotator")}
        )

    def __init__(
        self,
        n_neighbors=None,
        metric="rbf",
        metric_dict=None,
        classes=None,
        missing_label=MISSING_LABEL,
        cost_matrix=None,
        class_prior=0.0,
        random_state=None,
        target_type="auto",
    ):
        super().__init__(
            classes=classes,
            class_prior=class_prior,
            missing_label=missing_label,
            cost_matrix=cost_matrix,
            random_state=random_state,
            target_type=target_type,
        )
        self.metric = metric
        self.n_neighbors = n_neighbors
        self.metric_dict = metric_dict

    def fit(self, X, y, sample_weight=None):
        """Fit the model using `X` as samples and `y` as class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The feature matrix representing the samples.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            It contains the class labels of the training samples.
        sample_weight : array-like of shape (n_samples,) or \
                (n_samples, n_outputs), default=None
            It contains the weights of the training samples' class labels.
            One weight per sample or one weight per target entry can be
            provided.

        Returns
        -------
        self : ParzenWindowClassifier,
            The `ParzenWindowClassifier` is fitted on the training data.
        """
        # Resolve semantics and reject unsupported targets before fitted state
        # is changed.
        target_spec = self._resolve_target_spec(y)

        # Check input parameters.
        X, y, sample_weight = self._validate_data(
            X, y, sample_weight, target_spec=target_spec
        )

        # Check whether metric is available.
        if self.metric not in ParzenWindowClassifier.METRICS and not callable(
            self.metric
        ):
            raise ValueError(
                "The parameter 'metric' must be callable or "
                "in {}".format(KERNEL_PARAMS.keys())
            )

        # Check number of neighbors which must be a positive integer.
        if self.n_neighbors is not None:
            check_scalar(
                self.n_neighbors,
                name="n_neighbors",
                min_val=1,
                target_type=int,
            )

        # Ensure that metric_dict is a Python dictionary.
        self.metric_dict_ = (
            deepcopy(self.metric_dict) if self.metric_dict is not None else {}
        )
        if (
            "gamma" in self.metric_dict_
            and self.metric_dict["gamma"] == "mean"
            and self.metric == "rbf"
        ):
            is_lbld = is_labeled(
                y,
                missing_label=-1,
                target_type=target_spec.target_type,
            )
            N = np.max([2, np.sum(is_lbld)])
            variance = np.var(X, axis=0)
            n_features = X.shape[1]
            gamma = ParzenWindowClassifier._calculate_mean_gamma(
                N, variance, n_features
            )
            self.metric_dict_["gamma"] = gamma
        if not isinstance(self.metric_dict_, dict):
            raise TypeError("'metric_dict' must be a Python dictionary.")

        # Store train samples.
        self.X_ = X.copy()

        # Convert labels to count vectors.
        if self.n_features_in_ is None:
            self.V_ = 0
        else:
            self.V_ = self._compute_class_frequency_vectors(y, sample_weight)

        self.target_spec_ = target_spec

        return self

    def predict_freq(self, X):
        """Return class frequency estimates for the input samples `X`.

        Parameters
        ----------
        X : array-like or shape (n_samples, n_features) or shape \
                (n_samples, m_samples) if metric == 'precomputed'
            Input samples.

        Returns
        -------
        F : np.ndarray of shape (n_samples, n_classes) or \
                (n_samples, n_outputs, 2)
            The class frequency estimates of the input samples. Classes are
            ordered according to the attribute `classes_`.
        """
        check_is_fitted(self)
        X = check_array(X, ensure_all_finite=(self.metric != "precomputed"))

        # Predict zeros because of missing training data.
        if self.n_features_in_ is None:
            if self.target_spec_.target_type == "multi-label":
                return np.zeros((len(X), len(self.classes_), 2))
            return np.zeros((len(X), len(self.classes_)))

        # Compute kernel (metric) matrix.
        if self.metric == "precomputed":
            K = X
            if np.size(K, 0) != np.size(X, 0) or np.size(K, 1) != np.size(
                self.X_, 0
            ):
                raise ValueError(
                    "The kernel matrix 'X' must have the shape "
                    "(n_test_samples, n_train_samples)."
                )
        else:
            check_n_features(self, X, reset=False)
            K = pairwise_kernels(
                X, self.X_, metric=self.metric, **self.metric_dict_
            )

        # computing class frequency estimates
        if self.n_neighbors is None or np.size(self.X_, 0) <= self.n_neighbors:
            if self.target_spec_.target_type == "multi-label":
                F = np.einsum("nm,moc->noc", K, self.V_)
            else:
                F = K @ self.V_
        else:
            indices = np.argpartition(K, -self.n_neighbors, axis=1)
            indices = indices[:, -self.n_neighbors :]
            output_shape = (
                (np.size(X, 0), len(self.classes_), 2)
                if self.target_spec_.target_type == "multi-label"
                else (np.size(X, 0), len(self.classes_))
            )
            F = np.empty(output_shape)
            for i in range(np.size(X, 0)):
                if self.target_spec_.target_type == "multi-label":
                    F[i] = np.einsum(
                        "m,moc->oc",
                        K[i, indices[i]],
                        self.V_[indices[i]],
                    )
                else:
                    F[i, :] = K[i, indices[i]] @ self.V_[indices[i], :]
        return F

    @staticmethod
    def _calculate_mean_gamma(
        N, variance, n_features, delta=(np.sqrt(2) * 1e-6)
    ):
        denominator = 2 * N * np.sum(variance)
        numerator = (N - 1) * np.log((N - 1) / delta**2)
        if denominator <= 0:
            gamma = 1 / n_features
            warnings.warn(
                "The variance of the provided data is 0. Bandwidth of "
                + f"1/n_features={gamma} is used instead."
            )
        else:
            gamma = 0.5 * numerator / denominator
        return gamma
