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
    compute_vote_vectors,
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
    classes : array-like of shape (n_classes), default=None
        Holds the label for each class. If `None`, the classes are determined
        during the fit.
    missing_label : scalar or string or np.nan or None, default=np.nan
        Value to represent a missing label.
    cost_matrix : array-like of shape (n_classes, n_classes), default=None
        Cost matrix with `cost_matrix[i,j]` indicating cost of predicting class
        `classes[j]` for a sample of class `classes[i]`. Can be only set, if
        `classes` is not `None`.
    class_prior : float or array-like of shape (n_classes,), default=0
        Prior observations of the class frequency estimates. If `class_prior`
        is an array, the entry `class_prior[i]` indicates the non-negative
        prior number of samples belonging to class `classes_[i]`. If
        `class_prior` is a float, `class_prior` indicates the non-negative
        prior number of samples per class.
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

    Attributes
    ----------
    classes_ : numpy.ndarray of shape (n_classes,)
        Holds the label for each class after fitting.
    class_prior : np.ndarray of shape (n_classes,)
        Prior observations of the class frequency estimates. The entry
        `class_prior_[i]` indicates the non-negative prior number of samples
        belonging to class `classes_[i]`.
    cost_matrix_ : np.ndarray of shape (classes, classes)
        Cost matrix with `cost_matrix_[i,j]` indicating cost of predicting
        class `classes_[j]` for a sample of class `classes_[i]`.
    X_ : np.ndarray of shape (n_samples, n_features)
        The sample matrix `X` is the feature matrix representing the samples.
    V_ : np.ndarray of shape (n_samples, classes)
        The class labels are represented by counting vectors. An entry `V[i,j]`
        indicates how many class labels of `classes[j]` were provided for
        training sample `X_[i]`.

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
    ):
        super().__init__(
            classes=classes,
            class_prior=class_prior,
            missing_label=missing_label,
            cost_matrix=cost_matrix,
            random_state=random_state,
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
        y : array-like of shape (n_samples,)
            It contains the class labels of the training samples.
        sample_weight : array-like of shape (n_samples), default=None
            It contains the weights of the training samples' class labels.
            It must have the same shape as `y`.

        Returns
        -------
        self : ParzenWindowClassifier,
            The `ParzenWindowClassifier` is fitted on the training data.
        """
        # Check input parameters.
        X, y, sample_weight = self._validate_data(X, y, sample_weight)

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
            is_lbld = is_labeled(y, missing_label=1)
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
            self.V_ = compute_vote_vectors(
                y=y,
                w=sample_weight,
                classes=np.arange(len(self.classes_)),
                missing_label=-1,
            )

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
        F : np.ndarray of shape (n_samples, classes)
            The class frequency estimates of the input samples. Classes are
            ordered according to the attribute `classes_`.
        """
        check_is_fitted(self)
        X = check_array(X, ensure_all_finite=(self.metric != "precomputed"))

        # Predict zeros because of missing training data.
        if self.n_features_in_ is None:
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
            F = K @ self.V_
        else:
            indices = np.argpartition(K, -self.n_neighbors, axis=1)
            indices = indices[:, -self.n_neighbors :]
            F = np.empty((np.size(X, 0), len(self.classes_)))
            for i in range(np.size(X, 0)):
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
