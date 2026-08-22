"""
Classifier based on a Gaussian Mixture Model.
"""

# Author: Marek Herde <marek.herde@uni-kassel.de>

from copy import deepcopy

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.utils.validation import (
    check_array,
    check_is_fitted,
    NotFittedError,
)

from ..base import ClassFrequencyEstimator
from ..utils import MISSING_LABEL, check_n_features


class MixtureModelClassifier(ClassFrequencyEstimator):
    """Classifier based on a Mixture Model (CMM)

    The classifier based on a mixture model (CMM) is a generative classifier
    based on a (Bayesian) Gaussian mixture model (GMM).

    Parameters
    ----------
    mixture_model : sklearn.mixture.GaussianMixture or\
    sklearn.mixture.BayesianGaussianMixture or None, default=None
        (Bayesian) Gaussian Mixture model that is trained with unsupervised
        algorithm on train data. If the initial mixture model is not fitted, it
        will be refitted in each call of the `fit` method. If `None`,
        `mixture_model=BayesianGaussianMixture(n_components=n_classes)` will
        be used. Multi-label classification requires an explicit mixture
        model, because the number of label outputs does not define the number
        of mixture components.
    weight_mode : 'responsibilities' or 'similarities',\
            default='responsibilities'
        Determines whether the responsibilities outputted by the
        `mixture_model` or the exponential of the Mahalanobis distances as
        similarities are used to compute the class frequency estimates.
    classes : array-like of shape (n_classes,) or a list of array-like of \
            shape (2,), default=None
        Holds the label for each class. Nested binary vocabularies describe
        one vocabulary per output for multi-label classification. If `None`,
        the classes are determined during the fit.
    missing_label : scalar or str or np.nan or None, default=np.nan
        Value to represent a missing label.
    cost_matrix : array-like, shape (n_classes, n_classes)
        Cost matrix with `cost_matrix[i,j]` indicating cost of predicting class
        `classes[j]`  for a sample of class `classes[i]`. Can be only set, if
        `classes` is not `None`.
    class_prior : float or array-like of shape (n_classes,) or \
            (n_outputs, 2), default=0
        Prior observations of the class frequency estimates. If `class_prior`
        is an array for single-output classification, `class_prior[i]`
        indicates the non-negative prior number of samples belonging to class
        `classes_[i]`. For multi-label classification, an array contains one
        binary prior per output. If `class_prior` is a float, it indicates the
        non-negative prior number of samples per class for every output.
    random_state : int or RandomState instance or None, default=None
        Determines random number for `predict` method. Pass an int for
        reproducible results across multiple method calls.
    target_type : "auto" or "single-output" or "multi-label", default="auto"
        Declared target type. This estimator supports single-output and
        multi-label classification.

    Attributes
    ----------
    classes_ : np.ndarray of shape (n_classes,) or list of np.ndarray
        Holds the label for each class after fitting.
    class_prior_ : np.ndarray of shape (n_classes,) or (n_outputs, 2)
        Prior observations of the class frequency estimates, ordered like
        `classes_` for single-output targets and like each output's canonical
        binary vocabulary for multi-label targets.
    cost_matrix_ : np.ndarray, shape (classes, classes)
        Cost matrix with `cost_matrix_[i,j]` indicating cost of predicting
        class `classes_[j]` for a sample of class `classes_[i]`.
    F_components_ : numpy.ndarray of shape (n_components, n_classes) or \
            (n_components, n_outputs, 2)
        For single-output targets, `F_components_[j, c]` is a proxy for the
        number of samples of class `c` belonging to component `j`. For
        multi-label targets, the final two axes identify the output and its
        canonical binary class.
    mixture_model_ : sklearn.mixture.GaussianMixture or\
            sklearn.mixture.BayesianGaussianMixture
        (Bayesian) Gaussian Mixture model that is trained with unsupervised
        algorithm on train data.
    """

    @property
    def _target_capabilities(self):
        return super()._target_capabilities | frozenset(
            {("classification", "multi-label", "single-annotator")}
        )

    def __init__(
        self,
        mixture_model=None,
        weight_mode="responsibilities",
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
        self.mixture_model = mixture_model
        self.weight_mode = weight_mode

    def fit(self, X, y, sample_weight=None):
        """Fit the model using `X` as samples and `y` as class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The feature matrix representing the samples.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            It contains the class labels of the training samples.
        sample_weight : array-like of shape (n_samples,) or \
                (n_samples, n_outputs)
            It contains the weights of the training samples' class labels.
            One weight per sample or one weight per target entry can be
            provided.

        Returns
        -------
        self: skactiveml.classifier.MixtureModelClassifier
            The `MixtureModelClassifier` fitted on the training data.
        """
        target_spec = self._resolve_target_spec(y)
        if (
            target_spec.target_type == "multi-label"
            and self.mixture_model is None
        ):
            raise ValueError(
                "`mixture_model` must be provided for multi-label "
                "classification."
            )

        # Check input parameters.
        X, y, sample_weight = self._validate_data(
            X, y, sample_weight, target_spec=target_spec
        )

        # Check mixture model.
        if self.mixture_model is None:
            bgm = BayesianGaussianMixture(
                n_components=len(self.classes_),
                random_state=self.random_state_,
            )
            self.mixture_model_ = bgm
        else:
            if not isinstance(
                self.mixture_model, (GaussianMixture, BayesianGaussianMixture)
            ):
                raise TypeError(
                    f"`mixture_model` is of the type `{self.mixture_model}` "
                    f"but must be of the type "
                    f"`sklearn.mixture.GaussianMixture` or "
                    f"'sklearn.mixture.BayesianGaussianMixture'."
                )
            self.mixture_model_ = deepcopy(self.mixture_model)

        # Check weight mode.
        if self.weight_mode not in ["responsibilities", "similarities"]:
            raise ValueError(
                f"`weight_mode` must be either 'responsibilities' or "
                f"'similarities', got {self.weight_mode} instead."
            )

        if self.n_features_in_ is None:
            self.F_components_ = 0
        else:
            # Refit model if desired.
            try:
                check_is_fitted(self.mixture_model_)
            except NotFittedError:
                self.mixture_model_ = self.mixture_model_.fit(X)

            # Counts number of votes per class label for each sample.
            V = self._compute_class_frequency_vectors(y, sample_weight)

            # Stores responsibility for every given sample of training set.
            R = self.mixture_model_.predict_proba(X)

            # Stores class frequency estimates per component.
            if target_spec.target_type == "multi-label":
                self.F_components_ = np.einsum("nk,noc->koc", R, V)
            else:
                self.F_components_ = R.T @ V

        return self

    def predict_freq(self, X):
        """Return class frequency estimates for the input data `X`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        F : np.ndarray of shape (n_samples, n_classes) or \
                (n_samples, n_outputs, 2)
            The class frequency estimates of the input samples. Classes are
            ordered according to the attribute `classes_`.
        """
        check_is_fitted(self)
        X = check_array(X)
        check_n_features(self, X, reset=False)
        if np.sum(self.F_components_) > 0:
            if self.weight_mode == "similarities":
                S = np.exp(
                    -np.array(
                        [
                            cdist(
                                X,
                                [self.mixture_model_.means_[j]],
                                metric="mahalanobis",
                                VI=self.mixture_model_.precisions_[j],
                            ).ravel()
                            for j in range(self.mixture_model_.n_components)
                        ]
                    )
                ).T
            else:
                S = self.mixture_model_.predict_proba(X)
            if self.target_spec_.target_type == "multi-label":
                F = np.einsum("nk,koc->noc", S, self.F_components_)
            else:
                F = S @ self.F_components_
        else:
            output_shape = (
                (len(X), len(self.classes_), 2)
                if self.target_spec_.target_type == "multi-label"
                else (len(X), len(self.classes_))
            )
            F = np.zeros(output_shape)
        return F
