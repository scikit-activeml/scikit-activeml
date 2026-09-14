import numpy as np
from scipy.stats import norm, t
from sklearn.metrics.pairwise import pairwise_kernels, KERNEL_PARAMS
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from skactiveml.base import ProbabilisticRegressor
from skactiveml.utils import (
    MISSING_LABEL,
    check_scalar,
    check_type,
    check_n_features,
)
from skactiveml.utils._label import _observed_numerical_labels


class NICKernelRegressor(ProbabilisticRegressor):
    """NIC Kernel Regressor

    The NICKernelRegressor (Normal Inverse Chi-square Kernel Regressor)
    locally fits a t-distribution using the training data, weighting the
    samples by a kernel.

    Parameters
    ----------
    metric : str or callable, default='rbf'
        The metric must a be a valid kernel defined by the function
        `sklearn.metrics.pairwise.pairwise_kernels`, or `'precomputed'`.
        Its values weight the training targets, so it must not be negative
        for the given data.
        `'rbf'`, `'laplacian'` and `'chi2'` always satisfy this;
        `'linear'`, `'poly'`, `'polynomial'`, `'sigmoid'` and `'cosine'` do
        so only for some data, and `'additive_chi2'` never does because it
        is non-positive by construction. Prediction raises a `ValueError`
        when the resulting kernel evidence is negative.
    metric_dict : dict, default=None
        Any further parameters are passed directly to the kernel function.
    mu_0 : int or float, default=0
        The prior mean.
    kappa_0 : int or float, default=0.1
        The weight of the prior mean.
    sigma_sq_0: int or float, default=1.0
        The prior variance.
    nu_0 : int or float, default=2.5
        The weight of the prior variance.
    missing_label : scalar or string or np.nan or None, default=np.nan
        Value to represent a missing label.
    random_state : int, RandomState instance or None, default=None
        Determines random number for `predict` method. Pass an int for
        reproducible results across multiple method calls.
    target_type : "auto" or "single-output", default="auto"
        Declared target type. This estimator supports only single-output
        regression.
    """

    METRICS = list(KERNEL_PARAMS.keys()) + ["precomputed"]

    def __init__(
        self,
        metric="rbf",
        metric_dict=None,
        mu_0=0,
        kappa_0=0.1,
        sigma_sq_0=1.0,
        nu_0=2.5,
        missing_label=MISSING_LABEL,
        random_state=None,
        target_type="auto",
    ):
        super().__init__(
            random_state=random_state,
            missing_label=missing_label,
            target_type=target_type,
        )
        self.kappa_0 = kappa_0
        self.nu_0 = nu_0
        self.mu_0 = mu_0
        self.sigma_sq_0 = sigma_sq_0
        self.metric = metric
        self.metric_dict = metric_dict

    def fit(self, X, y, sample_weight=None):
        """Fit the model using `X` as training data and `y` as labels.

        Parameters
        ----------
        X : matrix-like of shape (n_samples, n_features) or \
                (n_samples, n_samples) if metric='precomputed'
            Training data set, usually complete, i.e., including the labeled
            and unlabeled samples. If `metric='precomputed'`, `X` contains the
            pairwise kernels between all training samples.
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Labels of the training data set (possibly including unlabeled ones
            indicated by `self.missing_label`).
        sample_weight : array-like of shape (n_samples,)
            It contains the weights of the training samples' values.

        Returns
        -------
        self: SkactivemlRegressor,
            The SkactivemlRegressor is fitted on the training data.
        """
        X, y, sample_weight = self._validate_data(X, y, sample_weight)
        if self.metric == "precomputed" and X.shape[0] != X.shape[1]:
            raise ValueError(
                "For metric='precomputed', the training kernel matrix `X` "
                "must have shape (n_train_samples, n_train_samples)."
            )
        is_lbld, y_observed = _observed_numerical_labels(
            y, self.missing_label_
        )
        for value, name in [
            (self.kappa_0, "self.kappa_0"),
            (self.nu_0, "self.nu_0"),
            (self.sigma_sq_0, "self.sigma_sq_0"),
        ]:
            check_scalar(value, name, (int, float), min_val=0)
        check_scalar(self.mu_0, "self.mu_0", (int, float))

        self._is_lbld = is_lbld
        self.X_ = X[is_lbld]
        self.y_ = y_observed

        self.prior_params_ = (
            self.kappa_0,
            self.nu_0,
            self.mu_0,
            self.sigma_sq_0,
        )

        if sample_weight is not None:
            self.weights_ = sample_weight[is_lbld]
            if np.sum(self.weights_) == 0:
                raise ValueError(
                    "The sample weights of the labeled samples "
                    "must not be all zero."
                )
        else:
            self.weights_ = None

        check_type(self.metric, "self.metric", target_vals=self.METRICS)
        self.metric_dict = {} if self.metric_dict is None else self.metric_dict
        check_type(
            self.metric_dict, "self.metric_dict", dict, target_vals=[None]
        )

        return self

    def _validate_prediction_data(self, X):
        """Validate feature data or a precomputed test kernel.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or \
                (n_samples, n_train_samples) if metric='precomputed'
            Input samples or kernels against all samples from the latest fit.

        Returns
        -------
        X : numpy.ndarray
            Validated input data.
        """
        X = check_array(X)
        if self.metric == "precomputed":
            if X.shape[1] != self.n_features_in_:
                raise ValueError(
                    "For metric='precomputed', the kernel matrix `X` must "
                    "have shape (n_test_samples, n_train_samples)."
                )
        else:
            check_n_features(self, X, reset=False)
        return X

    def _estimate_ml_params(self, X):
        if self.metric == "precomputed":
            K = X[:, self._is_lbld]
        else:
            K = pairwise_kernels(
                X, self.X_, metric=self.metric, **self.metric_dict
            )

        if self.weights_ is not None:
            K = self.weights_.reshape(1, -1) * K

        N = np.sum(K, axis=1)
        # Zero kernel mass contributes a neutral update to the prior.
        mu_ml = np.divide(K @ self.y_, N, out=np.zeros_like(N), where=N != 0)
        scatter = np.sum(
            K * (self.y_[np.newaxis, :] - mu_ml[:, np.newaxis]) ** 2, axis=1
        )
        # `N` is a pseudo-count and `scatter` a weighted sum of squares, so a
        # kernel with negative values makes either negative. Both then reach
        # the posterior and yield a `NaN` scale rather than any error. The
        # scatter is checked separately because it can be negative while the
        # mass stays positive.
        self._check_kernel_evidence(N, "kernel mass")
        self._check_kernel_evidence(
            scatter, "weighted sum of squared deviations (scatter)"
        )
        var_ml = np.divide(scatter, N, out=np.zeros_like(N), where=N != 0)

        return N, mu_ml, var_ml

    def _check_kernel_evidence(self, values, name):
        """Reject kernel evidence that cannot be a weighted count.

        Parameters
        ----------
        values : numpy.ndarray of shape (n_samples,)
            Kernel-weighted quantity to validate.
        name : str
            Name of the quantity reported in the error message.

        Raises
        ------
        ValueError
            If any value is negative.
        """
        if np.any(values < 0):
            raise ValueError(
                f"The {name} weighting the training targets is negative, the "
                f"smallest being {np.min(values)}, so it cannot serve as a "
                "pseudo-count. This happens when `sample_weight` contains "
                f"negative values, or when `metric={self.metric!r}` produces "
                "negative similarities on the given data. Use non-negative "
                "sample weights and a non-negative kernel, e.g. 'rbf' or "
                "'laplacian', or supply a non-negative 'precomputed' matrix."
            )

    def _estimate_update_params(self, X):
        if len(self.X_) != 0:
            N, mu_ml, var_ml = self._estimate_ml_params(X)
            update_params = (N, N, mu_ml, var_ml)
            return update_params
        else:
            neutral_params = (np.zeros(len(X)),) * 4
            return neutral_params

    def predict_target_distribution(self, X):
        """Returns the estimated target distribution conditioned on the test
        samples `X`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or \
                (n_samples, n_train_samples) if metric='precomputed'
            Input samples. If `metric='precomputed'`, `X` contains kernels
            between the test samples and every sample passed to the latest
            `fit` call.

        Returns
        -------
        dist : scipy.stats._distn_infrastructure.rv_frozen
            The distribution of the targets at the test samples.
        """
        check_is_fitted(self)
        X = self._validate_prediction_data(X)

        prior_params = self.prior_params_
        update_params = self._estimate_update_params(X)
        # Check before combining, because the combination is where the
        # undefined mean would be divided into existence.
        self._check_posterior_evidence(prior_params[0] + update_params[0])
        post_params = _combine_params(prior_params, update_params)

        kappa_post, nu_post, mu_post, sigma_sq_post = post_params

        df = nu_post
        loc = mu_post
        scale = np.sqrt((1 + kappa_post) / kappa_post * sigma_sq_post)
        return t(df=df, loc=loc, scale=scale)

    def _check_posterior_evidence(self, kappa_post):
        """Reject test samples the posterior says nothing about.

        The posterior weight on the target mean is `kappa_0` plus the kernel
        mass over the labeled training samples. A positive `kappa_0` lets a
        test sample the kernel does not reach fall back to the prior mean.
        When both are zero the mean is undefined, and the divisions
        computing it would silently return `NaN` instead.

        Parameters
        ----------
        kappa_post : numpy.ndarray of shape (n_samples,)
            Posterior weight on the target mean, per test sample.

        Raises
        ------
        ValueError
            If any test sample carries no posterior weight.
        """
        without_evidence = kappa_post <= 0
        if np.any(without_evidence):
            raise ValueError(
                f"{np.sum(without_evidence)} of {len(kappa_post)} test "
                "samples carry no evidence for the target mean, so it is "
                "undefined for them. The kernel gives them zero mass and "
                f"`kappa_0={self.kappa_0}` puts no weight on the prior mean "
                "either. Widen the kernel, e.g. with a smaller 'gamma' in "
                "`metric_dict`, or use a positive `kappa_0` so that such "
                "samples fall back to the prior mean."
            )


def _combine_params(prior_params, update_params):
    kappa_1, nu_1, mu_1, sigma_sq_1 = prior_params
    kappa_2, nu_2, mu_2, sigma_sq_2 = update_params

    kappa_com = kappa_1 + kappa_2
    nu_com = nu_1 + nu_2
    mu_com = (kappa_1 * mu_1 + kappa_2 * mu_2) / kappa_com
    scatter_com = (
        nu_1 * sigma_sq_1
        + nu_2 * sigma_sq_2
        + kappa_1 * kappa_2 * (mu_1 - mu_2) ** 2 / kappa_com
    )
    sigma_sq_com = scatter_com / nu_com
    return kappa_com, nu_com, mu_com, sigma_sq_com


class NadarayaWatsonRegressor(NICKernelRegressor):
    """Nadaraya Watson Regressor

    The Nadaraya Watson Regressor predicts the target value by taking a
    weighted average based on a kernel. It is implemented as a
    `NICKernelRegressor` with different prior values.

    Parameters
    ----------
    metric : str or callable, default='rbf'
        The metric must a be a valid kernel defined by the function
        `sklearn.metrics.pairwise.pairwise_kernels`, or `'precomputed'`.
        Its values weight the training targets, so it must not be negative
        for the given data.
        `'rbf'`, `'laplacian'` and `'chi2'` always satisfy this;
        `'linear'`, `'poly'`, `'polynomial'`, `'sigmoid'` and `'cosine'` do
        so only for some data, and `'additive_chi2'` never does because it
        is non-positive by construction. Prediction raises a `ValueError`
        when the resulting kernel evidence is negative.
    metric_dict : dict, default=None
        Any further parameters are passed directly to the kernel function.
    missing_label : scalar or string or np.nan or or None, default=np.nan
        Value to represent a missing label.
    random_state : int or RandomState instance or None, default=None
        Determines random number for `predict` method. Pass an int for
        reproducible results across multiple method calls.
    target_type : "auto" or "single-output", default="auto"
        Declared target type. This estimator supports only single-output
        regression.

    Notes
    -----
    Without observed targets, the Nadaraya-Watson estimate is undefined. This
    estimator then returns a standard normal fallback distribution, with mean
    zero and standard deviation one, until labeled data are fitted.
    """

    def __init__(
        self,
        metric="rbf",
        metric_dict=None,
        missing_label=MISSING_LABEL,
        random_state=None,
        target_type="auto",
    ):
        super().__init__(
            random_state=random_state,
            missing_label=missing_label,
            target_type=target_type,
            metric=metric,
            metric_dict=metric_dict,
            kappa_0=0,
            nu_0=3,
            sigma_sq_0=1,
        )

    def predict_target_distribution(self, X):
        """Return the estimated or fallback target distribution."""
        check_is_fitted(self)
        if len(self.X_) != 0:
            return super().predict_target_distribution(X)

        X = self._validate_prediction_data(X)
        return norm(loc=np.zeros(len(X)), scale=np.ones(len(X)))
