import numpy as np
from scipy.stats import t
from sklearn.metrics.pairwise import pairwise_kernels, KERNEL_PARAMS
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from skactiveml.base import ProbabilisticRegressor
from skactiveml.utils import (
    is_labeled,
    MISSING_LABEL,
    check_scalar,
    check_type,
    check_n_features,
)


class NICKernelRegressor(ProbabilisticRegressor):
    """NIC Kernel Regressor

    The NICKernelRegressor (Normal Inverse Chi-square Kernel Regressor)
    locally fits a t-distribution using the training data, weighting the
    samples by a kernel.

    Parameters
    __________
    metric : str or callable, default='rbf'
        The metric must a be a valid kernel defined by the function
        `sklearn.metrics.pairwise.pairwise_kernels`.
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
    ):
        super().__init__(
            random_state=random_state, missing_label=missing_label
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
        X : matrix-like of shape (n_samples, n_features)
            Training data set, usually complete, i.e., including the labeled
            and unlabeled samples.
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
        X, y, sample_weight = self._validate_data(
            X, y, sample_weight, reset=self.metric != "precomputed"
        )
        is_lbld = is_labeled(y, missing_label=self.missing_label_)
        for value, name in [
            (self.kappa_0, "self.kappa_0"),
            (self.nu_0, "self.nu_0"),
            (self.sigma_sq_0, "self.sigma_sq_0"),
        ]:
            check_scalar(value, name, (int, float), min_val=0)
        check_scalar(self.mu_0, "self.mu_0", (int, float))

        self.X_ = X[is_lbld]
        self.y_ = y[is_lbld]

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

    def _estimate_ml_params(self, X):
        K = pairwise_kernels(
            X, self.X_, metric=self.metric, **self.metric_dict
        )

        if self.weights_ is not None:
            K = self.weights_.reshape(1, -1) * K

        N = np.sum(K, axis=1)
        mu_ml = K @ self.y_ / N
        scatter = np.sum(
            K * (self.y_[np.newaxis, :] - mu_ml[:, np.newaxis]) ** 2, axis=1
        )
        var_ml = 1 / N * scatter

        return N, mu_ml, var_ml

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
        X :  array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        dist : scipy.stats._distn_infrastructure.rv_frozen
            The distribution of the targets at the test samples.
        """
        check_is_fitted(self)
        X = check_array(X)
        check_n_features(self, X, reset=False)

        prior_params = self.prior_params_
        update_params = self._estimate_update_params(X)
        post_params = _combine_params(prior_params, update_params)

        kappa_post, nu_post, mu_post, sigma_sq_post = post_params

        df = nu_post
        loc = mu_post
        scale = np.sqrt((1 + kappa_post) / kappa_post * sigma_sq_post)
        return t(df=df, loc=loc, scale=scale)


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
    __________
    metric : str or callable, default='rbf'
        The metric must a be a valid kernel defined by the function
        `sklearn.metrics.pairwise.pairwise_kernels`.
    metric_dict : dict, default=None
        Any further parameters are passed directly to the kernel function.
    missing_label : scalar or string or np.nan or or None, default=np.nan
        Value to represent a missing label.
    random_state : int or RandomState instance or None, default=None
        Determines random number for `predict` method. Pass an int for
        reproducible results across multiple method calls.
    """

    def __init__(
        self,
        metric="rbf",
        metric_dict=None,
        missing_label=MISSING_LABEL,
        random_state=None,
    ):
        super().__init__(
            random_state=random_state,
            missing_label=missing_label,
            metric=metric,
            metric_dict=metric_dict,
            kappa_0=0,
            nu_0=3,
            sigma_sq_0=1,
        )
