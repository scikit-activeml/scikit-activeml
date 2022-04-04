import numpy as np
from scipy.stats import t
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils import check_array

from ...base import SkactivemlConditionalEstimator
from ...utils import is_labeled


class NormalInverseChiKernelEstimator(SkactivemlConditionalEstimator):
    def __init__(
        self,
        metric="rbf",
        metric_dict=None,
        mu_0=0,
        kappa_0=0.1,
        nu_0=2.5,
        sigma_sq_0=1.0,
        random_state=None,
    ):
        super().__init__(random_state=random_state)
        self.kappa_0 = kappa_0
        self.nu_0 = nu_0
        self.mu_0 = mu_0
        self.sigma_sq_0 = sigma_sq_0
        self.metric = metric
        self.metric_dict = {} if metric_dict is None else metric_dict

        self.X_ = None
        self.y_ = None

    def fit(self, X, y, sample_weight=None):
        X, y, sample_weight = self._validate_data(X, y, sample_weight)
        is_lbld = is_labeled(y)
        self.X_ = X[is_lbld]
        self.y_ = y[is_lbld]
        if sample_weight is not None:
            weights_ = sample_weight[is_lbld]
            self.y_ = (weights_ * self.y_) / np.average(weights_)
        return self

    def _estimate_ml_params(self, X):
        K = pairwise_kernels(X, self.X_, metric=self.metric, **self.metric_dict)

        N = np.sum(K, axis=1)
        mu_ml = K @ self.y_ / N
        scatter = np.sum(
            K * (self.y_[np.newaxis, :] - mu_ml[:, np.newaxis]) ** 2, axis=1
        )
        var_ml = 1 / N * scatter

        return N, mu_ml, var_ml

    def _estimate_update_params(self, X):

        if self.X_ is not None and len(self.X_) != 0:
            N, mu_ml, var_ml = self._estimate_ml_params(X)
            update_params = (N, N, mu_ml, var_ml)
            return update_params
        else:
            neutral_params = (np.zeros(len(X)),) * 4
            return neutral_params

    def estimate_conditional_distribution(self, X):

        X = check_array(X)
        prior_params = (self.kappa_0, self.nu_0, self.mu_0, self.sigma_sq_0)
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
        + kappa_1 * kappa_com * (mu_1 - mu_2) ** 2 / (kappa_1 + kappa_2)
    )
    sigma_sq_com = scatter_com / nu_com
    return kappa_com, nu_com, mu_com, sigma_sq_com
