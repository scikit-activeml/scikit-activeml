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
        nu_0=2.0,
        sigma_sq_0=1.0,
        random_state=None,
    ):
        super().__init__(random_state=random_state)
        self.mu_0 = mu_0
        self.kappa_0 = kappa_0
        self.nu_0 = nu_0
        self.sigma_sq_0 = sigma_sq_0
        self.metric = metric
        self.metric_dict = {} if metric_dict is None else metric_dict

    def fit(self, X, y, sample_weight=None):
        X, y, sample_weight = self._validate_data(X, y, sample_weight)
        is_lbld = is_labeled(y)
        X_prior = np.zeros((1, len(X.T)))
        self.X_ = np.append(X_prior, X[is_lbld], axis=0)
        self.y_ = np.append([self.mu_0], y[is_lbld], axis=0)
        return self

    def estimate_conditional_distribution(self, X):
        check_array(X)

        K = pairwise_kernels(X, self.X_, metric=self.metric, **self.metric_dict)
        K[:, 0] = self.kappa_0
        N = np.sum(K[:, 1:], axis=1)
        kappa_L = self.kappa_0 + N
        nu_L = self.nu_0 + N
        mu_L = 1 / kappa_L * (K @ self.y_)
        sigma_sq_L = (
            1
            / nu_L
            * (
                np.sum((self.y_[np.newaxis, :] - mu_L[:, np.newaxis]) ** 2 * K, axis=1)
                + self.nu_0 * self.sigma_sq_0
            )
        )
        df = nu_L
        loc = mu_L
        scale = np.sqrt((1 + kappa_L) / kappa_L * sigma_sq_L)
        return t(df=df, loc=loc, scale=scale)

    def _estimate_likelihood_mu_var_N(self, X):
        K = pairwise_kernels(X, self.X_, metric=self.metric, **self.metric_dict)

        # maximum likelihood
        N = np.sum(K, axis=1)
        mu_ml = K @ self.y_ / N
        var_ml = np.sqrt(np.abs((K @ (self.y_**2) / N) - mu_ml**2))

        return mu_ml, var_ml, N

    def _estimate_conditional_distribution(self, X):
        mu_ml, var_ml, N = self._estimate_likelihood_mu_var_N(X)

        # normal inv chi squared
        mu_0 = self.mu_0
        kappa_0 = self.kappa_0
        nu_0 = self.nu_0
        sigma_sq_0 = self.sigma_sq_0
        mu_N = (kappa_0 * mu_0 + N * mu_ml) / (kappa_0 + N)
        kappa_N = kappa_0 + N
        nu_N = nu_0 + N
        sigma_sq_N = (
            nu_0 * sigma_sq_0
            + var_ml
            + (kappa_0 * N * (mu_ml - mu_0) ** 2) / (kappa_0 + N)
        ) / nu_N
        # posterior to marginal
        df = nu_N
        loc = mu_N
        scale = np.sqrt(((kappa_N + 1) / kappa_N) * sigma_sq_N)
        return t(df=df, loc=loc, scale=scale)
