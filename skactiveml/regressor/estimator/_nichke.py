import numpy as np
from scipy.stats import t
from sklearn.metrics.pairwise import pairwise_kernels, KERNEL_PARAMS

from ...base import SkactivemlConditionalEstimator


class NormalInverseChiKernelEstimator(SkactivemlConditionalEstimator):
    METRICS = list(KERNEL_PARAMS.keys()) + ["precomputed"]

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
        self.X_ = X.copy()[~np.isnan(y)]
        self.y_ = y.copy()[~np.isnan(y)]

        return self

    def _estimate_likelihood_mu_var_N(self, X):
        K = pairwise_kernels(X, self.X_, metric=self.metric, **self.metric_dict)

        # maximum likelihood
        N = np.sum(K, axis=1)
        mu_ml = K @ self.y_ / N
        var_ml = np.sqrt(np.abs((K @ (self.y_**2) / N) - mu_ml**2))

        return mu_ml, var_ml, N

    def estimate_conditional_distribution(self, X):
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
        print(scale)
        return t(df=df, loc=loc, scale=scale)
