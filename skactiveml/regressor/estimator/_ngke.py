import numpy as np
from scipy.stats import t
from sklearn.metrics.pairwise import pairwise_kernels, KERNEL_PARAMS

from ...base import SkactivemlContinuousEstimator


class NormalGammaKernelEstimator(SkactivemlContinuousEstimator):
    METRICS = list(KERNEL_PARAMS.keys()) + ['precomputed']

    def __init__(self, metric='rbf', metric_dict=None, mu_0=0,
                 lmbda_0=1.1, alpha_0=10.0, beta_0=1.0, random_state=None):
        super().__init__(random_state=random_state)
        self.mu_0 = mu_0
        self.lmbda_0 = lmbda_0
        self.alpha_0 = alpha_0
        self.beta_0 = beta_0
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
        var_ml = np.sqrt(np.abs((K @ (self.y_ ** 2) / N) - mu_ml ** 2))

        return mu_ml, var_ml, N

    def estimate_posterior_params_for_t(self, X):
        mu_ml, var_ml, N = self._estimate_likelihood_mu_var_N(X)

        # normal wishart
        mu_0 = self.mu_0
        lmbda_0 = self.lmbda_0
        alpha_0 = self.alpha_0
        beta_0 = self.beta_0
        mu_N = (lmbda_0 * mu_0 + N * mu_ml) / (lmbda_0 + N)
        lmbda_N = lmbda_0 + N
        # alpha and beta to variance
        alpha_N = alpha_0 + N / 2
        beta_N = beta_0 + 0.5 * N * var_ml \
                 + 0.5 * (lmbda_0 * N * (mu_ml - mu_0) ** 2) / (lmbda_0 + N)
        df = 2 * alpha_N
        loc = mu_N
        scale = (beta_N * (lmbda_N + 1)) / (alpha_N * lmbda_N)
        return {'df': df, 'loc': loc, 'scale': scale}

    def estimate_mu_cov(self, X):
        t_params = self.estimate_posterior_params_for_t(X)
        mu, var = t.stats(**t_params, moments='mv')
        return mu, var

    def estimate_random_variates(self, X, n_rvs):
        t_params = self.estimate_posterior_params_for_t(X)
        n_samples = len(X)
        rand_var = t.rvs(**t_params, size=(n_rvs, n_samples)).T
        return rand_var

    def predict(self, X):
        mu_ml, var_ml, N = self._estimate_likelihood_mu_var_N(X)
        mu = (self.lmbda_0 * self.mu_0 + N * mu_ml) / (self.lmbda_0 + N)
        return mu
