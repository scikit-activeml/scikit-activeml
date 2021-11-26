from copy import deepcopy

import numpy as np
from scipy.stats import t
from sklearn.metrics import pairwise_kernels

from skactiveml.base import SingleAnnotPoolBasedQueryStrategy
from skactiveml.regression._gsx import GSx
from skactiveml.utils import fit_if_not_fitted, rand_argmax


class RegxPal(SingleAnnotPoolBasedQueryStrategy):
    """Probablistic Active Learning Approach for Regression

    This class implements greedy sampling

    Parameters
    ----------
    random_state: numeric | np.random.RandomState, optional
        Random state for candidate selection.
    k_0: int, optional (default=1)
        The minimum number of samples the estimator requires.
    """

    def __init__(self, random_state=None, n_monte_carlo_samples=10,
                 metric='rbf'):
        super().__init__(random_state=random_state)
        self.metric = metric
        self.n_monte_carlo_samples = n_monte_carlo_samples
        self.metric_dict = {}

    def query(self, X_cand, reg, E, X, y, batch_size=1,
              assume_linear=False, return_utilities=False):

        """Query the next instance to be labeled.

        Parameters
        ----------
        X_cand: array-like, shape (n_candidates, n_features)
            Unlabeled candidate samples.
        reg: SkactivemlRegressor
            regressor to predict values of X_cand.
        E: array-like, shape (n_evaluator, n_features)
            Evaluation set.
        X: array-like, shape (n_samples, n_features)
            Complete training data set.
        y: array-like, shape (n_samples)
            Values of the training data set.
        batch_size: int, optional (default=1)
            The number of instances to be selected.
        assume_linear: bool, optional (default=2)
            Whether the model is assumed to be linear in the y-components of the
            given samples (`y`).
        return_utilities: bool, optional (default=False)
            If True, the utilities are additionally returned.

        Returns
        -------
        query_indices: np.ndarray, shape (batch_size)
            The index of the queried instance.
        utilities: np.ndarray, shape (batch_size, n_candidates)
            The utilities of all instances in X_cand
            (only returned if return_utilities is True).
        """
        n_candidates = len(X_cand)
        utilities = np.zeros(shape=n_candidates)
        t_params = self.estimate_posterior_params_for_t(X_cand, X, y)

        if assume_linear:
            return self._query_assume_lin(X_cand, reg, E, X, y, t_params,
                                          utilities, return_utilities)

        else:
            return self._query_not_assume_lin(X_cand, reg, E, X, y, t_params,
                                              utilities, return_utilities)

    def _query_assume_lin(self, X_cand, reg, E, X, y, t_params, utilities,
                          return_utilities):

        My_cand, Var_cand = t.stats(**t_params, moments='mv')

        for idx, (x_c, my_c, var_c) in enumerate(zip(X_cand, My_cand, Var_cand)):
            perf = self.x_perf_assume_linear(reg, E, x_c, my_c, var_c**(1/2), X, y)
            utilities[idx] = perf

        if return_utilities:
            return rand_argmax(utilities), utilities
        else:
            return rand_argmax(utilities)

    def _query_not_assume_lin(self, X_cand, reg, E, X, y, t_params,
                              utilities, return_utilities):

        n_candidates = len(X_cand)
        Y_cand = t.rvs(**t_params, size=(self.n_monte_carlo_samples,
                                         n_candidates)).T

        for idx_x_c, x_c in enumerate(X_cand):
            utilities_given_y_c = np.zeros(shape=self.n_monte_carlo_samples)
            for idx_y_c, y_c in enumerate(Y_cand[idx_x_c]):
                utilities_given_y_c[idx_y_c] = self.x_perf(reg, E, x_c, y_c, X, y)
            utilities[idx_x_c] = np.average(utilities_given_y_c)

        if return_utilities:
            return rand_argmax(utilities), utilities
        else:
            return rand_argmax(utilities)

    def estimate_posterior_params_for_t(self, X_eval, X, y):
        K = pairwise_kernels(X_eval, X, metric=self.metric,
                             **self.metric_dict)

        # maximum likelihood
        N = np.sum(K, axis=1)
        mu_ml = K @ y / N
        sigma_ml = np.sqrt((K @ y ** 2 / N) - mu_ml ** 2)

        # normal wishart
        mu_0 = 0
        lmbda_0 = 0.1
        alpha_0 = 2
        beta_0 = 0.1
        mu_N = (lmbda_0 * mu_0 + N * mu_ml) / (lmbda_0 + N)
        lmbda_N = lmbda_0 + N
        # alpha and beta to variance
        alpha_N = alpha_0 + N / 2
        beta_N = beta_0 + 0.5 * N * sigma_ml ** 2 \
                 + 0.5 * (lmbda_0 * N * (mu_ml - mu_0) ** 2) / (lmbda_0 + N)
        df = 2 * alpha_N
        loc = mu_N
        scale = (beta_N * (lmbda_N + 1)) / (alpha_N * lmbda_N)
        return {'df': df, 'loc': loc, 'scale': scale}

    def x_perf(self, reg, E, x_c, y_c, X, y):

        X_new = np.append(X, [x_c], axis=0)
        y_new = np.append(y, [y_c], axis=0)

        reg_old = reg
        reg_new = deepcopy(reg)
        reg_new.fit(X_new, y_new)

        t_params_new = self.estimate_posterior_params_for_t(E, X_new, y_new)
        my = t.stats(**t_params_new, moments='m')

        y_pred_old = reg_old.predict(E)
        y_pred_new = reg_new.predict(E)

        error_old = (y_pred_old - my) ** 2
        error_new = (y_pred_new - my) ** 2

        return np.average(error_old - error_new)

    def x_perf_assume_linear(self, reg, E, x_c, my_c, std_c, X, y):

        modes = [-std_c, 0, std_c]

        X_new = np.append(X, [x_c], axis=0)
        y_new_s = [np.append(y, [my_c + alpha], axis=0) for alpha in modes]

        reg_old = reg
        reg_new_s = [deepcopy(reg) for _ in modes]

        for reg_new, y_new in zip(reg_new_s, y_new_s):
            reg_new.fit(X_new, y_new)

        t_params_new_s = [self.estimate_posterior_params_for_t(E, X_new, y_new)
                          for y_new in y_new_s]

        my_s = [t.stats(**t_params_new, moments='m')
                for t_params_new in t_params_new_s]

        y_pred_old = reg_old.predict(E)
        y_pred_new_s = [reg_new.predict(E) for reg_new in reg_new_s]

        error_old = (y_pred_old - my_s[1]) ** 2
        error_new = (y_pred_new_s[1] - my_s[1]) ** 2

        var_new = 1/2*(((my_s[2] - my_s[1]) - (y_pred_new_s[2] - y_pred_new_s[1]))**2
                       + ((my_s[0] - my_s[1]) - (y_pred_new_s[0] - y_pred_new_s[1]))**2)

        var_old = 1/2*((my_s[2] - my_s[1])**2 + (my_s[0] - my_s[1])**2)

        return np.average((error_old - error_new) + (var_old - var_new))


