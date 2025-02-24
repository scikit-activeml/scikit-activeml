import numpy as np
from sklearn import clone

from skactiveml.base import (
    SingleAnnotatorPoolQueryStrategy,
    ProbabilisticRegressor,
)

from skactiveml.pool.utils import (
    _update_reg,
    _conditional_expect,
    _cross_entropy,
)
from skactiveml.utils import (
    check_type,
    simple_batch,
    MISSING_LABEL,
    is_unlabeled,
)


class KLDivergenceMaximization(SingleAnnotatorPoolQueryStrategy):
    """Regression based Kullback-Leibler Divergence Maximization

    This class implements a query  [1]_, which selects those samples
    that maximize the expected Kullback-Leibler divergence, where it is assumed
    that the target probabilities for different samples are independent.

    Parameters
    ----------
    integration_dict_target_val : dict, default=None
        Dictionary for integration arguments, i.e. `integration method` etc.,
        used for calculating the expected `y` value for the candidate samples.
        For details see method `skactiveml.pool.utils._conditional_expect`.
    integration_dict_cross_entropy : dict, default=None
        Dictionary for integration arguments, i.e. `integration method` etc.,
        used for calculating the cross entropy between the updated conditional
        estimator by the `X_cand` value and the old conditional estimator.
        For details see method `conditional_expect`.
    missing_label : scalar or string or np.nan or None, default=np.nan
        Value to represent a missing label.
    random_state : int or RandomState instance, default=None
        Random state for candidate selection.

    References
    ----------
    .. [1] D. Elreedy, A. F. Atiya, and S. I. Shaheen. A Novel Active Learning
       Regression Framework for Balancing the Exploration-Exploitation
       Trade-Off. Entropy, 21(7):651, 2019.
    """

    def __init__(
        self,
        integration_dict_target_val=None,
        integration_dict_cross_entropy=None,
        missing_label=MISSING_LABEL,
        random_state=None,
    ):
        super().__init__(
            random_state=random_state, missing_label=missing_label
        )
        self.integration_dict_target_val = integration_dict_target_val
        self.integration_dict_cross_entropy = integration_dict_cross_entropy

    def query(
        self,
        X,
        y,
        reg,
        fit_reg=True,
        sample_weight=None,
        candidates=None,
        batch_size=1,
        return_utilities=False,
    ):
        """Determines for which candidate samples labels are to be queried.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data set, usually complete, i.e. including the labeled and
            unlabeled samples.
        y : array-like of shape (n_samples)
            Labels of the training data set (possibly including unlabeled ones
            indicated by `self.missing_label`).
        reg : skactiveml.base.ProbabilisticRegressor
            Predicts the entropy and the cross entropy and the potential
            y-values for the candidate samples.
        fit_reg : bool, default=True
            Defines whether the regressor should be fitted on `X`, `y`, and
            `sample_weight`.
        sample_weight : array-like of shape (n_samples,), default=None
            Weights of training samples in `X`.
        candidates : None or array-like of shape (n_candidates), dtype=int or \
                array-like of shape (n_candidates, n_features), default=None
            - If `candidates` is `None`, the unlabeled samples from
              `(X,y)` are considered as `candidates`.
            - If `candidates` is of shape `(n_candidates,)` and of type
              `int`, `candidates` is considered as the indices of the
              samples in `(X,y)`.
            - If `candidates` is of shape `(n_candidates, *)`, the
              candidate samples are directly given in `candidates` (not
              necessarily contained in `X`). This is not supported by all
              query strategies.
        batch_size : int, default=1
            The number of samples to be selected in one AL cycle.
        return_utilities : bool, default=False
            If `True`, also return the utilities based on the query strategy.

        Returns
        -------
        query_indices : numpy.ndarray of shape (batch_size)
            The query indices indicate for which candidate sample a label is to
            be queried, e.g., `query_indices[0]` indicates the first selected
            sample.

            - If `candidates` is `None` or of shape
              `(n_candidates,)`, the indexing refers to the samples in
              `X`.
            - If `candidates` is of shape `(n_candidates, n_features)`,
              the indexing refers to the samples in `candidates`.
        utilities : numpy.ndarray of shape (batch_size, n_samples)
            The utilities of samples after each selected sample of the batch,
            e.g., `utilities[0]` indicates the utilities used for selecting
            the first sample (with index `query_indices[0]`) of the batch.
            Utilities for labeled samples will be set to np.nan.

            - If `candidates` is `None`, the indexing refers to the samples
              in `X`.
            - If `candidates` is of shape `(n_candidates,)` and of type
              `int`, `utilities` refers to the samples in `X`.
            - If `candidates` is of shape `(n_candidates, *)`, `utilities`
              refers to the indexing in `candidates`.
        """
        X, y, candidates, batch_size, return_utilities = self._validate_data(
            X, y, candidates, batch_size, return_utilities, reset=True
        )

        check_type(reg, "reg", ProbabilisticRegressor)
        check_type(fit_reg, "fit_reg", bool)

        X_eval = X[is_unlabeled(y, missing_label=self.missing_label_)]
        if len(X_eval) == 0:
            raise ValueError(
                "The training data contains no unlabeled " "data."
            )

        if self.integration_dict_target_val is None:
            self.integration_dict_target_val = {"method": "assume_linear"}

        if self.integration_dict_cross_entropy is None:
            self.integration_dict_cross_entropy = {
                "method": "gauss_hermite",
                "n_integration_samples": 10,
            }
        check_type(
            self.integration_dict_target_val, "self.integration_dict", dict
        )
        check_type(
            self.integration_dict_cross_entropy, "self.integration_dict", dict
        )

        X_cand, mapping = self._transform_candidates(candidates, X, y)

        if fit_reg:
            if sample_weight is None:
                reg = clone(reg).fit(X, y)
            else:
                reg = clone(reg).fit(X, y, sample_weight)

        utilities_cand = self._kullback_leibler_divergence(
            X_eval, X_cand, mapping, reg, X, y, sample_weight=sample_weight
        )

        if mapping is None:
            utilities = utilities_cand
        else:
            utilities = np.full(len(X), np.nan)
            utilities[mapping] = utilities_cand

        return simple_batch(
            utilities,
            self.random_state_,
            batch_size=batch_size,
            return_utilities=return_utilities,
        )

    def _kullback_leibler_divergence(
        self, X_eval, X_cand, mapping, reg, X, y, sample_weight=None
    ):
        """Calculates the expected Kullback-Leibler divergence over the
        evaluation set if each candidate sample where to be labeled.

        Parameters
        ----------
        X_eval : array-like of shape (n_samples, n_features)
            The samples where the information gain should be evaluated.
        X_cand : array-like of shape (n_candidate_samples, n_features)
            The candidate samples that determine the information gain.
        mapping : array-like of shape (n_candidate_samples,) or None
            A mapping between `X_cand` and `X` if it exists.
        reg: ProbabilisticRegressor
            Predicts the entropy, predicts values.
        X : array-like of shape (n_samples, n_features)
            Training data set, usually complete, i.e., including the labeled
            and unlabeled samples.
        y : array-like of shape (n_samples,)
            Labels of the training data set (possibly including unlabeled ones
            indicated by `self.missing_label`).
        sample_weight: array-like of shape (n_samples,), default=None
            Weights of training samples in `X`.

        Returns
        -------
        kl_div : numpy.ndarray of shape (n_candidate_samples,)
            The calculated expected Kullback-Leibler divergence.
        """

        def new_kl_divergence(idx, x_cand, y_pot):
            reg_new = _update_reg(
                reg,
                X,
                y,
                sample_weight=sample_weight,
                y_update=y_pot,
                idx_update=idx,
                X_update=x_cand,
                mapping=mapping,
            )
            entropy_post = np.sum(
                reg_new.predict(X_eval, return_entropy=True)[1]
            )
            cross_ent = np.sum(
                _cross_entropy(
                    X_eval,
                    reg_new,
                    reg,
                    integration_dict=self.integration_dict_cross_entropy,
                    random_state=self.random_state_,
                )
            )
            return cross_ent - entropy_post

        kl_div = _conditional_expect(
            X_cand,
            new_kl_divergence,
            reg,
            random_state=self.random_state_,
            **self.integration_dict_target_val
        )

        return kl_div
