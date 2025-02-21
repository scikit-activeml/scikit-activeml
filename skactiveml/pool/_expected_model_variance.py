import numpy as np
from sklearn import clone
from sklearn.utils.validation import check_array, _check_n_features

from skactiveml.base import (
    ProbabilisticRegressor,
    SingleAnnotatorPoolQueryStrategy,
)
from skactiveml.utils import check_type, simple_batch, MISSING_LABEL
from skactiveml.pool.utils import _update_reg, _conditional_expect


class ExpectedModelVarianceReduction(SingleAnnotatorPoolQueryStrategy):
    """Expected Model Variance Reduction (EMVR)

    This class implements the active learning strategy "Expected Model Variance
    Reduction" (EMVR) [1]_, which tries to select the sample that minimizes the
    expected model variance.

    Parameters
    ----------
    integration_dict : dict, default=None
        Dictionary for integration arguments, i.e. `integration method` etc.,
        used for calculating the expected `y` value for the candidate samples.
        For details see method `skactiveml.pool.utils._conditional_expect`.
    missing_label : scalar or string or np.nan or None, default=np.nan
        Value to represent a missing label.
    random_state : int or np.random.RandomState or None, default=None
        Random state for candidate selection.

    References
    ----------
    .. [1] Cohn, David A and Ghahramani, Zoubin and Jordan, Michael I. Active
       learning with statistical models, pages 129--145, 1996.
    """

    def __init__(
        self,
        integration_dict=None,
        missing_label=MISSING_LABEL,
        random_state=None,
    ):
        super().__init__(
            random_state=random_state, missing_label=missing_label
        )
        self.integration_dict = integration_dict

    def query(
        self,
        X,
        y,
        reg,
        fit_reg=True,
        sample_weight=None,
        candidates=None,
        X_eval=None,
        batch_size=1,
        return_utilities=False,
    ):
        """Determines for which candidate samples labels are to be queried.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data set, usually complete, i.e., including the labeled
            and unlabeled samples.
        y : array-like of shape (n_samples,)
            Labels of the training data set (possibly including unlabeled ones
            indicated by `self.missing_label`).
        reg : ProbabilisticRegressor
            Predicts the output and the conditional distribution.
        fit_reg : bool, optional (default=True)
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
              necessarily contained in `X`).
        batch_size : int, default=1
            The number of samples to be selected in one AL cycle.
        return_utilities : bool, default=False
            If `True`, also return the utilities based on the query strategy.

        Returns
        -------
        query_indices : numpy.ndarray of shape (batch_size,)
            The query indices indicate for which candidate sample a label is
            to be queried, e.g., `query_indices[0]` indicates the first
            selected sample.

            - If `candidates` is `None` or of shape
              `(n_candidates,)`, the indexing refers to the samples in
              `X`.
            - If `candidates` is of shape `(n_candidates, n_features)`,
              the indexing refers to the samples in `candidates`.
        utilities : numpy.ndarray of shape (batch_size, n_samples) or \
                numpy.ndarray of shape (batch_size, n_candidates)
            The utilities of samples after each selected sample of the batch,
            e.g., `utilities[0]` indicates the utilities used for selecting
            the first sample (with index `query_indices[0]`) of the batch.
            Utilities for labeled samples will be set to np.nan.

            - If `candidates` is `None` or of shape
              `(n_candidates,)`, the indexing refers to the samples in
              `X`.
            - If `candidates` is of shape `(n_candidates, n_features)`,
              the indexing refers to the samples in `candidates`.
        """
        X, y, candidates, batch_size, return_utilities = self._validate_data(
            X, y, candidates, batch_size, return_utilities, reset=True
        )

        check_type(reg, "reg", ProbabilisticRegressor)
        check_type(fit_reg, "fit_reg", bool)
        if X_eval is None:
            X_eval = X
        else:
            X_eval = check_array(X_eval)
            _check_n_features(self, X_eval, reset=False)
        if self.integration_dict is None:
            self.integration_dict = {"method": "assume_linear"}
        check_type(self.integration_dict, "self.integration_dict", dict)

        X_cand, mapping = self._transform_candidates(candidates, X, y)

        if fit_reg:
            if sample_weight is None:
                reg = clone(reg).fit(X, y)
            else:
                reg = clone(reg).fit(X, y, sample_weight)

        old_model_variance = np.average(
            reg.predict(X_eval, return_std=True)[1] ** 2
        )

        def new_model_variance(idx, x_cand, y_pot):
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
            _, new_model_std = reg_new.predict(X_eval, return_std=True)

            return np.average(new_model_std**2)

        ex_model_variance = _conditional_expect(
            X_cand,
            new_model_variance,
            reg,
            random_state=self.random_state_,
            **self.integration_dict
        )

        utilities_cand = old_model_variance - ex_model_variance

        if mapping is None:
            utilities = utilities_cand
        else:
            utilities = np.full(len(X), np.nan)
            utilities[mapping] = utilities_cand

        return simple_batch(
            utilities,
            batch_size=batch_size,
            random_state=self.random_state_,
            return_utilities=return_utilities,
        )
