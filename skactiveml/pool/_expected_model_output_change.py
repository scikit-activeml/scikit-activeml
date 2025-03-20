import numpy as np
from sklearn import clone
from sklearn.utils.validation import check_array, _check_n_features
from sklearn.metrics import mean_squared_error

from skactiveml.base import (
    ProbabilisticRegressor,
    SingleAnnotatorPoolQueryStrategy,
)
from skactiveml.pool.utils import _update_reg, _conditional_expect
from skactiveml.utils import (
    check_type,
    simple_batch,
    MISSING_LABEL,
    _check_callable,
    is_unlabeled,
)


class ExpectedModelOutputChange(SingleAnnotatorPoolQueryStrategy):
    """Regression based Expected Model Output Change (EMOC)

    This class implements an "Expected Model Output Change" (EMOC) based
    approach for regression [1]_, where samples are queried that change the
    output of the regression model the most.

    Parameters
    ----------
    integration_dict : dict, default=None
        Dictionary for integration arguments, i.e. `integration_method` etc.,
        used for calculating the expected `y` value for the candidate samples.
        For details see method `skactiveml.pool.utils._conditional_expect`.
        The default `integration_method` is `assume_linear`.
    loss : callable, default=None
        The loss for predicting a target value instead of the true value.
        Takes in the predicted values of an evaluation set and the true values
        of the evaluation set and returns the error, a scalar value.
        The default loss is `sklearn.metrics.mean_squared_error` an alternative
        might be `sklearn.metrics.mean_absolute_error`.
    missing_label : scalar or string or np.nan or None, default=np.nan
        Value to represent a missing label.
    random_state : int or np.random.RandomState or None, default=None
        Random state for candidate selection.

    References
    ----------
    .. [1] Christoph Kaeding, Erik Rodner, Alexander Freytag, Oliver Mothes,
       Oliver, Bjoern Barz and Joachim Denzler. Active Learning for Regression
       Tasks with Expected Model Output Change, BMVC, page 1-15, 2018.
    """

    def __init__(
        self,
        integration_dict=None,
        loss=None,
        missing_label=MISSING_LABEL,
        random_state=None,
    ):
        super().__init__(
            random_state=random_state, missing_label=missing_label
        )
        self.loss = loss
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
            Predicts the output and the target distribution.
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
              necessarily contained in `X`).
        X_eval : array-like of shape (n_eval_samples, n_features), default=None
            Evaluation data set that is used for estimating the probability
            distribution of the feature space. In the referenced paper it is
            proposed to use the unlabeled data, i.e.,
            `X_eval=X[is_unlabeled(y)]`.
        batch_size : int, default=1
            The number of samples to be selected in one AL cycle.
        return_utilities : bool, default=False
            If true, also return the utilities based on the query strategy.

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
        if self.integration_dict is None:
            self.integration_dict = {"method": "assume_linear"}
        check_type(self.integration_dict, "self.integration_dict", dict)
        if X_eval is None:
            X_eval = X[is_unlabeled(y, missing_label=self.missing_label_)]
            if len(X_eval) == 0:
                raise ValueError(
                    "The training data contains no unlabeled "
                    "data. This can be fixed by setting the "
                    "evaluation set manually, e.g. set "
                    "`X_eval=X`."
                )
        else:
            X_eval = check_array(X_eval)
            _check_n_features(self, X_eval, reset=False)
        check_type(fit_reg, "fit_reg", bool)
        if self.loss is None:
            self.loss = mean_squared_error
        _check_callable(self.loss, "self.loss", n_positional_parameters=2)

        X_cand, mapping = self._transform_candidates(candidates, X, y)

        if fit_reg:
            if sample_weight is None:
                reg = clone(reg).fit(X, y)
            else:
                reg = clone(reg).fit(X, y, sample_weight)

        y_pred = reg.predict(X_eval)

        def _model_output_change(idx, x_cand, y_pot):
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
            y_pred_new = reg_new.predict(X_eval)

            return self.loss(y_pred, y_pred_new)

        change = _conditional_expect(
            X_cand,
            _model_output_change,
            reg,
            random_state=self.random_state_,
            **self.integration_dict,
        )

        if mapping is None:
            utilities = change
        else:
            utilities = np.full(len(X), np.nan)
            utilities[mapping] = change

        return simple_batch(
            utilities,
            batch_size=batch_size,
            random_state=self.random_state_,
            return_utilities=return_utilities,
        )
