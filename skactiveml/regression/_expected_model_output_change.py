import numpy as np
from sklearn import clone

from skactiveml.base import (
    SkactivemlConditionalEstimator,
    SingleAnnotatorPoolQueryStrategy,
)
from skactiveml.utils import check_type, simple_batch
from skactiveml.utils._approximation import conditional_expect
from skactiveml.utils._functions import update_X_y_map
from skactiveml.utils._validation import check_callable


class ExpectedModelOutputChange(SingleAnnotatorPoolQueryStrategy):
    """Regression based Expected Model Output Change

    This class implements an expected model output change based approach for
    regression, where samples are queried that change the output of the model
    the most.

    Parameters
    ----------
    random_state: numeric | np.random.RandomState, optional (default=None)
        Random state for candidate selection.
    integration_dict: dict, optional (default=None)
        Dictionary for integration arguments, i.e. `integration method` etc.,
        used for calculating the expected `y` value for the candidate samples.
        For details see method `conditional_expect`.

    References
    ----------
    [1] Kaeding, Christoph and Rodner, Erik and Freytag, Alexander and Mothes,
        Oliver and Barz, Bjoern and Denzler, Joachim and AG, Carl Zeiss. Active
        Learning for Regression Tasks with Expected Model Output Change,
        page 103 and subsequently, 2018.

    """

    def __init__(self, random_state=None, integration_dict=None, loss=None):
        super().__init__(random_state=random_state)
        self.loss = loss if loss is not None else lambda x, y: np.sum((x - y) ** 2)
        if integration_dict is not None:
            self.integration_dict = integration_dict
        else:
            self.integration_dict = {"method": "assume_linear"}

    def query(
        self,
        X,
        y,
        cond_est,
        sample_weight=None,
        candidates=None,
        batch_size=1,
        return_utilities=False,
        fit_cond_est=None,
    ):
        """Determines for which candidate samples labels are to be queried.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data set, usually complete, i.e. including the labeled and
            unlabeled samples.
        y : array-like of shape (n_samples)
            Labels of the training data set (possibly including unlabeled ones
            indicated by self.MISSING_LABEL.
        cond_est: SkactivemlConditionalEstimator
            Estimates the output and the conditional distribution.
        fit_cond_est: bool
            Whether the conditional estimator is fitted.
        sample_weight: array-like of shape (n_samples), optional (default=None)
            Weights of training samples in `X`.
        candidates : None or array-like of shape (n_candidates), dtype=int or
            array-like of shape (n_candidates, n_features),
            optional (default=None)
            If candidates is None, the unlabeled samples from (X,y) are
            considered as candidates.
            If candidates is of shape (n_candidates) and of type int,
            candidates is considered as the indices of the samples in (X,y).
            If candidates is of shape (n_candidates, n_features), the
            candidates are directly given in candidates (not necessarily
            contained in X). This is not supported by all query strategies.
        batch_size : int, optional (default=1)
            The number of samples to be selected in one AL cycle.
        return_utilities : bool, optional (default=False)
            If true, also return the utilities based on the query strategy.

        Returns
        -------
        query_indices : numpy.ndarray of shape (batch_size)
            The query_indices indicate for which candidate sample a label is
            to queried, e.g., `query_indices[0]` indicates the first selected
            sample.
            If candidates is None or of shape (n_candidates), the indexing
            refers to samples in X.
            If candidates is of shape (n_candidates, n_features), the indexing
            refers to samples in candidates.
        utilities : numpy.ndarray of shape (batch_size, n_samples) or
            numpy.ndarray of shape (batch_size, n_candidates)
            The utilities of samples after each selected sample of the batch,
            e.g., `utilities[0]` indicates the utilities used for selecting
            the first sample (with index `query_indices[0]`) of the batch.
            Utilities for labeled samples will be set to np.nan.
            If candidates is None or of shape (n_candidates), the indexing
            refers to samples in X.
            If candidates is of shape (n_candidates, n_features), the indexing
            refers to samples in candidates.
        """
        X, y, candidates, batch_size, return_utilities = self._validate_data(
            X, y, candidates, batch_size, return_utilities, reset=True
        )

        check_type(cond_est, "cond_est", SkactivemlConditionalEstimator)
        check_type(self.integration_dict, "self.integration_dict", dict)

        loss = self.loss
        check_callable(loss, "self.loss", n_free_parameters=2)

        X_cand, mapping = self._transform_candidates(candidates, X, y)
        X_eval = X

        if fit_cond_est:
            cond_est = clone(cond_est).fit(X, y, sample_weight)

        y_pred = cond_est.predict(X_eval)

        def model_output_change(idx, x_cand, y_pot):
            X_new, y_new = update_X_y_map(X, y, y_pot, idx, x_cand, mapping)
            cond_est_new = clone(cond_est).fit(X_new, y_new, sample_weight)
            y_pred_new = cond_est_new.predict(X_eval)

            return loss(y_pred, y_pred_new)

        change = conditional_expect(
            X_cand,
            model_output_change,
            cond_est,
            random_state=self.random_state_,
            include_idx=True,
            include_x=True,
            **self.integration_dict
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
