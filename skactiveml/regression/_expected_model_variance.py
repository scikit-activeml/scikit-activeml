import numpy as np
from sklearn import clone

from skactiveml.base import (
    SkactivemlConditionalEstimator,
    SingleAnnotatorPoolQueryStrategy,
)
from skactiveml.utils import check_type, simple_batch
from skactiveml.utils._approximation import conditional_expect
from skactiveml.utils._functions import update_X_y


class ExpectedModelVarianceMinimization(SingleAnnotatorPoolQueryStrategy):
    """Expected model variance minimization

    This class implements the active learning strategy expected model variance
    minimization, which tries to select the sample that minimizes the expected
    model variance.

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
    [1] Cohn, David A and Ghahramani, Zoubin and Jordan, Michael I. Active
        learning with statistical models, pages 129--145, 1996.

    """

    def __init__(self, random_state=None, integration_dict=None):
        super().__init__(random_state=random_state)
        if integration_dict is not None:
            self.integration_dict = integration_dict
        else:
            self.integration_dict = {"integration_method": "assume_linear"}

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

        X_cand, mapping = self._transform_candidates(candidates, X, y)
        X_eval = X

        if fit_cond_est:
            cond_est = clone(cond_est).fit(X, y, sample_weight)

        def new_model_variance(x_idx, x_cand, y_pot):
            if mapping is not None:
                X_new, y_new = update_X_y(X, y, y_pot, idx_update=x_idx)
            else:
                X_new, y_new = update_X_y(X, y, y_pot, X_update=x_cand)

            cond_est_new = clone(cond_est).fit(X_new, y_new, sample_weight)
            _, new_model_std = cond_est_new.predict(X_eval, return_std=True)

            return np.average(new_model_std**2)

        ex_model_variance = conditional_expect(
            X_cand,
            new_model_variance,
            cond_est,
            random_state=self.random_state_,
            include_x=True,
            **self.integration_dict
        )

        # minimize the expected model variance by maximizing the negative
        # expected model variance

        return simple_batch(
            -ex_model_variance,
            batch_size=batch_size,
            random_state=self.random_state_,
            return_utilities=return_utilities,
        )
