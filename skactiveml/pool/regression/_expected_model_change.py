import numpy as np
from sklearn import clone

from skactiveml.base import (
    SkactivemlRegressor,
    SingleAnnotatorPoolQueryStrategy,
)
from skactiveml.utils import check_type, simple_batch, check_scalar, MISSING_LABEL
from skactiveml.pool.regression.utils._model_fitting import bootstrap_estimators
from skactiveml.utils._validation import check_callable


class ExpectedModelChange(SingleAnnotatorPoolQueryStrategy):
    """Expected Model Change

    This class implements expected model change, an active learning
    query strategy for linear regression.

    Parameters
    ----------
    k_bootstraps: int, optional (default=3)
        The number of bootstraps used to estimate the true model.
    n_train: int or float, optional (default=0.5)
        The size of a bootstrap compared to the training data.
    ord: int or string (default=2)
        The Norm to measure the gradient. Argument will be passed to
        `np.linalg.norm`.
    feature_map: callable
        The feature map of the linear regressor. Takes in the feature data. Must
        output a np.array of dimension 2.
    missing_label : scalar or string or np.nan or None, default=np.nan
        Value to represent a missing label.
    random_state: numeric | np.random.RandomState, optional
        Random state for candidate selection.

    References
    ----------
    [1] Cai, Wenbin, Ya Zhang, and Jun Zhou. Maximizing expected model change
    for active learning in regression, pages 51--60, 2013.

    """

    def __init__(
        self,
        k_bootstraps=3,
        n_train=0.5,
        ord=2,
        feature_map=None,
        missing_label=MISSING_LABEL,
        random_state=None,
    ):
        super().__init__(random_state=random_state, missing_label=missing_label)
        self.k_bootstraps = k_bootstraps
        self.n_train = n_train
        self.ord = ord
        self.feature_map = feature_map if feature_map is not None else lambda x: x

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
            indicated by self.MISSING_LABEL.
        reg: SkactivemlRegressor
            Regressor to predict the data. Assumes a linear regressor with respect
            to the parameters.
        fit_reg : bool, optional (default=True)
            Defines whether the regressor should be fitted on `X`, `y`, and
            `sample_weight`.
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

        check_type(reg, "reg", SkactivemlRegressor)
        check_type(fit_reg, "fit_reg", bool)
        check_scalar(
            self.n_train,
            "self.n_train",
            (int, float),
            min_val=0,
            max_val=1,
            min_inclusive=False,
        )
        check_scalar(self.k_bootstraps, "self.k_bootstraps", int, min_val=1)
        check_callable(self.feature_map, "self.feature_map")

        if fit_reg:
            reg = clone(reg).fit(X, y, sample_weight)

        X_cand, mapping = self._transform_candidates(candidates, X, y)

        learners = bootstrap_estimators(
            reg,
            X,
            y,
            k_bootstrap=self.k_bootstraps,
            n_train=self.n_train,
            sample_weight=sample_weight,
            random_state=self.random_state_,
        )

        results_learner = np.array([learner.predict(X_cand) for learner in learners])
        pred = reg.predict(X_cand).reshape(1, -1)
        scalars = np.average(np.abs(results_learner - pred), axis=0)
        X_cand_mapped_features = self.feature_map(X_cand)
        norms = np.linalg.norm(X_cand_mapped_features, ord=self.ord, axis=1)
        utilities_cand = np.multiply(scalars, norms)

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
