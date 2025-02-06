import math

import numpy as np
from sklearn import clone

from skactiveml.base import (
    SkactivemlRegressor,
    SingleAnnotatorPoolQueryStrategy,
    SkactivemlClassifier,
)
from skactiveml.utils import (
    check_type,
    simple_batch,
    check_scalar,
    MISSING_LABEL,
    check_X_y,
    check_random_state,
    _check_callable,
)


class ExpectedModelChangeMaximization(SingleAnnotatorPoolQueryStrategy):
    """Expected Model Change (EMC)

    This class implements "Expected Model Change" (EMC) [1]_, an active
    learning query strategy for linear regression.

    Parameters
    ----------
    bootstrap_size : int, default=3
        The number of bootstraps used to estimate the true model.
    n_train : int or float, default=0.5
        The size of a bootstrap compared to the training data if of type float.
        Must lie in the range of (0, 1]. The total size of a bootstrap if of
        type int. Must be greater or equal to 1.
    ord : int or string, default=2
        The norm to measure the gradient length. Argument will be passed to
        `np.linalg.norm`.
    feature_map : callable, default=None
        The feature map of the linear regressor. Takes in the feature data.
        Must output a np.array of dimension 2. The default value is the
        identity function. An example feature map is
        `sklearn.preprocessing.PolynomialFeatures().fit_transform`.
    missing_label : scalar or string or np.nan or None, default=np.nan
        Value to represent a missing label.
    random_state : int or np.random.RandomState or None, default=None
        Random state for candidate selection.

    References
    ----------
    .. [1] Cai, Wenbin, Ya Zhang, and Jun Zhou. Maximizing expected model
       change for active learning in regression, IEEE International Conference
       on Data Mining, pages 51--60, 2013.
    """

    def __init__(
        self,
        bootstrap_size=3,
        n_train=0.5,
        ord=2,
        feature_map=None,
        missing_label=MISSING_LABEL,
        random_state=None,
    ):
        super().__init__(
            random_state=random_state, missing_label=missing_label
        )
        self.bootstrap_size = bootstrap_size
        self.n_train = n_train
        self.ord = ord
        self.feature_map = feature_map

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
            Training data set, usually complete, i.e., including the labeled
            and unlabeled samples.
        y : array-like of shape (n_samples,)
            Labels of the training data set (possibly including unlabeled ones
            indicated by `self.missing_label`).
        reg : SkactivemlRegressor
            Regressor to predict the data. Assumes a linear regressor with
            respect to the parameters.
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
            - If `candidates` is of shape `(n_candidates, *)`, `candidates` is
              considered as the candidate samples in `(X,y)`.
        batch_size : int, default=1
            The number of samples to be selected in one AL cycle.
        return_utilities : bool, default=False
            If true, also return the utilities based on the query strategy.

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

        check_type(reg, "reg", SkactivemlRegressor)
        check_type(fit_reg, "fit_reg", bool)
        if self.feature_map is None:
            self.feature_map = lambda x: x
        _check_callable(self.feature_map, "self.feature_map")

        if fit_reg:
            if sample_weight is None:
                reg = clone(reg).fit(X, y)
            else:
                reg = clone(reg).fit(X, y, sample_weight)

        X_cand, mapping = self._transform_candidates(candidates, X, y)

        learners = _bootstrap_estimators(
            reg,
            X,
            y,
            bootstrap_size=self.bootstrap_size,
            n_train=self.n_train,
            sample_weight=sample_weight,
            random_state=self.random_state_,
        )

        results_learner = np.array(
            [learner.predict(X_cand) for learner in learners]
        )
        pred = reg.predict(X_cand).reshape(1, -1)
        scalars = np.average(np.abs(results_learner - pred), axis=0)
        X_cand_mapped_features = self.feature_map(X_cand)
        norms = np.linalg.norm(X_cand_mapped_features, ord=self.ord, axis=1)
        utilities_cand = scalars * norms

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


def _bootstrap_estimators(
    est,
    X,
    y,
    bootstrap_size=5,
    n_train=0.5,
    sample_weight=None,
    random_state=None,
):
    """Train the estimator on bootstraps of `X` and `y`.

    Parameters
    ----------
    est : SkactivemlClassifier or SkactivemlRegressor
        The estimator to be trained.
    X : array-like of shape (n_samples, n_features)
        Training data set, usually complete, i.e. including the labeled and
        unlabeled samples.
    y : array-like of shape (n_samples)
        Labels of the training data set.
    bootstrap_size : int, default=5
        The number of trained bootstraps.
    n_train : int or float, default=0.5
        The size of each bootstrap training data set.
    sample_weight: array-like of shape (n_samples,), default=None
        Weights of training samples in `X`.
    random_state : int or np.random.RandomState or None, default=None
        The random state to use.

    Returns
    -------
    bootstrap_est : list of SkactivemlClassifier or list of SkactivemlRegressor
        The estimators trained on different bootstraps.
    """

    check_X_y(X=X, y=y, sample_weight=sample_weight)
    check_scalar(bootstrap_size, "bootstrap_size", int, min_val=1)

    check_type(n_train, "n_train", int, float)
    if isinstance(n_train, int) and n_train < 1:
        raise ValueError(
            f"`n_train` has value `{type(n_train)}`, but must have a value "
            f"greater or equal to one, if of type `int`."
        )
    elif isinstance(n_train, float) and n_train <= 0 or n_train > 1:
        raise ValueError(
            f"`n_train` has value `{type(n_train)}`, but must have a value "
            f"between zero and one, excluding zero, if of type `float`."
        )
    if isinstance(n_train, float):
        n_train = math.ceil(n_train * len(X))

    check_type(est, "est", SkactivemlClassifier, SkactivemlRegressor)
    random_state = check_random_state(random_state)

    bootstrap_est = [clone(est) for _ in range(bootstrap_size)]
    sample_indices = np.arange(len(X))
    subsets_indices = [
        random_state.choice(sample_indices, size=int(len(X) * n_train + 1))
        for _ in range(bootstrap_size)
    ]

    for est_b, subset_indices in zip(bootstrap_est, subsets_indices):
        X_for_learner = X[subset_indices]
        y_for_learner = y[subset_indices]
        if sample_weight is None:
            est_b.fit(X_for_learner, y_for_learner)
        else:
            weight_for_learner = sample_weight[subset_indices]
            est_b.fit(X_for_learner, y_for_learner, weight_for_learner)

    return bootstrap_est
