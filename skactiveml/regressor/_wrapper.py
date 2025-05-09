import inspect
import warnings
from copy import deepcopy
from operator import attrgetter

import numpy as np
from scipy.stats import norm
from sklearn.base import MetaEstimatorMixin, is_regressor
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import (
    check_array,
    check_is_fitted,
    check_random_state,
)

from ..base import SkactivemlRegressor, ProbabilisticRegressor
from ..utils import (
    is_labeled,
    match_signature,
    check_n_features,
    check_scalar,
    MISSING_LABEL,
)


class SklearnRegressor(SkactivemlRegressor, MetaEstimatorMixin):
    """Sklearn Regressor

    Implementation of a wrapper class for scikit-learn regressors such that
    missing labels can be handled. Therefore, samples with missing values are
    filtered.

    Parameters
    ----------
    estimator : sklearn.base.RegressorMixin with predict method
        scikit-learn regressor.
    missing_label : scalar or string or np.nan or None, default=np.nan
        Value to represent a missing label.
    random_state : int or RandomState instance or None, default=None
        Determines random number for `predict` method. Pass an int for
        reproducible results across multiple method calls.
    """

    def __init__(
        self, estimator, missing_label=MISSING_LABEL, random_state=None
    ):
        super().__init__(
            random_state=random_state, missing_label=missing_label
        )
        self.estimator = estimator

    @match_signature("estimator", "fit")
    def fit(self, X, y, sample_weight=None, **fit_kwargs):
        """Fit the model using X as training data and y as labels.

        Parameters
        ----------
        X : matrix-like of shape (n_samples, n_features)
            The sample matrix X is the feature matrix representing the samples.
        y : array-like of shape (n_samples,)
            It contains the numeric target values of the training samples.
            Missing labels are represented as `self.missing_label`.
        sample_weight : array-like of shape (n_samples,), default=None
            It contains the weights of the training samples´ labels. It
            must have the same shape as y.
        fit_kwargs : dict-like
            Further parameters are passed as input to the `fit` method of the
            'estimator'.

        Returns
        -------
        self: SklearnRegressor,
            The SklearnRegressor is fitted on the training data.
        """
        return self._fit(
            fit_function="fit",
            X=X,
            y=y,
            sample_weight=sample_weight,
            **fit_kwargs,
        )

    @match_signature("estimator", "partial_fit")
    def partial_fit(self, X, y, sample_weight=None, **fit_kwargs):
        """Partially fitting the model using X as training data and y as class
        labels.

        Parameters
        ----------
        X : matrix-like of shape (n_samples, n_features)
            The sample matrix X is the feature matrix representing the samples.
        y : array-like of shape (n_samples,)
            It contains the numeric labels of the training samples.
            Missing labels are represented the attribute `self.missing_label`.
        sample_weight : array-like of shape (n_samples,)
            It contains the weights of the training samples' numeric labels. It
            must have the same shape as y.
        fit_kwargs : dict-like
            Further parameters as input to the `fit` method of the `estimator`.

        Returns
        -------
        self : SklearnRegressor,
            The `SklearnRegressor` is fitted on the training data.
        """
        return self._fit(
            fit_function="partial_fit",
            X=X,
            y=y,
            sample_weight=sample_weight,
            **fit_kwargs,
        )

    def _fit(self, fit_function, X, y, sample_weight, **fit_kwargs):
        if not is_regressor(estimator=self.estimator):
            raise TypeError(
                "'{}' must be a scikit-learn "
                "regressor.".format(self.estimator)
            )

        self.check_X_dict_ = {
            "ensure_min_samples": 0,
            "ensure_min_features": 0,
            "allow_nd": True,
            "dtype": None,
        }

        X, y, sample_weight = self._validate_data(
            X,
            y,
            sample_weight,
            check_X_dict=self.check_X_dict_,
            reset=fit_function == "fit" or not hasattr(self, "n_features_in_"),
        )

        is_lbld = is_labeled(y, missing_label=self.missing_label_)
        X_labeled = X[is_lbld]
        y_labeled = y[is_lbld]
        estimator_params = dict(fit_kwargs) if fit_kwargs is not None else {}

        if sample_weight is not None:
            estimator_params["sample_weight"] = sample_weight[is_lbld]

        self._label_mean = np.mean(y[is_lbld]) if np.sum(is_lbld) > 0 else 0
        self._label_std = np.std(y[is_lbld]) if np.sum(is_lbld) > 1 else 1
        self.estimator_ = deepcopy(self.estimator)
        try:
            attrgetter(fit_function)(self.estimator_)(
                X_labeled, y_labeled, **estimator_params
            )
        except Exception as e:
            warnings.warn(
                f"The 'estimator' could not be fitted because of"
                f" '{e}'. Therefore, the empirical label mean "
                f"`_label_mean={self._label_mean}` and the "
                f"empirical label standard deviation "
                f"`_label_std={self._label_std}` will be used to make "
                f"predictions."
            )

        return self

    @match_signature("estimator", "predict")
    def predict(self, X, **predict_kwargs):
        """Return label predictions for the input data `X`.

        Parameters
        ----------
        X :  array-like of shape (n_samples, n_features)
            Input samples.
        predict_kwargs : dict-like
            Further parameters are passed as input to the `predict` method of
            the `estimator`. If the estimator could not be fitted, only
            `return_std` is supported as keyword argument.

        Returns
        -------
        y :  ndarray of shape (n_samples,)
            Predicted labels of the input samples.
        """
        check_is_fitted(self)
        predict_dict = {"ensure_min_samples": 1, "ensure_min_features": 1}
        X = check_array(X, **(self.check_X_dict_ | predict_dict))
        check_n_features(self, X, reset=False)
        try:
            return self.estimator_.predict(X, **predict_kwargs)
        except NotFittedError:
            warnings.warn(
                f"Since the 'estimator' could not be fitted when"
                f" calling the `fit` method, the label "
                f"mean `_label_mean={self._label_mean}` and optionally the "
                f"label standard deviation `_label_std={self._label_std}` is "
                f"used to make the predictions."
            )
            has_std = predict_kwargs.pop("return_std", False)
            if has_std:
                return (
                    np.full(len(X), self._label_mean),
                    np.full(len(X), self._label_std),
                )
            else:
                return np.full(len(X), self._label_mean)

    @match_signature("estimator", "sample_y")
    def sample_y(self, X, n_samples=1, **sample_kwargs):
        """Assumes a probabilistic regressor. Samples are drawn from a
        predicted target distribution.

        Parameters
        ----------
        X : array-like of shape (n_samples_X, n_features)
            Input samples from which the target values are drawn.
        n_samples : int, default=1
            Number of random samples to be drawn.
        **sample_kwargs : dict
            Additional keyword arguments for sampling. For example:

            random_state : int, RandomState instance or None, default=None
                Determines the random number generation for drawing samples.
                Pass an int for reproducible results across multiple method
                calls.

        Returns
        -------
        y_samples : ndarray of shape (n_samples_X, n_samples)
            Drawn random target samples.
        """
        return self._sample(
            sample_function="sample_y",
            X=X,
            n_samples=n_samples,
            **sample_kwargs,
        )

    @match_signature("estimator", "sample")
    def sample(self, X, n_samples=1, **sample_kwargs):
        """Assumes a probabilistic regressor. Samples are drawn from a
        predicted target distribution.

        Parameters
        ----------
        X : array-like of shape (n_samples_X, n_features)
            Input samples from which the target values are drawn.
        n_samples : int, default=1
            Number of random samples to be drawn.
        **sample_kwargs : dict
            Additional keyword arguments for sampling. For example:

            random_state : int, RandomState instance or None, default=None
                Determines the random number generation for drawing samples.
                Pass an int for reproducible results across multiple method
                calls.

        Returns
        -------
        y_samples : ndarray of shape (n_samples_X, n_samples)
            Drawn random target samples.
        """
        return self._sample(
            sample_function="sample", X=X, n_samples=n_samples, **sample_kwargs
        )

    def _sample(self, sample_function, X, n_samples=1, **sample_kwargs):
        check_is_fitted(self)
        predict_dict = {"ensure_min_samples": 1, "ensure_min_features": 1}
        X = check_array(X, **(self.check_X_dict_ | predict_dict))
        check_n_features(self, X, reset=False)
        try:
            return attrgetter(sample_function)(self.estimator_)(
                X, n_samples, **sample_kwargs
            )
        except NotFittedError:
            warnings.warn(
                f"Since the 'estimator' could not be fitted when"
                f" calling the `fit` method, the label "
                f"mean `_label_mean={self._label_mean}` and optionally the "
                f"label standard deviation `_label_std={self._label_std}` is "
                f"used to make the predictions."
            )
            random_state = sample_kwargs.get("random_state", None)
            random_state = check_random_state(random_state)
            check_scalar(
                n_samples,
                "n_samples",
                min_val=1,
                min_inclusive=True,
                target_type=int,
            )
            y_samples = random_state.randn(len(X), n_samples)
            y_samples *= self._label_std
            y_samples += self._label_mean
            return y_samples

    def __sklearn_is_fitted__(self):
        if hasattr(self, "_label_mean"):
            return True

        try:
            check_is_fitted(self.estimator)
        except NotFittedError:
            return False

        # set attributes that would be set by the fit function
        self._label_mean = 0
        self._label_std = 1
        self.estimator_ = deepcopy(self.estimator)
        self.check_X_dict_ = {
            "ensure_min_samples": 0,
            "ensure_min_features": 0,
            "allow_nd": True,
            "dtype": None,
        }

        return True

    def __getattr__(self, item):
        if "estimator_" in self.__dict__:
            return getattr(self.estimator_, item)
        else:
            return getattr(self.estimator, item)


class SklearnNormalRegressor(ProbabilisticRegressor, SklearnRegressor):
    """Sklearn Normal Regressor

    Implementation of a wrapper class for scikit-learn probabilistic regressors
    such that missing labels can be handled and the target distribution can be
    estimated. Therefore, samples with missing values are filtered and a normal
    distribution is fitted using the predicted means and standard deviations.

    The wrapped regressor of sklearn needs `return_std` as a key_word argument
    for `predict`.

    Parameters
    ----------
    estimator : sklearn.base.RegressorMixin with predict method
        scikit-learn regressor.
    missing_label : scalar or string or np.nan or None, default=np.nan
        Value to represent a missing label.
    random_state : int or RandomState instance or None, default=None
        Determines random number for `predict` method. Pass an int for
        reproducible results across multiple method calls.
    """

    def __init__(
        self, estimator, missing_label=MISSING_LABEL, random_state=None
    ):
        super().__init__(
            estimator, missing_label=missing_label, random_state=random_state
        )

    def predict_target_distribution(self, X):
        """Returns the estimated target normal distribution conditioned on the
        test samples `X`.

        Parameters
        ----------
        X :  array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        dist : scipy.stats._distn_infrastructure.rv_frozen
            The distribution of the targets at the test samples.

        """
        check_is_fitted(self)

        if (
            "return_std"
            not in inspect.signature(self.estimator.predict).parameters.keys()
        ):
            raise ValueError(
                f"`{self.estimator}` must have key_word argument"
                f"`return_std` for predict."
            )

        loc, scale = SklearnRegressor.predict(self, X, return_std=True)
        return norm(loc=loc, scale=scale)
