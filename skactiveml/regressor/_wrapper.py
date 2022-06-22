import inspect
import warnings
from copy import deepcopy
from operator import attrgetter

import numpy as np
from scipy.stats import norm
from sklearn.base import MetaEstimatorMixin, is_regressor
from sklearn.exceptions import NotFittedError
from sklearn.utils import metaestimators
from sklearn.utils.validation import (
    has_fit_parameter,
    check_array,
    check_is_fitted,
)

from skactiveml.base import SkactivemlRegressor, ProbabilisticRegressor
from skactiveml.utils._functions import _available_if
from skactiveml.utils._label import is_labeled, MISSING_LABEL


class SklearnRegressor(SkactivemlRegressor, MetaEstimatorMixin):
    """SklearnRegressor

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
        Determines random number for 'predict' method. Pass an int for
        reproducible results across multiple method calls.
    """

    def __init__(
        self, estimator, missing_label=MISSING_LABEL, random_state=None
    ):
        super().__init__(
            random_state=random_state, missing_label=missing_label
        )
        self.estimator = estimator

    def fit(self, X, y, sample_weight=None, **fit_kwargs):
        """Fit the model using X as training data and y as labels.

        Parameters
        ----------
        X : matrix-like, shape (n_samples, n_features)
            The sample matrix X is the feature matrix representing the samples.
        y : array-like, shape (n_samples)
            It contains the values of the training samples.
            Missing labels are represented as 'np.nan'.
        sample_weight : array-like, shape (n_samples), optional (default=None)
            It contains the weights of the training samples´ labels. It
            must have the same shape as y.
        fit_kwargs : dict-like
            Further parameters are passed as input to the 'fit' method of the
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

    @_available_if("partial_fit", hasattr(metaestimators, "available_if"))
    def partial_fit(self, X, y, sample_weight=None, **fit_kwargs):
        """Partially fitting the model using X as training data and y as class
        labels.

        Parameters
        ----------
        X : matrix-like, shape (n_samples, n_features)
            The sample matrix X is the feature matrix representing the samples.
        y : array-like, shape (n_samples) or (n_samples, n_outputs)
            It contains the numeric labels of the training samples.
            Missing labels are represented the attribute 'missing_label'.
            In case of multiple labels per sample (i.e., n_outputs > 1), the
            samples are duplicated.
        sample_weight : array-like, shape (n_samples) or (n_samples, n_outputs)
            It contains the weights of the training samples' numeric labels. It
            must have the same shape as y.
        fit_kwargs : dict-like
            Further parameters as input to the 'fit' method of the 'estimator'.

        Returns
        -------
        self : SklearnRegressor,
            The SklearnRegressor is fitted on the training data.
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

        self.estimator_ = deepcopy(self.estimator)

        self._label_mean = 0
        self._label_std = 1

        self.check_X_dict_ = {
            "ensure_min_samples": 0,
            "ensure_min_features": 0,
            "allow_nd": True,
            "dtype": None,
        }

        X, y, sample_weight = self._validate_data(
            X, y, sample_weight, check_X_dict=self.check_X_dict_
        )

        is_lbld = is_labeled(y, missing_label=self.missing_label_)
        X_labeled = X[is_lbld]
        y_labeled = y[is_lbld]
        estimator_params = dict(fit_kwargs) if fit_kwargs is not None else {}

        if (
            has_fit_parameter(self.estimator_, "sample_weight")
            and sample_weight is not None
        ):
            sample_weight_labeled = sample_weight[is_lbld]
            estimator_params["sample_weight"] = sample_weight_labeled

        if np.sum(is_lbld) != 0:
            self._label_mean = np.mean(y[is_lbld])
            self._label_std = np.std(y[is_lbld]) if np.sum(is_lbld) > 1 else 1
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

    def predict(self, X, **predict_kwargs):
        """Return label predictions for the input data X.

        Parameters
        ----------
        X :  array-like, shape (n_samples, n_features)
            Input samples.
        predict_kwargs : dict-like
            Further parameters are passed as input to the 'predict' method of
            the 'estimator'. If the estimator could not be fitted, only
            `return_std` is supported as keyword argument.

        Returns
        -------
        y :  array-like, shape (n_samples)
            Predicted labels of the input samples.
        """
        check_is_fitted(self)
        X = check_array(X, **self.check_X_dict_)
        self._check_n_features(X, reset=False)
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

    @_available_if(
        ("sample_y", "sample"), hasattr(metaestimators, "available_if")
    )
    def sample_y(self, X, n_samples=1, random_state=None):
        """Assumes a probabilistic regressor. Samples are drawn from
        a predicted target distribution.

        Parameters
        ----------
        X :  array-like, shape (n_samples_X, n_features)
            Input samples, where the target values are drawn from.
        n_samples: int, optional (default=1)
            Number of random samples to be drawn.
        random_state : int, RandomState instance or None, optional
        (default=None)
            Determines random number generation to randomly draw samples. Pass
            an int for reproducible results across multiple method calls.

        Returns
        -------
        y_samples : ndarray of shape (n_samples_X, n_samples)
            Drawn random target samples.
        """
        check_is_fitted(self)
        if hasattr(self.estimator_, "sample_y"):
            return self.estimator_.sample_y(X, n_samples, random_state)
        else:
            return self.estimator_.sample(X, n_samples)

    def __getattr__(self, item):
        if "estimator_" in self.__dict__:
            return getattr(self.estimator_, item)
        else:
            return getattr(self.estimator, item)


class SklearnNormalRegressor(ProbabilisticRegressor, SklearnRegressor):
    """SklearnNormalRegressor

    Implementation of a wrapper class for scikit-learn probabilistic regressors
    such that missing labels can be handled and the target distribution can be
    estimated. Therefore, samples with missing values are filtered and a normal
    distribution is fitted to the predicted standard deviation.

    The wrapped regressor of sklearn needs `return_std` as a key_word argument
    for `predict`.

    Parameters
    ----------
    estimator : sklearn.base.RegressorMixin with predict method
        scikit-learn regressor.
    missing_label : scalar or string or np.nan or None, default=np.nan
        Value to represent a missing label.
    random_state : int or RandomState instance or None, default=None
        Determines random number for 'predict' method. Pass an int for
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
        X :  array-like, shape (n_samples, n_features)
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

        X = check_array(X)
        loc, scale = SklearnRegressor.predict(self, X, return_std=True)
        return norm(loc=loc, scale=scale)
