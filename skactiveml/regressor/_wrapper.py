import inspect
from copy import deepcopy
from operator import attrgetter

import numpy as np
from scipy.stats import norm
from sklearn.base import MetaEstimatorMixin, is_regressor
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
    missing labels can be handled and multiple labels per sample. Therefore,
    samples with missing values are filtered and one output regressor are
    wrapped by a multi output regressor.

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
        super().__init__(random_state=random_state, missing_label=missing_label)
        self.estimator = estimator

    def fit(self, X, y, sample_weight=None, **fit_kwargs):
        """Fit the model using X as training data and y as class labels.

        Parameters
        ----------
        X : matrix-like, shape (n_samples, n_features)
            The sample matrix X is the feature matrix representing the samples.
        y : array-like, shape (n_samples) or (n_samples, n_outputs)
            It contains the values of the training samples.
            Missing labels are represented as 'np.nan'.
        sample_weight : array-like, shape (n_samples) or (n_samples, n_outputs)
            It contains the weights of the training samples' class labels. It
            must have the same shape as y.
        fit_kwargs : dict-like
            Further parameters as input to the 'fit' method of the 'estimator'.

        Returns
        -------
        self: SklearnRegressor,
            The SklearnRegressor is fitted on the training data.
        """

        if not is_regressor(estimator=self.estimator):
            raise TypeError(
                "'{}' must be a scikit-learn "
                "regressor.".format(self.estimator)
            )

        self.estimator_ = deepcopy(self.estimator)

        X, y, sample_weight = self._validate_data(X, y, sample_weight)

        is_lbld = is_labeled(y, missing_label=self.missing_label_)
        X_labeled = X[is_lbld]
        y_labeled = y[is_lbld]
        estimator_parameters = (
            dict(fit_kwargs) if fit_kwargs is not None else {}
        )

        if (
            has_fit_parameter(self.estimator_, "sample_weight")
            and sample_weight is not None
        ):
            sample_weight_labeled = sample_weight[is_lbld]
            estimator_parameters["sample_weight"] = sample_weight_labeled

        if np.sum(is_lbld) != 0:
            self.estimator_.fit(X_labeled, y_labeled, **estimator_parameters)

        return self

    def predict(self, X, **predict_kwargs):
        """Return label predictions for the input data X.

        Parameters
        ----------
        X :  array-like, shape (n_samples, n_features)
            Input samples.
        predict_kwargs : dict-like
            Further parameters as input to the 'predict' method of the
            'estimator'.

        Returns
        -------
        y :  array-like, shape (n_samples) or (n_samples, n_targets)
            Predicted labels of the input samples.
        """
        return self.estimator_.predict(X, **predict_kwargs)

    @_available_if(
        ("sample_y", "sample"), hasattr(metaestimators, "available_if")
    )
    def sample_y(self, X, n_samples, random_state=None):
        """Assumes a conditional probability estimator. Samples are drawn from
        the posterior or prior conditional probability estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples_X, n_features) or list of object
            Query points where the GP is evaluated.
        n_samples : int, default=1
            Number of samples drawn from the Gaussian process per query point.
        random_state : int, RandomState instance or None, default=0
            Determines random number generation to randomly draw samples.
            Pass an int for reproducible results across multiple function
            calls.

        Returns
        -------
        y_samples : ndarray of shape (n_samples_X, n_samples), or \
            (n_samples_X, n_targets, n_samples)
            Values of n_samples samples drawn from Gaussian process and
            evaluated at query points.
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


class SklearnProbabilisticRegressor(ProbabilisticRegressor, SklearnRegressor):
    """SklearnProbabilisticRegressor

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
        super(SklearnProbabilisticRegressor, self).__init__(
            estimator, missing_label=missing_label, random_state=random_state
        )

    def predict_target_distribution(self, X):
        """Returns the estimated target distribution conditioned on the test
        samples `X`.

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
        loc, scale = self.estimator_.predict(X, return_std=True)
        return norm(loc=loc, scale=scale)
