import inspect
import numpy as np
from copy import deepcopy

from scipy.stats import norm
from sklearn.base import MetaEstimatorMixin, is_regressor
from sklearn.utils.metaestimators import _IffHasAttrDescriptor
from sklearn.utils.validation import has_fit_parameter, check_array

from skactiveml.base import SkactivemlRegressor, SkactivemlConditionalEstimator
from skactiveml.utils import check_type
from skactiveml.utils._label import is_labeled
from skactiveml.utils._validation import check_callable


def if_delegate_has_alternative_methods(delegate, *alternative_methods):
    """Create a decorator for methods that are delegated to alternative methods
     of a sub-estimator

    This enables ducktyping by hasattr returning True according to the
    sub-estimator. By
    Parameters
    ----------
    delegate : str, list of str or tuple of str
        Name of the sub-estimator that can be accessed as an attribute of the
        base object. If a list or a tuple of names are provided, the first
        sub-estimator that is an attribute of the base object will be used.
    alternative_methods : iterable of str
        Names of the alternative methods.
    """
    if isinstance(delegate, list):
        delegate = tuple(delegate)
    if not isinstance(delegate, tuple):
        delegate = (delegate,)

    return lambda fn: all(
        _IffHasAttrDescriptor(fn, delegate, attribute_name=method_name)
        for method_name in alternative_methods
    )


class SklearnRegressor(SkactivemlRegressor, MetaEstimatorMixin):
    """SklearnRegressor

    Implementation of a wrapper class for scikit-learn regressors such that
    missing labels can be handled and multiple labels per sample. Therefore,
    samples with missing values are filtered and one output regressor are
    wrapped by a multi output regressor.

    """

    def __init__(self, estimator, random_state=None):
        super().__init__(random_state=random_state)
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
                "'{}' must be a scikit-learn " "regressor.".format(self.estimator)
            )

        self.estimator_ = deepcopy(self.estimator)

        labeled_indices = is_labeled(y)
        X_labeled = X[labeled_indices]
        y_labeled = y[labeled_indices]
        estimator_parameters = dict(fit_kwargs) if fit_kwargs is not None else {}

        if (
            has_fit_parameter(self.estimator_, "sample_weight")
            and sample_weight is not None
        ):
            sample_weight_labeled = sample_weight[labeled_indices]
            estimator_parameters["sample_weight"] = sample_weight_labeled

        if np.sum(labeled_indices) != 0:
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

    @if_delegate_has_alternative_methods("est", "sample_y", "sample")
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
        if hasattr(self.estimator_, "sample_y"):
            return self.estimator_.sample_y(X, n_samples, random_state)
        else:
            return self.estimator_.sample(X, n_samples)

    def __getattr__(self, item):
        if "estimator_" in self.__dict__:
            return getattr(self.estimator_, item)
        else:
            return getattr(self.estimator, item)


class SklearnConditionalEstimator(SkactivemlConditionalEstimator, SklearnRegressor):
    """SklearnConditionalEstimator

    Implementation of a wrapper class for scikit-learn conditional estimators
    such that missing labels can be handled and the conditional distribution
    can be estimated. Therefore, samples with missing values are filtered and
    a normal distribution is fitted to the predicted standard deviation.

    The wrapped regressor of sklearn needs `return_std` as a key_word argument
    for `predict`.

    """

    def __init__(self, estimator, random_state=None, std=None):
        super(SklearnConditionalEstimator, self).__init__(estimator, random_state)
        self.std = std

    def estimate_conditional_distribution(self, X):
        """Returns the estimated target distribution conditioned on the test
        samples `X`.


        Parameters
        ----------
        X :  array-like, shape (n_samples, n_features)
            Input samples.
        Returns
        -------
        dist : scipy.stats.rv_continuous

        """
        if not hasattr(self, "estimator_"):
            if not is_regressor(estimator=self.estimator):
                raise TypeError(
                    f"`{self.estimator}` must be a scikit-learn " "regressor."
                )

            self.estimator_ = deepcopy(self.estimator)

        check_type(self.std, f"{self.std}", float, int, None)

        if callable(self.std):
            check_callable(self.std, "self.var")
            return self.estimator_.predict(X), self.std(X)

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
        if self.std is not None:
            scale = np.sqrt(self.std**2 + scale)
        return norm(loc=loc, scale=scale)
