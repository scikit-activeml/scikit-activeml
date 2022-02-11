from copy import deepcopy

import numpy as np
from sklearn.base import MetaEstimatorMixin, is_regressor
from sklearn.utils.validation import has_fit_parameter

from skactiveml.base import SkactivemlRegressor
from skactiveml.utils._label import is_all_labeled


class SklearnRegressor(SkactivemlRegressor, MetaEstimatorMixin):
    """SklearnRegressor

    Implementation of a wrapper class for scikit-learn regressors such that
    missing labels can be handled and multiple labels per sample. Therefore,
    samples with missing values are filtered.

    """

    def __init__(self, estimator, random_state=None):
        super().__init__(random_state=random_state)
        self.estimator = estimator
        self.estimator_ = None

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
            raise TypeError("'{}' must be a scikit-learn "
                            "classifier.".format(self.estimator))

        self.estimator_ = deepcopy(self.estimator)

        labeled_indices = is_all_labeled(y)
        X_labeled = X[labeled_indices]
        y_labeled = y[labeled_indices]
        estimator_parameters = {}
        if has_fit_parameter(self.estimator_,
                             'sample_weight') and sample_weight is not None:
            sample_weight_labeled = sample_weight[labeled_indices]
            estimator_parameters['sample_weight'] = sample_weight_labeled
        self.estimator_.fit(X_labeled, y_labeled, **estimator_parameters)

    def predict(self, X):
        return self.estimator_.predict(X)