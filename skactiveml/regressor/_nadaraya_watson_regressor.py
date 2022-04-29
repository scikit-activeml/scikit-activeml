from skactiveml.base import SkactivemlRegressor
from skactiveml.regressor._nic_kernel_regressor import NICKernelRegressor
from skactiveml.utils import MISSING_LABEL


class NadarayaWatsonRegressor(SkactivemlRegressor):
    """NadarayaWatsonRegressor

    The Nadaraya Watson Regressor predicts the target value by taking a weighted
    average based on a kernel.

    Parameters
    __________
    metric : str or callable, default='rbf'
        The metric must a be a valid kernel defined by the function
        `sklearn.metrics.pairwise.pairwise_kernels`.
    metric_dict : dict, optional (default=None)
        Any further parameters are passed directly to the kernel function.
    missing_label : scalar, string, np.nan, or None, default=np.nan
        Value to represent a missing label.
    random_state : int, RandomState instance or None, optional (default=None)
        Determines random number for 'predict' method. Pass an int for
        reproducible results across multiple method calls.
    """

    def __init__(
        self,
        metric="rbf",
        metric_dict=None,
        missing_label=MISSING_LABEL,
        random_state=None,
    ):
        super().__init__(random_state=random_state)
        self.nichkr = NICKernelRegressor(
            metric=metric,
            metric_dict=metric_dict,
            kappa_0=0,
            nu_0=3,
            sigma_sq_0=1,
            missing_label=missing_label,
            random_state=None,
        )

    def fit(self, X, y, sample_weight=None):
        """Fit the model using X as training data and y as class labels.

        Parameters
        ----------
        X : matrix-like, shape (n_samples, n_features)
            The sample matrix X is the feature matrix representing the samples.
        y : array-like, shape (n_samples) or (n_samples, n_targets)
            Contains the values of the samples, where
            missing values are represented the attribute 'np.nan'.
        sample_weight : array-like, shape (n_samples)
            It contains the weights of the training samples' values.

        Returns
        -------
        self: SkactivemlRegressor,
            The SkactivemlRegressor is fitted on the training data.
        """
        return self.nichkr.fit(X, y, sample_weight=sample_weight)

    def predict(self, X):
        """Return value predictions for the test samples X.

        Parameters
        ----------
        X :  array-like, shape (n_samples, n_features)
            Input samples.
        Returns
        -------
        y : numpy.ndarray, shape (n_samples)
            Predicted values of the test samples 'X'.
        """
        return self.nichkr.predict(X)
