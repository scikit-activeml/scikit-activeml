from skactiveml.regressor._nic_kernel_regressor import NICKernelRegressor
from skactiveml.utils import MISSING_LABEL


class NadarayaWatsonRegressor(NICKernelRegressor):
    """NadarayaWatsonRegressor

    The Nadaraya Watson Regressor predicts the target value by taking a weighted
    average based on a kernel. It is implemented asa `NICKernelRegressor` with
    different prior values.

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
        super().__init__(
            random_state=random_state,
            missing_label=missing_label,
            metric=metric,
            metric_dict=metric_dict,
            kappa_0=0,
            nu_0=3,
            sigma_sq_0=1,
        )
