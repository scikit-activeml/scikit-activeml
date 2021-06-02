import numpy as np


def minimum_aggregated_cost(n_acquired_labels, error, annotation_cost,
                            error_cost):
    """Calculate the minimum aggregated cost.

    Parameters
    ----------
    n_acquired_labels: array-like, shape (n_iterations) | int
        Number of acquired labels.
    error: array-like, shape (n_iterations) | float
        Number of errors with n acquired labels.
    annotation_cost: array-like, shape (n_costs) | float
        Cost for annotating a label or a list of different costs.
    error_cost: array-like, shape (n_costs) | float
        Cost for misclassifying an instance or a list of different costs.

    Returns
    -------
    mac: array-like, shape (n_costs) | float
        The minimum aggregated cost for every cost ratio.
    """
    is_scalar = np.isscalar(annotation_cost)
    n_acquired_labels = np.array(n_acquired_labels).reshape(-1, 1)
    error = np.array(error).reshape(-1, 1)
    annotation_cost = np.array(annotation_cost).reshape(1, -1)
    error_cost = np.array(error_cost).reshape(1, -1)

    if n_acquired_labels.shape != error.shape:
        raise ValueError(
            "'n_acquired_labels' and 'error' must have the same shape."
        )
    if annotation_cost.shape != error_cost.shape:
        raise ValueError(
            "'annotation_cost' and 'error_cost' must have the same shape."
        )

    agg_cost = n_acquired_labels * annotation_cost + error * error_cost
    mac = np.min(agg_cost, axis=0)
    if is_scalar:
        mac = mac.item()
    return mac
