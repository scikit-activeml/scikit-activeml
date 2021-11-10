import warnings

import numpy as np
from sklearn.utils import check_array

from skactiveml.utils import check_scalar
from skactiveml.utils._validation import check_bound


def mesh(bound, res):
    """
    Function to get instances of a mesh grid as well as x-mesh and y-mesh
    with given resolution in the specified bounds.

    Parameters
    ----------
    bound: array-like, [[xmin, ymin], [xmax, ymax]]
        The bounds of the mesh grid.
    res: int, optional (default=21)
        The resolution of the plot.

    Returns
    -------
    X_mesh: np.ndarray, shape (res, res)
        mesh grid over x
    Y_mesh: np.ndarray, shape (res, res)
        mesh grid over y
    mesh_instances: np.ndarray, shape (res*res,)
        instances of the mesh grid
    """

    check_scalar(res, 'res', int, min_val=1)
    if bound is None:
        raise TypeError("`bound` must not be `None`")
    check_bound(bound=bound)

    xmin, ymin, xmax, ymax = np.ravel(bound)

    x_vec = np.linspace(xmin, xmax, res)
    y_vec = np.linspace(ymin, ymax, res)
    X_mesh, Y_mesh = np.meshgrid(x_vec, y_vec)
    mesh_instances = np.array([X_mesh.reshape(-1), Y_mesh.reshape(-1)]).T
    return X_mesh, Y_mesh, mesh_instances
