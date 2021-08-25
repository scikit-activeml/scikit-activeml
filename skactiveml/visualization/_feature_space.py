import numpy as np

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from sklearn.neighbors import KNeighborsRegressor

from skactiveml.base import QueryStrategy, SkactivemlClassifier
from skactiveml.utils import check_scalar


def plot_decision_boundary(clf, bound, res=21, ax=None):
    """Plot the decision boundary of the given classifier.

    Parameters
    ----------
    clf: sklearn classifier # TODO correct?
        The classifier whose decision boundary is plotted.
    bound: array-like, [[xmin, ymin], [xmax, ymax]]
        Determines the area in which the boundary is plotted.
    res: int
        The resolution of the plot.
    ax: matplotlib.axes.Axes, optional (default=None)
        The axis on which the boundary is plotted.
    """
    # TODO which bound format sklearn, check bound

    # TODO: extend to multiclass, add parameter confidence [0,1] evtl. [0,0.5], or None
    # TODO: colors per class colormap or list of colors
    # TODO: boundary_dict, confidence_dict

    if not isinstance(clf, SkactivemlClassifier):
        raise TypeError("'clf' must be a SkactivemlClassifier.")
    check_scalar(res, 'res', int, min_val=1)
    if ax is None:
        ax = plt.gca()
    if not isinstance(ax, Axes):
        raise TypeError("ax must be a matplotlib.axes.Axes.")
    xmin, ymin, xmax, ymax = np.ravel(bound)

    # Create mesh for plotting
    x_vec = np.linspace(xmin, xmax, res)
    y_vec = np.linspace(ymin, ymax, res)
    X_mesh, Y_mesh = np.meshgrid(x_vec, y_vec)
    mesh_instances = np.array([X_mesh.reshape(-1), Y_mesh.reshape(-1)]).T

    posteriors = clf.predict_proba(mesh_instances)[:, 0].reshape(X_mesh.shape)

    ax.contour(X_mesh, Y_mesh, posteriors, [.5], colors='k', linewidths=[2],
               zorder=1)
    ax.contour(X_mesh, Y_mesh, posteriors, [.25, .75], cmap='coolwarm_r',
               linewidths=[2, 2], zorder=1, linestyles='--', alpha=.9, vmin=.2,
               vmax=.8)


def plot_utility(qs, qs_dict, X_cand=None, bound=None, res=21, ax=None):
    """
    TODO

    Parameters
    ----------
    qs
    qs_dict
    X_cand
    bound
    res
    ax

    Returns
    -------

    """

    # TODO: dict for contourf

    if not isinstance(qs, QueryStrategy):
        raise TypeError("'qs' must be a query strategy.")
    if not isinstance(qs_dict, dict):
        raise TypeError("'qs_dict' must be a dictionary.")
    if 'X_cand' in qs_dict.keys():
        raise ValueError("'X_cand' must be given as separate argument.")

    if bound is not None:
        xmin, ymin, xmax, ymax = np.ravel(bound)
    elif X_cand is not None:
        xmin = min(X_cand[:, 0])
        xmax = max(X_cand[:, 0])
        ymin = min(X_cand[:, 1])
        ymax = max(X_cand[:, 1])
    else:
        raise ValueError("If 'X_cand' is None, 'bound' must be given.")

    if ax is None:
        ax = plt.gca()
    if not isinstance(ax, Axes):
        raise TypeError("ax must be a matplotlib.axes.Axes.")
    check_scalar(res, 'res', int, min_val=1)

    # TODO check bound

    x_vec = np.linspace(xmin, xmax, res)
    y_vec = np.linspace(ymin, ymax, res)
    X_mesh, Y_mesh = np.meshgrid(x_vec, y_vec)
    mesh_instances = np.array([X_mesh.reshape(-1), Y_mesh.reshape(-1)]).T

    if X_cand is None:
        _, utilities = qs.query(mesh_instances, **qs_dict,
                                return_utilities=True)
        utilities = utilities.reshape(X_mesh.shape)
        ax.contourf(X_mesh, Y_mesh, utilities, cmap='Greens', alpha=.75)
    else:
        _, utilities = qs.query(X_cand, **qs_dict, return_utilities=True)
        utilities = utilities.reshape(-1)
        neighbors = KNeighborsRegressor(n_neighbors=1)
        print(X_cand.shape)
        print(utilities.shape)
        neighbors.fit(X_cand, utilities)
        scores = neighbors.predict(mesh_instances).reshape(X_mesh.shape)
        ax.contourf(X_mesh, Y_mesh, scores, cmap='Greens', alpha=.75)
