import numpy as np

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from sklearn.neighbors import KNeighborsRegressor
from sklearn.utils import check_array

from skactiveml.base import QueryStrategy, SkactivemlClassifier
from skactiveml.utils import check_scalar


def plot_decision_boundary(clf, bound, res=21, ax=None, confidence=0.75,
                           cmap='coolwarm_r', boundary_dict=None,
                           confidence_dict=None):
    """Plot the decision boundary of the given classifier.

    Parameters
    ----------
    clf: sklearn classifier # TODO correct?
        The classifier whose decision boundary is plotted.
    bound: array-like, [[xmin, ymin], [xmax, ymax]]
        Determines the area in which the boundary is plotted.
    res: int, optional (default=21)
        The resolution of the plot.
    ax: matplotlib.axes.Axes, optional (default=None)
        The axis on which the boundary is plotted.
    confidence: scalar | None, optional (default=0.5)
        The confidence interval plotted with dashed lines. It is not plotted if
        confidence is None. Must be in the open interval (0.5, 1). The value
        stands for the ratio best class / second best class.
    cmap: str | matplotlib.colors.Colormap, optional (default='coolwarm_r')
        The colormap for the confidence levels.
    boundary_dict: dict, optional (default=None)
        Additional parameters for the boundary contour.
    confidence_dict: dict, optional (default=None)
        Additional parameters for the confidence contour. Must not contain a
        colormap because cmap is used.
    """
    # TODO which type?
    if not isinstance(clf, SkactivemlClassifier):
        raise TypeError("'clf' must be a SkactivemlClassifier.")
    check_scalar(res, 'res', int, min_val=1)
    if ax is None:
        ax = plt.gca()
    if not isinstance(ax, Axes):
        raise TypeError("ax must be a matplotlib.axes.Axes.")
    check_array(bound)
    xmin, ymin, xmax, ymax = np.ravel(bound)

    # Check and convert the colormap
    if isinstance(cmap, str):
        cmap = plt.cm.get_cmap(cmap)
    if not isinstance(cmap, Colormap):
        raise TypeError("'cmap' must be a string or a ColorMap.")

    check_scalar(confidence, 'confidence', float, min_inclusive=False,
                 max_inclusive=False, min_val=0.5, max_val=1)

    # Create mesh for plotting
    x_vec = np.linspace(xmin, xmax, res)
    y_vec = np.linspace(ymin, ymax, res)
    X_mesh, Y_mesh = np.meshgrid(x_vec, y_vec)
    mesh_instances = np.array([X_mesh.reshape(-1), Y_mesh.reshape(-1)]).T

    # Update additional arguments
    boundary_args = {'colors': 'k', 'linewidths': [2], 'zorder': 1}
    if boundary_dict is not None:
        if not isinstance(boundary_dict, dict):
            raise TypeError("boundary_dict' must be a dictionary.")
        boundary_args.update(boundary_dict)
    confidence_args = {'linewidths': [2, 2], 'linestyles': '--', 'alpha': 0.9,
                       'vmin': 0.2, 'vmax': 0.8, 'zorder': 1}
    if confidence_dict is not None:
        if not isinstance(confidence_dict, dict):
            raise TypeError("confidence_dict' must be a dictionary.")
        confidence_args.update(confidence_dict)

    posterior_list = []
    classes = np.array(range(clf.predict_proba(mesh_instances).shape[1]))
    for y in classes:
        posteriors = clf.predict_proba(mesh_instances)[:, y]
        posteriors = posteriors.reshape(X_mesh.shape)
        posterior_list.append(posteriors)

    norm = plt.Normalize(vmin=min(classes), vmax=max(classes))

    for y in classes:
        posteriors = posterior_list[y]
        posteriors_2 = np.zeros_like(posteriors)
        for y2 in np.setdiff1d(classes, [y]):
            posteriors_2 = np.max([posteriors_2, posterior_list[y2]], axis=0)

        posteriors = posteriors / (posteriors + posteriors_2)
        ax.contour(X_mesh, Y_mesh, posteriors, [.5], **boundary_args)
        ax.contour(X_mesh, Y_mesh, posteriors, [.75], colors=[cmap(norm(y))],
                   **confidence_args)


def plot_utility(qs, qs_dict, X_cand=None, bound=None, res=21, ax=None,
                 contour_dict=None):
    """ Plot the utility for the given query strategy.

    Parameters
    ----------
    qs: QueryStrategy
        The query strategy for which the utility is plotted.
    qs_dict: dict
        Dictionary with the parameters for the qs.query method.
    X_cand: array-like, shape(n_candidates, n_features)
        Unlabeled candidate instances.
    bound: array-like, [[xmin, ymin], [xmax, ymax]]
        Determines the area in which the boundary is plotted.
    res: int, optional (default=21)
        The resolution of the plot.
    ax: matplotlib.axes.Axes, optional (default=None)
        The axis on which the boundary is plotted.
    contour_dict: dict, optional (default=None)
        Additional parameters for the utility contour.
    """

    # TODO: dict for contourf

    if not isinstance(qs, QueryStrategy):
        raise TypeError("'qs' must be a query strategy.")
    if not isinstance(qs_dict, dict):
        raise TypeError("'qs_dict' must be a dictionary.")
    if 'X_cand' in qs_dict.keys():
        raise ValueError("'X_cand' must be given as separate argument.")

    if bound is not None:
        check_array(bound)
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

    x_vec = np.linspace(xmin, xmax, res)
    y_vec = np.linspace(ymin, ymax, res)
    X_mesh, Y_mesh = np.meshgrid(x_vec, y_vec)
    mesh_instances = np.array([X_mesh.reshape(-1), Y_mesh.reshape(-1)]).T

    contour_args = {'cmap': 'Greens', 'alpha': 0.75}
    if contour_dict is not None:
        if not isinstance(contour_dict, dict):
            raise TypeError("contour_dict' must be a dictionary.")
        contour_args.update(contour_dict)

    if X_cand is None:
        _, utilities = qs.query(mesh_instances, **qs_dict,
                                return_utilities=True)
        utilities = utilities.reshape(X_mesh.shape)
        ax.contourf(X_mesh, Y_mesh, utilities, **contour_args)
    else:
        _, utilities = qs.query(X_cand, **qs_dict, return_utilities=True)
        utilities = utilities.reshape(-1)
        neighbors = KNeighborsRegressor(n_neighbors=1)
        neighbors.fit(X_cand, utilities)
        scores = neighbors.predict(mesh_instances).reshape(X_mesh.shape)
        ax.contourf(X_mesh, Y_mesh, scores, **contour_args)
