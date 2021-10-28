import warnings

import numpy as np

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from sklearn.base import ClassifierMixin
from sklearn.neighbors import KNeighborsRegressor
from sklearn.utils import check_array

from ..base import QueryStrategy
from ..utils import check_scalar


def plot_decision_boundary(clf, feature_bound, ax=None, res=21,
                           boundary_dict=None, confidence=0.75,
                           cmap='coolwarm', confidence_dict=None):
    """Plot the decision boundary of the given classifier.

    Parameters
    ----------
    clf: Sklearn classifier
        The fitted classifier whose decision boundary is plotted. If confidence
        is not None, the classifier must implement the predict_proba function.
    feature_bound: array-like, [[xmin, ymin], [xmax, ymax]]
        Determines the area in which the boundary is plotted.
    ax: matplotlib.axes.Axes, optional (default=None)
        The axis on which the decision boundary is plotted.
    res: int, optional (default=21)
        The resolution of the plot.
    boundary_dict: dict, optional (default=None)
        Additional parameters for the boundary contour.
    confidence: scalar | None, optional (default=0.5)
        The confidence interval plotted with dashed lines. It is not plotted if
        confidence is None. Must be in the open interval (0.5, 1). The value
        stands for the ratio best class / second best class.
    cmap: str | matplotlib.colors.Colormap, optional (default='coolwarm_r')
        The colormap for the confidence levels.
    confidence_dict: dict, optional (default=None)
        Additional parameters for the confidence contour. Must not contain a
        colormap because cmap is used.

    Returns
    -------
    matplotlib.axes.Axes: The axis on which the boundary was plotted.
    """
    if not isinstance(clf, ClassifierMixin):
        raise TypeError("'clf' must be an Sklearn classifier.")
    check_scalar(res, 'res', int, min_val=1)
    if ax is None:
        ax = plt.gca()
    if not isinstance(ax, Axes):
        raise TypeError("ax must be a matplotlib.axes.Axes.")
    check_array(feature_bound)
    xmin, ymin, xmax, ymax = np.ravel(feature_bound)

    # Check and convert the colormap
    if isinstance(cmap, str):
        cmap = plt.cm.get_cmap(cmap)
    if not isinstance(cmap, Colormap):
        raise TypeError("'cmap' must be a string or a Colormap.")

    if confidence is not None:
        check_scalar(confidence, 'confidence', float, min_inclusive=False,
                     max_inclusive=False, min_val=0.5, max_val=1)

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

    # Create mesh for plotting
    x_vec = np.linspace(xmin, xmax, res)
    y_vec = np.linspace(ymin, ymax, res)
    X_mesh, Y_mesh = np.meshgrid(x_vec, y_vec)
    mesh_instances = np.array([X_mesh.reshape(-1), Y_mesh.reshape(-1)]).T

    # Calculate predictions
    if hasattr(clf, 'predict_proba'):
        predictions = clf.predict_proba(mesh_instances)
        classes = np.array(range(predictions.shape[1]))
    elif hasattr(clf, 'predict'):
        if confidence is not None:
            warnings.warn("The given classifier does not implement "
                          "'predict_proba'. Thus, the confidence cannot be "
                          "plotted.")
            confidence = None
        predicted_classes = clf.predict(mesh_instances)
        classes = np.array(range(len(np.unique(predicted_classes))))
        predictions = np.zeros((len(predicted_classes), len(classes)))
        for idx, y in enumerate(predicted_classes):
            predictions[idx, y] = 1
    else:
        raise AttributeError("'clf' must implement 'predict' or "
                             "'predict_proba'")

    posterior_list = []

    for y in classes:
        posteriors = predictions[:, y].reshape(X_mesh.shape)
        posterior_list.append(posteriors)

    norm = plt.Normalize(vmin=min(classes), vmax=max(classes))

    for y in classes:
        posteriors = posterior_list[y]
        posteriors_best_alternative = np.zeros_like(posteriors)
        for y2 in np.setdiff1d(classes, [y]):
            posteriors_best_alternative = np.max([posteriors_best_alternative,
                                                  posterior_list[y2]], axis=0)

        posteriors = posteriors / (posteriors + posteriors_best_alternative)
        ax.contour(X_mesh, Y_mesh, posteriors, [.5], **boundary_args)
        if confidence is not None:
            ax.contour(X_mesh, Y_mesh, posteriors, [confidence],
                       colors=[cmap(norm(y))], **confidence_args)

    return ax


def plot_utility(qs, qs_dict, X_cand=None, feature_bound=None, ax=None, res=21,
                 contour_dict=None):
    """ Plot the utility for the given query strategy.

    Parameters
    ----------
    qs: QueryStrategy
        The query strategy for which the utility is plotted.
    qs_dict: dict
        Dictionary with the parameters for the qs.query method.
    X_cand: array-like, shape(n_candidates, n_features)
        Unlabeled candidate instances. If X_cand is given, the utility is
        calculated only for these instances and is interpolated. Otherwise, the
        utility is calculated for every point in the given area.
    feature_bound: array-like, [[xmin, ymin], [xmax, ymax]]
        Determines the area in which the boundary is plotted. If X_cand is not
        given, bound must not be None. Otherwise, the bound is determined based
        on the data.
    ax: matplotlib.axes.Axes, optional (default=None)
        The axis on which the utility is plotted.
    res: int, optional (default=21)
        The resolution of the plot.
    contour_dict: dict, optional (default=None)
        Additional parameters for the utility contour.

    Returns
    -------
    matplotlib.axes.Axes: The axis on which the utility was plotted.
    """
    if not isinstance(qs, QueryStrategy):
        raise TypeError("'qs' must be a query strategy.")
    if not isinstance(qs_dict, dict):
        raise TypeError("'qs_dict' must be a dictionary.")
    if 'X_cand' in qs_dict.keys():
        raise ValueError("'X_cand' must be given as separate argument.")

    if feature_bound is not None:
        check_array(feature_bound)
        xmin, ymin, xmax, ymax = np.ravel(feature_bound)
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

    return ax
