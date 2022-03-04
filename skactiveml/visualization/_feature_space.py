import warnings

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from sklearn.base import ClassifierMixin
from sklearn.neighbors import KNeighborsRegressor
from sklearn.utils import check_array

from ._auxiliary_functions import mesh, check_bound, _get_boundary_args, \
    _get_confidence_args, _get_contour_args, _get_cmap
from ..base import QueryStrategy
from ..utils import check_scalar
from ..utils._validation import check_type, check_indices


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
    ax: matplotlib.axes.Axes or List, optional (default=None)
        The axis on which the decision boundary is plotted. If ax is a List,
        each entry has to be an `matplotlib.axes.Axes`.
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
    ax: matplotlib.axes.Axes or List
        The axis on which the boundary was plotted or the list of axis if ax
        was a list.
    """
    check_type(clf, 'clf', ClassifierMixin)
    check_scalar(res, 'res', int, min_val=1)
    if ax is None:
        ax = plt.gca()
    check_type(ax, 'ax', Axes, list)
    if isinstance(ax, list):
        for ax_item in ax:
            check_type(ax_item, 'one item of ax', Axes)
        axs = ax
    else:
        axs = [ax, ]
    feature_bound = check_bound(bound=feature_bound)

    # Check and convert the colormap
    cmap = _get_cmap(cmap)

    if confidence is not None:
        check_scalar(confidence, 'confidence', float, min_inclusive=False,
                     max_inclusive=False, min_val=0.5, max_val=1)

    # Update additional arguments
    boundary_args = _get_boundary_args(boundary_dict)
    confidence_args = _get_confidence_args(confidence_dict)

    # Create mesh for plotting
    X_mesh, Y_mesh, mesh_instances = mesh(feature_bound, res)

    # Calculate predictions
    if hasattr(clf, 'predict_proba'):
        predictions = clf.predict_proba(mesh_instances)
        classes = np.arange(predictions.shape[1])
    elif hasattr(clf, 'predict'):
        if confidence is not None:
            warnings.warn("The given classifier does not implement "
                          "'predict_proba'. Thus, the confidence cannot be "
                          "plotted.")
            confidence = None
        predicted_classes = clf.predict(mesh_instances)
        classes = np.arange(len(np.unique(predicted_classes)))
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
        for ax_item in axs:
            ax_item.contour(X_mesh, Y_mesh, posteriors, [.5], **boundary_args)
        if confidence is not None:
            for ax_item in axs:
                ax_item.contour(X_mesh, Y_mesh, posteriors, [confidence],
                                colors=[cmap(norm(y))], **confidence_args)
    return ax


def plot_utility(qs, X, y, candidates=None, qs_dict=None, feature_bound=None,
                 ax=None, res=21, contour_dict=None):
    """ Plot the utility for the given query strategy.

    Parameters
    ----------
    qs: QueryStrategy
        The query strategy for which the utility is plotted.
    X : array-like of shape (n_samples, n_features)
        Training data set, usually complete, i.e. including the labeled and
        unlabeled samples.
    y : array-like of shape (n_samples)
        Labels of the training data set (possibly including unlabeled ones
        indicated by self.MISSING_LABEL.
    qs_dict: dict
        Dictionary with the parameters for the qs.query method.
    TODO: introduce flag enabling call_func
    candidates: array-like, shape(n_candidates, n_features)
        Unlabeled candidate instances. If `candidates` is not `None`, the
        utility is calculated only for the selected instances and is
        interpolated. Otherwise, the utility is calculated for every point in
        the given area.
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
    check_type(qs, 'qs', QueryStrategy)
    if qs_dict is None:
        qs_dict = {}
    check_type(qs_dict, 'qs_dict', dict)
    if 'candidates' in qs_dict.keys():
        raise ValueError("'candidates' must be given as separate argument.")
    for key, value in [('X', X), ('y', y)]:
        qs_dict[key] = value

    if candidates is None:
        X_cand = None
    else:
        candidates = check_array(candidates, allow_nd=True, ensure_2d=False,
                                 force_all_finite='allow-nan')
        if candidates.ndim == 2:
            X_cand = candidates
        else:
            check_indices(candidates, X)
            X_cand = X[candidates]

    if X_cand is None:
        X_check = X
    else:
        X_check = np.append(X, X_cand, axis=0)

    feature_bound = check_bound(bound=feature_bound, X=X_check)

    if ax is None:
        ax = plt.gca()
    check_type(ax, 'ax', Axes)
    check_scalar(res, 'res', int, min_val=1)

    X_mesh, Y_mesh, mesh_instances = mesh(feature_bound, res)

    contour_args = _get_contour_args(contour_dict)

    if X_cand is None:
        _, utilities = qs.query(candidates=mesh_instances, **qs_dict,
                                return_utilities=True)
        utilities = utilities.reshape(X_mesh.shape)
        ax.contourf(X_mesh, Y_mesh, utilities, **contour_args)
    else:
        _, utilities = qs.query(candidates=candidates, **qs_dict,
                                return_utilities=True)

        # if utilities refers to `X`
        if candidates.ndim == 1:
            utilities = utilities[0, candidates]
        utilities = utilities.reshape(-1)
        neighbors = KNeighborsRegressor(n_neighbors=1)
        neighbors.fit(X_cand, utilities)
        scores = neighbors.predict(mesh_instances).reshape(X_mesh.shape)
        ax.contourf(X_mesh, Y_mesh, scores, **contour_args)

    return ax


def plot_nearest_candidate_utility(X_cand, utilities_cand, feature_bound=None,
                                   ax=None, res=21, contour_dict=None):
    pass


def plot_utility(qs, X, y, candidates=None, qs_dict=None, feature_bound=None,
                 ax=None, res=21, contour_dict=None):
    pass
