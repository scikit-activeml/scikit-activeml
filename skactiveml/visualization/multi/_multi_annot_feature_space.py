import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from sklearn.neighbors import KNeighborsRegressor
from sklearn.utils import check_array, check_consistent_length

from .._auxiliary_functions import mesh, check_bound, _get_contour_args, \
    _get_tick_args, _get_legend_args, _get_cmap, _get_figure_for_ma
from .._feature_space import plot_decision_boundary
from ...base import MultiAnnotatorPoolQueryStrategy
from ...utils import is_labeled, check_scalar, check_type


def plot_ma_current_state(X, y, y_true, ma_qs, clf, ma_qs_arg_dict,
                          bound=None, epsilon=1, title=None, fontsize=15,
                          fig_size=None, plot_legend=True, legend_dict=None,
                          contour_dict=None, boundary_dict=None,
                          confidence_dict=None, tick_dict=None,
                          marker_size=10, res=21):
    """Shows the annotations from the different annotators, the decision
    boundary of the given classifier and the utilities expected of querying
    a sample from a given region based on the query strategy.

    Parameters
    ----------
    X : matrix-like of shape (n_samples, 2)
        The sample matrix `X` is the feature matrix representing the samples.
        The feature space must be two-dimensional.
    y : array-like of shape (n_samples, n_annotators)
        It contains the annotated values for each sample.
        The number of class labels may be variable for the samples, where
        missing labels are represented the attribute `missing_label`.
    y_true : array-like of shape (n_samples,)
        The correct labels.
    ma_qs: MultiAnnotatorPoolQueryStrategy
        The multiannotator-annotator query strategy.
    clf: sklearn classifier
        The classifier whose decision boundary is plotted.
    ma_qs_arg_dict: dict
        The argument dictionary for the multiple annotator query strategy.
    bound: array-like of shape [[xmin, ymin], [xmax, ymax]],
    optional (default=None)
        Determines the area in which the boundary is plotted.
    epsilon: float, optional (default=1)
        The minimal distance between the returned bound and the values of `X`,
        if `bound` is not specified.
    title : str, optional
        The title for the figure.
    fontsize: int, optional (default=15)
        The fontsize of the labels.
    fig_size: tuple of shape (width, height), optional (default=None)
        The size of the figure in inches. If `fig_size` is None, the size
        of the figure is set to 8 x 5 inches.
    plot_legend: bool, optional (default=True)
        Whether to plot the legend.
    legend_dict: dict, optional (default=None)
        Additional parameters for the legend.
    contour_dict: dict, optional (default=None)
        Additional parameters for the utility contour.
    boundary_dict: dict, optional (default=None)
        Additional parameters for the boundary contour.
    confidence_dict: dict, optional (default=None)
        Additional parameters for the confidence contour. Must not contain a
        colormap because `cmap` is used.
    tick_dict: dict, optional (default=None):
        Additional parameters for the ticks of the plots.
    marker_size: int, optional (default=10)
        The size of the markers on the plot.
    res: int, optional (default=21)
        The resolution of the plot.

    Returns
    ----------
    fig: matplotlib.figure.Figure
        The figure onto which the current state is plotted.
    """

    bound = check_bound(bound, X, epsilon=epsilon)

    fig = plot_ma_utility(fig_size=fig_size, ma_qs=ma_qs, X=X, y=y,
                          ma_qs_arg_dict=ma_qs_arg_dict, feature_bound=bound,
                          title=title, fontsize=fontsize, res=res,
                          contour_dict=contour_dict, tick_dict=tick_dict)
    plot_ma_data_set(fig=fig, X=X, y=y, y_true=y_true, feature_bound=bound,
                     plot_legend=plot_legend, legend_dict=legend_dict,
                     tick_dict=tick_dict, marker_size=marker_size)
    plot_ma_decision_boundary(clf, fig=fig, feature_bound=bound,
                              boundary_dict=boundary_dict,
                              confidence_dict=confidence_dict,
                              tick_dict=tick_dict)

    return fig


def plot_ma_data_set(X, y, y_true, fig=None, feature_bound=None, title=None,
                     fontsize=15, fig_size=None, plot_legend=True,
                     cmap='coolwarm', legend_dict=None, tick_dict=None,
                     marker_size=10):
    """Plots the annotations of a binary classification problem,
    differentiating between correctly and incorrectly labeled data.

    Parameters
    ----------
    X : array-like of shape (n_samples, 2)
        The sample matrix `X` is the feature matrix representing the samples.
        The feature space must be two-dimensional.
    y_true : array-like of shape (n_samples,)
        The correct labels.
    y : array-like of shape (n_samples, n_annotators)
        It contains the annotated values for each sample.
        The number of class labels may be variable for the samples, where
        missing labels are represented the attribute `missing_label`.
    fig: matplotlib.figure.Figure, optional (default=None)
        The figure to which axes the utilities will be plotted.
    fig_size: tuple of shape (width, height), optional (default=None)
        The size of the figure in inches. If `fig_size` is None, the size
        of the figure is set to 8 x 5 inches.
    feature_bound: array-like of shape [[xmin, ymin], [xmax, ymax]], optional
    (default=None)
        Determines the area in which the boundary is plotted.
    title : str, optional (default=none)
        The title for the figure.
    fontsize: int, optional (default=15)
        The fontsize of the labels.
    plot_legend: bool, optional (default=True)
        Whether to plot the legend.
    legend_dict: dict, optional (default=None)
        Additional parameters for the legend.
    tick_dict: dict, optional (default=None)
        Additional parameters for the ticks.
    cmap: str | matplotlib.colors.Colormap, optional (default='coolwarm_r')
        The colormap for the confidence levels.
    marker_size: int, optional (default=15)
        The size of the markers on the plot.

    Returns
    -------
    fig: matplotlib.figure.Figure
        The figure onto which the data set is plotted
    """

    # check input values
    X = check_array(X)
    if X.shape[1] != 2:
        raise ValueError(f"`X` along axis 1 must be of length two."
                         f"`X` along axis 1 is of length {X.shape[1]}.")

    y_true = check_array(y_true, ensure_2d=False)
    if y_true.ndim != 1:
        raise ValueError(f"`y_true` must be one dimensional."
                         f"`y_true` is {y_true.ndim} dimensional.")
    check_consistent_length(X, y_true)

    y = check_array(y, force_all_finite='allow-nan')
    check_consistent_length(y_true, y)

    n_annotators = y.shape[1]
    legend_args = _get_legend_args(legend_dict, fontsize)
    tick_args = _get_tick_args(tick_dict)
    fig = _get_figure_for_ma(fig, fig_size=fig_size, title=title,
                             fontsize=fontsize, n_annotators=n_annotators,
                             tick_args=tick_args)
    feature_bound = check_bound(feature_bound, X)
    check_scalar(plot_legend, 'plot_legend', bool)
    cmap = _get_cmap(cmap)
    labeled_indices = is_labeled(y)

    n_classes = len(np.unique(np.append(y[labeled_indices].flatten(), y_true,
                                        axis=0)))
    classes = np.arange(n_classes)

    norm = plt.Normalize(vmin=min(classes), vmax=max(classes))

    # plot data set
    for a, ax in enumerate(fig.get_axes()):
        ax.scatter(X[~labeled_indices[:, a], 0], X[~labeled_indices[:, a], 1],
                   c='gray', marker='o', s=marker_size)
        ax.set_xlim(feature_bound[:, 0])
        ax.set_ylim(feature_bound[:, 1])

        for cl, color in zip(classes, cmap(norm(classes))):
            for is_true, marker in zip([False, True], ['x', 's']):

                cl_is_true = np.logical_xor(y_true != cl, is_true)
                cl_current = np.logical_and(y[:, a] == cl, cl_is_true)

                cl_labeled = np.logical_and(cl_current, labeled_indices[:, a])

                ax.scatter(X[cl_labeled, 0], X[cl_labeled, 1], color=color,
                           marker=marker, s=marker_size)

    patch = Line2D([0], [0], marker='o', markerfacecolor='grey',
                   markersize=20, alpha=0.8, color='w')
    true_patches = (Line2D([0], [0], marker='s', markerfacecolor='b',
                           markersize=15, color='w'),
                    Line2D([0], [0], marker='s', markerfacecolor='r',
                           markersize=15, color='w'))
    false_patches = (Line2D([0], [0], marker='x', markerfacecolor='b',
                            markeredgecolor='b', markersize=15, color='w'),
                     Line2D([0], [0], marker='x', markerfacecolor='r',
                            markeredgecolor='r', markersize=15, color='w'))

    handles = [true_patches, false_patches]
    labels = ['true annotation', 'false annotation']

    if not(np.all(is_labeled(y))):
        handles = [patch, ] + handles
        labels = ['not acquired annotation', ] + labels

    if plot_legend:
        fig.legend(handles, labels, **legend_args)

    return fig


def plot_ma_utility(ma_qs, X, y, candidates=None, annotators=None,
                    ma_qs_arg_dict=None, fig=None, fig_size=None,
                    feature_bound=None, title=None, res=21,
                    fontsize=15, contour_dict=None, tick_dict=None):
    """Plots the utilities for the different annotators of the given
    multiannotator-annotator query strategy.

    Parameters
    ----------
    ma_qs: MultiAnnotatorPoolQueryStrategy
        The multiannotator-annotator query strategy.
    X : array-like of shape (n_samples, n_features)
        Training data set, usually complete, i.e., including the labeled and
        unlabeled samples.
    y : array-like of shape (n_samples, n_annotators)
        Labels of the training data set for each annotator (possibly
        including unlabeled ones indicated by self.MISSING_LABEL), meaning
        that `y[i, j]` contains the label annotated by annotator `i` for
        sample `j`.
    candidates : None or array-like of shape (n_candidates), dtype=int or
        array-like of shape (n_candidates, n_features),
        optional (default=None)
        If `candidates` is None, the samples from (X,y), for which an
        annotator exists such that the annotator sample pairs is
        unlabeled are considered as sample candidates.
        If `candidates` is of shape (n_candidates) and of type int,
        candidates is considered as the indices of the sample candidates in
        (X,y).
        If `candidates` is of shape (n_candidates, n_features), the
        sample candidates are directly given in candidates (not necessarily
        contained in X). This is not supported by all query strategies.
    annotators : array-like of shape (n_candidates, n_annotators), optional
    (default=None)
        If `annotators` is None, all annotators are considered as available
        annotators.
        If `annotators` is of shape (n_avl_annotators) and of type int,
        `annotators` is considered as the indices of the available
        annotators.
        If candidate samples and available annotators are specified:
        The annotator sample pairs, for which the sample is a candidate
        sample and the annotator is an available annotator are considered as
        candidate annotator sample pairs.
        If `annotators` is a boolean array of shape (n_candidates,
        n_avl_annotators) the annotator sample pairs, for which the sample
        is a candidate sample and the boolean matrix has entry `True` are
        considered as candidate sample pairs.
    ma_qs_arg_dict: dict, optional (default=None)
        The argument dictionary for the multiple annotator query strategy.
    fig: matplotlib.figure.Figure, optional (default=None)
        The figure to which axes the utilities will be plotted
    fig_size: tuple, shape (width, height) (default=None)
        The size of the figure in inches. If `fig_size` is None, the size
        of the figure is set to 8 x 5 inches.
    feature_bound: array-like of shape [[xmin, ymin], [xmax, ymax]], optional
    (default=None)
        Determines the area in which the boundary is plotted.
    title : str, optional (default=None)
        The title for the figure.
    res: int, optional (default=21)
        The resolution of the plot.
    fontsize: int, optional (default=15)
        The fontsize of the labels.
    contour_dict: dict, optional (default=None)
        Additional parameters for the utility contour.
    tick_dict: dict, optional (default=None)
        Additional parameters for the ticks.

    Returns
    -------
    fig: matplotlib.figure.Figure
        The figure onto which the utilities are plotted
    """

    # check arguments
    check_type(ma_qs, 'ma_qs', MultiAnnotatorPoolQueryStrategy)
    check_type(ma_qs_arg_dict, 'ma_qs_arg_dict', dict)
    for var in ['candidates', 'annotators', 'X', 'y']:
        if 'candidates' in ma_qs_arg_dict.keys():
            raise ValueError(f"'{var}' must be given as a separate argument.")

    y = check_array(y, force_all_finite='allow-nan')

    n_annotators = len(y.T)

    feature_bound = check_bound(feature_bound, X)
    contour_args = _get_contour_args(contour_dict)
    tick_args = _get_tick_args(tick_dict)
    fig = _get_figure_for_ma(fig, fig_size=fig_size, title=title,
                             fontsize=fontsize, n_annotators=n_annotators,
                             tick_args=tick_args)

    # plot the utilities
    X_mesh, Y_mesh, mesh_instances = mesh(feature_bound, res)

    if candidates is None:
        _, utilities = ma_qs.query(X=X, y=y, candidates=mesh_instances,
                                   annotators=annotators, **ma_qs_arg_dict,
                                   return_utilities=True,
                                   batch_size=1)

        for a, ax in enumerate(fig.get_axes()):
            a_utilities = utilities[0, :, a]
            a_utilities_mesh = a_utilities.reshape(X_mesh.shape)
            ax.contourf(X_mesh, Y_mesh, a_utilities_mesh, **contour_args)
    else:
        _, utilities = ma_qs.query(X=X, y=y, candidates=candidates,
                                   annotators=annotators, **ma_qs_arg_dict,
                                   return_utilities=True, batch_size=1)
        if candidates.ndim == 1:
            X_cand = X[candidates]
            utilities = utilities[:, candidates, :]
        else:
            X_cand = candidates

        for a, ax in enumerate(fig.get_axes()):
            utilities_a = utilities[0, :, a]
            neighbors = KNeighborsRegressor(n_neighbors=1)
            neighbors.fit(X_cand, utilities_a)
            scores = neighbors.predict(mesh_instances).reshape(X_mesh.shape)
            ax.contourf(X_mesh, Y_mesh, scores, **contour_args)

    return fig


def plot_ma_decision_boundary(clf, feature_bound, n_annotators=None, fig=None,
                              boundary_dict=None, confidence=0.75, title=None,
                              res=21, fig_size=None, fontsize=15,
                              cmap='coolwarm', confidence_dict=None,
                              tick_dict=None):
    """Plot the decision boundary of the given classifier for each annotator.

    Parameters
    ----------
    clf: sklearn classifier
        The classifier whose decision boundary is plotted.
    feature_bound: array-like of shape [[xmin, ymin], [xmax, ymax]]
        Determines the area in which the boundary is plotted.
    n_annotators: int, optional (default=None)
        The number of annotators for which the decision boundary will be
        plotted. `n_annotators` or `fig` have to be passed as an argument.
    fig: matplotlib.figure.Figure, optional (default=None)
        The figure to which axes the decision boundary will be plotted.
        `n_annotators` or `fig` have to be passed as an argument.
    confidence: scalar | None, optional (default=0.75)
        The confidence interval plotted with dashed lines. It is not plotted if
        confidence is None.
    title : str, optional
        The title for the figure.
    res: int, optional (default=21)
        The resolution of the plot.
    fig_size: tuple, shape (width, height) (default=None)
        The size of the figure in inches. If `fig_size` is None, the size
        of the figure is set to 8 x 5 inches.
    fontsize: int, optional (default=15)
        The fontsize of the labels
    cmap: str | matplotlib.colors.Colormap, optional (default='coolwarm_r')
        The colormap for the confidence levels.
    boundary_dict: dict, optional (default=None)
        Additional parameters for the boundary contour.
    confidence_dict: dict, optional (default=None)
        Additional parameters for the confidence contour. Must not contain a
        colormap because cmap is used.
    tick_dict: dict, optional (default=None)
        Additional parameters for the ticks.

    Returns
    -------
    fig: matplotlib.figure.Figure
        The figure onto which the decision boundaries are plotted.
    """

    # check arguments
    if n_annotators is None and fig is None:
        raise ValueError("`n_annotators` or `fig` must not be `None`")

    if n_annotators is not None:
        check_scalar(n_annotators, name='n_annotators', target_type=int)

    tick_args = _get_tick_args(tick_dict)
    fig = _get_figure_for_ma(fig, fig_size=fig_size, title=title,
                             fontsize=fontsize, n_annotators=n_annotators,
                             tick_args=tick_args)

    # plot decision boundary
    plot_decision_boundary(clf, feature_bound, ax=fig.get_axes(), res=res,
                           boundary_dict=boundary_dict, confidence=confidence,
                           cmap=cmap, confidence_dict=confidence_dict)
    return fig
