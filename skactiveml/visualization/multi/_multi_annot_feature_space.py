import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from sklearn.neighbors import KNeighborsRegressor
from sklearn.utils import check_array, check_consistent_length

from skactiveml.base import MultiAnnotPoolBasedQueryStrategy
from skactiveml.utils import is_labeled, check_scalar
from .. import plot_decision_boundary
from ...utils._validation import check_type
from ...utils._visualisation import mesh, check_bound, _get_contour_args, \
    _get_tick_args, _get_legend_args, _get_cmap, _get_figure_for_ma


def plot_current_state(X, y, y_true, ma_qs, clf, ma_qs_arg_dict,
                       bound=None, epsilon=1, title=None, fontsize=15,
                       fig_size=None, plot_legend=True, legend_dict=None,
                       contour_dict=None, boundary_dict=None,
                       confidence_dict=None, tick_dict=None):
    """Shows the annotations from the different annotators, the decision
    boundary of the given classifier and the utilities expected of querying
    a sample from a given region based on the query strategy.

    Parameters
    ----------
    X : matrix-like, shape (n_samples, 2)
        The sample matrix X is the feature matrix representing the samples.
        The feature space must be two dimensional.
    y : array-like, shape (n_samples, n_annotators)
        It contains the annotated values for each sample.
        The number of class labels may be variable for the samples, where
        missing labels are represented the attribute 'missing_label'.
    y_true : array-like, shape (n_samples,)
        The correct labels
    ma_qs: MultiAnnotPoolBasedQueryStrategy
        The multi-annotator query strategy.
    clf: sklearn classifier
        The classifier whose decision boundary is plotted.
    ma_qs_arg_dict: dict
        The argument dictionary for the multiple annotator query strategy.
    bound: array-like, [[xmin, ymin], [xmax, ymax]]
        Determines the area in which the boundary is plotted.
    epsilon: float, optional (default=1)
        The minimal distance between the returned bound and the values of `X`,
        if `bound` is not specified.
    title : str, optional
        The title for the figure.
    fontsize: int
        The fontsize of the labels.
    fig_size: tuple, shape (width, height) (default=None)
        The size of the figure in inches. If `fig_size` is None, the size
        of the figure is set to 8 x 5 inches.
    plot_legend: bool
        Whether to plot the legend.
    legend_dict: dict, optional (default=None)
        Additional parameters for the legend.
    contour_dict: dict, optional (default=None)
        Additional parameters for the utility contour.
    boundary_dict: dict, optional (default=None)
        Additional parameters for the boundary contour.
    confidence_dict: dict, optional (default=None)
        Additional parameters for the confidence contour. Must not contain a
        colormap because cmap is used.
    tick_dict: dict, optional (default=None):
        Additional parameters for the ticks of the plots.

    Returns
    ----------
    fig: matplotlib.figure.Figure
        The figure onto which the current state is plotted
    """

    bound = check_bound(bound, X, epsilon=epsilon)

    fig = plot_utility(fig_size=fig_size, ma_qs=ma_qs,
                       ma_qs_arg_dict=ma_qs_arg_dict,
                       bound=bound, title=title, fontsize=fontsize, res=5,
                       contour_dict=contour_dict, tick_dict=tick_dict)
    plot_data_set(fig=fig, X=X, y=y, y_true=y_true, bound=bound,
                  plot_legend=plot_legend, legend_dict=legend_dict,
                  tick_dict=tick_dict)
    plot_multi_annotator_decision_boundary(clf, fig=fig, bound=bound,
                                           boundary_dict=boundary_dict,
                                           confidence_dict=confidence_dict,
                                           tick_dict=tick_dict)

    return fig


def plot_data_set(X, y, y_true, fig=None, bound=None, title=None, fontsize=15,
                  fig_size=None, plot_legend=True, legend_dict=None,
                  tick_dict=None, cmap='coolwarm', marker_size=10):
    """Plots the annotations of a binary classification problem, differentiating
    between correctly and incorrectly labeled data.

    Parameters
    ----------
    X : array-like, shape (n_samples, 2)
        The sample matrix X is the feature matrix representing the samples.
        The feature space must be two dimensional.
    y_true : array-like, shape (n_samples,)
        The correct labels
    y : array-like, shape (n_samples, n_annotators)
        It contains the annotated values for each sample.
        The number of class labels may be variable for the samples, where
        missing labels are represented the attribute 'missing_label'.
    fig: matplotlib.figure.Figure, optional (default=None)
        The figure to which axes the utilities will be plotted.
    fig_size: tuple, shape (width, height) (default=None)
        The size of the figure in inches. If `fig_size` is None, the size
        of the figure is set to 8 x 5 inches.
    bound: array-like, [[xmin, ymin], [xmax, ymax]]
        Determines the area in which the boundary is plotted.
    title : str, optional
        The title for the figure.
    fontsize: int
        The fontsize of the labels.
    plot_legend: bool
        Whether to plot the legend.
    legend_dict: dict, optional (default=None)
        Additional parameters for the legend.
    tick_dict: dict, optional (default=None)
        Additional parameters for the ticks.
    cmap: str | matplotlib.colors.Colormap, optional (default='coolwarm_r')
        The colormap for the confidence levels.
    marker_size: int
        The size of the markers on the plot.

    Returns
    ----------
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
    fig = _get_figure_for_ma(fig, fig_size=fig_size, title=title, fontsize=fontsize,
                             n_annotators=n_annotators, tick_args=tick_args)
    bound = check_bound(bound, X)
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
        ax.set_xlim(bound[:, 0])
        ax.set_ylim(bound[:, 1])

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


def plot_utility(ma_qs, ma_qs_arg_dict, X_cand=None, A_cand=None, fig=None,
                 fig_size=None, bound=None, title=None, res=21, fontsize=15,
                 contour_dict=None, tick_dict=None):
    """Plots the utilities for the different annotators of the given
    multi-annotator query strategy.

    Parameters
    ----------
    ma_qs: MultiAnnotPoolBasedQueryStrategy
        The multi-annotator query strategy.
    ma_qs_arg_dict: dict
        The argument dictionary for the multiple annotator query strategy.
    fig: matplotlib.figure.Figure, optional (default=None)
        The figure to which axes the utilities will be plotted
    fig_size: tuple, shape (width, height) (default=None)
        The size of the figure in inches. If `fig_size` is None, the size
        of the figure is set to 8 x 5 inches.
    bound: array-like, [[xmin, ymin], [xmax, ymax]]
        Determines the area in which the boundary is plotted.
    X_cand : array-like, shape (n_samples, n_features)
        Candidate samples from which the strategy can select.
    A_cand : array-like, shape (n_samples, n_annotators), optional
             (default=None)
        Boolean matrix where `A_cand[i,j] = True` indicates that
        annotator `j` can be selected for annotating sample `X_cand[i]`,
        while `A_cand[i,j] = False` indicates that annotator `j` cannot be
        selected for annotating sample `X_cand[i]`. If A_cand=None, each
        annotator is assumed to be available for labeling each sample.
    title : str, optional
        The title for the figure.
    res: int
        The resolution of the plot.
    fontsize: int
        The fontsize of the labels.
    contour_dict: dict, optional (default=None)
        Additional parameters for the utility contour.
    tick_dict: dict, optional (default=None)
        Additional parameters for the ticks.

    Returns
    ----------
    fig: matplotlib.figure.Figure
        The figure onto which the utilities are plotted
    """

    # check arguments
    check_type(ma_qs, 'ma_qs', MultiAnnotPoolBasedQueryStrategy)
    check_type(ma_qs_arg_dict, 'ma_qs_arg_dict', dict)
    if 'X_cand' in ma_qs_arg_dict.keys():
        raise ValueError("'X_cand' must be given as separate argument.")

    n_annotators = None
    if fig is None:
        if A_cand is not None:
            check_array(A_cand)
            n_annotators = A_cand.shape[0]
        elif ma_qs.n_annotators is not None:
            check_scalar(ma_qs.n_annotators, "ma_qs.n_annotators",
                         target_type=int)
            n_annotators = ma_qs.n_annotators
        else:
            raise ValueError("`A_cand`, `fig` or `n_annotators` must be set in "
                             "the multi annotator query strategy, to determine "
                             "the number of annotators.")
    elif fig is not None and A_cand is not None:
        check_array(A_cand)
        if A_cand.shape[0] != len(fig.get_axes()):
            raise ValueError(f"`A_cand.shape[0]` must equal "
                             f"`len(fig.get_axes())`, but "
                             f"`A_cand.shape[0] == {A_cand.shape[0]})` and "
                             f"`len(fig.get_axes()) == {len(fig.get_axes())}`.")

    elif fig is not None and ma_qs.n_annotators is not None:
        check_scalar(ma_qs.n_annotators, "ma_qs.n_annotators", target_type=int)
        qs_n_a = ma_qs.n_annotators
        if qs_n_a != len(fig.get_axes()):
            raise ValueError(f"`ma_qs.n_annotators` must equal "
                             f"`len(fig.get_axes())`, but "
                             f"`ma_qs.n_annotators == {qs_n_a})` and "
                             f"`len(fig.get_axes()) == {len(fig.get_axes())}`.")

    bound = check_bound(bound, X_cand)
    contour_args = _get_contour_args(contour_dict)
    tick_args = _get_tick_args(tick_dict)
    fig = _get_figure_for_ma(fig, fig_size=fig_size, title=title,
                             fontsize=fontsize, n_annotators=n_annotators,
                             tick_args=tick_args)

    # plot the utilities
    X_mesh, Y_mesh, mesh_instances = mesh(bound, res)

    if X_cand is None:
        _, utilities = ma_qs.query(X_cand=mesh_instances,
                                   **ma_qs_arg_dict, return_utilities=True,
                                   batch_size=1)

        for a, ax in enumerate(fig.get_axes()):
            a_utilities = utilities[:, :, a]
            a_utilities_mesh = a_utilities.reshape(X_mesh.shape)
            ax.contourf(X_mesh, Y_mesh, a_utilities_mesh, **contour_args)
    else:
        _, utilities = ma_qs.query(X_cand, A_cand=A_cand, **ma_qs_arg_dict,
                                   return_utilities=True, batch_size=1)

        for a, ax in enumerate(fig.get_axes()):
            utilities_a = utilities[:, :, a]
            neighbors = KNeighborsRegressor(n_neighbors=1)
            neighbors.fit(X_cand, utilities_a)
            scores = neighbors.predict(mesh_instances).reshape(X_mesh.shape)
            ax.contourf(X_mesh, Y_mesh, scores, **contour_args)

    return fig


def plot_multi_annotator_decision_boundary(clf, bound, n_annotators=None,
                                           fig=None, boundary_dict=None,
                                           confidence=0.75, title=None, res=21,
                                           fig_size=None, fontsize=15,
                                           cmap='coolwarm',
                                           confidence_dict=None,
                                           tick_dict=None):
    """Plot the decision boundary of the given classifier for each annotator.

    Parameters
    ----------
    clf: sklearn classifier
        The classifier whose decision boundary is plotted.
    bound: array-like, [[xmin, ymin], [xmax, ymax]]
        Determines the area in which the boundary is plotted.
    n_annotators: int, optional (default=None)
        The number of annotators for which the decision boundary will be
        plotted. `n_annotators` or `fig` have to be passed as an argument.
    fig: matplotlib.figure.Figure, optional (default=None)
        The figure to which axes the decision boundary will be plotted.
        `n_annotators` or `fig` have to be passed as an argument.
    confidence: scalar | None, optional (default=0.5)
        The confidence interval plotted with dashed lines. It is not plotted if
        confidence is None.
    title : str, optional
        The title for the figure.
    res: int, optional (default=21)
        The resolution of the plot.
    fig_size: tuple, shape (width, height) (default=None)
        The size of the figure in inches. If `fig_size` is None, the size
        of the figure is set to 8 x 5 inches.
    fontsize: int
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
    ----------
    fig: matplotlib.figure.Figure
        The figure onto which the decision boundaries are plotted.
    """

    # check arguments
    if n_annotators is None and fig is None:
        raise TypeError("`n_annotators` or `fig` must not be `None`")

    if n_annotators is not None:
        n_annotators = check_scalar(n_annotators, name='n_annotators',
                                    target_type=int)

    tick_args = _get_tick_args(tick_dict)
    fig = _get_figure_for_ma(fig, fig_size=fig_size, title=title,
                             fontsize=fontsize, n_annotators=n_annotators,
                             tick_args=tick_args)

    # plot decision boundary
    plot_decision_boundary(clf, bound, ax=fig.get_axes(), res=res,
                           boundary_dict=boundary_dict, confidence=confidence,
                           cmap=cmap, confidence_dict=confidence_dict)
    return fig
