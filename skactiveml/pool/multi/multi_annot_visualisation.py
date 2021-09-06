import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from matplotlib.lines import Line2D

from mpl_toolkits.axes_grid1 import AxesGrid
from mpl_toolkits.axes_grid1 import axes_grid
from sklearn.neighbors import KNeighborsRegressor
from sklearn.utils import check_array, check_consistent_length

from ...base import MultiAnnotPoolBasedQueryStrategy, SkactivemlClassifier
from ...utils import is_labeled, check_scalar


def get_bound(X, bound):
    if bound is not None:
        bound = check_array(bound)
        return bound
    else:
        return np.array([[min(X[:, 0]) - 0.5, min(X[:, 1]) - 0.5],
                         [max(X[:, 0]) + 0.5, max(X[:, 1]) + 0.5]])


def check_or_get_figure(fig, fig_size, title, fontsize, n_annotators):
    if fig is None:
        if fig_size is None:
            fig_size = (8, 5)
        fig = plt.figure(figsize=fig_size)
        if title is not None:
            plt.title(title, fontsize=fontsize)
        AxesGrid(fig, 111,
                 nrows_ncols=(1, n_annotators),
                 axes_pad=0.05,
                 cbar_mode='single',
                 cbar_location='right',
                 cbar_pad=0.1
                 )
        return fig
    elif not isinstance(fig, Figure):
        raise TypeError("'fig' must be a matplotlib.figure.Figure")
    # type of axis has to equal axes_grid.Axes and must not be a subtype
    elif n_annotators is not None \
            and len([ax for ax in fig.axes if type(ax) == axes_grid.Axes]
            ) != n_annotators:
        raise ValueError("'fig' must contain an axes for each annotator")
    else:
        return fig


def set_up_annotator_axis(ax, annotator_index, bound, fontsize):
    x_min, y_min, x_max, y_max = np.ravel(bound)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title(r'annotator $a_{}$'.format(annotator_index + 1),
                 fontsize=fontsize)
    ax.set_xlabel(r'feature $x_1$', fontsize=fontsize, color='k')
    ax.set_ylabel(r'feature $x_2$', fontsize=fontsize, color='k')

    ax.tick_params(
        axis='both',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        direction='in',
        labelbottom=False,  # labels along the bottom edge are off
        labelleft=False)  # labels along the bottom edge are off

    return ax


def show_current_state(X, y, y_true, ma_qs, clf, ma_qs_arg_dict=None,
                       bound=None, title=None, fontsize=15, fig_size=None,
                       plot_legend=True, legend_dict=None, contour_dict=None,
                       boundary_dict=None, confidence_dict=None):
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
    clf: sklearn classifier # TODO correct?
        The classifier whose decision boundary is plotted.
    ma_qs_arg_dict: dict
        The argument dictionary for the multiple annotator query strategy.
    bound: array-like, [[xmin, ymin], [xmax, ymax]]
        Determines the area in which the boundary is plotted.
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
    """

    if ma_qs_arg_dict is None:
        ma_qs_arg_dict = {}
    ma_qs_arg_dict.update({"X": X, "y": y})

    bound = get_bound(X, bound)

    fig = plot_utility(fig_size=fig_size, ma_qs=ma_qs,
                       ma_qs_arg_dict=ma_qs_arg_dict,
                       bound=bound, title=title, fontsize=fontsize, res=5,
                       contour_dict=contour_dict)
    plot_data_set(fig=fig, X=X, y=y, y_true=y_true,
                  plot_legend=plot_legend, legend_dict=legend_dict)
    plot_multi_annot_decision_boundary(clf, fig=fig, bound=bound,
                                       boundary_dict=boundary_dict,
                                       confidence_dict=confidence_dict)

    plt.show()


def plot_data_set(X, y_true, y, fig=None, bound=None, title=None, fontsize=15,
                  fig_size=None, plot_legend=True, legend_dict=None):
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
    fig = check_or_get_figure(fig, fig_size=fig_size, title=title,
                              fontsize=fontsize, n_annotators=n_annotators)

    bound = get_bound(X, bound)

    check_scalar(plot_legend, 'plot_legend', bool)

    legend_args = {'fontsize': fontsize, 'loc': 'lower left'}
    if legend_dict is not None:
        if not isinstance(legend_dict, dict):
            raise TypeError("'legend_dict' must be a dictionary.")
        legend_args.update(legend_dict)

    # plot data set

    labeled_indices = is_labeled(y)

    # type of axis has to equal axes_grid.Axes and must not be a subtype
    axes = [ax for ax in fig.axes if type(ax) == axes_grid.Axes]
    for a, ax in enumerate(axes):

        set_up_annotator_axis(ax, annotator_index=a, bound=bound,
                              fontsize=fontsize)

        ax.scatter(X[labeled_indices[:, a], 0], X[labeled_indices[:, a], 1],
                   c=[[.2, .2, .2]], s=180, marker='o', zorder=3.8)
        ax.scatter(X[labeled_indices[:, a], 0], X[labeled_indices[:, a], 1],
                   c=[[.8, .8, .8]], s=120, marker='o', zorder=4)

        for cl, color in zip([0, 1], ['b', 'r']):
            for cl_true in [0, 1]:
                cl_current = np.logical_and(y[:, a] == cl, y_true == cl_true)

                cl_labeled = np.logical_and(cl_current, labeled_indices[:, a])

                ax.scatter(X[cl_labeled, 0],
                           X[cl_labeled, 1],
                           color=color, marker='x' if cl != cl_true else 's',
                           vmin=-0.2, vmax=1.2,
                           cmap='coolwarm', s=40, zorder=5)

    patch = Line2D([0], [0], marker='o', markerfacecolor='grey',
                   markeredgecolor='k', markersize=20, alpha=0.8, color='w')
    true_patches = (Line2D([0], [0], marker='s', markerfacecolor='b',
                           markersize=15, color='w'),
                    Line2D([0], [0], marker='s', markerfacecolor='r',
                           markersize=15, color='w'))
    false_patches = (Line2D([0], [0], marker='x', markerfacecolor='b',
                            markeredgecolor='b', markersize=15, color='w'),
                     Line2D([0], [0], marker='x', markerfacecolor='r',
                            markeredgecolor='r', markersize=15, color='w'))
    handles = [patch, true_patches, false_patches]
    labels = ['acquired annotation', 'true annotation', 'false annotation']

    if plot_legend:
        fig.legend(handles, labels, **legend_args)

    return fig


def plot_utility(ma_qs, ma_qs_arg_dict, X_cand=None, A_cand=None, fig=None,
                 fig_size=None, bound=None, title=None, res=21, fontsize=15,
                 contour_dict=None):
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
    """

    # check arguments

    if not isinstance(ma_qs, MultiAnnotPoolBasedQueryStrategy):
        raise TypeError("'ma_qs' must be a MultiAnnotPoolBasedQueryStrategy.")
    if not isinstance(ma_qs_arg_dict, dict):
        raise TypeError("'ma_qs_arg_dict' must be a dictionary.")
    if 'X_cand' in ma_qs_arg_dict.keys():
        raise ValueError("'X_cand' must be given as separate argument.")
    if ma_qs.n_annotators is None and A_cand is None:
        raise ValueError("'n_annotators' must be set in the multi annotator"
                         "query strategy or A_cand must be set, to determine"
                         "the number of annotators.")

    if A_cand is None:
        n_annotators = ma_qs.n_annotators
    else:
        n_annotators = A_cand.shape[1]

    bound = get_bound(X_cand, bound)
    x_min, y_min, x_max, y_max = np.ravel(bound)

    fig = check_or_get_figure(fig, fig_size=fig_size, title=title,
                              fontsize=fontsize, n_annotators=n_annotators)

    contour_args = {'cmap': 'Greens', 'alpha': 0.75}
    if contour_dict is not None:
        if not isinstance(contour_dict, dict):
            raise TypeError("'contour_dict' must be a dictionary.")
        contour_args.update(contour_dict)

    # plot the utilities

    x_vec = np.linspace(x_min, x_max, res)
    y_vec = np.linspace(y_min, y_max, res)
    X_mesh, Y_mesh = np.meshgrid(x_vec, y_vec)
    mesh_instances = np.array([X_mesh.reshape(-1), Y_mesh.reshape(-1)]).T

    # type of axis has to equal axes_grid.Axes and must not be a subtype
    axes = [ax for ax in fig.axes if type(ax) == axes_grid.Axes]
    if X_cand is None:
        _, utilities = ma_qs.query(X_cand=mesh_instances,
                                   **ma_qs_arg_dict, return_utilities=True)

        for a, ax in enumerate(axes):
            set_up_annotator_axis(ax, annotator_index=a, bound=bound,
                                  fontsize=fontsize)

            a_utilities = utilities[:, :, a]
            a_utilities_mesh = a_utilities.reshape(X_mesh.shape)

            ax.contourf(X_mesh, Y_mesh, a_utilities_mesh, **contour_args)
    else:
        _, utilities = ma_qs.query(X_cand, A_cand=A_cand, **ma_qs_arg_dict,
                                   return_utilities=True)
        for a, ax in enumerate(axes):
            set_up_annotator_axis(ax, annotator_index=a, bound=bound,
                                  fontsize=fontsize)

            utilities_a = utilities[0, :, a]
            neighbors = KNeighborsRegressor(n_neighbors=1)
            neighbors.fit(X_cand, utilities_a)
            scores = neighbors.predict(mesh_instances).reshape(X_mesh.shape)

            ax.contourf(X_mesh, Y_mesh, scores, cmap='Greens', alpha=.75)

    return fig


def plot_multi_annot_decision_boundary(clf, bound, n_annotators=None, fig=None,
                                       confidence=0.5, title=None, res=21,
                                       fig_size=None, fontsize=15,
                                       cmap='coolwarm_r', boundary_dict=None,
                                       confidence_dict=None):
    """Plot the decision boundary of the given classifier for each annotator.

    Parameters
    ----------
    clf: sklearn classifier # TODO correct?
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
    """
    def plot_decision_boundary(clf, bound, res=21, ax=None, confidence=0.5,
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
            confidence is None.
        cmap: str | matplotlib.colors.Colormap, optional (default='coolwarm_r')
            The colormap for the confidence levels.
        boundary_dict: dict, optional (default=None)
            Additional parameters for the boundary contour.
        confidence_dict: dict, optional (default=None)
            Additional parameters for the confidence contour. Must not contain a
            colormap because cmap is used.
        """

        # TODO: extend to multiclass, add parameter confidence [0,1] evtl. [0,0.5], or None
        # TODO: colors per class colormap or list of colors

        if not isinstance(clf, SkactivemlClassifier):
            raise TypeError("'clf' must be a SkactivemlClassifier.")
        check_scalar(res, 'res', int, min_val=1)
        if ax is None:
            ax = plt.gca()
        if not isinstance(ax, Axes):
            raise TypeError("ax must be a matplotlib.axes.Axes.")
        check_array(bound)
        xmin, ymin, xmax, ymax = np.ravel(bound)

        check_scalar(confidence, 'confidence', float, min_inclusive=False,
                     max_inclusive=False, min_val=0, max_val=1)

        # Create mesh for plotting
        x_vec = np.linspace(xmin, xmax, res)
        y_vec = np.linspace(ymin, ymax, res)
        X_mesh, Y_mesh = np.meshgrid(x_vec, y_vec)
        mesh_instances = np.array([X_mesh.reshape(-1), Y_mesh.reshape(-1)]).T

        posteriors = clf.predict_proba(mesh_instances)[:, 0].reshape(X_mesh.shape)

        boundary_args = {'colors': 'k', 'linewidths': [2], 'zorder': 1}
        if boundary_dict is not None:
            if not isinstance(boundary_dict, dict):
                raise TypeError("boundary_dict' must be a dictionary.")
            boundary_args.update(boundary_dict)
        ax.contour(X_mesh, Y_mesh, posteriors, [.5], **boundary_args)

        confidence_args = {'linewidths': [2, 2], 'linestyles': '--', 'alpha': 0.9,
                           'vmin': 0.2, 'vmax': 0.8, 'zorder': 1}
        if confidence_dict is not None:
            if not isinstance(confidence_dict, dict):
                raise TypeError("confidence_dict' must be a dictionary.")
            confidence_args.update(confidence_dict)
        ax.contour(X_mesh, Y_mesh, posteriors, [.25, .75], cmap=cmap,
                   **confidence_args)

    # check arguments
    if n_annotators is None and fig is None:
        raise TypeError("n_annotators or fig have to be passed as an argument")

    if n_annotators is not None:
        n_annotators = check_scalar(n_annotators, name='n_annotators',
                                    target_type=int)

    fig = check_or_get_figure(fig, fig_size=fig_size, title=title,
                              fontsize=fontsize, n_annotators=n_annotators)

    # plot decision boundary

    # type of axis has to equal axes_grid.Axes and must not be a subtype
    axes = [ax for ax in fig.axes if type(ax) == axes_grid.Axes]
    for a, ax in enumerate(axes):
        set_up_annotator_axis(ax, annotator_index=a, bound=bound,
                              fontsize=fontsize)

        plot_decision_boundary(clf, bound, res=res, ax=ax,
                               confidence=confidence, cmap=cmap,
                               boundary_dict=boundary_dict,
                               confidence_dict=confidence_dict)
