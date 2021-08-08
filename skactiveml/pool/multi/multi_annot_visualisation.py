import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from matplotlib.lines import Line2D

from mpl_toolkits.axes_grid1 import AxesGrid
from mpl_toolkits.axes_grid1.axes_grid import CbarAxes
from sklearn.neighbors import KNeighborsRegressor

from skactiveml.base import MultiAnnotPoolBasedQueryStrategy, ClassFrequencyEstimator
from skactiveml.utils import is_labeled, check_scalar


def set_up_bound(X, bound):
    if bound is not None:
        return bound
    else:
        return (min(X[:, 0]) - 0.5, max(X[:, 0]) + 0.5,
                min(X[:, 1]) - 0.5, max(X[:, 1]) + 0.5)


def set_up_figure(fig, fig_size, title, fontsize, n_annotators):
    if fig is None:
        if fig_size is None:
            fig_size = (8, 8)
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
    elif len([ax for ax in fig.axes if not isinstance(ax, CbarAxes)]
             ) != n_annotators:
        raise ValueError("'fig' must contain an axes for each annotator")
    else:
        return fig


def set_up_annotator_axis(ax, annotator_index, bound, fontsize):
    x_min, x_max, y_min, y_max = bound
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


# notebook for visualisation
def plot_data_set(X, y_true, y, fig=None, bound=None, title=None, fontsize=15,
                  fig_size=None):
    n_annotators = y.shape[1]

    bound = set_up_bound(X, bound)

    labeled_indices = is_labeled(y)
    unlabeled_indices = ~labeled_indices

    fig = set_up_figure(fig, fig_size=fig_size, title=title, fontsize=fontsize,
                        n_annotators=n_annotators)

    axes = [ax for ax in fig.axes if not isinstance(ax, CbarAxes)]
    for a, ax in enumerate(axes):

        set_up_annotator_axis(ax, annotator_index=a, bound=bound,
                              fontsize=fontsize)

        ax.scatter(X[labeled_indices[:, a], 0], X[labeled_indices[:, a], 1],
                   c=[[.2, .2, .2]], s=180, marker='o', zorder=3.8)
        ax.scatter(X[labeled_indices[:, a], 0], X[labeled_indices[:, a], 1],
                   c=[[.8, .8, .8]], s=120, marker='o', zorder=4)

        for cl, color in zip([0, 1], ['r', 'b']):
            for cl_true in [0, 1]:
                cl_labeled = np.logical_and(
                    y[labeled_indices[:, a], a] == cl,
                    y_true[labeled_indices[:, a]] == cl_true)

                cl_unlabeled = np.logical_and(
                    y[unlabeled_indices[:, a], a] == cl,
                    y_true[unlabeled_indices[:, a]] == cl_true)

                ax.scatter(X[cl_labeled, 0],
                           X[cl_labeled, 1],
                           color=color, marker='x' if cl != cl_true else 's',
                           vmin=-0.2, vmax=1.2,
                           cmap='coolwarm', s=40, zorder=5)

                ax.scatter(X[cl_unlabeled, 0],
                           X[cl_unlabeled, 1],
                           c=color, marker='x' if cl != cl_true else 's',
                           vmin=-0.2, vmax=1.2,
                           cmap='coolwarm', s=40, zorder=3)

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

    fig.legend(handles, labels, fontsize=fontsize, loc='lower left')

    return fig


def plot_utility(ma_qs, ma_qs_arg_dict, fig=None, fig_size=None, X_cand=None,
                 A_cand=None, bound=None, title=None, res=21, fontsize=15):

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

    bound = set_up_bound(X_cand, bound)
    x_min, x_max, y_min, y_max = bound

    fig = set_up_figure(fig, fig_size=fig_size, title=title, fontsize=fontsize,
                        n_annotators=n_annotators)

    x_vec = np.linspace(x_min, x_max, res)
    y_vec = np.linspace(y_min, y_max, res)
    X_mesh, Y_mesh = np.meshgrid(x_vec, y_vec)
    mesh_instances = np.array([X_mesh.reshape(-1), Y_mesh.reshape(-1)]).T

    axes = [ax for ax in fig.axes if not isinstance(ax, CbarAxes)]
    if X_cand is None:
        _, utilities = ma_qs.query(X_cand=mesh_instances, A_cand=A_cand,
                                   **ma_qs_arg_dict, return_utilities=True)

        for a, ax in enumerate(axes):
            set_up_annotator_axis(ax, annotator_index=a, bound=bound,
                                  fontsize=fontsize)

            a_utilities = utilities[:, :, a]
            a_utilities_mesh = a_utilities.reshape(X_mesh.shape)

            ax.contourf(X_mesh, Y_mesh, a_utilities_mesh, cmap='Greens',
                        alpha=.75)
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


def plot_multi_annot_decision_boundary(n_annotators, clf, bound, fig=None,
                                       title=None, res=21, fig_size=None,
                                       fontsize=15):

    def plot_decision_boundary(clf, bound, res=21, ax=None):
        """Plot the decision boundary of the given classifier.
        Parameters
        ----------
        clf: sklearn classifier # TODO correct?
            The classifier whose decision boundary is plotted.
        bound: array-like, (x_min, x_max, y_min, y_max)
            Determines the area in which the boundary is plotted.
        res: int
            The resolution of the plot.
        ax: matplotlib.axes.Axes, optional (default=None)
            The axis on which the boundary is plotted.
        """
        # TODO which bound format sklearn, check bound
        # TODO predict_proba - ClassFrequencyEstimator ?
        if not isinstance(clf, ClassFrequencyEstimator):
            raise TypeError("'clf' must be a ClassFrequencyEstimator.")
        check_scalar(res, 'res', int, min_val=1)
        if ax is None:
            ax = plt.gca()
        if not isinstance(ax, Axes):
            raise TypeError("ax must be a matplotlib.axes.Axes.")
        x_min, x_max, y_min, y_max = bound

        # Create mesh for plotting
        x_vec = np.linspace(x_min, x_max, res)
        y_vec = np.linspace(y_min, y_max, res)
        X_mesh, Y_mesh = np.meshgrid(x_vec, y_vec)
        mesh_instances = np.array([X_mesh.reshape(-1), Y_mesh.reshape(-1)]).T

        posteriors = clf.predict_proba(mesh_instances)[:, 0].reshape(X_mesh.shape)

        ax.contour(X_mesh, Y_mesh, posteriors, [.5], colors='k', linewidths=[2],
                   zorder=1)
        ax.contour(X_mesh, Y_mesh, posteriors, [.25, .75], cmap='coolwarm_r',
                   linewidths=[2, 2], zorder=1, linestyles='--', alpha=.9, vmin=.2,
                   vmax=.8)
        return ax

    fig = set_up_figure(fig, fig_size=fig_size, title=title, fontsize=fontsize,
                        n_annotators=n_annotators)

    axes = [ax for ax in fig.axes if not isinstance(ax, CbarAxes)]
    for a, ax in enumerate(axes):
        set_up_annotator_axis(ax, annotator_index=a, bound=bound,
                              fontsize=fontsize)

        plot_decision_boundary(clf, bound, res=res, ax=ax)
