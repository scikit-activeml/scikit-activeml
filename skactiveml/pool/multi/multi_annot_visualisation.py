import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from matplotlib.lines import Line2D

from mpl_toolkits.axes_grid1 import AxesGrid
from sklearn.neighbors import KNeighborsRegressor

from skactiveml.base import MultiAnnotPoolBasedQueryStrategy
from skactiveml.utils import is_labeled


def get_bound(X, bound):
    if bound is not None:
        return bound
    else:
        return (min(X[:, 0]) - 0.5, max(X[:, 0]) + 0.5,
                min(X[:, 1]) - 0.5, max(X[:, 1]) + 0.5)


# notebook for visualisation
def plot_scores_2d(figsize, X, y_true, y, bound=None,
                   title=None, fontsize=15):
    n_annotators = y.shape[1]

    x_min, x_max, y_min, y_max = get_bound(X, bound)

    labeled_indices = is_labeled(y)
    unlabeled_indices = ~labeled_indices

    # setup figure
    fig = plt.figure(figsize=figsize)
    if title is not None:
        plt.title(title, fontsize=fontsize)

    grid = AxesGrid(fig, 111,
                    nrows_ncols=(1, n_annotators),
                    axes_pad=0.05,
                    cbar_mode='single',
                    cbar_location='right',
                    cbar_pad=0.1
                    )

    for a, ax in enumerate(grid):
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_title(r'annotator $a_{}$'.format(a+1), fontsize=fontsize)
        ax.set_xlabel(r'feature $x_1$', fontsize=fontsize, color='k')
        ax.set_ylabel(r'feature $x_2$', fontsize=fontsize, color='k')

        ax.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            direction='in',
            labelbottom=False,  # labels along the bottom edge are off
            labelleft=False)  # labels along the bottom edge are off

        ax.scatter(X[labeled_indices[:, a], 0], X[labeled_indices[:, a], 1], c=[[.2, .2, .2]], s=180, marker='o', zorder=3.8)
        ax.scatter(X[labeled_indices[:, a], 0], X[labeled_indices[:, a], 1], c=[[.8, .8, .8]], s=120, marker='o', zorder=4)

        for cl, color in zip([0, 1], ['r', 'b']):
            for cl_true in [0, 1]:
                cl_labeled = np.logical_and(y[labeled_indices[:, a], a] == cl,
                                            y_true[labeled_indices[:, a]] == cl_true)

                cl_unlabeled = np.logical_and(y[unlabeled_indices[:, a], a] == cl,
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

    patch = Line2D([0], [0], marker='o', markerfacecolor='grey', markeredgecolor='k',
                   markersize=20, alpha=0.8, color='w')
    true_patches = (Line2D([0], [0], marker='s', markerfacecolor='b', markersize=15, color='w'),
                    Line2D([0], [0], marker='s', markerfacecolor='r', markersize=15, color='w'))
    false_patches = (Line2D([0], [0], marker='x', markerfacecolor='b', markeredgecolor='b', markersize=15, color='w'),
                     Line2D([0], [0], marker='x', markerfacecolor='r', markeredgecolor='r', markersize=15, color='w'))
    handles = [patch, true_patches, false_patches]
    labels = ['acquired annotation', 'true annotation', 'false annotation']

    fig.legend(handles, labels, fontsize=fontsize,loc='lower left')

    return fig


def plot_utility(figsize, qs, qs_dict, X_cand=None, A_cand=None, bound=None, title=None, res=21
                 , fontsize=15):
    if not isinstance(qs, MultiAnnotPoolBasedQueryStrategy):
        raise TypeError("'qs' must be a query MultiAnnotPoolBasedQueryStrategy.")
    if not isinstance(qs_dict, dict):
        raise TypeError("'qs_dict' must be a dictionary.")
    if 'X_cand' in qs_dict.keys():
        raise ValueError("'X_cand' must be given as separate argument.")

    if A_cand is None:
        n_annotators = qs.n_annotators
    else:
        n_annotators = A_cand.shape[1]

    x_min, x_max, y_min, y_max = get_bound(X_cand, bound)

    # setup figure
    fig = plt.figure(figsize=figsize)
    if title is not None:
        plt.title(title, fontsize=fontsize)

    # TODO check bound

    x_vec = np.linspace(x_min, x_max, res)
    y_vec = np.linspace(y_min, y_max, res)
    X_mesh, Y_mesh = np.meshgrid(x_vec, y_vec)
    mesh_instances = np.array([X_mesh.reshape(-1), Y_mesh.reshape(-1)]).T

    grid = AxesGrid(fig, 111,
                    nrows_ncols=(1, n_annotators),
                    axes_pad=0.05,
                    cbar_mode='single',
                    cbar_location='right',
                    cbar_pad=0.1
                    )

    if X_cand is None:
        _, utilities = qs.query(X_cand=mesh_instances, A_cand=A_cand, **qs_dict,
                                return_utilities=True)

        for a, ax in enumerate(grid):
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_title(r'annotator $a_{}$'.format(a + 1), fontsize=fontsize)
            ax.set_xlabel(r'feature $x_1$', fontsize=fontsize, color='k')
            ax.set_ylabel(r'feature $x_2$', fontsize=fontsize, color='k')

            ax.tick_params(
                axis='both',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                direction='in',
                labelbottom=False,  # labels along the bottom edge are off
                labelleft=False)  # labels along the bottom edge are off

            a_utilities = utilities[:, :, a]
            a_utilities_xy = a_utilities.reshape(X_mesh.shape)

            ax.contourf(X_mesh, Y_mesh, a_utilities_xy, cmap='Greens', alpha=.75)
    else:
        _, utilities = qs.query(X_cand, A_cand=A_cand, **qs_dict, return_utilities=True)
        for a, ax in enumerate(grid):
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_title(r'annotator $a_{}$'.format(a + 1), fontsize=fontsize)
            ax.set_xlabel(r'feature $x_1$', fontsize=fontsize, color='k')
            ax.set_ylabel(r'feature $x_2$', fontsize=fontsize, color='k')

            ax.tick_params(
                axis='both',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                direction='in',
                labelbottom=False,  # labels along the bottom edge are off
                labelleft=False)  # labels along the bottom edge are off

            utilities_a = utilities[0, :, a]
            neighbors = KNeighborsRegressor(n_neighbors=1)
            neighbors.fit(X_cand, utilities_a)
            scores = neighbors.predict(mesh_instances).reshape(X_mesh.shape)

            ax.contourf(X_mesh, Y_mesh, scores, cmap='Greens', alpha=.75)


def plot_utility_difference(figsize, qs, qs_dict, X_cand=None, A_cand=None, bound=None, title=None, res=21
                 , fontsize=15):
    if not isinstance(qs, MultiAnnotPoolBasedQueryStrategy):
        raise TypeError("'qs' must be a MultiAnnotPoolBasedQueryStrategy.")
    if not isinstance(qs_dict, dict):
        raise TypeError("'qs_dict' must be a dictionary.")
    if 'X_cand' in qs_dict.keys():
        raise ValueError("'X_cand' must be given as separate argument.")

    if A_cand is None:
        n_annotators = qs.n_annotators
    else:
        n_annotators = A_cand.shape[1]

    if bound is not None:
        x_min, x_max, y_min, y_max = bound
    else:
        x_min = min(X_cand[:, 0]) - 0.5
        x_max = max(X_cand[:, 0]) + 0.5
        y_min = min(X_cand[:, 1]) - 0.5
        y_max = max(X_cand[:, 1]) + 0.5

    # setup figure
    fig = plt.figure(figsize=figsize)
    if title is not None:
        plt.title(title, fontsize=fontsize)

    x_vec = np.linspace(x_min, x_max, res)
    y_vec = np.linspace(y_min, y_max, res)
    X_mesh, Y_mesh = np.meshgrid(x_vec, y_vec)
    mesh_instances = np.array([X_mesh.reshape(-1), Y_mesh.reshape(-1)]).T

    grid = AxesGrid(fig, 111,
                    nrows_ncols=(n_annotators, n_annotators),
                    axes_pad=0.05,
                    cbar_mode='single',
                    cbar_location='right',
                    cbar_pad=0.1
                    )

    _, utilities = qs.query(X_cand=mesh_instances, A_cand=A_cand, **qs_dict,
                            return_utilities=True)
    for a, ax in enumerate(grid):
        a_y = a // n_annotators
        a_x = a % n_annotators
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_title(f'annotator $a_{a_x+1},a_{a_y+1}$', fontsize=fontsize)
        ax.set_xlabel(r'feature $x_1$', fontsize=fontsize, color='k')
        ax.set_ylabel(r'feature $x_2$', fontsize=fontsize, color='k')
        ax.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            direction='in',
            labelbottom=False,  # labels along the bottom edge are off
            labelleft=False)  # labels along the bottom edge are off

        a_x_utilities = utilities[:, :, a_x]
        a_x_utilities_xy = a_x_utilities.reshape(X_mesh.shape)

        a_y_utilities = utilities[:, :, a_y]
        a_y_utilities_xy = a_y_utilities.reshape(X_mesh.shape)

        a_diff_utilities = a_y_utilities_xy - a_x_utilities_xy

        ax.contourf(X_mesh, Y_mesh, a_diff_utilities, cmap='Greens', alpha=.75)
