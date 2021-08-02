import numpy as np
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple

from mpl_toolkits.axes_grid1 import AxesGrid
from skactiveml.utils import call_func, is_unlabeled, is_labeled


# notebook for visualisation
def plot_scores_2d(figsize, X, y_true, y, res=21,  P=None,
                   title=None, vmin=0, vmax=1, fontsize=15):
    n_annotators = y.shape[1]

    x_1_vec = np.linspace(min(X[:, 0]), max(X[:, 0]), res)
    x_2_vec = np.linspace(min(X[:, 1]), max(X[:, 1]), res)
    X_1_mesh, X_2_mesh = np.meshgrid(x_1_vec, x_2_vec)
    X_mesh = np.array([X_1_mesh.reshape(-1), X_2_mesh.reshape(-1)]).T

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
        ax.set_xlim(min(X[:, 0]) - 0.5, max(X[:, 0]) + 0.5)
        ax.set_ylim(min(X[:, 1]) - 0.5, max(X[:, 1]) + 0.5)
        ax.set_title(r'annotator $a_{}$'.format(a+1), fontsize=fontsize)
        ax.set_xlabel(r'feature $x_1$', fontsize=fontsize, color='k')
        ax.set_ylabel(r'feature $x_2$', fontsize=fontsize, color='k')

        ax.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            direction='in',
            labelbottom=False,  # labels along the bottom edge are off
            labelleft=False)  # labels along the bottom edge are off
        r = labeled_indices[:, a]
        q = X[r, 0]
        ax.scatter(X[labeled_indices[:, a], 0], X[labeled_indices[:, a], 1], c=[[.2, .2, .2]], s=180, marker='o', zorder=3.8)
        ax.scatter(X[labeled_indices[:, a], 0], X[labeled_indices[:, a], 1], c=[[.8, .8, .8]], s=120, marker='o', zorder=4)
        is_false = y_true != y[:, a]
        for cl, color in zip([0, 1], ['r', 'b']):
            d = np.logical_and(y[labeled_indices[:, a], a] == cl, is_false[labeled_indices[:, a]])
            cl_labeled_idx_false = np.logical_and(y[labeled_indices[:, a], a] == cl,
                                                  is_false[labeled_indices[:, a]])
            cl_labeled_idx_correct = np.logical_and(y[labeled_indices[:, a], a] == cl,
                                                    ~is_false[labeled_indices[:, a]])
            cl_unlabeled_idx_false = np.logical_and(y[unlabeled_indices[:, a], a] == cl,
                                                    is_false[unlabeled_indices[:, a]])
            cl_unlabeled_idx_correct = np.logical_and(y[unlabeled_indices[:, a], a] == cl,
                                                      ~is_false[unlabeled_indices[:, a]])

            ax.scatter(X[cl_labeled_idx_false, 0],
                       X[cl_labeled_idx_false, 1],
                       color=color, marker='x',
                       vmin=-0.2, vmax=1.2,
                       cmap='coolwarm', s=40, zorder=5)
            ax.scatter(X[cl_labeled_idx_correct, 0],
                       X[cl_labeled_idx_correct, 1],
                       color=color, marker='s',
                       vmin=-0.2, vmax=1.2, cmap='coolwarm', s=40, zorder=5)
            ax.scatter(X[cl_unlabeled_idx_false, 0],
                       X[cl_unlabeled_idx_false, 1],
                       c=color, marker='x',
                       vmin=-0.2, vmax=1.2, cmap='coolwarm', s=40, zorder=3)
            ax.scatter(X[cl_unlabeled_idx_correct, 0],
                       X[cl_unlabeled_idx_correct, 1],
                       c=color, marker='s',
                       vmin=-0.2, vmax=1.2, cmap='coolwarm', s=40, zorder=3)
        # im = ax.contourf(X_1_mesh, X_2_mesh, scores[a], np.linspace(vmin, vmax, 10),
        #                  cmap='Greens', alpha=.75, vmin=vmin, vmax=vmax)
        if P is not None:
            ax.contour(X_1_mesh, X_2_mesh, P[a], [.49, .51], cmap='coolwarm', linewidths=[4, 4],
                        zorder=1, alpha=.8, vmin=.488, vmax=.512)
            ax.contour(X_1_mesh, X_2_mesh, P[a], [.5], colors='k', linewidths=[2], zorder=1)

        # handles, labels = ax.get_legend_handles_labels()

    line = Line2D([0], [0], color='k', linewidth=2)
    patch = Line2D([0], [0], marker='o', markerfacecolor='grey', markeredgecolor='k',
                   markersize=20, alpha=0.8, color='w')
    true_patches = (Line2D([0], [0], marker='s', markerfacecolor='b', markersize=15, color='w'),
                    Line2D([0], [0], marker='s', markerfacecolor='r', markersize=15, color='w'))
    false_patches = (Line2D([0], [0], marker='x', markerfacecolor='b', markeredgecolor='b', markersize=15, color='w'),
                     Line2D([0], [0], marker='x', markerfacecolor='r', markeredgecolor='r', markersize=15, color='w'))
    handles = [patch, true_patches, false_patches, line]
    labels = ['acquired annotation', 'true annotation', 'false annotation', 'decision boundary']

    vdiff = vmax - vmin
    # cbar = ax.cax.colorbar(im, ticks=[vmin, vmax])
    # cbar = grid.cbar_axes[0].colorbar(im, ticks=[vmin, vmax])
    # cbar.ax.set_yticklabels(['low', 'high'], fontsize=fontsize)
    return fig
