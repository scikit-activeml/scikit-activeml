import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Colormap
from matplotlib.figure import Figure

from ..utils import check_scalar, check_type, check_bound


def mesh(bound, res):
    """
    Function to get instances of a mesh grid as well as x-mesh and y-mesh
    with given resolution in the specified bounds.

    Parameters
    ----------
    bound: array-like, [[xmin, ymin], [xmax, ymax]]
        The bounds of the mesh grid.
    res: int, optional (default=21)
        The resolution of the plot.

    Returns
    -------
    X_mesh: np.ndarray, shape (res, res)
        mesh grid over x
    Y_mesh: np.ndarray, shape (res, res)
        mesh grid over y
    mesh_instances: np.ndarray, shape (res*res,)
        instances of the mesh grid
    """

    check_scalar(res, 'res', int, min_val=1)
    check_bound(bound=bound, bound_must_be_given=True)

    xmin, ymin, xmax, ymax = np.ravel(bound)

    x_vec = np.linspace(xmin, xmax, res)
    y_vec = np.linspace(ymin, ymax, res)
    X_mesh, Y_mesh = np.meshgrid(x_vec, y_vec)
    mesh_instances = np.array([X_mesh.reshape(-1), Y_mesh.reshape(-1)]).T
    return X_mesh, Y_mesh, mesh_instances


def _get_cmap(cmap):
    if isinstance(cmap, str):
        cmap = plt.cm.get_cmap(cmap)
    check_type(cmap, 'cmap', Colormap, str)
    return cmap


def _get_boundary_args(boundary_dict):
    boundary_args = {'colors': 'k', 'linewidths': [2], 'zorder': 1}
    if boundary_dict is not None:
        check_type(boundary_dict, 'boundary_dict', dict)
        boundary_args.update(boundary_dict)
    return boundary_args


def _get_confidence_args(confidence_dict):
    confidence_args = {'linewidths': [2, 2], 'linestyles': '--', 'alpha': 0.9,
                       'vmin': 0.2, 'vmax': 0.8, 'zorder': 1}
    if confidence_dict is not None:
        check_type(confidence_dict, 'confidence_dict', dict)
        confidence_args.update(confidence_dict)
    return confidence_args


def _get_contour_args(contour_dict):
    contour_args = {'cmap': 'Greens', 'alpha': 0.75}
    if contour_dict is not None:
        check_type(contour_dict, 'contour_dict', dict)
        contour_args.update(contour_dict)
    return contour_args


def _get_legend_args(legend_dict, fontsize):
    legend_args = {'fontsize': fontsize, 'loc': 'lower left',
                   'bbox_to_anchor': (0.0, -0.5)}
    if legend_dict is not None:
        check_type(legend_dict, 'legend_dict', dict)
        legend_args.update(legend_dict)
    return legend_args


def _get_tick_args(tick_dict):
    tick_args = {'axis': 'both', 'which': 'both', 'direction': 'in',
                 'labelbottom': False, 'labelleft': False}
    if tick_dict is not None:
        check_type(tick_dict, 'tick_dict', dict)
        tick_args.update(tick_dict)
    return tick_args


def _get_figure_for_ma(fig, fig_size, title, fontsize, n_annotators,
                       tick_args):
    if fig is None:
        if fig_size is None:
            fig_size = (8, 1.5)
        fig, _ = plt.subplots(nrows=1, ncols=n_annotators, figsize=fig_size,
                              sharex='all', sharey='all')
        axes = fig.get_axes()
        if title is not None:
            fig.suptitle(title, fontsize=fontsize)
        axes[0].set_ylabel(r'feature $x_2$', fontsize=fontsize, color='k')
        for a, ax in enumerate(axes):
            ax.set_title(fr'annotator $a_{a + 1}$', fontsize=fontsize)
            ax.set_xlabel(r'feature $x_1$', fontsize=fontsize, color='k')
            ax.tick_params(**tick_args)
        return fig
    elif not isinstance(fig, Figure):
        check_type(fig, 'fig', Figure)
    else:
        return fig
