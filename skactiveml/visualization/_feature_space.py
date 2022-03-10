import warnings

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from sklearn.base import ClassifierMixin
from sklearn.neighbors import KNeighborsRegressor
from sklearn.utils.validation import (
    check_array,
    check_consistent_length,
    column_or_1d,
)

from ._misc import (
    mesh,
    check_bound,
    _get_boundary_args,
    _get_confidence_args,
    _get_contour_args,
    _get_cmap,
)
from ..base import (
    QueryStrategy,
    SingleAnnotatorPoolQueryStrategy,
    MultiAnnotatorPoolQueryStrategy,
)
from ..exceptions import MappingError
from ..utils import (
    check_scalar,
    unlabeled_indices,
    call_func,
    check_type,
    check_indices,
)


def plot_utilities(qs, X, y, candidates=None, **kwargs):
    """Plot the utility for the given single-annotator query strategy.

    Parameters
    ----------
    qs : skactiveml.base.SingleAnnotatorPoolQueryStrategy
        The query strategy for which the utility is plotted.
    X : array-like of shape (n_samples, n_features)
        Training data set, usually complete, i.e. including the labeled and
        unlabeled samples.
    y : array-like of shape (n_samples, ) or (n_samples, n_annotators)
        Labels of the training data set (possibly including unlabeled ones
        indicated by self.MISSING_LABEL).
    candidates : None or array-like of shape (n_candidates,), dtype=int or
        array-like of shape (n_candidates, n_features),
        optional (default=None)
        If `candidates` is None, the unlabeled samples from (X,y) are
        considered as candidates.
        If `candidates` is of shape (n_candidates,) and of type int,
        candidates is considered as the indices of the samples in (X,y).
        If `candidates` is of shape (n_candidates, n_features), the
        candidates are directly given in candidates (not necessarily
        contained in X). This is not supported by all query strategies.

    Other Parameters
    ----------------
    replace_nan : numeric or None, optional (default=0.0)
        Only used if plotting with mesh instances is not possible.
        If numeric, the utility of labeled instances will be plotted with
        value `replace_nan`. If None, these samples will be ignored.
    ignore_undefined_query_params : bool, optional (default=False)
        If True, query parameters that are not defined in the query function
        are ignored and will not raise an exception.
    feature_bound : array-like of shape [[xmin, ymin], [xmax, ymax]], optional
    (default=None)
        Determines the area in which the boundary is plotted. If candidates is
        not given, bound must not be None. Otherwise, the bound is determined
        based on the data.
    ax : matplotlib.axes.Axes, optional (default=None)
        The axis on which the utility is plotted. Only if y.ndim = 1 (single
        annotator).
    res : int, optional (default=21)
        The resolution of the plot.
    contour_dict : dict, optional (default=None)
        Additional parameters for the utility contour.
    **kwargs
        Remaining keyword arguments are passed the query function of the query
        strategy.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axis on which the utilities were plotted.
    """
    check_type(qs, "qs", SingleAnnotatorPoolQueryStrategy)
    return _general_plot_utilities(
        qs=qs, X=X, y=y, candidates=candidates, **kwargs
    )


def plot_annotator_utilities(qs, X, y, candidates=None, **kwargs):
    """Plot the utility for the given query strategy.

    Parameters
    ----------
    qs : skactiveml.base.QueryStrategy
        The query strategy for which the utility is plotted.
    X : array-like of shape (n_samples, n_features)
        Training data set, usually complete, i.e. including the labeled and
        unlabeled samples.
    y : array-like of shape (n_samples, ) or (n_samples, n_annotators)
        Labels of the training data set (possibly including unlabeled ones
        indicated by self.MISSING_LABEL).
    candidates : None or array-like of shape (n_candidates,), dtype=int or
        array-like of shape (n_candidates, n_features),
        optional (default=None)
        If `candidates` is None, the unlabeled samples from (X,y) are
        considered as candidates.
        If `candidates` is of shape (n_candidates,) and of type int,
        candidates is considered as the indices of the samples in (X,y).
        If `candidates` is of shape (n_candidates, n_features), the
        candidates are directly given in candidates (not necessarily
        contained in X). This is not supported by all query strategies.

    Other Parameters
    ----------------
    replace_nan : numeric or None, optional (default=0.0)
        Only used if plotting with mesh instances is not possible.
        If numeric, the utility of labeled instances will be plotted with
        value `replace_nan`. If None, these samples will be ignored.
    ignore_undefined_query_params : bool, optional (default=False)
        If True, query parameters that are not defined in the query function
        are ignored and will not raise an exception.
    feature_bound : array-like of shape [[xmin, ymin], [xmax, ymax]], optional
    (default=None)
        Determines the area in which the boundary is plotted. If candidates is
        not given, bound must not be None. Otherwise, the bound is determined
        based on the data.
    axes : array-like of matplotlib.axes.Axes, optional (default=None)
        The axes on which the utilities for the annotators are plotted. Only
        supported for y.ndim = 2 (multi annotator).
    res : int, optional (default=21)
        The resolution of the plot.
    contour_dict : dict, optional (default=None)
        Additional parameters for the utility contour.
    plot_annotators : None or array-like of shape (n_annotators_to_plot,),
    optional (default=None)
        Contains the indices of the annotators to be plotted. If it is None,
        all annotators are plotted. Only supported for y.ndim = 2
         (multi annotator).
    **kwargs
        Remaining keyword arguments are passed the query function of the query
        strategy.

    Returns
    -------
    axes : array-like of shape (n_annotators_to_plot,)
        The axes on which the utilities were plotted.
    """
    check_type(qs, "qs", MultiAnnotatorPoolQueryStrategy)
    return _general_plot_utilities(
        qs=qs, X=X, y=y, candidates=candidates, **kwargs
    )


def plot_decision_boundary(
        clf,
        feature_bound,
        ax=None,
        res=21,
        boundary_dict=None,
        confidence=0.75,
        cmap="coolwarm",
        confidence_dict=None,
):
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
    check_type(clf, "clf", ClassifierMixin)
    check_scalar(res, "res", int, min_val=1)
    if ax is None:
        ax = plt.gca()
    check_type(ax, "ax", Axes)
    feature_bound = check_bound(bound=feature_bound)

    # Check and convert the colormap
    cmap = _get_cmap(cmap)

    if confidence is not None:
        check_scalar(
            confidence,
            "confidence",
            float,
            min_inclusive=False,
            max_inclusive=False,
            min_val=0.5,
            max_val=1,
        )

    # Update additional arguments
    boundary_args = _get_boundary_args(boundary_dict)
    confidence_args = _get_confidence_args(confidence_dict)

    # Create mesh for plotting
    X_mesh, Y_mesh, mesh_instances = mesh(feature_bound, res)

    # Calculate predictions
    if hasattr(clf, "predict_proba"):
        predictions = clf.predict_proba(mesh_instances)
        classes = np.arange(predictions.shape[1])
    elif hasattr(clf, "predict"):
        if confidence is not None:
            warnings.warn(
                "The given classifier does not implement "
                "'predict_proba'. Thus, the confidence cannot be "
                "plotted."
            )
            confidence = None
        predicted_classes = clf.predict(mesh_instances)
        classes = np.arange(len(np.unique(predicted_classes)))
        predictions = np.zeros((len(predicted_classes), len(classes)))
        for idx, y in enumerate(predicted_classes):
            predictions[idx, y] = 1
    else:
        raise AttributeError(
            "'clf' must implement 'predict' or " "'predict_proba'"
        )

    posterior_list = []

    for y in classes:
        posteriors = predictions[:, y].reshape(X_mesh.shape)
        posterior_list.append(posteriors)

    norm = plt.Normalize(vmin=min(classes), vmax=max(classes))

    for y in classes:
        posteriors = posterior_list[y]
        posteriors_best_alternative = np.zeros_like(posteriors)
        for y2 in np.setdiff1d(classes, [y]):
            posteriors_best_alternative = np.max(
                [posteriors_best_alternative, posterior_list[y2]], axis=0
            )

        posteriors = posteriors / (posteriors + posteriors_best_alternative)
        ax.contour(X_mesh, Y_mesh, posteriors, [0.5], **boundary_args)
        if confidence is not None:
            ax.contour(
                X_mesh,
                Y_mesh,
                posteriors,
                [confidence],
                colors=[cmap(norm(y))],
                **confidence_args,
            )
    return ax


def plot_contour_for_samples(
        X,
        values,
        replace_nan=0.0,
        feature_bound=None,
        ax=None,
        res=101,
        contour_dict=None,
):
    """Plot the utility for the given query strategy.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training data set, usually complete, i.e. including the labeled and
        unlabeled samples.
    values : array-like of shape (n_samples)
        Values to plot for samples `X` (may contain np.nan, can be replaced
        or ignored, see `replace_nan`).
    replace_nan : numeric or None, optional (default=0.0)
        If numeric, nan-values in `values` will be replaced by this number.
        If None, these samples will be ignored.
    feature_bound : array-like, [[xmin, ymin], [xmax, ymax]]
        Determines the area in which the boundary is plotted. If candidates is
        not given, bound must not be None. Otherwise, the bound is determined
        based on the data.
    ax : matplotlib.axes.Axes, optional (default=None)
        The axis on which the utility is plotted.
    res : int, optional (default=21)
        The resolution of the plot.
    contour_dict : dict, optional (default=None)
        Additional parameters for the utility contour.

    Returns
    -------
    matplotlib.axes.Axes: The axis on which the utility was plotted.
    """
    check_array(X, ensure_2d=True)
    check_array(values, ensure_2d=False, force_all_finite="allow-nan")

    feature_bound = check_bound(bound=feature_bound, X=X)

    X_mesh, Y_mesh, mesh_instances = mesh(feature_bound, res)

    if ax is None:
        ax = plt.gca()

    if replace_nan is None:
        valid_idx = ~np.isnan(values)
        X = X[valid_idx]
        values = values[valid_idx]
    else:
        values = np.nan_to_num(values, nan=replace_nan)

    contour_args = _get_contour_args(contour_dict)

    neighbors = KNeighborsRegressor(n_neighbors=1)
    neighbors.fit(X, values)

    scores = neighbors.predict(mesh_instances).reshape(X_mesh.shape)
    ax.contourf(X_mesh, Y_mesh, scores, **contour_args)
    return ax


def _general_plot_utilities(qs, X, y, candidates=None, **kwargs):
    """Plot the utility for the given query strategy.

    Parameters
    ----------
    qs : skactiveml.base.QueryStrategy
        The query strategy for which the utility is plotted.
    X : array-like of shape (n_samples, n_features)
        Training data set, usually complete, i.e. including the labeled and
        unlabeled samples.
    y : array-like of shape (n_samples, ) or (n_samples, n_annotators)
        Labels of the training data set (possibly including unlabeled ones
        indicated by self.MISSING_LABEL).
    candidates : None or array-like of shape (n_candidates,), dtype=int or
        array-like of shape (n_candidates, n_features),
        optional (default=None)
        If `candidates` is None, the unlabeled samples from (X,y) are
        considered as candidates.
        If `candidates` is of shape (n_candidates,) and of type int,
        candidates is considered as the indices of the samples in (X,y).
        If `candidates` is of shape (n_candidates, n_features), the
        candidates are directly given in candidates (not necessarily
        contained in X). This is not supported by all query strategies.

    Other Parameters
    ----------------
    replace_nan : numeric or None, optional (default=0.0)
        Only used if plotting with mesh instances is not possible.
        If numeric, the utility of labeled instances will be plotted with
        value `replace_nan`. If None, these samples will be ignored.
    ignore_undefined_query_params : bool, optional (default=False)
        If True, query parameters that are not defined in the query function
        are ignored and will not raise an exception.
    feature_bound : array-like of shape [[xmin, ymin], [xmax, ymax]], optional
    (default=None)
        Determines the area in which the boundary is plotted. If candidates is
        not given, bound must not be None. Otherwise, the bound is determined
        based on the data.
    ax : matplotlib.axes.Axes, optional (default=None)
        The axis on which the utility is plotted. Only if y.ndim = 1 (single
        annotator).
    axes : array-like of matplotlib.axes.Axes, optional (default=None)
        The axes on which the utilities for the annotators are plotted. Only
        supported for y.ndim = 2 (multi annotator).
    res : int, optional (default=21)
        The resolution of the plot.
    contour_dict : dict, optional (default=None)
        Additional parameters for the utility contour.
    plot_annotators : None or array-like of shape (n_annotators_to_plot,),
    optional (default=None)
        Contains the indices of the annotators to be plotted. If it is None,
        all annotators are plotted. Only supported for y.ndim = 2
         (multi annotator).
    **kwargs
        Remaining keyword arguments are passed the query function of the query
        strategy.

    Returns
    -------
     axes : array-like of shape (n_annotators_to_plot,)
         The axes on which the utilities were plotted.
    """
    replace_nan = kwargs.pop("replace_nan", 0.0)
    ignore_undefined_query_params = kwargs.pop(
        "ignore_undefined_query_params", False
    )
    feature_bound = kwargs.pop("feature_bound", None)
    ax = kwargs.pop("ax", None)
    axes = kwargs.pop("axes", None)
    res = kwargs.pop("res", 21)
    contour_dict = kwargs.pop("contour_dict", None)
    plot_annotators = kwargs.pop("plot_annotators", None)

    check_type(qs, "qs", QueryStrategy)
    X = check_array(X, allow_nd=False, ensure_2d=True)
    if X.shape[1] != 2:
        raise ValueError("Samples in `X` must have 2 features.")

    # Check labels
    y = check_array(y, ensure_2d=False, force_all_finite="allow-nan")
    check_consistent_length(X, y)

    if y.ndim == 2:
        if plot_annotators is None:
            n_annotators = y.shape[1]
            plot_annotators = np.arange(n_annotators)
        else:
            plot_annotators = column_or_1d(plot_annotators)
            check_indices(plot_annotators, y, dim=1)
            n_annotators = len(plot_annotators)
    else:
        n_annotators = None
        if plot_annotators is not None:
            raise TypeError(
                "`plot_annotator` can be only used in the multi-annotator "
                "setting."
            )
        else:
            plot_annotators = np.arange(1)
    if n_annotators is None:
        if axes is not None:
            raise TypeError(
                "`axes` can be only used in the multi-annotator setting. "
                "Use `ax` instead."
            )
        if ax is None:
            axes = np.array([plt.subplots(1, 1)[1]])
        else:
            check_type(ax, "ax", Axes)
            axes = np.array([ax])
    else:
        if ax is not None:
            raise ValueError(
                "`ax` can be only used in the single-annotator setting. "
                "Use `axes` instead."
            )
        if axes is None:
            axes = plt.subplots(1, n_annotators)[1]
        else:
            [check_type(ax_, "ax", Axes) for ax_ in axes]

    if n_annotators is not None and len(axes) != n_annotators:
        raise ValueError(
            "`axes` must contain one `Axes` object for each "
            "annotator to be plotted (indicated by `plot_annotators`)."
        )

    # ensure that utilities are returned
    kwargs["return_utilities"] = True

    if candidates is None:
        # plot mesh
        try:
            check_scalar(res, "res", int, min_val=1)
            feature_bound = check_bound(bound=feature_bound, X=X)

            X_mesh, Y_mesh, mesh_instances = mesh(feature_bound, res)

            contour_args = _get_contour_args(contour_dict)

            if ignore_undefined_query_params:
                _, utilities = call_func(
                    qs.query, X=X, y=y, candidates=mesh_instances, **kwargs
                )
            else:
                _, utilities = qs.query(
                    X=X, y=y, candidates=mesh_instances, **kwargs
                )

            for a_idx, ax_ in zip(plot_annotators, axes):
                if n_annotators is not None:
                    utilities_a_idx = utilities[0, :, a_idx]
                else:
                    utilities_a_idx = utilities[0, :]
                utilities_a_idx = utilities_a_idx.reshape(X_mesh.shape)
                ax_.contourf(X_mesh, Y_mesh, utilities_a_idx, **contour_args)

            if n_annotators is None:
                return axes[0]
            else:
                return axes

        except MappingError:
            candidates = unlabeled_indices(y, missing_label=qs.missing_label)
        except BaseException as err:
            warnings.warn(
                f"Unable to create utility plot with mesh because "
                f"of the following error. Trying plotting over "
                f"candidates. \n\n Unexpected {err.__repr__()}"
            )
            candidates = unlabeled_indices(y, missing_label=qs.missing_label)

    candidates = check_array(
        candidates,
        allow_nd=False,
        ensure_2d=False,
        force_all_finite="allow-nan",
    )
    if candidates.ndim == 1:
        X_utils = X
        candidates = check_indices(candidates, X)
    else:
        X_utils = candidates

    if ignore_undefined_query_params:
        _, utilities = call_func(
            qs.query, X=X, y=y, candidates=candidates, **kwargs
        )
    else:
        _, utilities = qs.query(X=X, y=y, candidates=candidates, **kwargs)

    for a_idx, ax_ in zip(plot_annotators, axes):
        if n_annotators is not None:
            utilities_a_idx = utilities[0, :, a_idx]
        else:
            utilities_a_idx = utilities[0, :]
        plot_contour_for_samples(
            X_utils,
            utilities_a_idx,
            replace_nan=replace_nan,
            feature_bound=feature_bound,
            ax=ax_,
            res=res,
            contour_dict=contour_dict,
        )

    if n_annotators is None:
        return axes[0]
    else:
        return axes
