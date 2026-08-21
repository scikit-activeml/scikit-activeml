import warnings

import numpy as np
from matplotlib import lines, pyplot as plt
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
from ..pool._target import (
    _collect_declared_authorities,
    _reconcile_target_declarations,
)
from ..utils._validation import _canonicalize_multilabel_probas
from ..utils import (
    check_scalar,
    unlabeled_indices,
    call_func,
    check_type,
    check_indices,
)


def _resolve_utility_target_type(qs, y, query_kwargs):
    """Resolve target semantics for utility-plot candidate fallback."""
    if isinstance(qs, MultiAnnotatorPoolQueryStrategy):
        return "single-output"

    wrapped_resolver = getattr(qs, "_resolve_wrapped_target_type", None)
    if wrapped_resolver is not None:
        return wrapped_resolver(y, query_kwargs)

    authorities = _collect_declared_authorities(
        qs._target_authority_params, query_kwargs
    )
    target_type, _ = _reconcile_target_declarations(
        [(qs.target_type, type(qs).__name__)],
        authorities,
        y,
        missing_label=qs.missing_label,
        owner_name=type(qs).__name__,
    )
    return target_type


def plot_utilities(qs, X, y, candidates=None, **kwargs):
    """Plot the utility for the given single-annotator query strategy.

    Parameters
    ----------
    qs : skactiveml.base.SingleAnnotatorPoolQueryStrategy
        The query strategy for which the utility is plotted.
    X : array-like of shape (n_samples, n_features)
        Training data set, usually complete, i.e., including the labeled and
        unlabeled samples.
    y : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Labels of the training data set (possibly including unlabeled ones
        indicated by `qs.missing_label`). Target semantics are resolved by the
        query strategy and any estimator it declares as authoritative. For
        multi-label targets, all label outputs contribute to one utility
        surface because the query strategy returns one utility per sample.
    candidates : None or array-like of shape (n_candidates), dtype=int or \
            array-like of shape (n_candidates, n_features), default=None
        - If `candidates` is `None`, the unlabeled samples from
          `(X,y)` are considered as `candidates`.
        - If `candidates` is of shape `(n_candidates,)` and of type
          `int`, `candidates` is considered as the indices of the
          samples in `(X,y)`.
        - If `candidates` is of shape `(n_candidates, ...)`, the
          candidate samples are directly given in `candidates` (not
          necessarily contained in `X`). This is not supported by all
          query strategies.

    Other Parameters
    ----------------
    replace_nan : numeric or None, default=0.0
        Only used if plotting with mesh samples is not possible.
        If numeric, the utility of labeled samples will be plotted with
        value `replace_nan`. If None, these samples will be ignored.
    ignore_undefined_query_params : bool, default=False
        If True, query parameters that are not defined in the query function
        are ignored and will not raise an exception.
    feature_bound : array-like of shape [[xmin, ymin], [xmax, ymax]],\
            default=None
        Determines the area in which the boundary is plotted. If candidates is
        not given, bound must not be None. Otherwise, the bound is determined
        based on the data.
    ax : matplotlib.axes.Axes, default=None
        The axis on which the utility is plotted.
    res : int, default=21
        The resolution of the plot.
    contour_dict : dict, default=None
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
    qs : skactiveml.base.MultiAnnotatorPoolQueryStrategy
        The query strategy for which the utility is plotted.
    X : array-like of shape (n_samples, n_features)
        Training data set, usually complete, i.e. including the labeled and
        unlabeled samples.
    y : array-like of shape (n_samples,) or (n_samples, n_annotators)
        Labels of the training data set (possibly including unlabeled ones
        indicated by `qs.missing_label`).
    candidates : None or array-like of shape (n_candidates), dtype=int or \
            array-like of shape (n_candidates, n_features), default=None
        - If `candidates` is `None`, the unlabeled samples from
          `(X,y)` are considered as `candidates`.
        - If `candidates` is of shape `(n_candidates,)` and of type
          `int`, `candidates` is considered as the indices of the
          samples in `(X,y)`.
        - If `candidates` is of shape `(n_candidates, ...)`, the
          candidate samples are directly given in `candidates` (not
          necessarily contained in `X`). This is not supported by all
          query strategies.

    Other Parameters
    ----------------
    replace_nan : numeric or None, default=0.0
        Only used if plotting with mesh samples is not possible.
        If numeric, the utility of labeled samples will be plotted with
        value `replace_nan`. If None, these samples will be ignored.
    ignore_undefined_query_params : bool, default=False
        If True, query parameters that are not defined in the query function
        are ignored and will not raise an exception.
    feature_bound : array-like of shape [[xmin, ymin], [xmax, ymax]],\
            default=None
        Determines the area in which the boundary is plotted. If candidates is
        not given, bound must not be None. Otherwise, the bound is determined
        based on the data.
    axes : array-like of matplotlib.axes.Axes, default=None
        The axes on which the utilities for the annotators are plotted. Only
        supported for y.ndim = 2 corresponding to a setting with multiple
        annotators.
    res : int, default=21
        The resolution of the plot.
    contour_dict : dict, default=None
        Additional parameters for the utility contour.
    plot_annotators : None or array-like of shape (n_annotators_to_plot,),\
            default=None
        Contains the indices of the annotators to be plotted. If it is None,
        all annotators are plotted. Only supported for y.ndim = 2 corresponding
        to a setting with multiple annotators.
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
    clf : sklearn.base.ClassifierMixin
        The fitted classifier whose decision boundary is plotted. If confidence
        is not None, the classifier must implement the `predict_proba` method.
        A multi-label classifier must publish its resolved semantics through
        `target_spec_` and implement `predict_proba`.
    feature_bound : array-like of shape [[xmin, ymin], [xmax, ymax]]
        Determines the area in which the boundary is plotted.
    ax : matplotlib.axes.Axes or array-like of matplotlib.axes.Axes, \
            default=None
        The axis on which the decision boundary is plotted. For multi-label
        classification, one axis overlays all label-output boundaries, while
        an array-like must contain one axis per label output and plots output
        `j` on axis `j`.
    res : int, default=21
        The resolution of the plot.
    boundary_dict : dict, default=None
        Additional parameters for the boundary contour.
    confidence : scalar or None, default=0.75
        The confidence interval plotted with dashed lines. It is not plotted if
        confidence is None. Must be in the open interval (0.5, 1). The value
        stands for the ratio best class / second best class. For each binary
        label output of a multi-label classifier, the dashed contours are drawn
        at positive-class probabilities `1 - confidence` and `confidence`.
    cmap : str or matplotlib.colors.Colormap, default='coolwarm'
        The colormap for the confidence levels and, unless overridden through
        `boundary_dict`, the multi-label output boundaries.
    confidence_dict : dict, default=None
        Additional parameters for the confidence contour. Must not contain a
        colormap because cmap is used.

    Returns
    -------
    ax : matplotlib.axes.Axes or array-like of matplotlib.axes.Axes
        The supplied axis or axes on which the boundaries were plotted. A
        multi-label boundary is the `0.5` contour of each label output's
        positive-class probability, whose positive class is
        `clf.target_spec_.classes[j][1]`.
    """
    check_type(clf, "clf", ClassifierMixin)
    check_scalar(res, "res", int, min_val=1)
    target_spec = getattr(clf, "target_spec_", None)
    is_multilabel = (
        target_spec is not None and target_spec.target_type == "multi-label"
    )
    if ax is None:
        ax = plt.gca()
    if is_multilabel and not isinstance(ax, Axes):
        axes = np.asarray(ax, dtype=object).reshape(-1)
        for ax_ in axes:
            check_type(ax_, "ax", Axes)
        n_outputs = len(target_spec.classes)
        if len(axes) != n_outputs:
            raise ValueError(
                "`ax` must contain one `Axes` object for each label output."
            )
    else:
        check_type(ax, "ax", Axes)
        axes = np.array([ax], dtype=object)
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
    X_mesh, Y_mesh, mesh_samples = mesh(feature_bound, res)

    if is_multilabel:
        if not hasattr(clf, "predict_proba"):
            raise AttributeError(
                "A multi-label `clf` must implement `predict_proba`."
            )
        n_outputs = len(target_spec.classes)
        predictions = _canonicalize_multilabel_probas(
            clf.predict_proba(mesh_samples),
            n_samples=len(mesh_samples),
            n_outputs=n_outputs,
        )
        output_axes = [axes[0]] * n_outputs if len(axes) == 1 else axes
        norm = plt.Normalize(vmin=0, vmax=max(n_outputs - 1, 1))

        for output_idx, ax_ in enumerate(output_axes):
            posteriors = predictions[:, output_idx].reshape(X_mesh.shape)
            output_boundary_args = boundary_args.copy()
            if boundary_dict is None or "colors" not in boundary_dict:
                output_boundary_args["colors"] = [cmap(norm(output_idx))]
            ax_.contour(
                X_mesh,
                Y_mesh,
                posteriors,
                [0.5],
                **output_boundary_args,
            )
            if confidence is not None:
                ax_.contour(
                    X_mesh,
                    Y_mesh,
                    posteriors,
                    [1 - confidence, confidence],
                    colors=[cmap(norm(output_idx))],
                    **confidence_args,
                )
        return ax

    # Calculate predictions
    if hasattr(clf, "predict_proba"):
        predictions = clf.predict_proba(mesh_samples)
        if isinstance(predictions, list):
            raise ValueError(
                "`clf.predict_proba` returned per-output probability "
                "matrices, but `clf` does not publish a resolved multi-label "
                "`target_spec_`."
            )
        classes = np.arange(predictions.shape[1])
    elif hasattr(clf, "predict"):
        if confidence is not None:
            warnings.warn(
                "The given classifier does not implement "
                "'predict_proba'. Thus, the confidence cannot be "
                "plotted."
            )
            confidence = None
        predicted_classes = clf.predict(mesh_samples)
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
    res=21,
    contour_dict=None,
):
    """Plot the utility for the given query strategy.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training data set, usually complete, i.e., including the labeled and
        unlabeled samples.
    values : array-like of shape (n_samples,)
        Values to plot for samples `X` (may contain np.nan, can be replaced
        or ignored, see `replace_nan`).
    replace_nan : numeric or None, default=0.0
        If numeric, nan-values in `values` will be replaced by this number.
        If None, these samples will be ignored.
    feature_bound : array-like of shape [[xmin, ymin], [xmax, ymax]]
        Determines the area in which the boundary is plotted. If candidates is
        not given, bound must not be None. Otherwise, the bound is determined
        based on the data.
    ax : matplotlib.axes.Axes, default=None
        The axis on which the utility is plotted. If no axis is given, the
        current axis (`plt.gca()`) will be used instead.
    res : int, default=21
        The resolution of the plot.
    contour_dict : dict, default=None
        Additional parameters for the utility contour.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axis on which the utility was plotted.
    """
    check_array(X, ensure_2d=True)
    values = check_array(
        values, ensure_2d=False, ensure_all_finite=False, copy=True
    )
    values[np.isinf(values)] = np.nan

    feature_bound = check_bound(bound=feature_bound, X=X)

    X_mesh, Y_mesh, mesh_samples = mesh(feature_bound, res)

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

    scores = neighbors.predict(mesh_samples).reshape(X_mesh.shape)
    ax.contourf(X_mesh, Y_mesh, scores, **contour_args)
    return ax


def plot_stream_training_data(
    ax,
    X,
    y,
    queried_indices,
    classes,
    feature_bound,
    unlabeled_color="grey",
    cmap="coolwarm",
    alpha=0.2,
    linewidth=3,
    plot_cand_highlight=True,
):
    """Plot the utility for the given query strategy.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis on which the utility is plotted. Only if y.ndim = 1 (single
        annotator).
    X : array-like of shape (n_samples, 1)
        Training data set, usually complete, i.e., including the labeled and
        unlabeled samples.
    y : array-like of shape (n_samples,)
        Labels of the training data set (possibly including unlabeled ones).
    queried_indices : array-like of shape (n_samples,)
        Indicates which samples in `X` have been queried.
    classes : array-like of shape (n_classes,)
        Holds the label for each class.
    feature_bound : array-like of shape [[xmin, ymin], [xmax, ymax]]
        Determines the area in which the boundary is plotted. If candidates is
        not given, bound must not be None. Otherwise, the bound is determined
        based on the data.
    unlabeled_color : str or matplotlib.colors.Colormap, default='grey'
        The color for the unlabeled samples.
    cmap : str or matplotlib.colors.Colormap, default='coolwarm_r'
        The colormap for the confidence levels.
    alpha : scalar, default=0.2
        Set the alpha value used for blending - not supported on all backends.
    linewidth : float, default=3
        Set the line width in points.
    plot_cand_highlight : bool, default=True
        The indicator to highlight the current candidate.

    Returns
    -------
     axes : array-like of shape (n_annotators_to_plot,)
         The axes on which the utilities were plotted.
    """
    column_or_1d(X)
    check_array(y, ensure_2d=False, ensure_all_finite="allow-nan")
    check_consistent_length(X, y)
    check_array(queried_indices, ensure_2d=False)
    check_array(classes, ensure_2d=False)
    check_type(unlabeled_color, "unlabeled_color", str)
    check_type(plot_cand_highlight, "plot_cand_highlight", bool)
    check_type(ax, "ax", Axes)

    data_lines = []
    cmap = _get_cmap(cmap)
    norm = plt.Normalize(vmin=min(classes), vmax=max(classes))

    highlight_color = (
        cmap(norm(y[-1])) if queried_indices[-1] else unlabeled_color
    )

    if plot_cand_highlight:
        data_lines.append(
            lines.Line2D(
                [0, feature_bound[0][1]],
                [X[-1], X[-1]],
                c=highlight_color,
                alpha=alpha,
                linewidth=linewidth * 2,
            )
        )

    for t, (x_t, a, y_t) in enumerate(zip(X, queried_indices, y)):
        line_color = cmap(norm(y_t)) if a else unlabeled_color
        zorder = 3 if a else 2
        alpha_tmp = alpha * 2 if a else alpha
        data_lines.append(
            lines.Line2D(
                [t, len(X) - 1],
                [x_t, x_t],
                zorder=zorder,
                color=line_color,
                alpha=alpha_tmp,
                linewidth=linewidth,
            )
        )
    for d_line in data_lines:
        ax.add_line(d_line)
    return data_lines


def plot_stream_decision_boundary(
    ax,
    t_x,
    plot_step,
    clf,
    X,
    pred_list,
    color="k",
    res=25,
):
    """Plot the decision boundary of the given classifier.

    Parameters
    ----------
    ax : matplotlib.axes.Axes or List
        The axis on which the decision boundary is plotted. If ax is a List,
        each entry has to be an `matplotlib.axes.Axes`.
    t_x : int
        The position of the newest instance for the x axies.
    plot_step : int
        The interval in which the clf should predict new samples.
    clf : sklearn.base.ClassifierMixin
        The fitted classifier whose decision boundary is plotted.
    X : array-like of shape (n_samples, 1)
        Training data set, usually complete, i.e. including the labeled and
        unlabeled samples.
    pred_list : array-like of shape (n_samples,)
        The list containing classifier prediction for the last steps.
    color : str or matplotlib.colors.Colormap, default='k'
        The color for the decision boundary.
    res : int, default=25
        The resolution of the plot.

    Returns
    -------
    ax : matplotlib.axes.Axes or List
        The axis on which the boundary was plotted or the list of axis if ax
        was a list.
    pred_list : array-like of shape (n_samples,)
        The list containing classifier prediction for the last steps.
    """
    X = column_or_1d(X)
    check_array(pred_list, ensure_2d=False, ensure_min_samples=0)
    check_scalar(t_x, "t_x", int, min_val=0)
    check_scalar(plot_step, "plot_step", int, min_val=1)
    check_type(ax, "ax", Axes)
    check_type(clf, "clf", ClassifierMixin)
    x_vec = np.linspace(np.min(X), np.max(X), res)
    t_vec = np.arange(1, t_x // plot_step + 1) * plot_step
    t_mesh, x_mesh = np.meshgrid(t_vec, x_vec)
    predictions = np.array([clf.predict(x_vec.reshape([-1, 1]))])
    pred_list.extend(predictions)

    if len(pred_list) > 2 and np.sum(pred_list) > 0:
        ax.contour(
            t_mesh,
            x_mesh,
            np.array(pred_list[1:]).T,
            levels=[0.5],
            colors=color,
        )
    return ax, pred_list


def _general_plot_utilities(qs, X, y, candidates=None, **kwargs):
    """Plot the utility for the given query strategy.

    Parameters
    ----------
    qs : skactiveml.base.QueryStrategy
        The query strategy for which the utility is plotted.
    X : array-like of shape (n_samples, n_features)
        Training data set, usually complete, i.e. including the labeled and
        unlabeled samples.
    y : array-like of shape (n_samples,) or (n_samples, n_outputs) or \
            (n_samples, n_annotators)
        Labels of the training data set (possibly including unlabeled ones
        indicated by `qs.missing_label`). A single-annotator strategy produces
        one utility surface, including for resolved multi-label targets. For
        multi-annotator strategies, the columns of a two-dimensional `y`
        represent annotators and produce one surface per selected annotator.
    candidates : None or array-like of shape (n_candidates), dtype=int or \
            array-like of shape (n_candidates, n_features), default=None
        - If `candidates` is `None`, the unlabeled samples from
          `(X,y)` are considered as `candidates`.
        - If `candidates` is of shape `(n_candidates,)` and of type
          `int`, `candidates` is considered as the indices of the
          samples in `(X,y)`.
        - If `candidates` is of shape `(n_candidates, ...)`, the
          candidate samples are directly given in `candidates` (not
          necessarily contained in `X`). This is not supported by all
          query strategies.

    Other Parameters
    ----------------
    replace_nan : numeric or None, default=0.0
        Only used if plotting with mesh samples is not possible.
        If numeric, the utility of labeled samples will be plotted with
        value `replace_nan`. If None, these samples will be ignored.
    ignore_undefined_query_params : bool, default=False
        If True, query parameters that are not defined in the query function
        are ignored and will not raise an exception.
    feature_bound : array-like of shape [[xmin, ymin], [xmax, ymax]],\
            default=None
        Determines the area in which the boundary is plotted. If candidates is
        not given, bound must not be None. Otherwise, the bound is determined
        based on the data.
    ax : matplotlib.axes.Axes, default=None
        The axis on which the utility of a single-annotator strategy is
        plotted.
    axes : array-like of matplotlib.axes.Axes, default=None
        The axes on which the utilities for the annotators are plotted. Only
        supported for a multi-annotator strategy with a two-dimensional `y`.
    res : int, default=21
        The resolution of the plot.
    contour_dict : dict, default=None
        Additional parameters for the utility contour.
    plot_annotators : None or array-like of shape (n_annotators_to_plot,),\
            default=None
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
    y = check_array(y, ensure_2d=False, ensure_all_finite="allow-nan")
    check_consistent_length(X, y)

    is_multi_annotator = isinstance(qs, MultiAnnotatorPoolQueryStrategy)
    if is_multi_annotator and y.ndim == 2:
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

            X_mesh, Y_mesh, mesh_samples = mesh(feature_bound, res)

            contour_args = _get_contour_args(contour_dict)

            if ignore_undefined_query_params:
                _, utilities = call_func(
                    qs.query, X=X, y=y, candidates=mesh_samples, **kwargs
                )
            else:
                _, utilities = qs.query(
                    X=X, y=y, candidates=mesh_samples, **kwargs
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
            pass
        except BaseException as err:
            warnings.warn(
                f"Unable to create utility plot with mesh because "
                f"of the following error. Trying plotting over "
                f"candidates. \n\n Unexpected {err.__repr__()}"
            )
        target_type = _resolve_utility_target_type(qs, y, kwargs)
        fallback_y = (
            y[:, 0]
            if not is_multi_annotator
            and target_type == "single-output"
            and y.ndim == 2
            else y
        )
        candidates = unlabeled_indices(
            fallback_y,
            missing_label=qs.missing_label,
            target_type=target_type,
        )

    candidates = check_array(
        candidates,
        allow_nd=False,
        ensure_2d=False,
        ensure_all_finite="allow-nan",
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
