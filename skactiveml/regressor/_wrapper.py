import inspect
import warnings
from copy import deepcopy
from operator import attrgetter

import numpy as np
from scipy.stats import norm
from sklearn.base import MetaEstimatorMixin, is_regressor
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import (
    check_array,
    check_is_fitted,
    check_random_state,
)

from ..base import SkactivemlRegressor, ProbabilisticRegressor
from ..utils import (
    is_labeled,
    match_signature,
    check_n_features,
    check_scalar,
    check_type,
    MISSING_LABEL,
)

successful_skorch_torch_import = False
try:
    import torch
    from torch import nn
    from skactiveml.base import SkorchMixin
    from skactiveml.utils import make_criterion_tuple_aware

    successful_skorch_torch_import = True
except ImportError:  # pragma: no cover
    pass


class SklearnRegressor(SkactivemlRegressor, MetaEstimatorMixin):
    """Sklearn Regressor

    Implementation of a wrapper class for scikit-learn regressors such that
    missing labels can be handled. Therefore, samples with missing values are
    filtered.

    Parameters
    ----------
    estimator : sklearn.base.RegressorMixin with predict method
        scikit-learn regressor.
    include_unlabeled_samples : bool, default=False
        - If `False`, only labeled samples are passed to the `fit` method of
          the `estimator`.
        - If `True`, all samples including the unlabeled ones are passed to
          the `fit` method of the `estimator`. Ensure that your `estimator`
          is able to handle unlabeled samples marked by `missing_label`.
          Otherwise, `missing_label` is interpreted as a regular target value.
    missing_label : scalar or string or np.nan or None, default=np.nan
        Value to represent a missing label.
    random_state : int or RandomState instance or None, default=None
        Determines random number for `predict` method. Pass an int for
        reproducible results across multiple method calls.
    """

    def __init__(
        self,
        estimator,
        include_unlabeled_samples=False,
        missing_label=MISSING_LABEL,
        random_state=None,
    ):
        super().__init__(
            random_state=random_state, missing_label=missing_label
        )
        self.estimator = estimator
        self.include_unlabeled_samples = include_unlabeled_samples

    @match_signature("estimator", "fit")
    def fit(self, X, y, sample_weight=None, **fit_kwargs):
        """Fit the model using X as training data and y as labels.

        Parameters
        ----------
        X : matrix-like of shape (n_samples, n_features)
            The sample matrix X is the feature matrix representing the samples.
        y : array-like of shape (n_samples,)
            It contains the numeric target values of the training samples.
            Missing labels are represented as `self.missing_label`.
        sample_weight : array-like of shape (n_samples,), default=None
            It contains the weights of the training samples´ labels. It
            must have the same shape as y.
        fit_kwargs : dict-like
            Further parameters are passed as input to the `fit` method of the
            'estimator'.

        Returns
        -------
        self: SklearnRegressor,
            The SklearnRegressor is fitted on the training data.
        """
        return self._fit(
            fit_function="fit",
            X=X,
            y=y,
            sample_weight=sample_weight,
            **fit_kwargs,
        )

    @match_signature("estimator", "partial_fit")
    def partial_fit(self, X, y, sample_weight=None, **fit_kwargs):
        """Partially fitting the model using X as training data and y as
        labels.

        Parameters
        ----------
        X : matrix-like of shape (n_samples, n_features)
            The sample matrix X is the feature matrix representing the samples.
        y : array-like of shape (n_samples,)
            It contains the numeric labels of the training samples.
            Missing labels are represented the attribute `self.missing_label`.
        sample_weight : array-like of shape (n_samples,)
            It contains the weights of the training samples' numeric labels. It
            must have the same shape as y.
        fit_kwargs : dict-like
            Further parameters as input to the `fit` method of the `estimator`.

        Returns
        -------
        self : SklearnRegressor,
            The `SklearnRegressor` is fitted on the training data.
        """
        return self._fit(
            fit_function="partial_fit",
            X=X,
            y=y,
            sample_weight=sample_weight,
            **fit_kwargs,
        )

    def _fit(self, fit_function, X, y, sample_weight, **fit_kwargs):
        if not is_regressor(estimator=self.estimator):
            raise TypeError(
                "'{}' must be a scikit-learn "
                "regressor.".format(self.estimator)
            )
        check_type(
            self.include_unlabeled_samples, "include_unlabeled_samples", bool
        )

        self.check_X_dict_ = {
            "ensure_min_samples": 0,
            "ensure_min_features": 0,
            "allow_nd": True,
            "dtype": None,
        }

        X, y, sample_weight = self._validate_data(
            X,
            y,
            sample_weight,
            check_X_dict=self.check_X_dict_,
            reset=fit_function == "fit" or not hasattr(self, "n_features_in_"),
        )

        is_lbld = is_labeled(y, missing_label=self.missing_label_)
        if self.include_unlabeled_samples:
            is_included = np.full_like(y, True, dtype=bool)
        else:
            is_included = is_lbld
        X_train = X[is_included]
        y_train = y[is_included]
        estimator_params = dict(fit_kwargs) if fit_kwargs is not None else {}

        if sample_weight is not None:
            estimator_params["sample_weight"] = sample_weight[is_included]

        self._label_mean = np.mean(y[is_lbld]) if np.sum(is_lbld) > 0 else 0
        self._label_std = np.std(y[is_lbld]) if np.sum(is_lbld) > 1 else 1
        self.estimator_ = deepcopy(self.estimator)
        try:
            attrgetter(fit_function)(self.estimator_)(
                X_train, y_train, **estimator_params
            )
            self.is_fitted_ = True
        except Exception as e:
            warnings.warn(
                f"The 'estimator' could not be fitted because of"
                f" '{e}'. Therefore, the empirical label mean "
                f"`_label_mean={self._label_mean}` and the "
                f"empirical label standard deviation "
                f"`_label_std={self._label_std}` will be used to make "
                f"predictions."
            )
            self.is_fitted_ = False

        return self

    @match_signature("estimator", "predict")
    def predict(self, X, **predict_kwargs):
        """Return label predictions for the input data `X`.

        Parameters
        ----------
        X :  array-like of shape (n_samples, n_features)
            Input samples.
        predict_kwargs : dict-like
            Further parameters are passed as input to the `predict` method of
            the `estimator`. If the estimator could not be fitted, only
            `return_std` is supported as keyword argument.

        Returns
        -------
        y :  ndarray of shape (n_samples,)
            Predicted labels of the input samples.
        """
        check_is_fitted(self)
        predict_dict = {"ensure_min_samples": 1, "ensure_min_features": 1}
        X = check_array(X, **(self.check_X_dict_ | predict_dict))
        check_n_features(self, X, reset=False)
        if self.is_fitted_:
            return self.estimator_.predict(X, **predict_kwargs)

        warnings.warn(
            f"Since the 'estimator' could not be fitted when"
            f" calling the `fit` method, the label "
            f"mean `_label_mean={self._label_mean}` and optionally the "
            f"label standard deviation `_label_std={self._label_std}` is "
            f"used to make the predictions."
        )
        has_std = predict_kwargs.pop("return_std", False)
        if has_std:
            return (
                np.full(len(X), self._label_mean),
                np.full(len(X), self._label_std),
            )
        else:
            return np.full(len(X), self._label_mean)

    @match_signature("estimator", "sample_y")
    def sample_y(self, X, n_samples=1, **sample_kwargs):
        """Assumes a probabilistic regressor. Samples are drawn from a
        predicted target distribution.

        Parameters
        ----------
        X : array-like of shape (n_samples_X, n_features)
            Input samples from which the target values are drawn.
        n_samples : int, default=1
            Number of random samples to be drawn.
        **sample_kwargs : dict
            Additional keyword arguments for sampling. For example:

            random_state : int, RandomState instance or None, default=None
                Determines the random number generation for drawing samples.
                Pass an int for reproducible results across multiple method
                calls.

        Returns
        -------
        y_samples : ndarray of shape (n_samples_X, n_samples)
            Drawn random target samples.
        """
        return self._sample(
            sample_function="sample_y",
            X=X,
            n_samples=n_samples,
            **sample_kwargs,
        )

    @match_signature("estimator", "sample")
    def sample(self, X, n_samples=1, **sample_kwargs):
        """Assumes a probabilistic regressor. Samples are drawn from a
        predicted target distribution.

        Parameters
        ----------
        X : array-like of shape (n_samples_X, n_features)
            Input samples from which the target values are drawn.
        n_samples : int, default=1
            Number of random samples to be drawn.
        **sample_kwargs : dict
            Additional keyword arguments for sampling. For example:

            random_state : int, RandomState instance or None, default=None
                Determines the random number generation for drawing samples.
                Pass an int for reproducible results across multiple method
                calls.

        Returns
        -------
        y_samples : ndarray of shape (n_samples_X, n_samples)
            Drawn random target samples.
        """
        return self._sample(
            sample_function="sample", X=X, n_samples=n_samples, **sample_kwargs
        )

    def _sample(self, sample_function, X, n_samples=1, **sample_kwargs):
        check_is_fitted(self)
        predict_dict = {"ensure_min_samples": 1, "ensure_min_features": 1}
        X = check_array(X, **(self.check_X_dict_ | predict_dict))
        check_n_features(self, X, reset=False)
        try:
            return attrgetter(sample_function)(self.estimator_)(
                X, n_samples, **sample_kwargs
            )
        except NotFittedError:
            warnings.warn(
                f"Since the 'estimator' could not be fitted when"
                f" calling the `fit` method, the label "
                f"mean `_label_mean={self._label_mean}` and optionally the "
                f"label standard deviation `_label_std={self._label_std}` is "
                f"used to make the predictions."
            )
            random_state = sample_kwargs.get("random_state", None)
            random_state = check_random_state(random_state)
            check_scalar(
                n_samples,
                "n_samples",
                min_val=1,
                min_inclusive=True,
                target_type=int,
            )
            y_samples = random_state.randn(len(X), n_samples)
            y_samples *= self._label_std
            y_samples += self._label_mean
            return y_samples

    def __sklearn_is_fitted__(self):
        if hasattr(self, "is_fitted_"):
            return True

        try:
            check_is_fitted(self.estimator)
        except NotFittedError:
            return False

        # set attributes that would be set by the fit function
        self.is_fitted_ = True
        self._label_mean = 0
        self._label_std = 1
        self.estimator_ = deepcopy(self.estimator)
        self.check_X_dict_ = {
            "ensure_min_samples": 0,
            "ensure_min_features": 0,
            "allow_nd": True,
            "dtype": None,
        }

        return True

    def __getattr__(self, item):
        if "estimator_" in self.__dict__:
            return getattr(self.estimator_, item)
        else:
            return getattr(self.estimator, item)


class SklearnNormalRegressor(ProbabilisticRegressor, SklearnRegressor):
    """Sklearn Normal Regressor

    Implementation of a wrapper class for scikit-learn probabilistic regressors
    such that missing labels can be handled and the target distribution can be
    estimated. Therefore, samples with missing values are filtered and a normal
    distribution is fitted using the predicted means and standard deviations.

    The wrapped regressor of sklearn needs `return_std` as a keyword argument
    for `predict`.

    Parameters
    ----------
    estimator : sklearn.base.RegressorMixin with predict method
        scikit-learn regressor.
    missing_label : scalar or string or np.nan or None, default=np.nan
        Value to represent a missing label.
    random_state : int or RandomState instance or None, default=None
        Determines random number for `predict` method. Pass an int for
        reproducible results across multiple method calls.
    """

    def __init__(
        self, estimator, missing_label=MISSING_LABEL, random_state=None
    ):
        super().__init__(
            estimator, missing_label=missing_label, random_state=random_state
        )

    def _fit(self, fit_function, X, y, sample_weight, **fit_kwargs):
        if (
            hasattr(self.estimator, "predict")
            and "return_std"
            not in inspect.signature(self.estimator.predict).parameters.keys()
            and inspect.getfullargspec(self.estimator.predict).varkw is None
        ):
            raise ValueError(
                f"`{self.estimator}` must have keyword argument"
                f"`return_std` for predict."
            )

        return super()._fit(fit_function, X, y, sample_weight, **fit_kwargs)

    def predict_target_distribution(self, X):
        """Returns the estimated target normal distribution conditioned on the
        test samples `X`.

        Parameters
        ----------
        X :  array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        dist : scipy.stats._distn_infrastructure.rv_frozen
            The distribution of the targets at the test samples.

        """
        check_is_fitted(self)

        try:
            loc, scale = SklearnRegressor.predict(self, X, return_std=True)
            return norm(loc=loc, scale=scale)
        except TypeError as e:
            if (
                "predict() got an unexpected keyword argument 'return_std'"
                in str(e)
            ):
                raise ValueError(
                    "SklearnNormalRegressors require the Regressor from"
                    "`sklearn` to accept `return_std`."
                ) from e


if successful_skorch_torch_import:

    class SkorchRegressor(SkactivemlRegressor, SkorchMixin):
        """SkorchRegressor

        Implement a regression wrapper class, to make it possible to use
        `torch` with `skactiveml`. This is achieved by providing a wrapper
        around `torch` that has a `skactiveml` interface and can handle
        missing labels. This wrapper is based on the open-source library
        `skorch` [1]_.

        Notes
        -----
        Adjust your `criterion` and `module.forward` outputs consistently.
        See the documentation of the parameters `forward_outputs` and
        `criterion_output_keys` for further details.

        Parameters
        ----------
        module : torch.nn.Module.__class__ or torch.nn.Module
            A PyTorch `torch.nn.Module`. In general, the uninstantiated class
            should be passed, although instantiated modules will also work.
        criterion : torch.nn.Module or torch.nn.Module.__class__, \
                default=torch.nn.MSELoss
            The loss (criterion) used to optimize the module.

            - If a class (subclass of `torch.nn.Module`) is passed
              (e.g. `torch.nn.MSELoss`), it is instantiated
              internally.
            - If an instance is passed (e.g. `torch.nn.MSELoss()`),
              that instance (or a wrapped copy of it) is used.

            By default, `torch.nn.MSELoss` is used as criterion.
        forward_outputs : dict[str, tuple[int, Callable | None]] or None,\
                default=None
            Dictionary that describes how to get and post-process the outputs
            of `module.forward` for prediction. This parameter replaces the
            functionality of `predict_nonlinearity` in a `skorch.net.NeuralNet`
            (see documentation of `neural_net_param_dict`).

            Given `raw_outputs = module.forward(x)`, each entry
            `name -> (idx, transform)` in `forward_outputs` is interpreted as:

            - `idx` : int
              Index into `raw_outputs` (0-based).
            - `transform` : callable or `None`
              If not `None`, it is applied to the selected raw tensor
              `raw_outputs[idx]`. Otherwise, the raw tensor is used.

            This allows multiple named outputs to reference the same raw tensor
            with different transforms, for example::

                forward_outputs = {
                    "raw-pred": (0, None),      # raw predicted targets
                    "log-pred": (0, torch.log), # log predicted targets
                    "emb":      (1, None),      # embeddings
                }

            The first entry in `forward_outputs` defines the primary
            scores used for prediction:

            - In `predict`, the transformed first output is interpreted as
              predicted targets.

            If ``forward_outputs`` is ``None``, a sensible default is chosen
            for common single-output regressors based on the ``criterion``:

            - If ``criterion`` is ``torch.nn.MSELoss``, ``torch.nn.L1Loss``, or
              ``torch.nn.SmoothL1Loss``, it is assumed that ``module.forward``
              returns the regression predictions directly and the effective
              mapping is::

                  {"output": (0, torch.ravel)}

            - For all other criteria, a single-output module is assumed to
              already produce values in the target space, and the effective
              mapping is::

                  {"output": (0, None)}

        criterion_output_keys : str or sequence of str or None, default=None
            Name or names of the forward outputs that are passed to the
            loss / criterion during training. Use this when
            `module.forward` returns multiple outputs
            (e.g. `(logits, embeddings, ...)`), but the criterion expects
            a single tensor input or a specific tuple of inputs.

            The names must refer to keys of the effective `forward_outputs`
            mapping. If `criterion_output_keys` is not `None` and
            `forward_outputs` is `None`, a `ValueError` is raised
            because the names cannot be resolved.

            - If a `str`, the corresponding named output of
              `module.forward` (i.e., the raw tensor selected via its
              index in `forward_outputs` before applying the transform)
              is passed to the criterion (e.g. `"raw-pred"` to use only the
              raw predicted targets).
            - If a sequence of `str`, the selected named outputs are passed to
              the criterion in that order. Each raw forward output index may
              appear at most once: using multiple names that resolve to the
              same underlying index (e.g. `"raw-pred"` and `"log-pred"` both
              pointing to index 0) is not allowed and results in a
              `ValueError`.
            - If `None`, the first output defined by the effective
              `forward_outputs` mapping is used as criterion input.

            To pass all distinct forward outputs to the criterion in the
            same order as `forward_outputs`, choose one representative name
            per raw output index and set, for example::

                # assuming that each key refers to a different raw index
                criterion_output_keys = tuple(forward_outputs.keys())

            If `forward_outputs` contains multiple names that refer to the
            same raw output index (aliases such as `"raw-pred"` and`"log-pred"`
            both mapping to index 0), you must select at most one name per
            raw index in `criterion_output_keys`.
        neural_net_param_dict : dict, default=None
            Additional arguments for `skorch.net.NeuralNet`. If
            `neural_net_param_dict` is `None`, no additional arguments are
            added.
        sample_dtype : str or type, default=np.float32
            Dtype to which input samples are cast inside the estimator. If set
            to `None`, the input dtype is preserved. The label data type is
            always cast to  `np.float32`.
        include_unlabeled_samples : bool, default=False
            - If `False`, only labeled samples are passed to the `fit` method
              of the `estimator`.
            - If `True`, all samples including the unlabeled ones are passed to
              the `fit` method of the `estimator`. Ensure that the `criterion`
              is able to handle unlabeled samples marked by `missing_label`.
              Otherwise, `missing_label` is interpreted as a regular target
              value.
        missing_label : scalar or string or np.nan or None, default=np.nan
            Value to represent a missing label.
        random_state : int or RandomState instance or None, default=None
            Determines random number for 'predict' method. Pass an int for
            reproducible results across multiple method calls.

        References
        ----------
        .. [1] Marian Tietz, Thomas J. Fan, Daniel Nouri, Benjamin Bossan, and
           skorch Developers. skorch: A scikit-learn compatible neural network
           library that wraps PyTorch, July 2017.
        """

        def __init__(
            self,
            module,
            criterion=nn.MSELoss,
            forward_outputs=None,
            criterion_output_keys=None,
            neural_net_param_dict=None,
            sample_dtype=np.float32,
            include_unlabeled_samples=False,
            missing_label=MISSING_LABEL,
            random_state=None,
        ):
            super(SkorchRegressor, self).__init__(
                missing_label=missing_label,
                random_state=random_state,
            )
            self.module = module
            self.criterion = criterion
            self.forward_outputs = forward_outputs
            self.criterion_output_keys = criterion_output_keys
            self.neural_net_param_dict = neural_net_param_dict
            self.include_unlabeled_samples = include_unlabeled_samples
            self.sample_dtype = sample_dtype

        def fit(self, X, y, **fit_params):
            """Initialize and fit the module.

            If the module was already initialized, by calling fit, the module
            will be re-initialized (unless `warm_start` is True).

            Parameters
            ----------
            X : matrix-like, shape (n_samples, n_features)
                Training data set, usually complete, i.e. including the labeled
                and unlabeled samples
            y : array-like of shape (n_samples,)
                Labels of the training data set (possibly including unlabeled
                ones indicated by self.missing_label)
            fit_params : dict-like
                Further parameters as input to the 'fit' method of the
                `skorch.net.NeuralNet`.

            Returns
            -------
            self: SkorchRegressor,
                `SkorchRegressor` fitted on the training data.
            """
            return self._fit("fit", X, y, **fit_params)

        def partial_fit(self, X, y, **fit_params):
            """Fit the module without re-initialization.

            If the module was already initialized, by calling `partial_fit`,
            the module will not be re-initialized again.

            Parameters
            ----------
            X : matrix-like, shape (n_samples, n_features)
                Training data set, usually complete, i.e. including the labeled
                and unlabeled samples
            y : array-like of shape (n_samples, )
                Labels of the training data set (possibly including unlabeled
                ones indicated by `self.missing_label`)
            fit_params : dict-like
                Further parameters as input to the 'partial_fit' method of the
                `skorch.net.NeuralNet`.

            Returns
            -------
            self: SkorchRegressor,
                `SkorchRegressor` object fitted on the training data.
            """
            return self._fit("partial_fit", X, y, **fit_params)

        def predict(self, X, extra_outputs=None):
            """Return predicted targets for the test data `X`.

            By default, this method returns only the predicted targets
            `y_pred`. If `extra_outputs` is provided, a tuple is returned whose
            first element is `y_pred` and whose remaining elements are the
            requested additional forward outputs, in the order specified by
            `extra_outputs`.

            Parameters
            ----------
            X : array-like of shape (n_samples, ...)
                Test samples.
            extra_outputs : None or str or sequence of str, default=None
                Names of additional outputs to return next to `y_pred`. The
                names must be a subset of the keys of the effective
                `forward_outputs` mapping.

                For example, if::

                    self.forward_outputs = {
                        "raw-pred": (0, None),
                        "log-pred": (0, None),
                        "emb":      (1, None),
                    }

                then valid values for `extra_outputs` include `"emb"` or
                `["emb", "log-pred"]`.

                - If `extra_outputs is None`, only `y_pred` is returned.
                - If `extra_outputs` is a string, e.g. `"emb"`, the
                  return value is `(y_pred, emb)`.
                - If `extra_outputs` is a sequence of strings, the return
                  value is `(y_pred, out_1, out_2, ...)`, where `out_i`
                  corresponds to the i-th name in `extra_outputs`.

            Returns
            -------
            y_pred : numpy.ndarray of shape (n_samples,)
                Predicted targets of the test samples.
            *extras : numpy.ndarray, optional
                Additional outputs. Only present if `extra_outputs` is not
                `None`. In that case, the method returns a single tuple whose
                first element is `y_pred` and whose remaining elements
                (`extras`) correspond to the requested forward outputs in the
                order given by `extra_outputs`.
            """
            # Initialize module, if not done yet.
            if not hasattr(self, "neural_net_"):
                self.initialize()

            # Check input parameters.
            X = check_array(X, **self.check_X_dict_)
            check_n_features(
                self, X, reset=not hasattr(self, "n_features_in_")
            )

            # Resolve effective forward_outputs (either user-provided or
            # defaulted based on the criterion).
            forward_outputs = self._effective_forward_outputs()

            # Forward propagation whose return values depends on the request
            # ones.
            return self._forward_with_named_outputs(
                X, forward_outputs=forward_outputs, extra_outputs=extra_outputs
            )

        def _effective_forward_outputs(self):
            """Return the effective `forward_outputs` mapping.

            If the user did not specify `forward_outputs`, choose a reasonable
            default for common criteria (e.g., `nn.MSELoss`) and a
            simple single-output module.

            The returned mapping has the form::

                {name: (idx, transform)}

            where `idx` is the index into the tuple returned by
            `module.forward` (0-based) and `transform` is a callable or
            `None`. For the defaults below, a single-output module is assumed,
            i.e., `idx == 0`.
            """
            # User explicitly provided a mapping: trust it.
            if self.forward_outputs is not None:
                return self.forward_outputs

            # No explicit mapping: handle common single-output cases.
            crit_cls = (
                self.criterion
                if isinstance(self.criterion, type)
                else self.criterion.__class__
            )

            if crit_cls in [nn.MSELoss, nn.L1Loss, nn.SmoothL1Loss]:
                # Module returns raw predictions.
                return {"output": (0, torch.ravel)}

            # Fallback: treat the single forward output as already in desired
            # target space. Caller is responsible for making this true.
            return {"output": (0, None)}

        def _net_parts(self, X=None, y=None):
            """Assemble and validate network components.

            Implementations should perform any optional checks or normalization
            of constructor/init parameters (e.g., shape consistency, dtype
            checks, wrapping criteria), then return the ready-to-use pieces for
            `skorch.NeuralNet`.

            Parameters
            ----------
            X : array-like of shape (n_samples, ...), default=None
                Input samples for optional validation.
            y : array-like of shape (n_samples, ...), default=None
                Target values for optional validation.

            Returns
            -------
            module : torch.nn.Module.__class__ or torch.nn.Module
                A PyTorch `torch.nn.Module`. In general, the uninstantiated
                class should be passed, although instantiated modules will also
                work.
            criterion : torch.nn.Module.__class__
                The uninitialized criterion (loss) used to optimize the module.
            predict_nonlinearity : Callable
                The nonlinearity to be applied to the prediction.
            params : dict
                Keyword arguments (excluding `predict_non_linearity`) for
                `skorch.NeuralNet` construction. Must be a mapping and may be
                empty.
            """
            criterion = self.criterion
            criterion = make_criterion_tuple_aware(
                criterion=criterion,
                criterion_output_keys=self.criterion_output_keys,
                forward_outputs=self._effective_forward_outputs(),
            )
            return (
                self.module,
                criterion,
                self.neural_net_param_dict or {},
            )

        def _validate_data_kwargs(self):
            """Return kwargs forwarded to `_validate_data`.

            Returns
            -------
            kwargs : dict or None
                Keyword arguments consumed by `_validate_data`.
            """
            self.check_X_dict_ = {
                "ensure_min_samples": 0,
                "ensure_min_features": 0,
                "allow_nd": True,
                "dtype": self.sample_dtype,
            }
            check_type(
                self.include_unlabeled_samples,
                "include_unlabeled_samples",
                bool,
            )
            return {"check_X_dict": self.check_X_dict_}

        def _return_training_data(self, X, y):
            """
            Return only samples and labels required for training.

            Parameters
            ----------
            X : array-like of shape (n_samples, ...)
                Input samples.
            y : array-like of shape (n_samples, ...)
                Targets with unlabeled entries following the subclass'
                convention.

            Returns
            -------
            X_train : ndarray or None
                Training samples or `None` if none exist.
            y_train : ndarray or None
                Training labels or `None` if none exist.
            """
            X_train, y_train = None, None
            if self.include_unlabeled_samples:
                is_included = np.full_like(y, fill_value=True, dtype=bool)
            else:
                is_included = is_labeled(y, missing_label=self.missing_label_)
            if np.sum(is_included) > 0:
                X_train = X[is_included]
                y_train = y[is_included]
            if y_train is not None:
                y_train = y_train.astype(np.float32, copy=True).reshape(-1, 1)
            return X_train, y_train
