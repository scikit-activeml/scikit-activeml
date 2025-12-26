try:
    import numpy as np
    import torch

    from abc import ABC, abstractmethod
    from torch import nn
    from torch.utils.data import default_collate

    from ...classifier import SkorchClassifier
    from ...utils import (
        MISSING_LABEL,
        is_labeled,
        check_classes,
    )

    class _SkorchMultiAnnotatorClassifier(SkorchClassifier, ABC):
        """
        Abstract base class for neural multi-annotator classifiers built on top
        of `SkorchClassifier`.

        This estimator wraps a multi-annotator module that operates on
        annotation matrices `y` of shape `(n_samples, n_annotators)` and
        internally uses a classifier module `clf_module`. The class takes care
        of:

        - validating and storing the class labels `classes_`,
        - validating and storing the number of annotators `n_annotators_`,
        - splitting and normalizing `neural_net_param_dict` into parameters
          for the multi-annotator module and the underlying classifier, and
        - enforcing skorch parameter overrides provided by subclasses via
          `_build_neural_net_param_overrides`.

        Subclasses are expected to implement
        `_build_neural_net_param_overrides` to inject architecture-specific
        parameters (e.g. embedding dimensions, number of annotators, number of
        classes) into the skorch network configuration.

        Parameters
        ----------
        multi_annotator_module : nn.Module.__class__ or nn.Module
            PyTorch module (or module class) implementing the multi-annotator
            model. It is passed as the `module` argument to
            `SkorchClassifier`. If a class is given, it is instantiated
            by `skorch` using `neural_net_param_dict`.
        criterion : nn.Module.__class__ or nn.Module
            Loss function used by skorch. Passed through to `SkorchClassifier`
            as `criterion`.
        clf_module : nn.Module or type
            Backbone / head used by the multi-annotator module to map inputs
            `x` to class logits (and optionally embeddings). This is exposed
            to the module via the enforced skorch parameters
            `"module__clf_module"` and `"module__clf_module_param_dict"`.
        n_annotators : int or None, default=None
            Number of annotators (i.e., columns in `y`). If not given, it is
            inferred from the shape of `y` on the first call to
            `_net_parts`. If given and inconsistent with `y.shape[1]`,
            a `ValueError` is raised.
        neural_net_param_dict : dict or None, default=None
            Dictionary of keyword arguments that configure the underlying
            skorch `NeuralNet`. Keys starting with `"module__"` are
            treated as arguments for the multi-annotator module and are split
            off into `clf_module_param_dict` when building the network.
            Subclasses may add further enforced parameters via
            `_build_neural_net_param_overrides`.
        sample_dtype : str or type, default=np.float32
            Dtype to which input samples are cast inside the estimator. If set
            to `None`, the input dtype is preserved.
        classes : array-like of shape (n_classes,), default=None
            List or array of class labels. If provided, it is validated and
            stored as `classes_` before network initialization. If omitted,
            subclasses must ensure that class information is available before
            fitting; otherwise a `RuntimeError` is raised.
        cost_matrix : array-like of shape (n_classes, n_classes), default=None
            Misclassification cost matrix used by `SkorchClassifier`.
        missing_label : int or float, default=MISSING_LABEL
            Value in `y` indicating an unlabeled entry. The exact convention
            is respected by downstream utilities such as :func:`is_labeled`.
        random_state : int, numpy.random.RandomState, or None, default=None
            Random seed or random state used for reproducible initialization
            and training, forwarded to `SkorchClassifier`.

        Notes
        -----
        This class is intended as an internal base class. Concrete
        multi-annotator estimators should:

        - implement `_build_neural_net_param_overrides` to supply
          model-specific skorch parameters, and
        - rely on `_net_parts` to construct the final module, criterion,
          and parameter dictionary that are passed to the skorch
          `NeuralNet` backend.
        """

        def __init__(
            self,
            multi_annotator_module,
            criterion,
            clf_module,
            n_annotators=None,
            neural_net_param_dict=None,
            sample_dtype=np.float32,
            classes=None,
            cost_matrix=None,
            missing_label=MISSING_LABEL,
            random_state=None,
        ):
            super(_SkorchMultiAnnotatorClassifier, self).__init__(
                module=multi_annotator_module,
                criterion=criterion,
                classes=classes,
                missing_label=missing_label,
                cost_matrix=cost_matrix,
                random_state=random_state,
                neural_net_param_dict=neural_net_param_dict,
                sample_dtype=sample_dtype,
            )
            self.clf_module = clf_module
            self.n_annotators = n_annotators

        def fit(self, X, y, **fit_params):
            """Initialize and fit the module.

            If the module was already initialized, by calling fit, the module
            will be re-initialized (unless `warm_start` is True).

            Parameters
            ----------
            X : matrix-like, shape (n_samples, ...)
                Training data set, usually complete, i.e., including the
                labeled and unlabeled samples
            y : array-like of shape (n_samples, n_annotators)
                It contains the class labels of the training samples, where
                missing labels are represented via `missing_label`.
                Specifically, label `y[n, m]` refers to the label of sample
                `X[n]` from annotator `m`.
            fit_params : dict-like
              Further parameters as input to the 'fit' method of the
              `skorch.net.NeuralNet`.

            Returns
            -------
            self: _SkorchMultiAnnotatorClassifier,
              `_SkorchMultiAnnotatorClassifier` object fitted on the training
              data.
            """
            super().fit(X, y, **fit_params)

        def partial_fit(self, X, y, **fit_params):
            """Fit the module without re-initialization.

            If the module was already initialized, by calling `partial_fit`,
            the module will not be re-initialized again.

            Parameters
            ----------
            X : matrix-like, shape (n_samples, ...)
                Training data set, usually complete, i.e., including the
                labeled and unlabeled samples
            y : array-like of shape (n_samples, n_annotators)
                It contains the class labels of the training samples, where
                missing labels are represented via `missing_label`.
                Specifically, label `y[n, m]` refers to the label of sample
                `X[n]` from annotator `m`.
            fit_params : dict-like
                Further parameters as input to the 'partial_fit' method of the
                `skorch.net.NeuralNet`.

            Returns
            -------
            self: _SkorchMultiAnnotatorClassifier,
              `_SkorchMultiAnnotatorClassifier` object fitted on the training
              data.
            """
            super().partial_fit(X, y, **fit_params)

        def _net_parts(self, X, y):
            # Check module parameters.
            if self.neural_net_param_dict is None:
                neural_net_param_dict = {"train_split": None}
                clf_module_param_dict = {}
            elif isinstance(self.neural_net_param_dict, dict):
                neural_net_param_dict = self.neural_net_param_dict.copy()
                prefix = "module__"
                clf_module_param_dict = {
                    k[len(prefix) :]: neural_net_param_dict.pop(k)
                    for k in list(neural_net_param_dict)
                    if k.startswith(prefix)
                }
            else:
                raise TypeError(
                    "`neural_net_param_dict` must be a `dict` or `None`."
                )
            if "train_split" not in neural_net_param_dict:
                neural_net_param_dict["train_split"] = None
            if neural_net_param_dict.get("train_split") is not None:
                raise ValueError("`train_split` must be `None`.")

            # Check `classes` parameter.
            if not hasattr(self, "classes_") and self.classes is not None:
                check_classes(self.classes)
                self.classes_ = self.classes
            if not hasattr(self, "classes_") and self.classes is None:
                raise ValueError(
                    "Number of classes must be known before init."
                )

            # Check `n_annotators` parameter.
            if self.n_annotators is None and y is None:
                raise ValueError(
                    "Provide `n_annotators` or pass `y` when initializing."
                )
            if (
                self.n_annotators is not None
                and isinstance(y, np.ndarray)
                and self.n_annotators != y.shape[1]
            ):
                raise ValueError("`n_annotators` mismatch.")
            self.n_annotators_ = (
                self.n_annotators
                if self.n_annotators is not None
                else y.shape[1]
            )

            # Build `neural_net_param_dict`.
            invariant = {
                "module__clf_module": self.clf_module,
                "module__clf_module_param_dict": clf_module_param_dict,
            }
            override = self._build_neural_net_param_overrides(X=X, y=y)
            if not isinstance(override, dict):
                raise TypeError(
                    "`build_neural_net_param_overrides` must return a `dict`."
                )
            illegal = invariant.keys() & override.keys()
            if illegal:
                raise ValueError(
                    f"Do not set these in overrides: {sorted(illegal)}"
                )
            for k, v in (invariant | override).items():
                if (
                    k in neural_net_param_dict
                    and neural_net_param_dict[k] != v
                ):
                    raise ValueError(
                        f"`neural_net_param_dict[{k!r}]` must be {v!r} "
                        f"or unset."
                    )
                neural_net_param_dict[k] = v

            return (
                self.module,
                self.criterion,
                neural_net_param_dict,
            )

        def _return_training_data(self, X, y):
            """
            Return samples and labels required for training.

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
            is_train = is_labeled(y, missing_label=-1).any(axis=1)
            if np.sum(is_train) > 0:
                net = self.neural_net_.module_
                net.set_forward_return()
                X_train = X[is_train]
                y_train = y[is_train].astype(np.int64)
            return X_train, y_train

        def _validate_data_kwargs(self):
            """
            Return kwargs forwarded to `_validate_data`.

            Returns
            -------
            kwargs : dict
                Keyword arguments consumed by `_validate_data`.
            """
            vd_kwargs = super()._validate_data_kwargs()
            vd_kwargs["y_ensure_1d"] = False
            return vd_kwargs

        @abstractmethod
        def _build_neural_net_param_overrides(self, X, y):
            """Subclasses must return the enforced skorch param overrides."""

    class _MultiAnnotatorClassificationModule(nn.Module):
        """
        Auxiliary module that wraps a classifier backbone and standardizes
        its output to `(logits_class, x_embed)`.

        Parameters
        ----------
        clf_module : nn.Module or type
            Classifier backbone/head that maps `x -> logits_class` or
            `(logits_class, x_embed)`. If it returns only logits, `x_embed` is
            set to the input `x` (or to `None` if `x` is not an embedding).
        clf_module_param_dict : dict
            Keyword args for constructing `clf_module` if a class is passed.
        default_forward_outputs : sequence of str
            Default names to be used by `set_forward_return`.
        full_forward_outputs : sequence of str
            All possible names that `set_forward_return` can accept.
        """

        def __init__(
            self,
            clf_module,
            clf_module_param_dict,
            default_forward_outputs,
            full_forward_outputs,
        ):
            super().__init__()
            self.clf_module = self._as_module(
                clf_module, clf_module_param_dict
            )
            self.default_forward_outputs = default_forward_outputs
            self.full_forward_outputs = full_forward_outputs
            self.set_forward_return()

        def set_forward_return(self, values=None):
            """
            Select logical outputs to return.

            Parameters
            ----------
            values : str or sequence of str, optional
                Any subset of `full_forward_outputs`. Currently stored in
                `self.forward_return` and may be used by code that wraps this
                module.

            Returns
            -------
            self : _MultiAnnotatorClassificationModule
                The module itself for chaining.

            Raises
            ------
            ValueError
                If an unknown name is requested.
            """
            if values is None:
                values = self.default_forward_outputs
            if isinstance(values, str):
                values = [values]
            unknown = set(values) - set(self.full_forward_outputs)
            if unknown:
                raise ValueError(f"Unknown forward return(s): {unknown}")
            self.forward_return = set(values)
            return self

        def clf_module_forward(self, x):
            """
            Run the classifier backbone and standardize its output.

            Parameters
            ----------
            x : torch.Tensor of shape (batch_size, ...)
                Input batch. Shape depends on `clf_module`.

            Returns
            -------
            logits_class : torch.Tensor
                Class logits tensor.
            x_embed : torch.Tensor or None
                Embedding tensor if provided by `clf_module`, otherwise `x`
                (or `None` if `x` is not an embedding).
            """
            cls_out = self.clf_module(x)
            if isinstance(cls_out, tuple):
                logits_class, x_embed = cls_out
            else:
                logits_class, x_embed = cls_out, x  # fallback
            return logits_class, x_embed

        @staticmethod
        def _as_module(maybe_cls_or_mod, kwargs):
            if isinstance(maybe_cls_or_mod, nn.Module):
                return maybe_cls_or_mod
            if isinstance(maybe_cls_or_mod, type):
                return maybe_cls_or_mod(**(kwargs or {}))
            raise TypeError(
                "Expected nn.Module instance or class for a submodule."
            )

    class _MultiAnnotatorCollate:
        """
        Collate that expands a batch into all (sample, annotator) pairs and
        returns labels for labeled pairs only.

        Parameters
        ----------
        missing_label : int or float, default=-1
            Value in `y` indicating an unlabeled sample. Rows whose sample
            label equals `missing_label` are excluded from the
            (sample, annotator) pairs. If set to `float('nan')` or `numpy.nan`,
            NaN labels are treated as missing.

        Notes
        -----
        - This collate runs on CPU (inside DataLoader workers).
        - The returned batch has:
            * `x_out["x"]`: original sample batch
            * `x_out["input_ids"]`: tensor of shape (n_pairs, 2) with
              `(sample_index, annotator_index)` for each labeled pair
            * `y`: a 1D tensor of class indices for these pairs.
        """

        def __init__(self, missing_label=-1):
            self.missing_label = missing_label

        def __call__(self, batch):
            # Basic collation (supports tensors/ndarrays/nested dicts of X, y).
            x = default_collate([b[0] for b in batch])
            y = default_collate([b[1] for b in batch])

            # Flatten labels to (n_samples * n_annotators,)
            n_samples, n_annotators = y.shape
            y = y.view(-1)

            # Sample indices: 0..B-1 repeated for each annotator.
            idx_s = torch.arange(
                n_samples, dtype=torch.long
            ).repeat_interleave(n_annotators)

            # Annotator indices: 0..A-1 repeated for each annotator.
            idx_a = torch.arange(n_annotators, dtype=torch.long).repeat(
                n_samples
            )

            # Mask out pairs whose sample is unlabeled.
            if isinstance(self.missing_label, float) and (
                self.missing_label != self.missing_label
            ):  # NaN
                mask = ~torch.isnan(y.to(torch.float32))
            else:
                mask = y != self.missing_label
            idx_s = idx_s[mask]
            idx_a = idx_a[mask]
            y = y[mask]

            # Return batches including sample and annotator indices.
            x_out = {"x": x, "input_ids": torch.column_stack((idx_s, idx_a))}
            return x_out, y

except ImportError:  # pragma: no cover
    pass
