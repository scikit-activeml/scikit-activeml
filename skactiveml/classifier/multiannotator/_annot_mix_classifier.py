try:
    import math
    import numpy as np
    import torch

    from sklearn.utils.validation import check_array
    from torch import nn
    from torch.nn import KLDivLoss
    from torch.nn import functional as F
    from torch.utils.data import default_collate

    from ...base import SkactivemlClassifier
    from ...utils import (
        MISSING_LABEL,
        check_n_features,
        check_scalar,
    )
    from ._utils import (
        _MultiAnnotatorClassificationModule,
        _SkorchMultiAnnotatorClassifier,
    )

    class AnnotMixClassifier(_SkorchMultiAnnotatorClassifier):
        """Annot-Mix

        Annot-Mix [1]_ trains a multi-annotator classifier using an extension
        of MixUp [2]_. The main idea is to apply MixUp not only to samples
        and class labels, but to sample–annotator pairs: it convexly combines
        inputs and their annotator-specific noisy labels and trains a one-stage
        model that jointly estimates the true label distribution and each
        annotator’s reliability. In this way, Annot-Mix can handle multiple,
        potentially conflicting labels per sample while using MixUp-style
        regularization to become more robust to label noise.

        Parameters
        ----------
        clf_module : nn.Module or nn.Module.__class__
            A PyTorch module as classification model outputting logits for
            samples as input. In general, the uninstantiated class should
            be passed, although instantiated modules will also work. The
            `forward` module must return logits as first element and optional
            sample embeddings as
            second element. If no sample embeddings are returned, the
            implementation uses the original samples.
        alpha : float, default=0.5
            MixUp concentration parameter. The mix coefficient `lambda` is
            drawn from `Beta(alpha, alpha)`. Use `alpha=0` to disable MixUp.
        annotator_embed_dim : int, default=16
            Dimensionality of the annotator embedding used to model
            annotator-specific behavior.
        sample_embed_dim : int, default=0
            Dimensionality of an optional learnable sample-embedding used to
            model sample-specific behavior of each annotator. If
            `sample_embed_dim=0`, the annotator performances are only
            modeled as class-specific.
        hidden_dim : int or None, default=None
            Hidden size of the fusion multi-layer perceptron that propagates
            sample and annotator representations. If `None`, a sensible
            default is used, which depends on the other input parameters.
            Note that this parameter has no effect for `n_hidden_layers=0`.
        n_hidden_layers : int, default=0
            Number of hidden layers in the fusion multi-layer perceptron.
        hidden_dropout : float, default=0.1
            Dropout probability applied in the fusion multi-layer perceptron.
            Note that this parameter has no effect for `n_hidden_layers=0`.
        eta : float in (0, 1), default=0.9
            Prior annotator performance, i.e., the probability of obtaining a
            correct annotation from an arbitrary annotator for an arbitrary
            sample of an arbitrary class.
        n_annotators : int, default=None
            Number of annotators. If `n_annotators=None`, the number of
            annotators is inferred by the shape of `y` during training.
        neural_net_param_dict : dict, default=None
            Additional arguments for `skorch.net.NeuralNet`. If
            `neural_net_param_dict` is `None`, no extra arguments are added.
            `module`, `criterion`, `predict_nonlinearity`, and `train_split`
            are not allowed in this dictionary.
        sample_dtype : str or type, default=np.float32
            Dtype to which input samples are cast inside the estimator. If set
            to `None`, the input dtype is preserved.
        classes : array-like of shape (n_classes,), default=None
            Holds the label for each class. If `None`, the classes are
            determined during the fit.
        missing_label : scalar or string or np.nan or None, default=np.nan
            Value to represent a missing label.
        cost_matrix : array-like of shape (n_classes, n_classes), default=None
            Cost matrix with `cost_matrix[i,j]` indicating cost of predicting
            class `classes[j]` for a sample of class `classes[i]`. Can be only
            set, if `classes` is not `None`.
        random_state : int or RandomState instance or None, default=None
            Determines random number for `predict` method. Pass an int for
            reproducible results across multiple method calls.

        References
        ----------
        .. [1] Herde, M., Lührs, L., Huseljic, D., & Sick, B. (2024).
           Annot-Mix: Learning with Noisy Class Labels from Multiple Annotators
           via a Mixup Extension. Eur. Conf. Artif. Intell.
        .. [2] Zhang, H., Cisse, M., Dauphin, Y. N., & Lopez-Paz, D. (2018).
           mixup: Beyond Empirical Risk Minimization. Int. Conf. Learn.
           Represent.
        """

        _ALLOWED_EXTRA_OUTPUTS = {
            "logits",
            "embeddings",
            "annotator_perf",
            "annotator_class",
            "annotator_embeddings",
        }

        def __init__(
            self,
            clf_module,
            alpha=0.5,
            sample_embed_dim=0,
            annotator_embed_dim=16,
            hidden_dim=None,
            n_hidden_layers=0,
            hidden_dropout=0.1,
            eta=0.9,
            n_annotators=None,
            neural_net_param_dict=None,
            sample_dtype=np.float32,
            classes=None,
            cost_matrix=None,
            missing_label=MISSING_LABEL,
            random_state=None,
        ):
            super(AnnotMixClassifier, self).__init__(
                multi_annotator_module=_AnnotMixModule,
                clf_module=clf_module,
                criterion=KLDivLoss,
                classes=classes,
                missing_label=missing_label,
                cost_matrix=cost_matrix,
                random_state=random_state,
                neural_net_param_dict=neural_net_param_dict,
                sample_dtype=sample_dtype,
            )
            self.clf_module = clf_module
            self.alpha = alpha
            self.sample_embed_dim = sample_embed_dim
            self.annotator_embed_dim = annotator_embed_dim
            self.hidden_dim = hidden_dim
            self.n_hidden_layers = n_hidden_layers
            self.hidden_dropout = hidden_dropout
            self.eta = eta
            self.n_annotators = n_annotators

        def predict(
            self,
            X,
            extra_outputs=None,
        ):
            """Return class predictions for the test samples `X`.

            By default, this method returns only the class predictions
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
                names must be a subset of the following keys:

                - "logits" : Additionally return the class-membership logits
                  `L_class` for the samples in `X`.
                - "embeddings" : Additionally return the learned embeddings
                  `X_embed` for the samples in `X`.
                - "annotator_perf" : additionally return the estimated
                  annotator performance probabilities `P_perf` for each
                  sample–annotator pair.
                - "annotator_class" : Additionally return the annotator–class
                  probability estimates `P_annot` for each sample, class, and
                  annotator.
                - "annotator_embeddings" : Additionally return the
                  learned embeddings `A_embed` for the annotators as the next
                  element of the output tuple.

            Returns
            -------
            y_pred : numpy.ndarray of shape (n_samples,)
                Class labels of the test samples.
            *extras : numpy.ndarray, optional
                Only returned if `extra_outputs` is not `None`. In that
                case, the method returns a tuple whose first element is
                `y_pred` and whose remaining elements correspond to the
                requested forward outputs in the order given by
                `extra_outputs`. Potential outputs are:

                - `L_class` : `np.ndarray` of shape `(n_samples, n_classes)`,
                  where `L_class[n, c]` is the logit for the class
                  `classes_[c]` of sample `X[n]`.
                - `X_embed` : `np.ndarray` of shape `(n_samples, ...)`, where
                  `X_embed[n]` refers to the learned embedding for sample
                  `X[n]`.
                - `P_perf` : `np.ndarray` of shape `(n_samples, n_annotators)`,
                  where `P_perf[n, m]` refers to the estimated label
                  correctness probability (performance) of annotator `m` when
                  labeling sample `X[n]`.
                - `P_annot` : `np.ndarray` of shape
                  `(n_samples, n_annotators, n_classes)`, where
                  `P_annot[n, m, c]` refers to the probability that annotator
                  `m` provides the class label `c` for sample `X[n]`.
                - `A_embed` : `np.ndarray` of shape
                  `(n_annotators, annotator_embed_dim)`, where `A_embed[m]`
                  refers to the learned embedding for annotator `m`.
            """
            return SkactivemlClassifier.predict(
                self,
                X=X,
                extra_outputs=extra_outputs,
            )

        def predict_proba(
            self,
            X,
            extra_outputs=None,
        ):
            """Return class probability estimates for the test samples `X`.

            By default, this method returns only the class probabilities `P`.
            If `extra_outputs` is provided, a tuple is returned whose first
            element is `P` and whose remaining elements are the requested
            additional forward outputs, in the order specified by
            `extra_outputs`.

            Parameters
            ----------
            X : array-like of shape (n_samples, ...)
                Test samples.
            extra_outputs : None or str or sequence of str, default=None
                Names of additional outputs to return next to `P`. The names
                must be a subset of the following keys:

                - "logits" : Additionally return the class-membership logits
                  `L_class` for the samples in `X`.
                - "embeddings" : Additionally return the learned embeddings
                  `X_embed` for the samples in `X`.
                - "annotator_perf" : additionally return the estimated
                  annotator performance probabilities `P_perf` for each
                  sample–annotator pair.
                - "annotator_class" : Additionally return the annotator–class
                  probability estimates `P_annot` for each sample, class, and
                  annotator.
                - "annotator_embeddings" : Additionally return the
                  learned embeddings `A_embed` for the annotators as the next
                  element of the output tuple.

            Returns
            -------
            P : numpy.ndarray of shape (n_samples, n_classes)
                Class probabilities of the test samples. Classes are ordered
                according to `self.classes_`.
            *extras : numpy.ndarray, optional
                Only returned if `extra_outputs` is not `None`. In that
                case, the method returns a tuple whose first element is `P`
                and whose remaining elements correspond to the requested
                forward outputs in the order given by `extra_outputs`.
                Potential outputs are:

                - `L_class` : `np.ndarray` of shape `(n_samples, n_classes)`,
                  where `L_class[n, c]` is the logit for the class
                  `classes_[c]` of sample `X[n]`.
                - `X_embed` : `np.ndarray` of shape `(n_samples, ...)`, where
                  `X_embed[n]` refers to the learned embedding for sample
                  `X[n]`.
                - `P_perf` : `np.ndarray` of shape `(n_samples, n_annotators)`,
                  where `P_perf[n, m]` refers to the estimated label
                  correctness probability (performance) of annotator `m` when
                  labeling sample `X[n]`.
                - `P_annot : `np.ndarray` of shape
                  `(n_samples, n_annotators, n_classes)`, where
                  `P_annot[n, m, c]` refers to the probability that annotator
                  `m` provides the class label `c` for sample `X[n]`.
                - `A_embed` : `np.ndarray` of shape
                  `(n_annotators, annotator_embed_dim)`, where `A_embed[m]`
                  refers to the learned embedding for annotator `m`.
            """
            # Check input parameters.
            self._validate_data_kwargs()
            X = check_array(X, **self.check_X_dict_)
            check_n_features(
                self, X, reset=not hasattr(self, "n_features_in_")
            )
            extra_outputs = self._normalize_extra_outputs(
                extra_outputs=extra_outputs,
                allowed_names=AnnotMixClassifier._ALLOWED_EXTRA_OUTPUTS,
            )

            # Initialize module, if not done yet.
            if not hasattr(self, "neural_net_"):
                self.initialize()

            # Set forward options to obtain the different outputs required
            # by the input parameters.
            net = self.neural_net_.module_
            old_forward_return = net.forward_return
            forward_outputs = {"probas": (0, nn.Softmax(dim=-1))}
            forward_returns = ["logits_class"]
            out_idx = 1

            if "logits" in extra_outputs:
                forward_outputs["logits"] = (0, None)

            if "embeddings" in extra_outputs:
                forward_outputs["embeddings"] = (out_idx, None)
                forward_returns.append("x_embed")
                out_idx += 1

            if "annotator_perf" in extra_outputs:

                def _transform_annotator_perf(P_perf):
                    P_perf = P_perf.exp()
                    return P_perf.reshape(-1, self.n_annotators_)

                forward_outputs["annotator_perf"] = (
                    out_idx,
                    _transform_annotator_perf,
                )
                forward_returns.append("log_p_annotator_perf")
                out_idx += 1

            if "annotator_class" in extra_outputs:

                def _transform_annotator_class(P_annot):
                    P_annot = P_annot.exp()
                    return P_annot.reshape(
                        -1, self.n_annotators_, len(self.classes_)
                    )

                forward_outputs["annotator_class"] = (
                    out_idx,
                    _transform_annotator_class,
                )
                forward_returns.append("log_p_annotator_class")
                out_idx += 1

            if "annotator_embeddings" in extra_outputs:

                def _transform_annotator_embeddings(A_embed):
                    return A_embed[: self.n_annotators_]

                forward_outputs["annotator_embeddings"] = (
                    out_idx,
                    _transform_annotator_embeddings,
                )
                forward_returns.append("a_embed")

            # Compute predictions for the different outputs required
            # by the input parameters.
            try:
                net.set_forward_return(forward_returns)
                fw_out = self._forward_with_named_outputs(
                    X=X,
                    forward_outputs=forward_outputs,
                    extra_outputs=extra_outputs,
                )
            finally:
                net.set_forward_return(old_forward_return)

            # Initialize fallbacks if the classifier hasn't been fitted before.
            self._initialize_fallbacks(
                fw_out[0] if isinstance(fw_out, tuple) else fw_out
            )
            return fw_out

        def _build_neural_net_param_overrides(self, X, y):
            """Initialize the internal `sklearn` wrapper from `skorch`."""
            # Check parameters specific to `AnnotMixClassifier`.
            check_scalar(
                self.alpha,
                name="alpha",
                target_type=float,
                min_val=0.0,
                min_inclusive=True,
            )
            check_scalar(
                self.sample_embed_dim,
                name="sample_embed_dim",
                target_type=int,
                min_val=0,
                min_inclusive=True,
            )
            check_scalar(
                self.annotator_embed_dim,
                name="annotator_embed_dim",
                target_type=int,
                min_val=1,
                min_inclusive=True,
            )
            hidden_dim = self.hidden_dim
            if hidden_dim is None:
                hidden_dim = min(
                    4 * len(self.classes_),
                    max(
                        128,
                        2 * (self.annotator_embed_dim + self.sample_embed_dim),
                    ),
                )
            check_scalar(
                hidden_dim,
                name="hidden_dim",
                target_type=int,
                min_val=1,
                min_inclusive=True,
            )
            check_scalar(
                self.n_hidden_layers,
                name="n_hidden_layers",
                target_type=int,
                min_val=0,
                min_inclusive=True,
            )
            check_scalar(
                self.hidden_dropout,
                name="hidden_dropout",
                target_type=float,
                min_val=0.0,
                min_inclusive=True,
                max_val=1.0,
                max_inclusive=False,
            )
            check_scalar(
                self.eta,
                name="eta",
                target_type=float,
                min_val=1 / len(self.classes_),
                min_inclusive=False,
                max_val=1.0,
                max_inclusive=False,
            )
            collate_fn = _MixUpCollate(
                n_classes=len(self.classes_),
                n_annotators=self.n_annotators_,
                alpha=self.alpha,
                missing_label=-1,
            )
            return {
                "criterion__reduction": "batchmean",
                "module__n_classes": len(self.classes_),
                "module__n_annotators": self.n_annotators_,
                "module__sample_embed_dim": self.sample_embed_dim,
                "module__annotator_embed_dim": self.annotator_embed_dim,
                "module__hidden_dim": hidden_dim,
                "module__n_hidden_layers": self.n_hidden_layers,
                "module__hidden_dropout": self.hidden_dropout,
                "module__eta": self.eta,
                "iterator_train__collate_fn": collate_fn,
            }

    class _AnnotMixModule(_MultiAnnotatorClassificationModule):
        """
        Auxiliary module for Annot-Mix [1]_ that produces class logits and
        annotator-conditioned outputs, while training with MixUp [2]_.

        Parameters
        ----------
        n_classes : int
            Number of classes.
        n_annotators : int
            Number of annotators.
        clf_module : nn.Module or nn.Module.__class__
            Classifier backbone/head that maps `x -> logits_class` or
            `(logits_class, x_embed)`. If it returns only logits, `x_embed` is
            set to the input `x` (or to `None` if `x` is not an embedding).
        clf_module_param_dict : dict
            Keyword args for constructing `clf_module` if a class is passed.
        annotator_embed_dim : int
            Dimensionality of the annotator embedding used to model
            annotator-specific behavior.
        sample_embed_dim : int or None
            Dimensionality of an optional learnable sample-embedding used to
            model sample-specific behavior of each annotator. If
            `sample_embed_dim=0`, no additional sample embedding is learned.
        hidden_dim : int or None
            Hidden size of the fusion multi-layer perceptron that propagates
            sample and annotator representations. If `None`, a sensible
            default is used, which depends on the other input parameters.
        n_hidden_layers : int
            Number of hidden layers in the fusion multi-layer perceptron.
        hidden_dropout : float
            Dropout probability applied in the fusion multi-layer perceptron.
        eta : float in (0, 1)
            Prior annotator performance, i.e., the probability of obtaining a
            correct annotation from an arbitrary annotator for an arbitrary
            sample.

        References
        ----------
        .. [1] Herde, M., Lührs, L., Huseljic, D., & Sick, B. (2024).
           Annot-Mix: Learning with Noisy Class Labels from Multiple Annotators
           via a Mixup Extension. Eur. Conf. Artif. Intell.
        .. [2] Zhang, H., Cisse, M., Dauphin, Y. N., & Lopez-Paz, D. (2018).
           mixup: Beyond Empirical Risk Minimization. Int. Conf. Learn.
           Represent.
        """

        # Optional names that can be returned *after* logits_class
        OUTPUTS = (
            "logits_class",
            "x_embed",
            "a_embed",
            "log_p_annotator_class",
            "log_p_annotator_perf",
        )

        def __init__(
            self,
            n_classes,
            n_annotators,
            clf_module,
            clf_module_param_dict,
            sample_embed_dim,
            annotator_embed_dim,
            hidden_dim,
            n_hidden_layers,
            hidden_dropout,
            eta,
        ):
            super().__init__(
                clf_module=clf_module,
                clf_module_param_dict=clf_module_param_dict,
                default_forward_outputs="log_p_annotator_class",
                full_forward_outputs=[
                    "logits_class",
                    "x_embed",
                    "a_embed",
                    "log_p_annotator_class",
                    "log_p_annotator_perf",
                ],
            )
            # Define integer variables.
            self.n_classes = n_classes
            self.annotator_embed_dim = annotator_embed_dim

            # Set up layer to learn annotator embeddings.
            self.register_buffer(
                "a", torch.eye(n_annotators, dtype=torch.float32)
            )
            self.sample_embed = None
            if sample_embed_dim > 0:
                self.sample_embed = nn.LazyLinear(
                    out_features=sample_embed_dim,
                )
            self.annotator_embed = nn.Linear(
                in_features=n_annotators,
                out_features=annotator_embed_dim,
            )

            # Post-scale diagonal bump as inductive bias.
            eta = math.log(eta / (1.0 - eta)) + math.log(n_classes - 1.0)
            prior_conf = nn.Parameter(
                eta * torch.eye(n_classes, dtype=torch.float32).flatten()
            )

            # Set up annotator confusion head.
            full_dim = sample_embed_dim + annotator_embed_dim
            blocks, dim = [], full_dim
            for _ in range(n_hidden_layers):
                blocks += [
                    nn.Dropout(hidden_dropout),
                    nn.Linear(dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.SiLU(),
                ]
                dim = hidden_dim
            out = nn.Linear(dim, n_classes * n_classes)
            out.bias = prior_conf
            blocks += [out]
            self.annotator_confusion_head = nn.Sequential(*blocks)

        def forward(self, x, a=None):
            """
            Parameters
            ----------
            x : torch.Tensor of shape (batch_size, ...)
                Input batch. Shape depends on `clf_module`.
            a : torch.Tensor of shape (batch_size, ...) or None
                Annotator features/IDs. Needed if any of {"a_embed",
                "log_p_annotator_class", "p_perf"} are requested.

            Returns
            -------
            out : torch.Tensor or tuple
                Given `set_forward_return`, tensors are appended in the order:

                - `"logits_class"`,
                - `"x_embed"`,
                - "log_p_annotator_class"`
                - "log_p_annotator_perf"`
                - `"a_embed"`.
            """
            # Obtain classifier outputs.
            logits_class, x_embed = self.clf_module_forward(x)

            # Append classifier output if required.
            out = []
            if "logits_class" in self.forward_return:
                out.append(logits_class)
            if "x_embed" in self.forward_return:
                out.append(x_embed)

            need_annotator_output = any(
                k in self.forward_return
                for k in (
                    "a_embed",
                    "log_p_annotator_class",
                    "log_p_annotator_perf",
                )
            )
            if need_annotator_output:
                a = a if a is not None else self.a

                # Sample/annotator embeddings for annotator head.
                if self.sample_embed:
                    x_embed = self.sample_embed(
                        x_embed.detach().flatten(start_dim=1)
                    )
                a_embed = self.annotator_embed(a)

                # Generate pairs of samples and annotator if not done yet.
                if not self.training:
                    combs = torch.cartesian_prod(
                        torch.arange(len(x), device=x.device),
                        torch.arange(len(a_embed), device=a_embed.device),
                    )
                    if self.sample_embed:
                        x_embed = x_embed[combs[:, 0]]
                    a_embed_return = a_embed.clone().detach()
                    a_embed = a_embed[combs[:, 1]]
                    logits_class = logits_class[combs[:, 0]]
                else:
                    a_embed_return = a_embed

                # Compute confusion matrix logits per sample-annotator pair.
                annot_head_input = a_embed
                if self.sample_embed:
                    annot_head_input = torch.cat([x_embed, a_embed], dim=-1)
                logits_conf = self.annotator_confusion_head(annot_head_input)
                logits_conf = logits_conf.view(
                    -1, self.n_classes, self.n_classes
                )

                # Compute log-probabilities for class and confusion matrices.
                p_conf_log = F.log_softmax(logits_conf, dim=-1)
                p_class_log = F.log_softmax(logits_class, dim=-1)

                # Compute and append annotator correctness log-probabilities.
                if "log_p_annotator_perf" in self.forward_return:
                    log_diag_conf = torch.diagonal(
                        p_conf_log, dim1=-2, dim2=-1
                    )
                    p_perf = torch.logsumexp(
                        p_class_log + log_diag_conf, dim=-1
                    )
                    out.append(p_perf)

                # Compute and append annotator class log-probabilities.
                if "log_p_annotator_class" in self.forward_return:
                    log_p_annotator_class = torch.logsumexp(
                        p_class_log[:, :, None] + p_conf_log, dim=1
                    )
                    out.append(log_p_annotator_class)

                if "a_embed" in self.forward_return:
                    out.append(a_embed_return)

            return out[0] if len(out) == 1 else tuple(out)

    class _MixUpCollate:
        """
        Collate that expands a batch into all (sample, annotator) pairs and
        optionally applies MixUp [1]_  jointly to samples, annotators, and
        labels [2]_.

        Parameters
        ----------
        n_classes : int
            Number of classes (for one-hot encoding).
        n_annotators : int
            Number of annotators (for one-hot encoding)
        alpha : float, default=1.0
            MixUp Beta(alpha, alpha) parameter. If <= 0, no MixUp is applied.
        missing_label : int or float, default=-1
            Value in `y` indicating an unlabeled sample. Rows whose sample
            label equals `missing_label` are excluded from the
            (sample, annotator) pairs. If set to `float('nan')` or `numpy.nan`,
            NaN labels are treated as missing.

        Notes
        -----
        Labels are returned as one-hot encoded vectors of length `n_classes`.

        References
        ----------
        .. [1] Zhang, H., Cisse, M., Dauphin, Y. N., & Lopez-Paz, D. (2018).
           mixup: Beyond Empirical Risk Minimization. Int. Conf. Learn.
           Represent.
        .. [2] Herde, M., Lührs, L., Huseljic, D., & Sick, B. (2024).
           Annot-Mix: Learning with Noisy Class Labels from Multiple Annotators
           via a Mixup Extension. Eur. Conf. Artif. Intell.
        """

        def __init__(
            self, n_classes, n_annotators, alpha=1.0, missing_label=-1
        ):
            if n_classes <= 0:
                raise ValueError("`n_classes` must be a positive integer.")
            if n_annotators <= 0:
                raise ValueError("`n_annotators` must be a positive integer.")
            alpha = float(alpha)
            if alpha < 0:
                raise ValueError("`alpha` must be >= 0 for MixUp.")

            self.n_classes = int(n_classes)
            self.n_annotators = int(n_annotators)
            self.a = torch.eye(self.n_annotators, dtype=torch.float32)
            self.alpha = alpha
            self.missing_label = missing_label

        def __call__(self, batch):
            # 1) Basic collation (supports tensors/ndarrays/nested dicts) of
            # samples X, labels y, and annotators a.
            x = default_collate([b[0] for b in batch])
            y = default_collate([b[1] for b in batch])

            # Expect labels of shape (n_samples, n_annotators)
            if y.dim() != 2 or y.shape[1] != self.n_annotators:
                raise ValueError(
                    f"y must have shape (n_samples, {self.n_annotators}), "
                    f"got {tuple(y.shape)}."
                )

            n_samples, _ = y.shape

            # Flatten labels to (n_samples * n_annotators,).
            y = y.view(-1)

            # 2) Build all (sample, annotator) combinations
            # sample indices: 0..B-1 repeated for each annotator.
            idx_s = torch.arange(
                n_samples, dtype=torch.long
            ).repeat_interleave(self.n_annotators)
            # Annotator indices: 0..A-1 tiled B times.
            idx_a = torch.arange(self.n_annotators, dtype=torch.long).repeat(
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
            y_pairs = y[mask]

            # 3) Select data per pair.
            x_pairs = x.index_select(0, idx_s)
            a_pairs = self.a.index_select(0, idx_a)

            # One-hot labels (ensure integer dtype for F.one_hot).
            y_pairs = y_pairs.to(torch.long)
            y_oh = F.one_hot(y_pairs, num_classes=self.n_classes).to(
                dtype=torch.float32
            )

            # 4) Optional MixUp across pairs (jointly mixing x, a, and y).
            if self.alpha > 0:
                x_pairs, a_pairs, y_oh, _, _ = _mix_up(
                    x_pairs, a_pairs, y_oh, alpha=self.alpha
                )

            x_out = {"x": x_pairs, "a": a_pairs}
            return x_out, y_oh

    def _mix_up(*arrays, alpha=1.0, lmbda=None, permute_indices=None):
        """
        MixUp [1]_ multiple arrays using the same permutation and
        lambdas.

        Parameters
        ----------
        arrays : sequence of torch.Tensor
            Tensors with the same length `N` along the first dimension.
            Each will be mixed with the same permutation and mixing
            coefficients.
        alpha : float, default=1.0
            Beta(alpha, alpha) parameter. Used only if `lmbda is None`.
            If `alpha == 0`, returns inputs unchanged (with `lmbda` all ones).
            If `alpha < 0`, a ValueError is raised.
        lmbda : torch.Tensor of shape (N,), default=None
            Precomputed mixing coefficients in [0, 1]. If not provided, sampled
            from `Beta(alpha, alpha)` on the same device as the first array
            when `alpha > 0`, or set to ones if `alpha == 0`.
        permute_indices : torch.Tensor of shape (N,), default=None
            Precomputed permutation indices. If not provided, a random
            permutation is generated on the same device as the first array.

        Returns
        -------
        outputs : tuple
            Tuple of mixed tensors in the same order as `arrays`, followed by
            `(lmbda, permute_indices)`.

        References
        ----------
        .. [1] Zhang, H., Cisse, M., Dauphin, Y. N., & Lopez-Paz, D. (2018).
           mixup: Beyond Empirical Risk Minimization. Int. Conf. Learn.
           Represent.
        """
        if len(arrays) == 0:
            raise ValueError("At least one array must be provided to _mix_up.")

        # All arrays must share the same leading dimension.
        N = arrays[0].shape[0]
        for arr in arrays[1:]:
            if arr.shape[0] != N:
                raise ValueError(
                    "All arrays must have the same length in dim 0."
                )

        first = arrays[0]
        device = first.device
        alpha = float(alpha)
        if alpha < 0:
            raise ValueError("alpha must be >= 0 for MixUp.")

        # Handle lambda.
        if lmbda is None:
            if alpha == 0:
                lmbda = torch.ones(N, device=device, dtype=torch.float32)
            else:
                lmbda = (
                    torch.distributions.Beta(alpha, alpha)
                    .sample((N,))
                    .to(device=device, dtype=torch.float32)
                )
        else:
            lmbda = torch.as_tensor(lmbda, device=device, dtype=torch.float32)
            if lmbda.dim() != 1 or lmbda.shape[0] != N:
                raise ValueError(
                    f"`lmbda` must have shape ({N},), "
                    f"got {tuple(lmbda.shape)}."
                )

        # Handle permutation.
        if permute_indices is None:
            permute_indices = torch.randperm(N, device=device)
        else:
            permute_indices = torch.as_tensor(
                permute_indices, device=device, dtype=torch.long
            )
            if permute_indices.dim() != 1 or permute_indices.shape[0] != N:
                raise ValueError(
                    f"`permute_indices` must have shape ({N},), "
                    f"got {tuple(permute_indices.shape)}."
                )

        # Broadcast lmbda to array shapes and mix.
        outputs = []
        for arr in arrays:
            view_shape = (N,) + (1,) * (arr.dim() - 1)
            lam_view = lmbda.view(view_shape)
            mixed = lam_view * arr + (1.0 - lam_view) * arr.index_select(
                0, permute_indices
            )
            outputs.append(mixed.to(arr.dtype))

        outputs.extend([lmbda, permute_indices])
        return tuple(outputs)

except ImportError:  # pragma: no cover
    pass
