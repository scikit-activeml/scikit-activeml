try:
    import numpy as np
    import torch

    from sklearn.utils.validation import check_array
    from torch import nn
    from torch.nn import CrossEntropyLoss
    from torch.nn import functional as F

    from ...base import SkactivemlClassifier
    from ...utils import (
        MISSING_LABEL,
        check_n_features,
    )
    from ._utils import (
        _SkorchMultiAnnotatorClassifier,
        _MultiAnnotatorClassificationModule,
        _MultiAnnotatorCollate,
    )

    class CrowdLayerClassifier(_SkorchMultiAnnotatorClassifier):
        """Crowd Layer

        Crowd Layer [1]_ is a layer added at the end of a classifying neural
        network and allows us to train deep neural networks end-to-end,
        directly from the noisy labels of multiple annotators, using only
        backpropagation. The main idea is to insert an annotator-specific
        transformation on top of a shared latent prediction: for each
        annotator, the layer models how their noisy labels are generated from
        the underlying class probabilities (e.g., via a confusion-matrix-like
        mapping). By learning these annotator-specific mappings jointly with
        the base network, Crowd Layer can separate systematic annotator biases
        from the true label signal and thus leverage multiple noisy labels per
        sample during training.

        Parameters
        ----------
        clf_module : nn.Module or nn.Module.__class__
            A PyTorch module as classification model outputting logits for
            samples as input. In general, the uninstantiated class should
            be passed, although instantiated modules will also work. The
            `forward` module must return logits as first element and optional
            sample embeddings as second element. If no sample embeddings are
            returned, the implementation uses the original samples.
        n_annotators : int, default=None
            Number of annotators. If `n_annotators=None`, the number of
            annotators is inferred from `y` when calling `fit`.
        neural_net_param_dict : dict, default=None
            Additional arguments for `skorch.net.NeuralNet`. If
            `neural_net_param_dict` is None, no additional arguments are added.
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
        random_state : int or RandomState sample or None, default=None
            Determines random number for `predict` method. Pass an int for
            reproducible results across multiple method calls.

        References
        ----------
        .. [1] Rodrigues, Filipe, and Francisco Pereira. "Deep Learning from
           Crowds." AAAI Conference on Artificial Intelligence, 2018.
        """

        _ALLOWED_EXTRA_OUTPUTS = {
            "logits",
            "embeddings",
            "annotator_perf",
            "annotator_class",
        }

        def __init__(
            self,
            clf_module,
            n_annotators=None,
            neural_net_param_dict=None,
            sample_dtype=np.float32,
            classes=None,
            cost_matrix=None,
            missing_label=MISSING_LABEL,
            random_state=None,
        ):
            super(CrowdLayerClassifier, self).__init__(
                multi_annotator_module=_CrowdLayerModule,
                clf_module=clf_module,
                n_annotators=n_annotators,
                criterion=CrossEntropyLoss,
                sample_dtype=sample_dtype,
                classes=classes,
                missing_label=missing_label,
                cost_matrix=cost_matrix,
                random_state=random_state,
                neural_net_param_dict=neural_net_param_dict,
            )

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

            Returns
            -------
            y_pred : numpy.ndarray of shape (n_samples,)
                Class predictions of the test samples.
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
                - `P_annot` : `np.ndarray` of shape
                  `(n_samples, n_annotators, n_classes)`, where
                  `P_annot[n, m, c]` refers to the probability that annotator
                  `m` provides the class label `c` for sample `X[n]`.
            """
            # Check input parameters.
            self._validate_data_kwargs()
            X = check_array(X, **self.check_X_dict_)
            check_n_features(
                self, X, reset=not hasattr(self, "n_features_in_")
            )
            extra_outputs = self._normalize_extra_outputs(
                extra_outputs=extra_outputs,
                allowed_names=CrowdLayerClassifier._ALLOWED_EXTRA_OUTPUTS,
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
                forward_outputs["annotator_perf"] = (out_idx, None)
                forward_returns.append("p_annot_perf")
                out_idx += 1

            if "annotator_class" in extra_outputs:
                forward_outputs["annotator_class"] = (
                    out_idx,
                    nn.Softmax(dim=-1),
                )
                forward_returns.append("logits_annot")

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
            collate_fn = _MultiAnnotatorCollate(missing_label=-1)
            return {
                "criterion__reduction": "mean",
                "criterion__ignore_index": -1,
                "module__n_classes": len(self.classes_),
                "module__n_annotators": self.n_annotators_,
                "iterator_train__collate_fn": collate_fn,
            }

    class _CrowdLayerModule(_MultiAnnotatorClassificationModule):
        """Crowd Layer Module

        Crowd Layer [1]_ is a layer added at the end of a classifying neural
        network and allows us to train deep neural networks end-to-end,
        directly from the noisy labels of multiple annotators, using only
        backpropagation.

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

        References
        ----------
        .. [1] Rodrigues, Filipe, and Francisco Pereira. "Deep Learning from
           Crowds." AAAI Conference on Artificial Intelligence, 2018.
        """

        def __init__(
            self, n_classes, n_annotators, clf_module, clf_module_param_dict
        ):
            super().__init__(
                clf_module=clf_module,
                clf_module_param_dict=clf_module_param_dict,
                default_forward_outputs="logits_annot",
                full_forward_outputs=[
                    "logits_class",
                    "x_embed",
                    "p_annot_perf",
                    "logits_annot",
                ],
            )
            self.n_classes = n_classes
            self.n_annotators = n_annotators

            # Setup crowd layer.
            self.W_annot = torch.eye(n_classes).repeat(n_annotators, 1, 1)
            self.W_annot = nn.Parameter(self.W_annot)

        def forward(self, x, input_ids=None):
            """
            Forward pass through the classification module and optionally
            through the crowd layer.

            Parameters
            ----------
            x : torch.Tensor of shape (batch_size, ...)
                Input samples.
            input_ids : torch.Tensor of shape (batch_size, 2), default=None
                - If a tensor is given, `input_ids[:, 0]` are sample indices
                  and `input_ids[:, 1]` are annotator indices. One output row
                  is produced per (sample, annotator) pair.
                - If `input_ids=None`, all combinations of samples and
                  annotators are propagated through the crowd-layer.

            Returns
            -------
            logits_class : torch.Tensor of shape (batch_size, n_classes)
                Class-membership logits.
            x_embed : torch.Tensor of shape (batch_size, ...), optional
                Learned embeddings of samples. Only returned if "x_embed" in
                `self.forward_return`.
            p_annot_perf : torch.Tensor of shape (batch_size, n_annotators), \
                    optional
                Estimated performance, i.e., label correctness probability, per
                sample-annotator pair.
            logits_annot : torch.Tensor of shape (batch_size, n_annotators,\
                    n_classes) or (len(input_ids), n_classes), optional
                Annotation logits for sample-annotator pairs. Only returned
                if "logits_annot" in self.forward_return. Shape depends on
                whether `input_ids` is given or `None`.
            """
            # Inference of classification model.
            logits_class, x_embed = self.clf_module_forward(x)

            # Append classifier outputs to `out` if required.
            out = []
            if "logits_class" in self.forward_return:
                out.append(logits_class)
            if "x_embed" in self.forward_return:
                out.append(x_embed.detach().flatten(start_dim=1))

            # Add annotator logits / performances to `out` if required.
            if (
                "logits_annot" in self.forward_return
                or "p_annot_perf" in self.forward_return
            ):
                p_class = F.softmax(logits_class, dim=-1)

                # Expected annotator performance: (n_samples, n_annotators).
                if "p_annot_perf" in self.forward_return:
                    # Convert logits to confusion probabilities:
                    # confusion[a, c, o] = P(z=o | y=c, annotator=a).
                    confusion = F.softmax(self.W_annot, dim=-1)
                    # Compute perf_per_class[a, c]
                    # = P(correct | y=c, annotator=a) = confusion[a, c, c].
                    diag_idx = torch.arange(
                        self.n_classes, device=confusion.device
                    )
                    perf_per_class = confusion[:, diag_idx, diag_idx]
                    # Compute expected performance per sample-annotator pair:
                    # logits_annot_perf[n, a] =
                    # sum_c p_class[n, c] * perf_per_class[a, c].
                    logits_annot_perf = torch.einsum(
                        "nc,ac->na", p_class, perf_per_class
                    )
                    out.append(logits_annot_perf)

                # Expected annotator outputs / logits for labels.
                if "logits_annot" in self.forward_return:
                    if isinstance(input_ids, torch.Tensor):
                        # input_ids: (m, 2)  -> (sample_idx, annotator_idx)
                        p_sel = p_class.index_select(0, input_ids[:, 0])
                        W_sel = self.W_annot.index_select(0, input_ids[:, 1])
                        logits_annot = torch.einsum("mc,mco->mo", p_sel, W_sel)
                    else:
                        logits_annot = torch.einsum(
                            "nc,aco->nao", p_class, self.W_annot
                        )
                    out.append(logits_annot)

            return out[0] if len(out) == 1 else tuple(out)

except ImportError:  # pragma: no cover
    pass
