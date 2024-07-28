import numpy as np
import torch
from skorch import NeuralNet
from skorch.callbacks import regularization
from skorch.dataset import unpack_data
from torch import nn
from torch.nn import functional as F

from ...base import AnnotatorModelMixin
from ...classifier import SkorchClassifier


class RegCrowdNetClassifier(SkorchClassifier, AnnotatorModelMixin):
    """RegCrowdNetClassifier

    The "regularized crowd network" (RegCrowdNet) [1, 2] jointly learns the underlying ground truth (GT) distribution
    and the individual confusion matrix as proxy of each annotator's performance. Therefor, a regularization term is
    added to the loss function that encourages convergence to the true annotator confusion matrix.

    Parameters
    ----------
    n_classes : int
        Number of classes
    n_annotators : int
        Number of annotators.
    module__gt_net : nn.Module
        Pytorch module of the GT model taking samples
        as input to predict class-membership logits.
    module__output_net
    module__n_annotators : int
         Number of annotators.
    arguments
        more possible arguments for initialize your neural network
        see: https://skorch.readthedocs.io/en/stable/net.html
    classes : array-like of shape (n_classes,), default=None
        Holds the label for each class. If none, the classes are determined
        during the fit.
    missing_label : scalar or string or np.nan or None, default=np.nan
        Value to represent a missing label.
    cost_matrix : array-like of shape (n_classes, n_classes)
        Cost matrix with `cost_matrix[i,j]` indicating cost of predicting class
        `classes[j]` for a sample of class `classes[i]`. Can be only set, if
        `classes` is not none.
    random_state : int or RandomState instance or None, default=None
        Determines random number for 'predict' method. Pass an int for
        reproducible results across multiple method calls.
    **kwargs : keyword arguments
        more possible parameters to customizing your neural network
        see: https://skorch.readthedocs.io/en/stable/net.html
    ATTENTION: Criterion is in this methode predefined. Please don't overwrite the
    'criterion' parameter.

    References
    ----------
    [1] Rodrigues, Filipe, and Francisco Pereira. "Deep learning from crowds." In Proceedings of the AAAI conference on
        artificial intelligence, vol. 32, no. 1. 2018.
    """

    def __init__(
        self,
        n_classes,
        n_annotators,
        *args,
        lmbda="auto",
        regularization="trace-reg",
        **kwargs,
    ):
        if regularization not in ["trace-reg", "geo-reg-f", "geo-reg-w"]:
            raise ValueError(
                "`regularization` must be in ['trace-reg', 'geo-reg-f', 'geo-reg-w']."
            )

        super(RegCrowdNetClassifier, self).__init__(
            module=RegCrowdNetModule,
            module__n_classes=n_classes,
            module__n_annotators=n_annotators,
            module__regularization=regularization,
            criterion=nn.NLLLoss,
            criterion__reduction="mean",
            criterion__ignore_index=-1,
            *args,
            **kwargs,
        )

        self.n_classes = n_classes
        self.n_annotators = n_annotators
        self.lmbda = lmbda
        self.regularization = regularization
        if self.lmbda == "auto":
            self.lmbda = 1e-2 if self.regularization == "trace-reg" else 1e-4

    def get_loss(self, y_pred, y_true, *args, **kwargs):
        """Return the loss for this batch.

        Parameters
        ----------
        y_pred : torch.Tensor
          Predicted target values
        y_true : torch.Tensor
          True target values.

        Returns
        ---------
        loss : torch.Tensor
            Loss for this batch
        """
        n_samples, n_annotators = y_true.shape[0], y_true.shape[1]
        combs = torch.cartesian_prod(
            torch.arange(n_samples, device=y_true.device),
            torch.arange(n_annotators, device=y_true.device),
        )
        z = y_true.ravel()
        is_lbld = z != -1  # missing_label is -1
        combs, z = combs[is_lbld], z[is_lbld]
        p_class_log = F.log_softmax(y_pred, dim=-1)
        p_class_log_ext = p_class_log[combs[:, 0]]
        p_perf_log = F.log_softmax(self.module_.ap_confs_, dim=-1)
        p_perf_log_ext = p_perf_log[combs[:, 1]]
        p_annot_log = torch.logsumexp(
            p_class_log_ext[:, :, None] + p_perf_log_ext, dim=1
        )
        loss = NeuralNet.get_loss(self, p_annot_log, z)
        if self.lmbda > 0:
            if self.regularization == "trace-reg":
                p_perf = F.softmax(self.module_.ap_confs_, dim=-1)
                reg_term = self._compute_reg_term(p_class=None, p_perf=p_perf)
            elif self.regularization == "geo-reg-f":
                p_class = p_class_log.exp()
                reg_term = self._compute_reg_term(p_class=p_class, p_perf=None)
            elif self.regularization == "geo-reg-w":
                p_perf = (
                    p_perf_log.exp()
                    .swapaxes(1, 2)
                    .flatten(start_dim=0, end_dim=1)
                )
                reg_term = self._compute_reg_term(p_class=None, p_perf=p_perf)
            else:
                raise ValueError(
                    "`regularization` must be in ['trace-reg', 'geo-ref-f`, `geo-reg-w']."
                )
            loss += self.lmbda * reg_term
        return loss

    def _compute_reg_term(self, p_class, p_perf):
        if self.regularization == "trace-reg":
            reg_term = (
                p_perf.diagonal(offset=0, dim1=-2, dim2=-1).sum(-1).mean()
            )
        elif self.regularization in ["geo-reg-f", "geo-reg-w"]:
            if self.regularization == "geo-reg-f":
                reg_term = -torch.logdet(p_class.T @ p_class)
            else:
                reg_term = -torch.logdet(p_perf.T @ p_perf)
            if (
                torch.isnan(reg_term)
                or torch.isinf(torch.abs(reg_term))
                or reg_term > 100
            ):
                reg_term = 0
        return reg_term

    def fit(self, X, y, **fit_params):
        self.check_X_dict_ = {
            "ensure_min_samples": 0,
            "ensure_min_features": 0,
            "allow_nd": True,
            "dtype": None,
        }
        X, y, _ = self._validate_data(
            X=X,
            y=y,
            check_X_dict=self.check_X_dict_,
            y_ensure_1d=False,
        )

        self._check_n_features(X, reset=True)

        return NeuralNet.fit(self, X, y, **fit_params)

    def validation_step(self, batch, **fit_params):
        Xi, yi = unpack_data(batch)
        with torch.no_grad():
            y_pred = self.predict(Xi)
            y_pred_tensor = torch.from_numpy(y_pred)
            acc = torch.mean((y_pred_tensor == yi).float())
        return {
            "loss": acc,
            "y_pred": y_pred,
        }

    def predict_proba(self, X):
        logits_class = self.forward(X)
        P_class = F.softmax(logits_class, dim=-1)
        P_class = P_class.numpy()
        return P_class

    def predict_annotator_perf(self, X, return_confusion_matrix=False):
        p_class = self.predict_proba(X)
        p_perf = F.softmax(self.module_.ap_confs_, dim=-1)
        p_perf = p_perf.detach().numpy()
        p_perf = p_class[:, None, :, None] * p_perf[None, :, :, :]
        if return_confusion_matrix:
            return p_perf
        perf = p_perf.diagonal(axis1=-2, axis2=-1).sum(axis=-1)
        return perf


class RegCrowdNetModule(nn.Module):
    """RegCrowdNetModule"""

    def __init__(
        self, gt_embed_x, gt_output, n_classes, n_annotators, regularization
    ):
        super().__init__()
        self.gt_embed_x = gt_embed_x
        self.gt_output = gt_output
        if regularization == "trace-reg":
            self.ap_confs_ = nn.Parameter(
                torch.stack([6.0 * torch.eye(n_classes) - 5.0] * n_annotators)
            )
        elif regularization in ["geo-reg-f", "geo-reg-w"]:
            self.ap_confs_ = nn.Parameter(
                torch.stack([torch.eye(n_classes)] * n_annotators)
            )

    def forward(self, x):
        x_learned = self.gt_embed_x(x)
        logits_class = self.gt_output(x_learned)
        return logits_class
