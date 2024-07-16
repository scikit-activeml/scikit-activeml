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
        super(RegCrowdNetClassifier, self).__init__(
            module=RegCrowdNetModule,
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
        if regularization == "trace-reg":
            self.ap_confs = nn.Parameter(torch.stack([6.0 * torch.eye(n_classes) - 5.0] * n_annotators))
        elif regularization in ["geo-reg-f", "geo-reg-w"]:
            self.ap_confs = nn.Parameter(torch.stack([torch.eye(n_classes)] * n_annotators))
        else:
            raise ValueError("`regularization` must be in ['trace-reg', 'geo-reg-f', 'geo-reg-w'].")

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
            torch.arange(n_samples, device=y_true.device), torch.arange(n_annotators, device=y_true.device)
        )
        z = y_true.ravel()
        is_lbld = z != -1  # missing_label is -1
        combs, z = combs[is_lbld], z[is_lbld]
        p_class_log = F.log_softmax(y_pred, dim=-1)
        p_class_log_ext = p_class_log[combs[:, 0]]
        p_perf_log = F.log_softmax(self.ap_confs, dim=-1)
        p_perf_log_ext = p_perf_log[combs[:, 1]]
        p_annot_log = torch.logsumexp(p_class_log_ext[:, :, None] + p_perf_log_ext, dim=1)
        loss = NeuralNet.get_loss(self, p_annot_log, z)
        if self.lmbda > 0:
            if self.regularization == "trace-reg":
                p_perf = F.softmax(self.ap_confs, dim=-1)
                reg_term = p_perf.diagonal(offset=0, dim1=-2, dim2=-1).sum(-1).mean()
            elif self.regularization == "geo-reg-f":
                p_class = p_class_log.exp()
                reg_term = -torch.logdet(p_class.T @ p_class)
                if torch.isnan(reg_term) or torch.isinf(torch.abs(reg_term)) or reg_term > 100:
                    reg_term = 0
            elif self.regularization == "geo-reg-w":
                p_perf = p_perf_log.exp().swapaxes(1, 2).flatten(start_dim=0, end_dim=1)
                reg_term = -torch.logdet(p_perf.T @ p_perf)
                if torch.isnan(reg_term) or torch.isinf(torch.abs(reg_term)) or reg_term > 100:
                    reg_term = 0
            else:
                raise ValueError("`regularization` must be in ['trace-reg', 'geo-ref-f`, `geo-reg-w'].")
            loss += self.lmbda * reg_term
        return loss

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

    def predict_annotator_perf(self, return_confusion_matrix=False):
        p_perf = F.softmax(self.ap_confs, dim=-1)
        p_perf_numpy = p_perf.detach().numpy()
        if return_confusion_matrix:
            return p_perf_numpy
        perf = p_perf.diagonal(offset=0, dim1=-2, dim2=-1).sum(-1).mean()
        return perf.detach().numpy()


class RegCrowdNetModule(nn.Module):
    """RegCrowdNetModule
    """
    def __init__(self, gt_embed_x, gt_output):
        super().__init__()
        self.gt_embed_x = gt_embed_x
        self.gt_output = gt_output

    def forward(self, x):
        x_learned = self.gt_embed_x(x)
        logits_class = self.gt_output(x_learned)
        return logits_class