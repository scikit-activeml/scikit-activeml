import numpy as np
import torch
from skorch import NeuralNet
from skorch.dataset import unpack_data
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F

from ...base import AnnotatorModelMixin
from ...classifier import SkorchClassifier
from ...utils import ExtLabelEncoder


class CrowdLayerClassifier(SkorchClassifier, AnnotatorModelMixin):
    """
    CrowdLayerClassifier

    CrowdLayer [1] is a layer added at the end of a classifying neural network
    and allows us to train deep neural networks end-to-end, directly from the
    noisy labels of multiple annotators, using only backpropagation.

    Parameters
    ----------
    module__n_annotators : int
         Number of annotators.
    module__gt_net : nn.Module
        Pytorch module of the GT model taking samples
        as input to predict class-membership logits.
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
    def __init__(self, *args, **kwargs):
        super(CrowdLayerClassifier, self).__init__(
            module=CrowdLayerModule,
            *args,
            criterion=CrossEntropyLoss(),
            criterion__reduction="mean",
            criterion__ignore_index=-1,
            **kwargs,
        )

    def get_loss(self, y_pred, y_true, *args, **kwargs):
        # unpack the tuple from the forward function
        p_class, logits_annot = y_pred
        loss = NeuralNet.get_loss(self, logits_annot, y_true, *args, **kwargs)
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

    def predict_annotator_perf(self, X, return_confusion_matrix=False):
        n_annotators = self.module__n_annotators
        P_class, logits_annot = self.forward(X)
        P_class = P_class.numpy()
        P_annot = F.softmax(logits_annot, dim=1)
        P_annot = P_annot.numpy()
        P_perf = np.array([np.einsum("ij,ik->ijk", P_class, P_annot[:, :, i]) for i in range(n_annotators)])
        P_perf = P_perf.swapaxes(0, 1)
        if return_confusion_matrix:
            return P_perf
        return P_perf.diagonal(axis1=-2, axis2=-1).sum(axis=-1)

    def predict(self, X):
        # maybe flag to switch between mode
        p_class, logits_annot = self.forward(X)
        return p_class.argmax(axis=1)

    def predict_proba(self, X):
        P_class, logits_annot = self.forward(X)
        P_class = P_class.numpy()
        return P_class

    def validation_step(self, batch, **fit_params):
        # not for loss but for acc
        Xi, yi = unpack_data(batch)
        with torch.no_grad():
            y_pred = self.predict(Xi)
            acc = torch.mean((y_pred == yi).float())
        return {
            'loss': acc,
            'y_pred': y_pred,
        }

    def predict_P_annot(self, X):
        _, logits_annot = self.forward(X)
        P_annot = F.softmax(logits_annot, dim=1)
        P_annot = P_annot.numpy()
        return P_annot


class CrowdLayerModule(nn.Module):
    """
    CrowdLayerModule

    CrowdLayer [1] is a layer added at the end of a classifying neural network
    and allows us to train deep neural networks end-to-end, directly from the
    noisy labels of multiple annotators, using only backpropagation.

    Parameters
    ----------
    n_classes : int
        Number of classes.
    n_annotators : int
        Number of annotators.
    gt_net : nn.Module
        Pytorch module of the GT model taking samples
        as input to predict class-membership logits.

    References
    ----------
    [1] Rodrigues, Filipe, and Francisco Pereira. "Deep learning from crowds." In Proceedings
    of the AAAI conference on artificial intelligence, vol. 32, no. 1. 2018.
    """
    def __init__(
            self,
            n_classes,
            n_annotators,
            gt_net,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.n_annotators = n_annotators
        self.gt_net = gt_net

        # Setup crowd layer.
        self.annotator_layers = nn.ModuleList()
        for i in range(n_annotators):
            layer = nn.Linear(n_classes, n_classes, bias=False)
            layer.weight = nn.Parameter(torch.eye(n_classes))
            self.annotator_layers.append(layer)

    def forward(self, x):
        """Forward propagation of samples through the GT and AP (optional) model.

        Parameters
        ----------
        x : torch.Tensor of shape (batch_size, *)
            Samples.

        Returns
        -------
        p_class : torch.Tensor of shape (batch_size, n_classes)
            Class-membership probabilities.
        logits_annot : torch.Tensor of shape (batch_size, n_classes, n_annotators)
            Annotation logits for each sample-annotator pair.
        """
        # Compute class-membership logits.
        logit_class = self.gt_net(x)

        # Compute class-membership probabilities.
        p_class = F.softmax(logit_class, dim=-1)

        # Compute logits per annotator.
        logits_annot = []
        for layer in self.annotator_layers:
            logits_annot.append(layer(p_class))
        logits_annot = torch.stack(logits_annot, dim=2)

        return p_class, logits_annot

