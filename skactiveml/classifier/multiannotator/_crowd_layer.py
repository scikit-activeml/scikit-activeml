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
        """Fit the model using X as training data and y as annotation from annotators.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The sample matrix X is the feature matrix representing the samples.
        y : array-like of shape (n_samples, n_estimators)
            It contains the class labels of the training samples.
            The number of class labels may be variable for the samples, where
            missing labels are represented the attribute `missing_label`.
        sample_weight : array-like of shape (n_samples, n_estimators)
            It contains the weights of the training samples' class labels.
            It must have the same shape as `y`.

        Returns
        -------
        self: skactiveml.classifier.multiannotator.AnnotatorEnsembleClassifier,
            The `AnnotatorEnsembleClassifier` object fitted on the training
            data.
        """

        self.check_X_dict_ = {
            "ensure_min_samples": 0,
            "ensure_min_features": 0,
            "allow_nd": True,
            "dtype": None,
        }
        X, y, sample_weight = self._validate_data(
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
        P_perf = np.array(
            [
                np.einsum("ij,ik->ijk", P_class, P_annot[:, :, i])
                for i in range(n_annotators)
            ]
        )
        P_perf = P_perf.swapaxes(0, 1)
        if return_confusion_matrix:
            return P_perf
        return P_perf.diagonal(axis1=-2, axis2=-1).sum(axis=-1)

    def predict(self, X):
        p_class = self.predict_proba(X)
        return p_class.argmax(axis=1)

    def predict_proba(self, X):
        p_class = self.forward(X, return_logits_annotators=False)
        return p_class.numpy()

    def validation_step(self, batch, **fit_params):
        # not for loss but for acc
        Xi, yi = unpack_data(batch)
        with torch.no_grad():
            y_pred = self.predict(Xi)
            acc = torch.mean((y_pred == yi).float())
        return {
            "acc": acc,
            "y_pred": y_pred,
        }


class CrowdLayerModule(nn.Module):
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
            layer.weight = nn.Parameter(torch.eye(n_classes) * 10)
            self.annotator_layers.append(layer)

    def forward(self, x, return_logits_annotators=True):
        """Forward propagation of samples through the GT and AP (optional) model.

        Parameters
        ----------
        x : torch.Tensor of shape (batch_size, *)
            Samples.
        return_logits_annot: bool, optional (default=True)
            Flag whether the annotation logits are to be returned, next to the class-membership probabilities.

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

        if not return_logits_annotators:
            return p_class

        # Compute logits per annotator.
        logits_annot = []
        for layer in self.annotator_layers:
            logits_annot.append(layer(p_class))
        logits_annot = torch.stack(logits_annot, dim=2)

        return p_class, logits_annot
