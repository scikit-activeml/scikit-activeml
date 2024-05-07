import torch
from torch import nn
from torch.nn import functional as F

from skactiveml.base import AnnotatorModelMixin
from skactiveml.classifier import SkorchClassifier


class CrowdLayerClassifier(SkorchClassifier, AnnotatorModelMixin):
    def __init__(self, *args, **kwargs):
        super(CrowdLayerClassifier, self).__init__(*args, **kwargs)

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
            nn.init.eye_(layer.weight)
            self.annotator_layers.append(layer)

    def forward(self, x, return_logits_annot=Tru):
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
        logits_annot : torch.Tensor of shape (batch_size, n_annotators, n_classes)
            Annotation logits for each sample-annotator pair.
        """
        # Compute class-membership logits.
        logit_class = self.gt_net(x)

        # Compute class-membership probabilities.
        p_class = F.softmax(logit_class, dim=-1)

        if not return_logits_annot:
            return p_class

        # Compute logits per annotator.
        logits_annot = []
        for layer in self.annotator_layers:
            logits_annot.append(layer(p_class))
        logits_annot = torch.stack(logits_annot, dim=2)

        return p_class, logits_annot


