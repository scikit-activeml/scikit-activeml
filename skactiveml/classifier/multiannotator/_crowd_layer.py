import numpy as np
import torch
from skorch import NeuralNet
from skorch.dataset import unpack_data
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F

from ...base import AnnotatorModelMixin
from ...classifier import SkorchClassifier


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
        # unpack the tuple from the forward function
        p_class, logits_annot = y_pred
        loss = NeuralNet.get_loss(self, logits_annot, y_true, *args, **kwargs)
        return loss

    def fit(self, X, y, **fit_params):
        """Initialize and fit the module.

        If the module was already initialized, by calling fit, the
        module will be re-initialized (unless ``warm_start`` is True).

        Parameters
        ----------
        X : matrix-like, shape (n_samples, n_features)
            Training data set, usually complete, i.e. including the labeled and
            unlabeled samples
        y : array-like of shape (n_samples, )
            Labels of the training data set (possibly including unlabeled ones
            indicated by self.missing_label)
        fit_params : dict-like
            Further parameters as input to the 'fit' method of the 'estimator'.

        Returns
        -------
        self: CrowdLayerClassifier,
            The CrowdLayerClassifier is fitted on the training data.
        """
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
        """Calculates the probability that an annotator provides the true label for a given sample.

        Parameters
        ----------
        X : matrix-like, shape (n_samples, n_features)
            Test samples.
        return_confusion_matrix : bool, optional (default=False)
            If `return_confusion_matrix=True`, the entire confusion matrix per annotator is returned.

        Returns
        -------
        P_perf : numpy.ndarray of shape (n_samples, n_annotators) or (n_samples, n_annotators, n_classes, n_classes)
            If `return_confusion_matrix=False`, `P_perf[n, m]` is the probability, that annotator `A[m]` provides the
            correct class label for sample `X[n]`. If `return_confusion_matrix=False`, `P_perf[n, m, c, j]` is the
            probability, that annotator `A[m]` provides the correct class label `classes_[j]` for sample `X[n]` and
            that this sample belongs to class `classes_[c]`. If `return_cond=True`, `P_perf[n, m, c, j]` is the
            probability that annotator `A[m]` provides the class label `classes_[j]` for sample `X[n]` conditioned
            that this sample belongs to class `classes_[c]`.
        """
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

    def predict_proba(self, X):
        """Returns class-membership probability estimates for the test data `X`.

        Parameters
        ----------
        X : matrix-like, shape (n_samples, n_features)
            Test samples.

        Returns
        -------
        P_class : numpy.ndarray of shape (n_samples, classes)
            `P_class[n, c]` is the probability, that instance `X[n]` belongs to the `classes_[c]`.
        """
        P_class, logits_annot = self.forward(X)
        P_class = P_class.numpy()
        return P_class

    def validation_step(self, batch, **fit_params):
        """Perform a single validation step.

        Parameters
        ----------
        batch : list
            A list containing the input data (Xi) and the target labels (yi) for the validation batch.
        fit_params : dict
            Additional fit parameters (not used in this function).

        Returns
        -------
        A dictionary containing:
        - 'loss' : float
            The accuracy of the predictions for the validation batch.
        - 'y_pred' : numpy.ndarray
            The predicted labels for the input data in the validation batch.
        """
        Xi, yi = unpack_data(batch)
        with torch.no_grad():
            y_pred = self.predict(Xi)
            y_pred_tensor = torch.from_numpy(y_pred)
            acc = torch.mean((y_pred_tensor == yi).float())
        return {
            "loss": acc,
            "y_pred": y_pred,
        }

    def predict_P_annot(self, X):
        """Predict the probabilities of annotator assign for a label for the given input data.

        Parameters
        ----------
        X : matrix-like, shape (n_samples, n_features)
            Test samples.

        Returns
        -------
        numpy.ndarray
            The predicted probabilities for each annotator, obtained by applying softmax to the logits.
        """
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
