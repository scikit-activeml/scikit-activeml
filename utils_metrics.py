import numpy as np
from torchmetrics.classification import MultilabelAveragePrecision

def topk_accuracy(preds, targets, topk=1, include_nocalls=False, threshold=0.5):
    """
    Compute the Top-K Accuracy for multi-class or multi-label predictions using NumPy.

    Args:
        preds (np.ndarray): Prediction probabilities of shape (N, C).
        targets (np.ndarray): One-hot encoded ground truth labels of shape (N, C).
        topk (int): Number of top predictions to consider.
        include_nocalls (bool): Whether to include no-call cases in accuracy computation.
        threshold (float): Threshold for considering predictions as no-calls.

    Returns:
        float: Top-K accuracy.
    """
    # Get the indices of the top-k predictions
    topk_pred_indices = np.argsort(-preds, axis=1)[:, :topk]

    # Identify no-call targets (rows where all targets are 0)
    no_call_targets = np.sum(targets, axis=1) == 0

    # Process no-call instances if specified
    if include_nocalls:
        # Check if all top-k predictions are below the threshold
        no_positive_predictions = np.all(np.sort(-preds, axis=1)[:, :topk] < -threshold, axis=1)
        correct_all_negative = no_call_targets & no_positive_predictions
    else:
        correct_all_negative = np.zeros_like(no_call_targets, dtype=bool)

    # Convert one-hot encoded targets to indices for positive cases
    correct_positive = np.any(targets[np.arange(targets.shape[0])[:, None], topk_pred_indices], axis=1)

    # Combine correct predictions
    correct = np.sum(correct_positive) + np.sum(correct_all_negative)

    # Determine total number of samples to consider
    if not include_nocalls:
        total = len(targets) - np.sum(no_call_targets)
    else:
        total = len(targets)

    # Compute accuracy
    accuracy = correct / total if total > 0 else 0.0

    return accuracy

class cmAP(MultilabelAveragePrecision):
    def __init__(
            self,
            num_labels,
            thresholds=None
        ):
        super().__init__(
            num_labels=num_labels,
            average="macro",
            thresholds=thresholds
        )

    def __call__(self, logits, labels):
        macro_cmap = super().__call__(logits, labels)
        return macro_cmap