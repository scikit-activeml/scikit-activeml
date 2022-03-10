import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.utils.validation import (
    check_consistent_length,
    column_or_1d,
    check_array,
)

from ._label import MISSING_LABEL, is_labeled, is_unlabeled
from ._label_encoder import ExtLabelEncoder


def ext_confusion_matrix(
        y_true, y_pred, classes=None, missing_label=MISSING_LABEL,
        normalize=None
):
    """Compute confusion matrix to evaluate the accuracy of a classification.

    This is an extension of the 'sklearn.metric.confusion_matrix function' by
    allowing missing labels and labels predicted by multiple annotators.

    By definition a confusion matrix :math:`C` is such that :math:`C_{i, j}`
    is equal to the number of observations known to be in group :math:`i` and
    predicted to be in group :math:`j`.

    Thus in binary classification, the count of true negatives is
    :math:`C_{0,0}`, false negatives is :math:`C_{1,0}`, true positives is
    :math:`C_{1,1}` and false positives is :math:`C_{0,1}`.

    Parameters
    ----------
    y_true: array-like, shape (n_samples)
        Array of true labels. Is not allowed to contain any missing labels.
    y_pred: array-like, shape (n_samples) or (n_samples, n_annotators)
            Estimated targets as returned by multiple annotators.
    classes : array-like of shape (n_classes), default=None
        List of class labels to index the matrix. This may be used to reorder
        or select a subset of labels. If ``None`` is given, those that appear
        at least once in ``y_true`` or ``y_pred`` are used in sorted order.
    missing_label : {scalar, string, np.nan, None}, default=np.nan
        Value to represent a missing label.
    normalize : {'true', 'pred', 'all'}, default=None
        Normalizes confusion matrix over the true (rows), predicted (columns)
        conditions or all the population. If None, confusion matrix will not be
        normalized.

    Returns
    -------
    conf_matrices : numpy.ndarray, shape (n_annotators, n_classes, n_classes)
        Confusion matrix whose i-th row and j-th column entry indicates the
        number of samples with true label being i-th class and prediced label
        being j-th class.

    References
    ----------
    [1] `Wikipedia entry for the Confusion matrix
        <https://en.wikipedia.org/wiki/Confusion_matrix>`_
        (Wikipedia and other references may use a different convention for
        axes)
    [2] `Scikit-learn Confusion Matrix
        <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.
        confusion_matrix.html>`_
    """
    # Check input.
    y_true = column_or_1d(y_true)
    y_pred = check_array(
        y_pred, force_all_finite=False, ensure_2d=False, dtype=None
    )
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    check_consistent_length(y_true, y_pred)
    if normalize not in ["true", "pred", "all", None]:
        raise ValueError(
            "'normalize' must be one of {'true', 'pred', 'all', " "None}."
        )
    le = ExtLabelEncoder(classes=classes, missing_label=missing_label)
    y = np.column_stack((y_true, y_pred))
    y = le.fit_transform(y)
    if np.sum(is_unlabeled(y[:, 0], missing_label=-1)):
        raise ValueError("'y_true' is not allowed to contain missing labels.")
    n_classes = len(le.classes_)
    n_annotators = y_pred.shape[1]

    # Determine confusion matrix for each annotator.
    conf_matrices = np.zeros((n_annotators, n_classes, n_classes))
    for a in range(n_annotators):
        is_not_nan_a = is_labeled(y[:, a + 1], missing_label=-1)
        if np.sum(is_not_nan_a) > 0:
            cm = confusion_matrix(
                y_true=y[is_not_nan_a, 0],
                y_pred=y[is_not_nan_a, a + 1],
                labels=np.arange(n_classes),
            )
        else:
            cm = np.zeros((n_classes, n_classes))
        with np.errstate(all="ignore"):
            if normalize == "true":
                cm = cm / cm.sum(axis=1, keepdims=True)
                conf_matrices[a] = np.nan_to_num(cm, nan=1 / n_classes)
            elif normalize == "pred":
                cm = cm / cm.sum(axis=0, keepdims=True)
                conf_matrices[a] = np.nan_to_num(cm, nan=1 / n_classes)
            elif normalize == "all":
                cm = cm / cm.sum()
                conf_matrices[a] = np.nan_to_num(cm, nan=1 / cm.size)

    return conf_matrices
