import numpy as np

from ._label import ExtLabelEncoder
from sklearn.utils import check_array, check_consistent_length


def compute_vote_vectors(y, w=None, classes=None, missing_label=np.nan):
    """Counts number of votes per class label for each sample.

    Parameters
    ----------
    y : array-like, shape (n_samples) or (n_samples, n_annotators)
        Class labels.
    w : array-like, shape (n_samples) or (n_samples, n_annotators),
    default=np.ones_like(y)
        Class label weights.
    classes : array-like, shape (n_classes), default=None
        Holds the label for each class.
    missing_label : scalar|string|np.nan|None, default=np.nan
        Value to represent a missing label.

    Returns
    -------
    v : array-like, shape (n_samples, classes)
        V[i,j] counts number of votes per class j for sample i.
    """
    # check input parameters
    le = ExtLabelEncoder(classes=classes, missing_label=missing_label)
    y = le.fit_transform(y)
    n_classes = len(le.classes_)
    y = y if y.ndim == 2 else y.reshape((-1, 1))
    is_unlabeled_y = np.isnan(y)
    y[is_unlabeled_y] = 0
    y = y.astype(int)
    n_classes = max(len(np.unique(y)) if n_classes is None else n_classes, 1)
    w = np.ones_like(y) if w is None else check_array(w, ensure_2d=False,
                                                      force_all_finite=False,
                                                      dtype=None, copy=True)
    w = w if w.ndim == 2 else w.reshape((-1, 1))
    check_consistent_length(y, w)
    check_consistent_length(y.T, w.T)
    w[is_unlabeled_y] = 1

    # count class labels per class and weight by confidence scores
    w[np.logical_or(np.isnan(w), is_unlabeled_y)] = 0
    y_off = y + np.arange(y.shape[0])[:, None] * n_classes
    v = np.bincount(y_off.ravel(), minlength=y.shape[0] * n_classes,
                    weights=w.ravel())
    v = v.reshape(-1, n_classes)

    return v
