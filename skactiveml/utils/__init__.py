"""
The :mod:`skactiveml.utils` module includes various utilities.
"""
import numpy as np
import pandas as pd

from sklearn.utils import check_array, check_consistent_length
from .selection import rand_argmax, rand_argmin

__all__ = ['rand_argmax', 'rand_argmin', 'compute_vote_vectors', 'is_unlabeled', 'is_labeled']


def is_unlabeled(y, unlabeled_class=-1):
    """Creates a boolean mask indicating missing labels.

    Parameters
    ----------
    y: array-like, shape (n_samples) or (n_samples, n_outputs)
        Class labels to be checked w.r.t. to missing labels.
    unlabeled_class: scalar | str | None | np.nan
        Symbol to represent a missing label. Important: We do not differ between None and np.nan.

    Returns
    -------
    is_unlabeled: array-like, shape (n_samples) or (n_samples, n_outputs)
        Boolean mask indicating missing labels.
    """
    if pd.isnull(unlabeled_class):
        return pd.isnull(y)
    return np.equal(y, unlabeled_class)


def is_labeled(y, unlabeled_class=-1):
    """Creates a boolean mask indicating available labels.

    Parameters
    ----------
    y: array-like, shape (n_samples) or (n_samples, n_outputs)
        Class labels to be checked w.r.t. to available labels.
    unlabeled_class: scalar | str | None | np.nan
        Symbol to represent a missing label. Important: We do not differ between None and np.nan.

    Returns
    -------
    is_unlabeled: array-like, shape (n_samples) or (n_samples, n_outputs)
        Boolean mask indicating available labels.
    """
    return ~is_unlabeled(y, unlabeled_class)


def compute_vote_vectors(y, w=None, n_classes=None, unlabeled=np.nan):
    """Counts number of votes for each sample and class label.

    Parameters
    ----------
    y: array-like, shape (n_samples) or (n_samples, n_annotators)
        Class labels.
    w: array-like, shape (n_samples) or (n_samples, n_annotators)
        Class label weights.
    n_classes: int
        Number of classes. If none, the number is inferred from the number of unique classes in y.
    unlabeled: scalar|string|np.nan|None, default=np.nan
        Value to represent a missing label.

    Returns
    -------
    V: array-like, shape (n_samples, classes)
        V[i,j] counts number of votes for class j and sample i.
    """
    # check input parameters
    y = check_array(y, ensure_2d=False, force_all_finite=False, dtype=None, copy=True)
    y = y if y.ndim == 2 else y.reshape((-1, 1))
    is_unlabeled_y = np.equal(y, unlabeled)
    y[is_unlabeled_y] = 0
    y = y.astype(int)
    n_classes = len(np.unique(y)) if n_classes is None else n_classes
    w = np.ones_like(y) if w is None else check_array(w, ensure_2d=False, force_all_finite=False, dtype=None, copy=True)
    w = w if w.ndim == 2 else w.reshape((-1, 1))
    check_consistent_length(y, w)
    check_consistent_length(y.T, w.T)
    w[np.logical_and(np.isnan(w), ~is_unlabeled_y)] = 1

    # count class labels per class and weight by confidence scores
    w[np.logical_or(np.isnan(w), is_unlabeled_y)] = 0
    y_off = y + np.arange(y.shape[0])[:, None] * n_classes
    V = np.bincount(y_off.ravel(), minlength=y.shape[0] * n_classes, weights=w.ravel())
    V = V.reshape(-1, n_classes)

    return V
