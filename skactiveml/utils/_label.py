import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import column_or_1d
from sklearn.preprocessing import LabelEncoder


def is_unlabeled(y, unlabeled_class=np.nan):
    """Creates a boolean mask indicating missing labels.

    Parameters
    ----------
    y : array-like, shape (n_samples) or (n_samples, n_outputs)
        Class labels to be checked w.r.t. to missing labels.
    unlabeled_class : scalar | str | None | np.nan
        Symbol to represent a missing label. Important: We do not differ between None and np.nan.

    Returns
    -------
    is_unlabeled : array-like, shape (n_samples) or (n_samples, n_outputs)
        Boolean mask indicating missing labels.
    """
    if pd.isnull(unlabeled_class):
        return pd.isnull(y)
    else:
        y = np.asarray(y)
        return y == unlabeled_class


def is_labeled(y, unlabeled_class=np.nan):
    """Creates a boolean mask indicating available labels.

    Parameters
    ----------
    y : array-like, shape (n_samples) or (n_samples, n_outputs)
        Class labels to be checked w.r.t. to available labels.
    unlabeled_class : scalar | str | None | np.nan
        Symbol to represent a missing label. Important: We do not differ between None and np.nan.

    Returns
    -------
    is_unlabeled : array-like, shape (n_samples) or (n_samples, n_outputs)
        Boolean mask indicating available labels.
    """
    return ~is_unlabeled(y, unlabeled_class)

class ExtLabelEncoder(BaseEstimator, TransformerMixin):
    """Encode class labels with value between 0 and classes-1.
    This transformer should be used to encode class labels, *i.e.* `y`, and
    not the input `X`.

    Parameters
    ----------
    classes: array-like, shape (n_classes), default=None
        Holds the label for each class.
    unlabeled_class: scalar|string|np.nan|None, default=np.nan
        Value to represent a missing label.

    Attributes
    ----------
    classes: array-like, shape (n_classes)
        Holds the label for each class.
    unlabeled_class: scalar|string|np.nan|None, default=np.nan
        Value to represent a missing label.
    """

    def __init__(self, classes=None, unlabeled_class=np.nan):
        self.classes = column_or_1d(classes, warn=True) if classes is not None else None
        self.unlabeled_class = unlabeled_class
        self.le = LabelEncoder()

    def fit(self, y):
        """Fit label encoder.

        Parameters
        ----------
        y: array-like, shape (n_samples) or (n_samples, n_outputs)
            Class labels.

        Returns
        -------
        self: returns an instance of self.
        """
        is_lbld = is_labeled(y, unlabeled_class=self.unlabeled_class)
        y = np.asarray(y)
        self.dtype_ = y.dtype
        if self.classes is None:
            self.le.fit(y[is_lbld])
        else:
            self.le.fit(self.classes)
        self.classes = self.le.classes_

        return self

    def fit_transform(self, y):
        """Fit label encoder and return encoded labels.

        Parameters
        ----------
        y: array-like, shape (n_samples) or (n_samples, n_outputs)
            Class labels.

        Returns
        -------
        y: array-like, shape (n_samples) or (n_samples, n_outputs)
            Class labels.
        """
        return self.fit(y).transform(y)

    def transform(self, y):
        """Transform labels to normalized encoding.
        Parameters
        ----------
        y : array-like of shape [n_samples]
            Target values.
        Returns
        -------
        y_enc : array-like of shape [n_samples]
        """
        is_lbld = is_labeled(y, unlabeled_class=self.unlabeled_class)
        y = np.asarray(y)
        y_enc = np.empty_like(y, dtype=float)
        y_enc[is_lbld] = self.le.transform(y[is_lbld].ravel())
        y_enc[~is_lbld] = np.nan
        return y_enc

    def inverse_transform(self, y):
        """Transform labels back to original encoding.
        Parameters
        ----------
        y : numpy array of shape [n_samples]
            Target values.
        Returns
        -------
        y_dec : numpy array of shape [n_samples]
        """
        is_lbld = is_labeled(y, unlabeled_class=np.nan)
        y = np.asarray(y)
        y_dec = np.empty_like(y, dtype=self.dtype_)
        y_dec[is_lbld] = self.le.inverse_transform(np.array(y[is_lbld].ravel(), dtype=int))
        y_dec[~is_lbld] = self.unlabeled_class
        return y_dec


