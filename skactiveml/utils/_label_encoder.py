import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from ._label import MISSING_LABEL, is_labeled
from ._validation import check_classifier_params


class ExtLabelEncoder(TransformerMixin, BaseEstimator):
    """Encode class labels with value between 0 and classes-1 and uses -1 for
    unlabeled samples.
    This transformer should be used to encode class labels, *i.e.* `y`, and
    not the input `X`.

    Parameters
    ----------
    classes: array-like, shape (n_classes), default=None
        Holds the label for each class.
    missing_label: scalar|string|np.nan|None, default=np.nan
        Value to represent a missing label.

    Attributes
    ----------
    classes_: array-like, shape (n_classes)
        Holds the label for each class.
    """

    def __init__(self, classes=None, missing_label=MISSING_LABEL):
        self.classes = classes
        self.missing_label = missing_label

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
        check_classifier_params(
            classes=self.classes, missing_label=self.missing_label
        )
        y = check_array(y, ensure_2d=False, force_all_finite=False, dtype=None)
        self._le = LabelEncoder()
        if self.classes is None:
            y = np.asarray(y)
            is_lbld = is_labeled(y, missing_label=self.missing_label)
            self._dtype = np.append(y, self.missing_label).dtype
            self._le.fit(y[is_lbld])
        else:
            self._dtype = np.append(self.classes, self.missing_label).dtype
            self._le.fit(self.classes)
            self.classes_ = self._le.classes_
        self.classes_ = self._le.classes_

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
        y : array-like of shape (n_samples)
            Target values.

        Returns
        -------
        y_enc : array-like of shape (n_samples
        """
        check_is_fitted(self, attributes=["classes_"])
        y = check_array(
            y,
            ensure_2d=False,
            force_all_finite=False,
            ensure_min_samples=0,
            dtype=None,
        )
        is_lbld = is_labeled(y, missing_label=self.missing_label)
        y = np.asarray(y)
        y_enc = np.empty_like(y, dtype=int)
        y_enc[is_lbld] = self._le.transform(y[is_lbld].ravel())
        y_enc[~is_lbld] = -1
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
        check_is_fitted(self, attributes=["classes_"])
        y = check_array(
            y,
            ensure_2d=False,
            force_all_finite=False,
            ensure_min_samples=0,
            dtype=None,
        )
        is_lbld = is_labeled(y, missing_label=-1)
        y = np.asarray(y)
        y_dec = np.empty_like(y, dtype=self._dtype)
        y_dec[is_lbld] = self._le.inverse_transform(
            np.array(y[is_lbld].ravel())
        )
        y_dec[~is_lbld] = self.missing_label
        return y_dec
