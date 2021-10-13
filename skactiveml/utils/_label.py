import numpy as np
from iteration_utilities import deepflatten
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted, check_array

from ..utils._validation import check_classifier_params, check_missing_label, \
    MISSING_LABEL


def is_unlabeled(y, missing_label=MISSING_LABEL):
    """Creates a boolean mask indicating missing labels.

    Parameters
    ----------
    y : array-like, shape (n_samples) or (n_samples, n_outputs)
        Class labels to be checked w.r.t. to missing labels.
    missing_label : number | str | None | np.nan, optional (default=np.nan)
        Symbol to represent a missing label.

    Returns
    -------
    is_unlabeled : numpy.ndarray, shape (n_samples) or (n_samples, n_outputs)
        Boolean mask indicating missing labels in y.
    """
    check_missing_label(missing_label)
    if len(y) == 0:
        raise ValueError("'y' is not allowed to be empty.")
    if not isinstance(y, np.ndarray):
        types = set(
            t.__qualname__ for t in set(type(v) for v in deepflatten(y)))
        types.add(type(missing_label).__qualname__)
        is_number = False
        is_character = False
        for t in types:
            t = object if t == 'NoneType' else t
            is_character = True \
                if np.issubdtype(t, np.character) else is_character
            is_number = True if np.issubdtype(t, np.number) else is_number
            if is_character and is_number:
                raise TypeError(
                    "'y' must be uniformly strings or numbers. "
                    "'NoneType' is allowed. Got {}".format(types))
        y = np.asarray(y)
    target_type = np.append(y.ravel(), missing_label).dtype
    check_missing_label(missing_label, target_type=target_type, name='y')
    if (y.ndim == 2 and np.size(y, axis=1) == 0) or y.ndim > 2:
        raise ValueError("'y' must be of shape (n_samples) or '(n_samples, "
                         "n_features)' with 'n_samples > 0' and "
                         "'n_features > 0'.")
    if missing_label is np.nan:
        return np.isnan(y)
    else:
        return y == missing_label


def is_labeled(y, missing_label=MISSING_LABEL):
    """Creates a boolean mask indicating present labels.

    Parameters
    ----------
    y : array-like, shape (n_samples) or (n_samples, n_outputs)
        Class labels to be checked w.r.t. to present labels.
    missing_label : number | str | None | np.nan, optional (default=np.nan)
        Symbol to represent a missing label.

    Returns
    -------
    is_unlabeled : numpy.ndarray, shape (n_samples) or (n_samples, n_outputs)
        Boolean mask indicating present labels in y.
    """
    return ~is_unlabeled(y, missing_label)


def unlabeled_indices(y, missing_label=MISSING_LABEL):
    """Return an array of indices indicating missing labels.

    Parameters
    ----------
    y : array-like, shape (n_samples) or (n_samples, n_outputs)
        Class labels to be checked w.r.t. to present labels.
    missing_label : number | str | None | np.nan, optional (default=np.nan)
        Symbol to represent a missing label.

    Returns
    -------
    unlbld_indices : numpy.ndarray, shape (n_samples) or (n_samples, 2)
        Index array of missing labels. If y is a 2D-array, the indices
        have shape `(n_samples, 2), otherwise it has the shape `(n_samples)`.
    """
    is_unlbld = is_unlabeled(y, missing_label)
    unlbld_indices = np.argwhere(is_unlbld)
    return unlbld_indices[:, 0] if is_unlbld.ndim == 1 else unlbld_indices


def labeled_indices(y, missing_label=MISSING_LABEL):
    """Return an array of indices indicating present labels.

    Parameters
    ----------
    y : array-like, shape (n_samples) or (n_samples, n_outputs)
        Class labels to be checked w.r.t. to present labels.
    missing_label : number | str | None | np.nan, optional (default=np.nan)
        Symbol to represent a missing label.

    Returns
    -------
    lbld_indices : numpy.ndarray, shape (n_samples) or (n_samples, 2)
        Index array of present labels. If y is a 2D-array, the indices
        have shape `(n_samples, 2), otherwise it has the shape `(n_samples)`.
    """
    is_lbld = is_labeled(y, missing_label)
    lbld_indices = np.argwhere(is_lbld)
    return lbld_indices[:, 0] if is_lbld.ndim == 1 else lbld_indices


class ExtLabelEncoder(TransformerMixin, BaseEstimator):
    """Encode class labels with value between 0 and classes-1.
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
        check_classifier_params(classes=self.classes,
                                missing_label=self.missing_label)
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
        check_is_fitted(self, attributes=['classes_'])
        y = check_array(y, ensure_2d=False, force_all_finite=False, dtype=None)
        is_lbld = is_labeled(y, missing_label=self.missing_label)
        y = np.asarray(y)
        y_enc = np.empty_like(y, dtype=float)
        y_enc[is_lbld] = self._le.transform(y[is_lbld].ravel())
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
        check_is_fitted(self, attributes=['classes_'])
        y = check_array(y, ensure_2d=False, force_all_finite=False, dtype=None)
        is_lbld = is_labeled(y, missing_label=np.nan)
        y = np.asarray(y)
        y_dec = np.empty_like(y, dtype=self._dtype)
        y_dec[is_lbld] = self._le.inverse_transform(
            np.array(y[is_lbld].ravel(), dtype=int))
        y_dec[~is_lbld] = self.missing_label
        return y_dec
