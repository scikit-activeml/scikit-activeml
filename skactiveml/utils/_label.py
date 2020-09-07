import numpy as np

from iteration_utilities import deepflatten
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_array
from sklearn.preprocessing import LabelEncoder
from ..utils._validation import check_missing_label, check_classes

MISSING_LABEL = np.nan


def is_unlabeled(y, missing_label=MISSING_LABEL):
    """Creates a boolean mask indicating missing labels.

    Parameters
    ----------
    y : array-like, shape (n_samples) or (n_samples, n_outputs)
        Class labels to be checked w.r.t. to missing labels.
    missing_label : number | str | None | np.nan
        Symbol to represent a missing label.

    Returns
    -------
    is_unlabeled : numpy.ndarray, shape (n_samples) or (n_samples, n_outputs)
        Boolean mask indicating missing labels in y.
    """
    missing_label = check_missing_label(missing_label)
    if not isinstance(y, np.ndarray):
        types = set(
            t.__qualname__ for t in set(type(v) for v in deepflatten(y)))
        types.add(type(missing_label).__qualname__)
        is_number = False
        is_character = False
        for t in types:
            t = np.object if t == 'NoneType' else t
            is_character = True \
                if np.issubdtype(t, np.character) else is_character
            is_number = True if np.issubdtype(t, np.number) else is_number
            if is_character and is_number:
                raise TypeError(
                    "'y' must be uniformly strings or numbers. "
                    "'NoneType' is allowed. Got {}".format(types))
        y = np.asarray(y)
    target_type = np.append(y.ravel(), missing_label).dtype
    missing_label = check_missing_label(missing_label, target_type=target_type,
                                        name='y')
    if len(y) == 0:
        return np.array([], dtype=bool)
    elif missing_label is np.nan:
        return np.isnan(y)
    else:
        return y == missing_label


def is_labeled(y, missing_label=MISSING_LABEL):
    """Creates a boolean mask indicating present labels.

    Parameters
    ----------
    y : array-like, shape (n_samples) or (n_samples, n_outputs)
        Class labels to be checked w.r.t. to present labels.
    missing_label : number | str | None | np.nan
        Symbol to represent a missing label.

    Returns
    -------
    is_unlabeled : numpy.ndarray, shape (n_samples) or (n_samples, n_outputs)
        Boolean mask indicating present labels in y.
    """
    return ~is_unlabeled(y, missing_label)


class ExtLabelEncoder(BaseEstimator, TransformerMixin):
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
    missing_label: scalar|string|np.nan|None, default=np.nan
        Value to represent a missing label.
    _dtype: numpy data type
        Inferred from classes or y through fitting.
    _le: sklearn.preprocessing.LabelEncoder
        LabelEncoder created through fitting.
    """

    def __init__(self, classes=None, missing_label=MISSING_LABEL):
        self.missing_label = check_missing_label(missing_label)
        self._le = None
        if classes is not None:
            self.classes_ = np.array(check_classes(classes))
            self._le = LabelEncoder().fit(self.classes_)
            self._dtype = np.append(self.classes_, self.missing_label).dtype
            self.missing_label = check_missing_label(missing_label,
                                                     target_type=self._dtype,
                                                     name='classes')
        self._no_init_classes = classes is None

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
        y = check_array(y, ensure_2d=False, force_all_finite=False, dtype=None)
        if self._no_init_classes:
            y = np.asarray(y)
            is_lbld = is_labeled(y, missing_label=self.missing_label)
            self._dtype = np.append(y, self.missing_label).dtype
            self._le = LabelEncoder()
            self._le.fit(y[is_lbld])
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
