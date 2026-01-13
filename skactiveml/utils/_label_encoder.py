import numpy as np
from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from ._label import MISSING_LABEL, is_labeled, check_missing_label
from ._validation import check_classifier_params, _is_multioutput_classes


class ExtLabelEncoder(BaseEstimator):
    """Encode class labels with integers in `[0, ..., n_classes-1]` and use
    `-1` for unlabeled.

    Parameters
    ----------
    classes : array-like of shape (n_classes,) or a list of such array-likes, \
            default=None
        TODO: Allow `classes=None` for multioutput.
        TODO: Add `interpret_y_columns_as_separate_tasks=False`.
        - If `classes` is not nested (`None` or one-dimensional), a single task
          problem is assumed such that `y` can be shape `(n_samples,)` or
          `(n_samples, n_annotators)`. Same encoder is applied to all entries.
        - If `classes` is nested (list of array-like objects), a multioutput
          (tasks) problem `y` must be shape `(n_samples, n_tasks)` with
          `n_tasks == len(classes)`. Each column is encoded with its
          task-specific encoder.
    missing_label : scalar or string or np.nan or None, default=np.nan
        Value to represent a missing label. In the case of a multioutput
        setting, we expect that the missing label is identical across all
        tasks.

    """

    def __init__(self, classes=None, missing_label=MISSING_LABEL):
        self.classes = classes
        self.missing_label = missing_label

    def fit(self, y):
        """Fit label encoder.

        Parameters
        ----------
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Class labels.

        Returns
        -------
        self : ExtLabelEncoder
            Returns an instance of `ExtLabelEncoder`.
        """
        y = check_array(
            y,
            ensure_2d=False,
            ensure_all_finite=False,
            ensure_min_samples=0,
            dtype=None,
        )
        check_missing_label(
            missing_label=self.missing_label, target_type=y.dtype
        )
        self.multioutput_ = _is_multioutput_classes(classes=self.classes)
        check_classifier_params(
            classes=self.classes, missing_label=self.missing_label
        )
        if self.multioutput_:
            classes_outer = list(self.classes)
            self.n_outputs_ = len(classes_outer)
            if y.ndim != 2 or y.shape[1] != self.n_outputs_:
                raise ValueError(
                    f"Expected y with shape `(n_samples, {self.n_outputs_})` "
                    f"in multioutput mode, got {y.shape}."
                )
            self._le = []
            self.classes_ = []
            self._dtype = []
            for t, cls_t in enumerate(classes_outer):
                cls_arr = np.asarray(list(cls_t))
                le = LabelEncoder()
                le.fit(cls_arr)
                self._le.append(le)
                self.classes_.append(le.classes_)
                self._dtype.append(le.classes_.dtype)
            self._dtype.append(np.asarray(self.missing_label).dtype)
            self._dtype = np.result_type(*self._dtype)
            return self

        self._le = LabelEncoder()
        if self.classes is None:
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
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Class labels.

        Returns
        -------
        y : np.ndarray shape (n_samples,) or (n_samples, n_outputs)
            Class labels.
        """
        return self.fit(y).transform(y)

    def transform(self, y):
        """Transform labels to new class encoding.

        Parameters
        ----------
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Original class labels.

        Returns
        -------
        y_enc : array-like of shape (n_samples) or (n_samples, n_outputs)
            Encoded class labels.
        """
        check_is_fitted(self, attributes=["classes_"])
        y = check_array(
            y,
            ensure_2d=False,
            ensure_all_finite=False,
            ensure_min_samples=0,
            dtype=None,
        )
        is_lbld = is_labeled(y, missing_label=self.missing_label)
        y_enc = np.full_like(y, -1, dtype=int)

        if self.multioutput_:
            if y.ndim != 2 or y.shape[1] != self.n_outputs_:
                raise ValueError(
                    f"Expected y with shape `(n_samples, {self.n_outputs_})` "
                    f"in multioutput mode, got {y.shape}."
                )
            for t in range(self.n_outputs_):
                y_t = y[:, t]
                is_lbld_t = is_lbld[:, t]
                if is_lbld_t.any():
                    y_enc[is_lbld_t, t] = self._le[t].transform(y_t[is_lbld_t])
            return y_enc

        if is_lbld.any():
            y_enc[is_lbld] = self._le.transform(y[is_lbld].ravel())
        return y_enc

    def inverse_transform(self, y):
        """Transform labels back to original encoding.

        Parameters
        ----------
        y : numpy array of shape (n_samples,) or (n_samples, n_outputs)
            Encoded class labels.

        Returns
        -------
        y_dec : np.ndarray of shape (n_samples,) or (n_samples, n_outputs)
            Decoded (original) class labels.
        """
        check_is_fitted(self, attributes=["classes_"])
        y = check_array(
            y,
            ensure_2d=False,
            ensure_all_finite=False,
            ensure_min_samples=0,
            dtype=int,
        )
        is_lbld = is_labeled(y, missing_label=-1)
        y_dec = np.full_like(
            y, dtype=self._dtype, fill_value=self.missing_label
        )

        if self.multioutput_:
            if y.ndim != 2 or y.shape[1] != self.n_outputs_:
                raise ValueError(
                    f"Expected y with shape `(n_samples, {self.n_outputs_})` "
                    f"in multioutput mode, got {y.shape}."
                )
            for t in range(self.n_outputs_):
                y_t = y[:, t]
                is_lbld_t = is_lbld[:, t]
                if is_lbld_t.any():
                    y_dec[is_lbld_t, t] = self._le[t].inverse_transform(
                        y_t[is_lbld_t]
                    )
            return y_dec

        if is_lbld.any():
            y_dec[is_lbld] = self._le.inverse_transform(y[is_lbld].ravel())
        return y_dec
