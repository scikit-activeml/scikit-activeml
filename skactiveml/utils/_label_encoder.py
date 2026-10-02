import numpy as np
from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from ._label import MISSING_LABEL, is_labeled, check_missing_label
from ._validation import check_classifier_params, _has_nested_classes


class ExtLabelEncoder(BaseEstimator):
    """Encode class labels with integers in `[0, ..., n_classes-1]` and use
    `-1` for unlabeled.

    Parameters
    ----------
    classes : array-like of shape (n_classes,) or a list of such array-likes, \
            default=None
        - If `classes` is not nested (`None` or one-dimensional), a single task
          problem is assumed such that `y` can be shape `(n_samples,)` or
          `(n_samples, n_annotators)`. Same encoder is applied to all entries.
        - If `classes` is nested, `target_type` must be `"multi-label"`, and
          `y` must contain one column per binary class vocabulary.
    missing_label : scalar or string or np.nan or None, default=np.nan
        Value to represent a missing label.
    target_type : "single-output" or "multi-label", default="single-output"
        Resolved target type controlling whether one shared encoder or one
        encoder per label is used.

    """

    def __init__(
        self,
        classes=None,
        missing_label=MISSING_LABEL,
        target_type="single-output",
    ):
        self.classes = classes
        self.missing_label = missing_label
        self.target_type = target_type

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
        if self.target_type not in {"single-output", "multi-label"}:
            raise ValueError(
                "`target_type` must be either 'single-output' or "
                "'multi-label'."
            )
        has_nested_classes = _has_nested_classes(self.classes)
        if has_nested_classes != (self.target_type == "multi-label"):
            raise ValueError(
                "Nested `classes` require `target_type='multi-label'`, and "
                "multi-label encoding requires nested `classes`."
            )
        if y.size > 0:
            # An empty `y` carries no dtype evidence: NumPy defaults it to
            # `float64`, which would reject a string
            # `missing_label`.
            check_missing_label(
                missing_label=self.missing_label, target_type=y.dtype
            )
        else:
            check_missing_label(missing_label=self.missing_label)
        check_classifier_params(
            classes=self.classes, missing_label=self.missing_label
        )
        if self.target_type == "multi-label":
            classes_outer = list(self.classes)
            if not all(len(classes_t) == 2 for classes_t in classes_outer):
                raise ValueError(
                    "Each multi-label class vocabulary must contain exactly "
                    "two classes."
                )
            n_labels = len(classes_outer)
            if y.ndim != 2 or y.shape[1] != n_labels:
                raise ValueError(
                    f"Expected y with shape `(n_samples, {n_labels})` "
                    f"for multi-label targets, got {y.shape}."
                )
            is_labeled(
                y,
                missing_label=self.missing_label,
                target_type="multi-label",
            )
            self.n_labels_ = n_labels
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
        y_enc = np.full_like(y, -1, dtype=int)

        if self.target_type == "multi-label":
            if y.ndim != 2 or y.shape[1] != self.n_labels_:
                raise ValueError(
                    f"Expected y with shape `(n_samples, {self.n_labels_})` "
                    f"for multi-label targets, got {y.shape}."
                )
            # A multi-label row is either fully observed or fully missing, so
            # one row mask covers every label output.
            is_lbld = is_labeled(
                y,
                missing_label=self.missing_label,
                target_type="multi-label",
            )
            if is_lbld.any():
                for t in range(self.n_labels_):
                    y_enc[is_lbld, t] = self._le[t].transform(y[is_lbld, t])
            return y_enc

        is_lbld = is_labeled(y, missing_label=self.missing_label)
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
            dtype=None,
        )
        y_dec = np.full_like(
            y, dtype=self._dtype, fill_value=self.missing_label
        )

        if self.target_type == "multi-label":
            if y.ndim != 2 or y.shape[1] != self.n_labels_:
                raise ValueError(
                    f"Expected y with shape `(n_samples, {self.n_labels_})` "
                    f"for multi-label targets, got {y.shape}."
                )
            # A multi-label row is either fully observed or fully missing, so
            # one row mask covers every label output.
            is_lbld = is_labeled(
                y,
                missing_label=-1,
                target_type="multi-label",
            )
            if is_lbld.any():
                for t in range(self.n_labels_):
                    y_dec[is_lbld, t] = self._le[t].inverse_transform(
                        y[is_lbld, t]
                    )
            return y_dec

        is_lbld = is_labeled(y, missing_label=-1)
        if is_lbld.any():
            y_dec[is_lbld] = self._le.inverse_transform(y[is_lbld].ravel())
        return y_dec
