import numpy as np
from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from ._label import (
    MISSING_LABEL,
    is_labeled,
    check_missing_label,
)
from ._label_dtype import (
    _as_class_vocabulary_array,
    _as_label_array,
    _check_compatible_kinds,
    _lossless_decode_dtype,
)
from ._validation import (
    check_classifier_params,
    check_type,
    _has_nested_classes,
)


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
            _as_label_array(y),
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
            for t, cls_t in enumerate(classes_outer):
                cls_arr = _as_class_vocabulary_array(
                    cls_t, name=f"classes[{t}]"
                )
                le = LabelEncoder()
                le.fit(cls_arr)
                self._le.append(le)
                self.classes_.append(le.classes_)
            self._dtype = _lossless_decode_dtype(
                self.classes_, self.missing_label
            )
            return self

        # `y` is the evidence the label contract is checked against, whether
        # or not the class vocabulary is inferred from it. An empty `y`
        # carries none, because NumPy defaults it to `float64`, which would
        # reject a string `missing_label`; the label helpers accept it.
        is_lbld = is_labeled(y, missing_label=self.missing_label)
        if self.classes is None and not is_lbld.any():
            raise ValueError(
                "No class label is observed and `classes` is not defined."
            )
        if self.classes is not None:
            # The wrapped `LabelEncoder` would report a string label beside
            # numeric classes as an unparsable integer rather than as the
            # incompatible label kind it is.
            _check_compatible_kinds(y[is_lbld], self.classes, name="classes")
        self._le = LabelEncoder()
        classes = (
            _as_class_vocabulary_array(self.classes)
            if self.classes is not None
            else y[is_lbld]
        )
        self._le.fit(classes)
        self.classes_ = self._le.classes_
        self._dtype = _lossless_decode_dtype(
            [self.classes_], self.missing_label
        )

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
            _as_label_array(y),
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

    def inverse_transform(self, y, *, prefer_class_dtype=False):
        """Transform labels back to original encoding.

        Parameters
        ----------
        y : numpy array of shape (n_samples,) or (n_samples, n_outputs)
            Encoded class labels.
        prefer_class_dtype : bool, default=False
            If `True`, decode fully observed targets directly into the class
            dtype (a lossless common vocabulary dtype for multi-label targets,
            using `object` if no common numeric dtype preserves the values).
            Targets containing missing entries still use a lossless dtype
            accommodating the missing label. If `False`, always use that
            missing-capable dtype, preserving the default behavior.

        Returns
        -------
        y_dec : np.ndarray of shape (n_samples,) or (n_samples, n_outputs)
            Decoded (original) class labels. Unless `prefer_class_dtype=True`
            and no entries are missing, the dtype accommodates the class
            vocabulary and missing label without losing values. It is
            `object` if their ordinary common dtype would lose information.
        """
        check_type(prefer_class_dtype, "prefer_class_dtype", bool, np.bool_)
        return self._inverse_transform(
            y, allow_missing=True, prefer_class_dtype=prefer_class_dtype
        )

    def _inverse_transform(
        self, y, *, allow_missing, prefer_class_dtype=False
    ):
        """Decode into a lossless missing-capable or fully observed dtype.

        With `allow_missing=False`, reject missing codes and allocate directly
        in the common class dtype, preserving integer prediction identities.
        Otherwise `prefer_class_dtype` chooses the class dtype only when all
        entries are observed. All paths share shape and code validation.

        Parameters
        ----------
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Encoded class labels, with `-1` indicating missing entries.
        allow_missing : bool
            Whether missing codes are permitted. If `False`, require fully
            observed targets and use the lossless common class dtype.
        prefer_class_dtype : bool, default=False
            Prefer the class-only dtype for fully observed targets when
            missing codes are permitted.

        Returns
        -------
        y_dec : numpy.ndarray of shape (n_samples,) or (n_samples, n_outputs)
            Original labels in a lossless dtype.
        """
        check_is_fitted(self, attributes=["classes_"])
        y = check_array(
            y,
            ensure_2d=False,
            ensure_all_finite=False,
            ensure_min_samples=0,
            dtype=None,
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
        else:
            is_lbld = is_labeled(y, missing_label=-1)

        has_missing = not np.all(is_lbld)
        if not allow_missing and has_missing:
            raise ValueError(
                "`y` contains the encoded missing-label value -1; this "
                "decoding path requires fully observed labels."
            )

        if allow_missing and (not prefer_class_dtype or has_missing):
            y_dec = np.full_like(
                y, dtype=self._dtype, fill_value=self.missing_label
            )
        else:
            classes = (
                self.classes_
                if self.target_type == "multi-label"
                else [self.classes_]
            )
            dtype = _lossless_decode_dtype(classes)
            y_dec = np.empty_like(y, dtype=dtype)

        if self.target_type == "multi-label":
            if is_lbld.any():
                for t in range(self.n_labels_):
                    y_dec[is_lbld, t] = self._le[t].inverse_transform(
                        y[is_lbld, t]
                    )
            return y_dec

        if is_lbld.any():
            y_dec[is_lbld] = self._le.inverse_transform(y[is_lbld].ravel())
        return y_dec
