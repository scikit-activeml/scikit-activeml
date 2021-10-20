import numpy as np

from ._label import ExtLabelEncoder, is_labeled
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
    v : array-like, shape (n_samples, n_classes)
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

    if n_classes == 0:
        raise ValueError(
            "Number of classes can not be inferred. "
            "There must be at least one assigned label or classes must not be"
            "None. "
        )

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


def majority_vote(y, w=None, classes=None, missing_label=np.nan):
    """ Assigns a label to each sample based on weighted voting.
    Samples with no labels are assigned with `missing_label`.

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
    y_aggregated : array-like, shape (n_samples)
        Assigned labels for each sample.

    """
    # check input parameters
    y = check_array(y, ensure_2d=False, dtype=None, force_all_finite=False)
    y = y if y.ndim == 2 else y.reshape((-1, 1))
    n_samples = y.shape[0]

    # extract labeled samples
    is_labeled_y = np.any(is_labeled(y, missing_label), axis=1)
    y_labeled = y[is_labeled_y]

    # infer encoding
    le = ExtLabelEncoder(classes=classes, missing_label=missing_label)
    le.fit(y_labeled)
    y_aggregated = np.full((n_samples,), missing_label, dtype=le._dtype)

    if np.any(is_labeled_y):
        # transform labels
        y_labeled_transformed = le.transform(y_labeled)
        max_value_y_l_t = np.nanmax(y_labeled_transformed)

        # perform voting
        vote_matrix = compute_vote_vectors(y_labeled_transformed, w,
                                           classes=np.arange(max_value_y_l_t+1))
        vote_vector = vote_matrix.argmax(axis=1)

        # inverse transform labels
        y_labeled_inverse_transformed = le.inverse_transform(vote_vector)
        # assign labels
        y_aggregated[is_labeled_y] = y_labeled_inverse_transformed

    return y_aggregated


