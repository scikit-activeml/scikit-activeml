import warnings

import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels, KERNEL_PARAMS

from skactiveml.base import SingleAnnotatorPoolQueryStrategy
from skactiveml.utils import (
    check_scalar,
    simple_batch,
    MISSING_LABEL,
    is_labeled,
    ExtLabelEncoder,
)


class Quire(SingleAnnotatorPoolQueryStrategy):
    """QUerying Informative and Representative Examples (QUIRE)

    This class implements "QUerying Informative and Representative
    Examples" (QUIRE) [1]_, which uses the min–max view of active learning with
    a regularized least-squares classifier in a reproducing kernel Hilbert
    space (RKHS) under quadratic loss. It assigns each unlabeled sample a
    single score derived from the kernel ridge regression objective that
    jointly reflects informativeness with respect to the labeled set and
    representativeness with respect to the unlabeled set.

    Parameters
    ----------
    classes : array-like of shape (n_classes,)
        Array of class labels.
    lmbda : float, default=1.0
        Controls informativeness (high) and representativeness (low). Values
        must be greater than 0.
    metric : str or callable, default='rbf'
        The metric must be a valid kernel defined by the function
        `sklearn.metrics.pairwise.pairwise_kernels` or 'precomputed'.
    metric_dict : dict, default=None
        Any further parameters are passed directly to the metric function.
    missing_label : scalar or string or np.nan or None, default=MISSING_LABEL
        Value to represent a missing label.
    random_state : int or np.random.RandomState, default=None
        The random state to use.
    target_type : "auto" or "single-output", default="auto"
        Declared target type. This strategy supports only single-output
        classification.

    References
    ----------
    .. [1] S.-J. Huang, R. Jin, and Z.-H. Zhou. Active Learning by Querying
       Informative and Representative Examples. In Adv. Neural Inf. Process.
       Syst., 2010.
    """

    METRICS = list(KERNEL_PARAMS.keys()) + ["precomputed"]

    def __init__(
        self,
        classes,
        lmbda=1.0,
        metric="rbf",
        metric_dict=None,
        missing_label=MISSING_LABEL,
        random_state=None,
        target_type="auto",
    ):
        super().__init__(
            missing_label=missing_label,
            random_state=random_state,
            target_type=target_type,
        )
        self.classes = classes
        self.lmbda = lmbda
        self.metric = metric
        self.metric_dict = metric_dict

    def query(
        self,
        X,
        y,
        candidates=None,
        batch_size=1,
        return_utilities=False,
    ):
        """Determines for which candidate samples labels are to be queried.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data set, usually complete, i.e., including the labeled
            and unlabeled samples.
        y : array-like of shape (n_samples,)
            Labels of the training data set (possibly including unlabeled ones
            indicated by `self.missing_label`).
        candidates : None or array-like of shape (n_candidates), dtype=int or \
                array-like of shape (n_candidates, n_features), default=None
            - If `candidates` is `None`, the unlabeled samples from
              `(X,y)` are considered as `candidates`.
            - If `candidates` is of shape `(n_candidates,)` and of type
              `int`, `candidates` is considered as the indices of the
              samples in `(X,y)`.
        batch_size : int, default=1
            The number of samples to be selected in one AL cycle.
        return_utilities : bool, default=False
            If `True`, also return the utilities based on the query strategy.

        Returns
        -------
        query_indices : numpy.ndarray of shape (batch_size)
            The query indices indicate for which candidate sample a label is
            to be queried, e.g., `query_indices[0]` indicates the first
            selected sample. The indexing refers to the samples in `X`.
        utilities : numpy.ndarray of shape (batch_size, n_samples) or \
                numpy.ndarray of shape (batch_size, n_candidates)
            The utilities of samples after each selected sample of the batch,
            e.g., `utilities[0]` indicates the utilities used for selecting
            the first sample (with index `query_indices[0]`) of the batch.
            Utilities for labeled samples will be set to np.nan. The indexing
            refers to the samples in `X`.
        """
        # --- Validation -----------------------------------------------------
        # Check standard parameters.
        (
            X,
            y,
            candidates,
            batch_size,
            return_utilities,
        ) = self._validate_data(
            X=X,
            y=y,
            candidates=candidates,
            batch_size=batch_size,
            return_utilities=return_utilities,
            reset=True,
        )

        # Obtain candidates plus mapping.
        X_cand, mapping = self._transform_candidates(
            candidates, X, y, enforce_mapping=True
        )
        mask_l = is_labeled(y=y, missing_label=self.missing_label)
        le = ExtLabelEncoder(self.classes, self.missing_label)
        y = le.fit_transform(y)
        classes_ = le.transform(self.classes)

        # Check whether metric is available.
        if self.metric not in Quire.METRICS and not callable(self.metric):
            raise ValueError(
                "The parameter 'metric' must be callable or "
                "in {}".format(KERNEL_PARAMS.keys())
            )

        # Ensure that metric_dict is a Python dictionary.
        self.metric_dict_ = (
            self.metric_dict if self.metric_dict is not None else {}
        )
        if not isinstance(self.metric_dict_, dict):
            raise TypeError("'metric_dict' must be a Python dictionary.")

        # Check lmbda.
        lmbda = self.lmbda
        check_scalar(
            lmbda,
            target_type=(float, int),
            name="lmbda",
            min_val=0,
            min_inclusive=False,
        )

        # --- Computation ----------------------------------------------------
        # Preserve sklearn's full-pool symmetrization of callable kernels.
        K = None
        if self.metric == "precomputed":
            K = X
            if K.shape != (len(y), len(y)):
                raise ValueError(
                    "The kernel matrix 'K' must have the shape "
                    "(n_samples, n_samples)."
                )
        elif callable(self.metric):
            K = pairwise_kernels(X, X, metric=self.metric, **self.metric_dict_)
        y_labeled_ovr = _one_versus_rest_transform(
            y[mask_l], classes_, l_rest=-1
        )

        if K is not None and not np.array_equal(K, K.T):
            # The reduced quadratic form assumes symmetry. Keep the previous
            # calculation and its warning for nonsymmetric precomputed inputs.
            utilities = _full_kernel_utilities(
                K, lmbda, mask_l, mapping, y_labeled_ovr
            )
        else:
            if K is None:
                X_l = X[mask_l]
                if len(X_l):
                    K_ll = pairwise_kernels(
                        X_l, metric=self.metric, **self.metric_dict_
                    )
                    K_lc = pairwise_kernels(
                        X_l, X_cand, metric=self.metric, **self.metric_dict_
                    )
                else:
                    K_ll = np.empty((0, 0))
                    K_lc = np.empty((0, len(X_cand)))
                # Bounded blocks support every built-in kernel diagonal and
                # retain its parameter validation without an M-by-M matrix.
                diagonal = np.empty(len(X_cand))
                for start in range(0, len(X_cand), 256):
                    stop = start + 256
                    block = X_cand[start:stop]
                    diagonal[start:stop] = np.diag(
                        pairwise_kernels(
                            block, metric=self.metric, **self.metric_dict_
                        )
                    )
            else:
                labeled = np.flatnonzero(mask_l)
                K_ll = K[np.ix_(labeled, labeled)]
                K_lc = K[np.ix_(labeled, mapping)]
                diagonal = K[mapping, mapping]

            # Eliminating all other unlabeled variables from Eq. (9) gives
            # z.T A^-1 z + (1 - k.T A^-1 z)^2 / (k_ss + lambda - k.T A^-1 k),
            # where A = K_ll + lambda I. Solve once for all classes/candidates.
            # A general solve also permits symmetric indefinite kernels.
            n_classes = len(classes_)
            solved = np.linalg.solve(
                K_ll + lmbda * np.eye(len(K_ll)),
                np.column_stack((y_labeled_ovr, K_lc)),
            )
            solved_y, solved_k = solved[:, :n_classes], solved[:, n_classes:]
            residual = 1 - K_lc.T @ solved_y
            denominator = diagonal + lmbda - np.sum(K_lc * solved_k, axis=0)
            scores = np.sum(y_labeled_ovr * solved_y, axis=0)
            scores = scores + residual**2 / denominator[:, None]
            utilities = np.full(len(X), np.nan)
            utilities[mapping] = -np.max(scores, axis=1)

        return simple_batch(
            utilities,
            self.random_state_,
            batch_size=batch_size,
            return_utilities=return_utilities,
        )


def _full_kernel_utilities(K, lmbda, mask_l, mapping, y_labeled_ovr):
    """Retain the original calculation for nonsymmetric kernel matrices."""
    mask_a = ~mask_l
    L = np.linalg.inv(K + lmbda * np.eye(len(K)))
    L_aa_inv = _L_aa_inv(K, lmbda, mask_a, mask_l)
    unlabeled = np.flatnonzero(mask_a)
    utilities = np.full(len(K), np.nan)
    for s in mapping:
        mask_u = mask_a.copy()
        mask_u[s] = False
        position = np.searchsorted(unlabeled, s)
        L_uu_inv = _del_i_inv(L_aa_inv, position, "L_aa_inv")
        utilities[s] = -L[s, s] - np.max(
            [
                yl.T.dot(L[mask_l][:, mask_l]).dot(yl)
                + 2 * L[s][mask_l].dot(yl)
                - (L[mask_u][:, mask_l].dot(yl) + L[mask_u][:, [s]])
                .T.dot(L_uu_inv)
                .dot(L[mask_u][:, mask_l].dot(yl) + L[mask_u][:, [s]])
                for yl in y_labeled_ovr.T[:, :, np.newaxis]
            ]
        )
    return utilities


def _one_versus_rest_transform(y, classes, l_one=1, l_rest=-1):
    missing_label = np.nan
    dtype = np.float64
    y_ovr = np.full((len(classes), len(y)), fill_value=l_rest, dtype=dtype)
    for i, c in enumerate(classes):
        y_ovr[i, (y == c)] = l_one
        y_ovr[i, (np.isnan(y))] = missing_label
    return y_ovr.T


def _del_i_inv(A_inv, s, name="A"):
    if not np.allclose(A_inv, A_inv.T):
        err = np.abs(A_inv - A_inv.T)
        warnings.warn(
            f"The approximation of the inverse of matrix `{name}` "
            f"may be inaccurate because the matrix is not symmetric "
            f"with an absolut error of \n{err}.\n To avoid this "
            f"warning you can increase `lmbda`."
        )

    a = A_inv[s, s]
    b = np.delete(A_inv[:, [s]], s, axis=0)
    D = np.delete(np.delete(A_inv, [s], axis=0), [s], axis=1)
    B_inv = D - (1 / a) * np.dot(b, b.T)
    return B_inv


def _L_aa_inv(K, lmbda, is_unlabeled, is_labeled):
    L_aa_inv = (
        lmbda * np.eye(sum(is_unlabeled)) + K[is_unlabeled][:, is_unlabeled]
    )
    L_aa_inv -= (
        K[is_unlabeled][:, is_labeled]
        .dot(
            np.linalg.inv(
                lmbda * np.eye(sum(is_labeled)) + K[is_labeled][:, is_labeled]
            )
        )
        .dot(K[is_labeled][:, is_unlabeled])
    )
    return L_aa_inv
