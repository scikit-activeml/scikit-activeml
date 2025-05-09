import warnings

import numpy as np

from copy import copy
from sklearn import clone
from sklearn.cluster import KMeans
from sklearn.metrics import (
    pairwise_distances_argmin_min,
    pairwise_distances_argmin,
    pairwise_distances,
)
from sklearn.tree import DecisionTreeRegressor

from skactiveml.base import SingleAnnotatorPoolQueryStrategy
from skactiveml.regressor import SklearnRegressor
from skactiveml.utils import (
    MISSING_LABEL,
    check_type,
    check_equal_missing_label,
    is_labeled,
    simple_batch,
    rand_argmax,
    check_scalar,
    labeled_indices,
)


class RegressionTreeBasedAL(SingleAnnotatorPoolQueryStrategy):
    """Regression Tree-based Active Learning (RT-AL)

    This class implements the query strategy Regression Tree-based Active
    Learning (RT-AL) [1]_, which is based on a regression tree and selects the
    number `n_k` of samples to be selected from each leaf `k` given a certain
    `batch size`. It than uses one of the three methods 'random', 'diversity',
    or 'representativity' to select `n_k` samples from each leaf `k`.

    Parameters
    ----------
    method : str, default='random'
        Possible values are 'random', 'diversity', and 'representativity'.
    missing_label : scalar or string or np.nan or None,
      default=skactiveml.utils.MISSING_LABEL
        Value to represent a missing label.
    random_state : int or np.random.RandomState, default=None
        The random state to use.
    max_iter_representativity : int, default=5
        Maximum number of optimisation iterations.
        Only used if `method='representativity'`.

    References
    ----------
    .. [1] A. Jose, J. P. A. de Mendonça, E. Devijver, N. Jakse, V. Monbet,
       and R. Poloni. Regression Tree-based Active Learning. Data Min. Knowl.
       Discov., pages 420–460, 2023.
    """

    def __init__(
        self,
        method="random",
        missing_label=MISSING_LABEL,
        random_state=None,
        max_iter_representativity=5,
    ):
        super().__init__(
            random_state=random_state, missing_label=missing_label
        )
        self.method = method
        self.max_iter_representativity = max_iter_representativity

    def query(
        self,
        X,
        y,
        reg,
        fit_reg=True,
        sample_weight=None,
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
            indicated by self.MISSING_LABEL).
        reg : SkactivemlRegressor
            The regressor must be `sklearn.tree.DecisionTreeRegressor` to
            predict the data. Ensure that the number of samples in the leaf is
            greater than 1. For example, by setting `min_samples_leaf >= 2` or
            by restricting the tree's depth.
        fit_reg : bool, default=True
            Defines whether the regressor should be fitted on `X`, `y`, and
            `sample_weight`.
        sample_weight : array-like of shape (n_samples), default=None
            Weights of training samples in `X`.
        candidates : None or array-like of shape (n_candidates), dtype=int or \
                array-like of shape (n_candidates, n_features), default=None
            - If `candidates` is `None`, the unlabeled samples from
              `(X,y)` are considered as `candidates`.
            - If `candidates` is of shape `(n_candidates,)` and of type
              `int`, `candidates` is considered as the indices of the
              samples in `(X,y)`.
            - If `candidates` is of shape `(n_candidates, *)`, the
              candidate samples are directly given in `candidates` (not
              necessarily contained in `X`). This is not supported by all
              query strategies.
        batch_size : int, default=1
            The number of samples to be selected in one AL cycle. Originally,
            this query strategy is developed for `batch_sizes > 1`.
        return_utilities : bool, default=False
            If true, also return the utilities based on the query strategy.

        Returns
        -------
        query_indices : numpy.ndarray of shape (batch_size,)
            The query indices indicate for which candidate sample a label is
            to be queried, e.g., `query_indices[0]` indicates the first
            selected sample.

            - If `candidates` is `None` or of shape
              `(n_candidates,)`, the indexing refers to the samples in
              `X`.
            - If `candidates` is of shape `(n_candidates, n_features)`,
              the indexing refers to the samples in `candidates`.
        utilities : numpy.ndarray of shape (batch_size, n_samples) or \
                numpy.ndarray of shape (batch_size, n_candidates)
            The utilities of samples after each selected sample of the batch,
            e.g., `utilities[0]` indicates the utilities used for selecting
            the first sample (with index `query_indices[0]`) of the batch.
            Utilities for labeled samples will be set to np.nan.

            - If `candidates` is `None` or of shape
              `(n_candidates,)`, the indexing refers to the samples in
              `X`.
            - If `candidates` is of shape `(n_candidates, n_features)`,
              the indexing refers to the samples in `candidates`.
        """
        # Validate input parameters.
        X, y, candidates, batch_size, return_utilities = self._validate_data(
            X, y, candidates, batch_size, return_utilities, reset=True
        )
        if batch_size == 1:
            warnings.warn(
                "This query strategy was originally developed for "
                "`batch_sizes > 1`."
            )
        X_cand, mapping = self._transform_candidates(candidates, X, y)
        labeled_idxs = labeled_indices(y, self.missing_label_)

        # Validate regressor type.
        check_type(reg, "reg", SklearnRegressor)
        check_type(reg.estimator, "reg.estimator", DecisionTreeRegressor)
        check_equal_missing_label(reg.missing_label, self.missing_label_)

        # Validate boolean flag
        check_type(fit_reg, "fit_reg", bool)

        # Validate method type.
        check_type(self.method, "method", str)

        # Validate max_iter_representativity type.
        check_scalar(
            self.max_iter_representativity,
            "max_iter_representativity",
            int,
            min_val=1,
        )

        # Fallback to random sampling if no sample is labeled.
        if len(labeled_idxs) <= 1:
            warnings.warn("No sample is labeled. Fallback to random sampling.")
            if mapping is None:
                utilities = np.ones(len(X_cand))
            else:
                utilities = np.full(len(X), np.nan)
                utilities[mapping] = np.ones(len(mapping))

            return simple_batch(
                utilities,
                self.random_state_,
                batch_size=batch_size,
                return_utilities=return_utilities,
                method="proportional",
            )

        # Fit the regressor.
        if fit_reg:
            if sample_weight is None:
                reg = clone(reg).fit(X, y)
            else:
                reg = clone(reg).fit(X, y, sample_weight)

        # Calculate the number of samples to be selected from each leaf k.
        n_k = _calc_acquisitions_per_leaf(
            X, y, reg, missing_label=self.missing_label_, batch_size=batch_size
        )

        # Discretize number of leaf acquisitions.
        n_k_discrete = _discretize_acquisitions_per_leaf(
            n_k, batch_size, self.random_state_
        )

        # Calculate the number of candidates per leaf.
        leaf_indices_cand = reg.apply(X_cand)

        if self.method == "random":
            batch_utilities_cand = np.full((batch_size, len(X_cand)), -np.inf)
            query_indices = []

            for leaf_idx in np.unique(leaf_indices_cand):
                for _ in range(n_k_discrete[leaf_idx]):
                    batch_utilities_cand[
                        len(query_indices), leaf_indices_cand == leaf_idx
                    ] = 1
                    batch_utilities_cand[len(query_indices), query_indices] = (
                        np.nan
                    )
                    query_indices.append(
                        rand_argmax(
                            batch_utilities_cand[len(query_indices)],
                            random_state=self.random_state_,
                        )
                    )

        elif self.method == "diversity":
            batch_utilities_cand = np.full((batch_size, len(X_cand)), -np.inf)
            query_indices = []

            X_labeled = X[labeled_idxs]
            leaf_indices_labeled = reg.apply(X_labeled)
            for leaf_idx in np.unique(leaf_indices_cand):
                X_cand_leaf = X_cand[leaf_indices_cand == leaf_idx]
                X_labeled_leaf = X_labeled[leaf_indices_labeled == leaf_idx]

                # Calculate the L2 distance of each unlabeled sample in leaf k
                # to all the labeled samples using equation (6) and (7).
                # Compute the shortest distance from x_j to all labeled
                # samples using equation (7).
                _, d_min = pairwise_distances_argmin_min(
                    X_cand_leaf, X_labeled_leaf, axis=1
                )
                for _ in range(n_k_discrete[leaf_idx]):
                    batch_utilities_cand[
                        len(query_indices), leaf_indices_cand == leaf_idx
                    ] = d_min
                    batch_utilities_cand[len(query_indices), query_indices] = (
                        np.nan
                    )
                    query_indices.append(
                        rand_argmax(
                            batch_utilities_cand[len(query_indices)],
                            random_state=self.random_state_,
                        )
                    )

        elif self.method == "representativity":
            # Perform a k-means clustering in leaf k with n_k clusters.
            query_indices = np.empty(shape=batch_size, dtype=int)
            l_cand = np.full(len(X_cand), fill_value=-1, dtype=int)
            for leaf in np.argwhere(n_k_discrete != 0).flatten():
                X_cand_leaf = X_cand[leaf_indices_cand == leaf]
                kmeans = KMeans(
                    n_k_discrete[leaf], random_state=self.random_state_
                ).fit(X_cand_leaf)

                l_cand[leaf_indices_cand == leaf] = kmeans.predict(
                    X_cand_leaf
                ) + np.sum(n_k_discrete[0:leaf])

                centroids = kmeans.cluster_centers_
                query_indices[
                    np.sum(n_k_discrete[0:leaf]) + range(n_k_discrete[leaf])
                ] = pairwise_distances_argmin(centroids, X_cand, axis=1)

            # Calculate R using Eq. (9)
            R_cand = np.zeros(len(X_cand))
            for l_idx in np.unique(l_cand):
                C_l = X_cand[l_cand == l_idx]
                if l_idx != -1 and len(C_l) > 1:
                    R_cand[l_cand == l_idx] = pairwise_distances(C_l, C_l).sum(
                        axis=1
                    ) / (len(C_l) - 1)

            batch_utilities_cand = np.full((batch_size, len(X_cand)), -np.inf)
            for i in range(self.max_iter_representativity):
                prev_best_indices = copy(query_indices)
                for l_idx in range(batch_size):
                    # Update DELTA using the current centroids.
                    X_M = X[labeled_idxs]
                    X_M = np.append(X_M, X_cand[query_indices[:l_idx]], axis=0)
                    X_M = np.append(
                        X_M, X_cand[query_indices[l_idx + 1 :]], axis=0
                    )
                    X_cand_l = X_cand[l_cand == l_idx]
                    _, delta_l = pairwise_distances_argmin_min(
                        X_cand_l, X_M, axis=1
                    )  # Equation (10)

                    # Use Eq. (8) to find the next sample to be labeled.
                    R_l = R_cand[l_cand == l_idx]
                    batch_utilities_cand[l_idx, l_cand == l_idx] = (
                        delta_l - R_l
                    )
                    query_indices[l_idx] = rand_argmax(
                        batch_utilities_cand[l_idx],
                        random_state=self.random_state_,
                    )[0]

                if np.all(prev_best_indices == query_indices):
                    break
            for l_idx in range(batch_size):
                batch_utilities_cand[l_idx, query_indices[:l_idx]] = np.nan
        else:
            raise ValueError(
                f'The given method "{self.method}" is not valid. Supported '
                f'methods are "random", "diversity", and "representativity".'
            )

        if mapping is None:
            batch_utilities = batch_utilities_cand
        else:
            batch_utilities = np.full((batch_size, len(X)), np.nan)
            batch_utilities[:, mapping] = batch_utilities_cand
            query_indices = mapping[query_indices]

        # Check whether utilities are to be returned.
        if return_utilities:
            return query_indices, batch_utilities
        else:
            return query_indices


def _discretize_acquisitions_per_leaf(n_k, batch_size, random_state):
    """Discretizes a given array of non-negative floats corresponding to the
    number of acquisitions per leaf of the regression tree. Guarantees that we
    acquire a minimum number (i.e., floored floats) per leaf.

    Parameters
    ----------
    n_k : numpy.ndarray of shape (n_leafs,)
        Float acquisitions per leaf of the regression tree.
    batch_size : int
        Number of acquisitions.
    random_state : np.random.RandomState
        Random state for reproducibility.

    Returns
    -------
    n_k_discrete : numpy.ndarray of shape (n_leafs,)
        Integer acquisitions per leaf of the regression tree.
    """
    n_k_rest, n_k_discrete = np.modf(n_k)
    rest_size = n_k_rest.sum()
    leaf_indices = np.arange(len(n_k))
    if rest_size > 0:
        sampled_leaf_indices = random_state.choice(
            leaf_indices,
            p=n_k_rest / rest_size,
            size=batch_size - np.sum(np.round(n_k_discrete)).astype(int),
            replace=False,
        )
        add_leaf_indices, add_leaf_counts = np.unique(
            sampled_leaf_indices, return_counts=True
        )
        n_k_discrete[add_leaf_indices] += add_leaf_counts
    return n_k_discrete.astype(int)


def _calc_acquisitions_per_leaf(X, y, reg, missing_label, batch_size=1):
    """Computes the number of samples to be selected from each leaf of the
    regression tree.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training data set, usually complete, i.e. including the labeled and
        unlabeled samples.
    y : array-like of shape (n_samples)
        Labels of the training data set (possibly including unlabeled ones
        indicated by `self.missing_label`).
    missing_label : scalar or string or np.nan or None
        Value to represent a missing label.
    reg: SkactivemlRegressor
        Fitted regressor to predict the data.
    batch_size : int, default=1
        The number of samples to be selected in one AL cycle.

    Returns
    -------
    n_samples_per_leaf : numpy.ndarray of shape (n_leafs)
        Number of samples per leaf.
    """
    is_lbld = is_labeled(y, missing_label=missing_label)

    # Compute the variance v_k on labeled samples in leaf k.
    leaf_labeled = reg.apply(X[is_lbld])
    y_labeled = y[is_lbld]
    v_k = np.zeros(reg.tree_.node_count)
    for leaf in range(len(v_k)):
        y_labeled_leaf = y_labeled[leaf_labeled == leaf]
        if len(y_labeled_leaf) > 1:
            v_k[leaf] = np.var(y_labeled_leaf, ddof=1)

    v_k[np.isnan(v_k)] = 0
    if 0 in v_k[np.unique(leaf_labeled)]:
        warnings.warn(
            "There are leaves with less than two labeled samples, "
            "which causes a variance of zero. To avoid this, set "
            "parameter `min_samples_leaf` of `reg` to >= 2."
        )

    # Compute the probability p_k that an unlabeled sample belongs to leaf k.
    leaf_unlabeled = reg.apply(X[~is_lbld])
    samples_per_leaf = np.bincount(leaf_unlabeled, minlength=len(v_k))
    p_k = samples_per_leaf / sum(~is_lbld)

    # Compute the number of sample to be selected from each leaf of the
    # regression tree.
    n_k = np.sqrt(p_k * v_k)
    if np.sum(n_k) == 0:
        n_k = np.full_like(n_k, fill_value=batch_size / reg.tree_.node_count)
    else:
        n_k = batch_size * n_k / np.sum(n_k)

    return n_k
