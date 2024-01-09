import warnings
from collections import Counter

import numpy as np
from sklearn import clone
from sklearn.tree import DecisionTreeRegressor

from skactiveml.base import SingleAnnotatorPoolQueryStrategy
from skactiveml.regressor import SklearnRegressor
from skactiveml.utils import MISSING_LABEL, check_type, \
    check_equal_missing_label, is_unlabeled, is_labeled, simple_batch


class RegressionTreeBasedAL(SingleAnnotatorPoolQueryStrategy):
    """Regression Tree-based Active Learning

    This strategy is based on a regression tree and selects the number `n_k` of
    samples to be selected from each leaf `k` given a certain `batch size`. It
    than uses one of the three methods 'random', 'diversity', ore
    'representativity' to select `n_k` samples from each leaf `k`.

    Parameters
    ----------
    method : str, default="random"
        Possible values are 'random', 'diversity', and 'representativity'.
    missing_label : scalar or string or np.nan or None,
      default=skactiveml.utils.MISSING_LABEL
        Value to represent a missing label.
    random_state : int | np.random.RandomState, optional
        Random state for candidate selection.

    References
    ----------

    """

    def __init__(
            self,
            method='random',
            missing_label=MISSING_LABEL,
            random_state=None,
    ):
        super().__init__(
            random_state=random_state, missing_label=missing_label
        )
        self.method = method

    def query(
            self,
            X,
            y,
            reg,
            fit_reg=True,
            sample_weight=None,
            candidates=None,
            batch_size=1,
            return_utilities=False
    ):
        """Determines for which candidate samples labels are to be queried.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data set, usually complete, i.e. including the labeled and
            unlabeled samples.
        y : array-like of shape (n_samples)
            Labels of the training data set (possibly including unlabeled ones
            indicated by self.MISSING_LABEL).
        reg: SkactivemlRegressor
            Regressor to predict the data.
        fit_reg : bool, optional (default=True)
            Defines whether the regressor should be fitted on `X`, `y`, and
            `sample_weight`.
        sample_weight: array-like of shape (n_samples), optional (default=None)
            Weights of training samples in `X`.
        candidates : None or array-like of shape (n_candidates), dtype=int or
            array-like of shape (n_candidates, n_features),
            optional (default=None)
            If candidates is None, the unlabeled samples from (X,y) are
            considered as candidates.
            If candidates is of shape (n_candidates) and of type int,
            candidates is considered as the indices of the samples in (X,y).
            If candidates is of shape (n_candidates, n_features), the
            candidates are directly given in candidates (not necessarily
            contained in X).
        batch_size : int, optional (default=1)
            The number of samples to be selected in one AL cycle.
        return_utilities : bool, optional (default=False)
            If true, also return the utilities based on the query strategy.

        Returns
        -------
        query_indices : numpy.ndarray of shape (batch_size)
            The query_indices indicate for which candidate sample a label is
            to queried, e.g., `query_indices[0]` indicates the first selected
            sample.
            If candidates is None or of shape (n_candidates), the indexing
            refers to samples in X.
            If candidates is of shape (n_candidates, n_features), the indexing
            refers to samples in candidates.
        utilities : numpy.ndarray of shape (batch_size, n_samples) or
            numpy.ndarray of shape (batch_size, n_candidates)
            The utilities of samples after each selected sample of the batch,
            e.g., `utilities[0]` indicates the utilities used for selecting
            the first sample (with index `query_indices[0]`) of the batch.
            Utilities for labeled samples will be set to np.nan.
            If candidates is None or of shape (n_candidates), the indexing
            refers to samples in X.
            If candidates is of shape (n_candidates, n_features), the indexing
            refers to samples in candidates.
        """

        # Validate input parameters.
        X, y, candidates, batch_size, return_utilities = self._validate_data(
            X, y, candidates, batch_size, return_utilities, reset=True
        )
        X_cand, mapping = self._transform_candidates(candidates, X, y)

        # Validate regressor type.
        check_type(reg, "reg", SklearnRegressor)
        check_type(reg.estimator, "reg.estimator", DecisionTreeRegressor)
        check_equal_missing_label(reg.missing_label, self.missing_label)

        check_type(fit_reg, "fit_reg", bool)

        # Validate method type.
        check_type(self.method, "method", str)

        # Fallback to random sampling if no sample is labeled.
        if np.sum(is_labeled(y)) == 0:
            warnings.warn("No sample is labeled. Fallback to random sampling.")
            if mapping is None:
                utilities = np.full(len(X_cand), fill_value=1/len(X_cand))
            else:
                utilities = np.full(len(X), np.nan)
                utilities[mapping] = np.full(len(mapping),
                                             fill_value=1/len(mapping))

            return simple_batch(
                utilities,
                self.random_state_,
                batch_size=batch_size,
                return_utilities=return_utilities,
                method='proportional',
            )

        # Fit the regressor.
        if fit_reg:
            reg = clone(reg).fit(X, y, sample_weight)

        # Calculate the number of samples to be selected from each leaf k.
        n_k = _rt_al(X, y, reg, batch_size)

        # Calculate the number of candidates per leaf.
        leaf_indices_cand = reg.apply(X_cand)
        n_cand_per_leaf = np.bincount(leaf_indices_cand, minlength=len(n_k))

        # n_cand_per_leaf = np.zeros(reg.tree_.node_count)
        # for i in range(reg.tree_.node_count):
        #     n_cand_per_leaf[i] = Counter(np.array(leaf_indices_cand))[i]
        # point_proportions = \
        #     n_k/np.max(
        #         [np.ones(reg.tree_.node_count), n_cand_per_leaf],
        #         axis=0
        #     )

        # leaf_probas = np.array(
        #     [point_proportions[key] for key in leaf_indices_cand]
        # )
        # leaf_probas = leaf_probas/sum(leaf_probas)
        #
        # leaves_to_sample = self.random_state_.choice(
        #     leaf_indices_cand,
        #     batch_size,
        #     p=leaf_probas,
        #     replace=False,
        # )

        if self.method == 'random':
            utilities_cand = (n_k/n_cand_per_leaf)[leaf_indices_cand]
            selection_method = 'proportional'
            # query_indices_cand = np.empty(len(leaves_to_sample), dtype=int)
            # i = 0
            # for leaf, n in Counter(leaves_to_sample).items():
            #     query_indices_cand[i:i+n] = self.random_state_.choice(
            #         np.argwhere(leaf_indices_cand == leaf).flatten(),
            #         size=n,
            #     )
            #     i += n

        elif self.method == 'diversity':
            raise NotImplementedError
        elif self.method == 'representativity':
            raise NotImplementedError
        else:
            raise ValueError(
                f'The given method "{self.method}" is not valid. Supported '
                f'methods are "random", "diversity", and "representativity".'
            )

        if mapping is None:
            utilities = utilities_cand
        else:
            utilities = np.full(len(X), np.nan)
            utilities[mapping] = utilities_cand

        return simple_batch(
            utilities,
            random_state=self.random_state_,
            batch_size=batch_size,
            return_utilities=return_utilities,
            method=selection_method
        )


def _rt_al(X, y, reg, batch_size=1):
    """Regression Tree-based Active Learning

    Computes the number of sample to be selected from each leaf of the
    regression tree.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training data set, usually complete, i.e. including the labeled and
        unlabeled samples.
    y : array-like of shape (n_samples)
        Labels of the training data set (possibly including unlabeled ones
        indicated by `self.missing_label`).
    reg: SkactivemlRegressor
        Fitted regressor to predict the data.
    batch_size : int, default=1
        The number of samples to be selected in one AL cycle.

    Returns
    -------
    n_samples_per_leaf : numpy.ndarray of shape (n_leafs)
    """

    # Compute the variance v_k on labeled samples in leaf k.
    leaf_labeled = reg.apply(X[~is_unlabeled(y)])
    y_labeled = y[~is_unlabeled(y)]
    v_k = np.empty(reg.tree_.node_count)
    for leaf in range(len(v_k)):
        v_k[leaf] = np.var(
            y_labeled[np.argwhere(leaf_labeled == leaf).flatten()],
            ddof=1
        )
    v_k[np.isnan(v_k)] = 0
    if 0 in v_k[np.unique(leaf_labeled)]:
        warnings.warn('There are leaves with less than two labeled samples, '
                      'which causes a variance of zero. To avoid this, set '
                      'parameter `min_samples_leaf` of `reg` to >= 2')

    # Compute the probability p_k that an unlabeled sample belongs to leaf k.
    samples_per_leaf = np.zeros(reg.tree_.node_count)
    samples_per_leaf[np.unique(leaf_labeled)] = np.array(list(Counter(leaf_labeled).values()))
    p_k = np.full(shape=reg.tree_.node_count, fill_value=1/sum(is_unlabeled(y)))
    p_k *= samples_per_leaf

    # Compute the number of sample to be selected from each leaf of the
    # regression tree.
    n_k = np.sqrt(p_k*v_k)
    if np.sum(n_k) == 0:
        n_k = np.full_like(n_k, fill_value=batch_size/reg.tree_.node_count)
    else:
        n_k = batch_size*n_k/np.sum(n_k)

    leaf_indices_cand = reg.apply(X[is_unlabeled(y)])
    n_cand_per_leaf = np.bincount(leaf_indices_cand, minlength=len(n_k))
    utilities_cand = (n_k / n_cand_per_leaf)[leaf_indices_cand]
    # if np.all(utilities_cand == 0):
    #     raise ValueError

    return n_k
