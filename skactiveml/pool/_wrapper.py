"""
Wrapper for skactiveml PoolQueryStrategy to deal utility weights.
"""

# Authors: Pascal Mergard <Pascal.Mergard@student.uni-kassel.de>

from sklearn.utils import check_array
from ..base import PoolQueryStrategy
from ..utils import rand_argmax


class UtilityWrapper(PoolQueryStrategy):
    """UtilityWrapper

    Implementation of a wrapper class for skactiveml query strategies to deal
    utility weights.

    Parameters
    ----------
    query_strategy : skaktiveml.PoolQueryStrategy
        The query strategy to wrap.
    """
    def __init__(self, query_strategy):
        if not isinstance(query_strategy, PoolQueryStrategy):
            raise TypeError(
                f'query_strategy has to be of type PoolQueryStrategy, but it '
                f'is of type {type(query_strategy)}.')
        super().__init__(
            missing_label=query_strategy.missing_label,
            random_state=query_strategy.random_state
        )
        self.query_strategy = query_strategy

    def query(self, X, y, *args, candidates=None, batch_size=1,
              return_utilities=False, utility_weights=None, **kwargs):
        """

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data set, usually complete, i.e. including the labeled and
            unlabeled samples.
        y : array-like of shape (n_samples)
            Labels of the training data set (possibly including unlabeled ones
            indicated by self.MISSING_LABEL.
        candidates : None or array-like of shape (n_candidates), dtype=int or
            array-like of shape (n_candidates, n_features),
            optional (default=None)
            If candidates is None, the unlabeled samples from (X,y) are
            considered as candidates.
            If candidates is of shape (n_candidates) and of type int,
            candidates is considered as the indices of the samples in (X,y).
            If candidates is of shape (n_candidates, n_features), the
            candidates are directly given in candidates (not necessarily
            contained in X). This is not supported by all query strategies.
        batch_size : int, default=1
            The number of samples to be selected in one AL cycle.
        return_utilities : bool, default=False
            If true, also return the utilities based on the query strategy.
        utility_weights : array-like of shape (n_samples)
            Used to weight the utilities.

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
            The weighted utilities of samples after each selected sample of the
            batch, e.g., `utilities[0]` indicates the utilities used for
            selecting the first sample (with index `query_indices[0]`) of the
            batch. Utilities for labeled samples will be set to np.nan.
            If candidates is None or of shape (n_candidates), the indexing
            refers to samples in X.
            If candidates is of shape (n_candidates, n_features), the indexing
            refers to samples in candidates.

        """
        # Validate input parameters.
        self._validate_data(X, y, candidates, batch_size, return_utilities)
        utility_weights = check_array(utility_weights, ensure_2d=False)

        # Get the utilities.
        _, utilities = self.query_strategy.query(X, y, *args,
                                                 candidates=candidates,
                                                 batch_size=batch_size,
                                                 return_utilities=True,
                                                 **kwargs)
        if utility_weights is not None:
            utilities *= utility_weights

        if return_utilities:
            return rand_argmax(utilities, random_state=self.random_state_,
                               axis=1), utilities
        else:
            return rand_argmax(utilities, random_state=self.random_state_,
                               axis=1)

    def __getattr__(self, item):
        return getattr(self.query_strategy, item)
