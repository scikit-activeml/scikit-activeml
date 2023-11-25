"""
Module implementing various Core-set Selection strategies.

Core-set selection problem aims to find a small subset given a large labeled
dataset such that a model learned over the small subset is competitve over the
whole dataset.
"""

import numpy as np

from ..base import SingleAnnotatorPoolQueryStrategy
from ..utils import MISSING_LABEL, labeled_indices, unlabeled_indices
from sklearn.metrics import pairwise_distances


class CoreSet(SingleAnnotatorPoolQueryStrategy):
    """ Core Set Selection

    This class implement various core-set based query strategies, i.e., the
    standard greedy algorithm for k-center problem [1], the robust k-center
    algorithm [1].

    Parameters
    ----------
    method: {'greedy', 'robust'}, default='greedy'
        The method to solve the k-center problem, k-center-greedy and robust
        k-center are possible
    missing_label: scalar or string or np.nan or None, default=np.nan
        Value to represent a missing label
    random_state: int or np.random.RandomState
        The random state to use

    References
    ----------
    [1] O. Sener und S. Savarese, „ACTIVE LEARNING FOR CONVOLUTIONAL NEURAL 
    NETWORKS: A CORE-SET APPROACH“, 2018.
    """

    def __init__(
            self, method='greedy', missing_label=MISSING_LABEL, random_state=None
    ):
        super().__init__(
            missing_label=missing_label, random_state=random_state
        )

        self.method = method

    def query(
            self,
            X,
            y,
            candidates=None,
            batch_size=1,
            return_utilities=False,
            **kwargs
    ):

        """ Query the next instances to be labeled

         Parameters
         ----------
         **kwargs
         X: array-like of shape (n_samples, n_features)
            Training data set, usually complete, i.e. including the labeled and
            unlabeled samples
         y: array-like of shape (n_samples, )
            Labels of the training data set (possibly including unlabeles ones
            indicated by self.missing_label)
         candidates: None or array-like of shape (n_candidates), dtype = int or
            array-like of shape (n_candidates, n_features),
            optional (default=None)
            If candidates is None, the unlabeled samples from (X,y) are considered
            as candidates
         batch_size: int, optional(default=1)
            The number of samples to be selectes in one AL cycle.
         return_utilities: bool, optional(default=False)
            If True, also return the utilites based on the query strategy

         Returns
         ----------
         query_indices: numpy.ndarry of shape (batch_size, )
            The query_indices indicate for which candidate sample a label is
            to queried, e.g., `query_indices[0]` indicates the first selected
            sample.
         utilities: numpy.ndarray of shape (n_samples, )
            The distance between each data point and its nearest center after
            each selected sample of the batch
         """

        X, y, candidates, batch_size, return_utilities = self._validate_data(
            X, y, candidates, batch_size, return_utilities, reset=True
        )

        X_cand, mapping = self._transform_candidates(candidates, X, y, enforce_mapping=True)
        """
        X_cand unlabeled samples
        mapping: indices of the original array
        """

        if self.method == 'greedy':
            query_indices, utilities = k_greedy_center(X, y, batch_size, self.random_state_, self.missing_label, mapping)


        if return_utilities:
            return query_indices, utilities
        else:
            return query_indices

def k_greedy_center(X, y, batch_size, random_state, missing_label=np.nan, mapping=None):
    """
     An active learning method that greedily forms a batch to minimize
     the maximum distance to a cluster center among all unlabeled
     datapoints.

     Parameters:
     ----------
     X: array-like of shape (n_samples, n_features)
        Training data set, usually complete, i.e. including the labeled and
        unlabeled samples
     selected_samples: np.ndarray of shape (n_selected_samples, )
        index of datapoints already selectes
     batch_size: int, optional(default=1)
        The number of samples to be selected in one AL cycle.
        
     Return:
     ----------
     new_samples: numpy.ndarry of shape (batch_size, )
         The query_indices indicate for which candidate sample a label is
         to queried from the candidates
     utilities: numpy.ndarray of shape (batch_size, n_samples)
         The distance between each data point and its nearest center that used
         for selecting the next sample.
        """
    # read the labeled aka selected samples from the y vector
    selected_samples = labeled_indices(y, missing_label=missing_label)
    if mapping is None:
        mapping = unlabeled_indices(y, missing_label=missing_label)
    # initialize the utilities matrix with
    utilities = np.empty(shape=(batch_size, X.shape[0]))

    query_indices = np.array([], dtype=int)

    for i in range(batch_size):
        utilities[i] = update_distances(X, selected_samples, mapping)

        # select index
        idx = np.nanargmax(utilities[i])

        if len(selected_samples) == 0:
            idx = random_state.choice(np.arange(X.shape[0]))
            # because np.nanargmax always return the first occurrence is returned

        query_indices = np.append(query_indices, [idx])
        selected_samples = np.append(selected_samples, [idx])

    return query_indices, utilities

def update_distances(X, cluster_centers, mapping=None):
    """
    Update min distances by given cluster centers.

    Parameters:
    ----------
    X: array-like of shape (n_samples, n_features)
        Training data set, usually complete, i.e. including the labeled and
        unlabeled samples
    cluster_centers: indices of cluster centers

    Return:
    ---------
    dist: numpy.ndarray of shape (1, n_samples)
        - if there aren't any cluster centers existed, the default distance
            will be 0
        - if there are some cluster center existed, the return will be the
            distance between each data point and its nearest center after
            each selected sample of the batch
        """
    dist = np.empty(shape=(1, X.shape[0]))

    if len(cluster_centers) > 0:
        cluster_center_feature = X[cluster_centers]
        dist_matrix = pairwise_distances(X, cluster_center_feature)
        dist = np.min(dist_matrix, axis=1).reshape(1, -1)

    result_dist = np.full((1, X.shape[0]), np.nan)
    result_dist[0, mapping] = dist[0, mapping]
    result_dist[0, cluster_centers] = np.nan

    return result_dist
