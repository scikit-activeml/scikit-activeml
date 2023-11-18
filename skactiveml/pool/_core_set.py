"""
Module implementing various Core-set Selection strategies.

Core-set selection problem aims to find a small subset given a large labeled
dataset such that a model learned over the small subset is competitve over the
whole dataset.
"""

import numpy as np

from ..base import SingleAnnotatorPoolQueryStrategy
from ..utils import MISSING_LABEL, labeled_indices
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
        ):
         
         """ Query the next instances to be labeled

         Parameters
         ----------
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
         
         X_cand, mapping = self._transform_candidates(candidates, X, y)
         selected_samples = labeled_indices(y, missing_label=self.missing_label)
         
         if self.method == 'greedy':
             query_indices, utilities = self.k_greedy_center(X, selected_samples, batch_size)

         if return_utilities:
             return query_indices, utilities
         else:
             return query_indices
    
    def k_greedy_center(self, X, selected_samples, batch_size):
        """
         An active learning method that greedily forms a batch to minimize 
         the maximum distance to a cluster center among all unlabeled
         datapoints.

         Parameters:
         ----------
         X: array-like of shape (n_samples, n_features)
            Training data set, usually complete, i.e. including the labeled and
            unlabeled samples
         selected_samples: np.ndarray of shape (n_seleted_samples, )
            index of datapoints already selectes
         batch_size: int, optional(default=1)
            The number of samples to be selectes in one AL cycle.
        
         Return:
         ----------
         new_samples: numpy.ndarry of shape (batch_size, )
            The query_indices indicate for which candidate sample a label is
            to queried from the candidates
         utilities: numpy.ndarray of shape (n_samples, )
            The distance between each data point and its nearest center after
            each selected sample of the batch
        """

        if len(selected_samples) > 0:
            min_distances = self.update_distances(X, selected_samples)

        query_indices = np.array([], dtype=int)

        for _ in range(batch_size):
            if len(selected_samples) == 0:
                idx = self.random_state_.choice(np.arange(X.shape[0]))
            else:
                idx = np.argmax(min_distances)
            assert idx not in selected_samples

            query_indices = np.append(query_indices, [idx])
            selected_samples = np.append(selected_samples, [idx])
            min_distances = self.update_distances(X, selected_samples)
                
        return query_indices, min_distances
    
    def update_distances(self, X, cluster_centers):
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
         dist: numpy.ndarray of shape (n_samples, )
            The distance between each data point and its nearest center after
            each selected sample of the batch
        """
        if len(cluster_centers) == 0:
            return np.full(shape=len(X), fill_value=np.nan)

        cluster_center_feature = X[cluster_centers]
        dist_matrix = pairwise_distances(X, cluster_center_feature)
        dist = np.min(dist_matrix, axis=1)

        return dist
    
    