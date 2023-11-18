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
        if len(selected_samples) > 0:
            min_distances = self.update_distances(X, selected_samples)

        new_samples = []

        for _ in range(batch_size):
            if len(selected_samples) == 0:
                idx = self.random_state_.choice(np.arange(X.shape[0]))
            else:
                idx = np.argmax(min_distances)
            assert idx not in selected_samples

            selected_samples = np.append(selected_samples, [idx])

            min_distances = self.update_distances(X, selected_samples)

            new_samples.append(idx)
                
        return new_samples, min_distances
    
    def update_distances(self, X, cluster_centers):
        """

        """
        if len(cluster_centers) == 0:
            return np.full(shape=len(X), fill_value=np.nan)

        cluster_center_feature = X[cluster_centers]
        dist_matrix = pairwise_distances(X, cluster_center_feature)
        dist = np.min(dist_matrix, axis=1).reshape(-1,1)

        return dist
    
    