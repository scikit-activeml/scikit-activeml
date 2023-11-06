from ..base import PoolQueryStrategy
from ..utils import MISSING_LABEL, check_random_state, is_labeled, unlabeled_indices
import numpy as np

class SubSamplingWrapper(PoolQueryStrategy):
    def __init__(self, query_strategy=None, max_candidates=None, missing_label=MISSING_LABEL, random_state=None):
        super().__init__(missing_label, random_state)
        self.query_strategy = query_strategy
        self.max_candidates = max_candidates
    
    def query(self, X, y, candidates=None, batch_size=1, return_utilities=False, **query_kwargs):
        X, y, candidates, batch_size, return_utilities = self._validate_data(
            X, y, candidates, batch_size, return_utilities, reset=True
        )
        
        seed_multiplier = int(is_labeled(y).sum())
        random_state = check_random_state(self.random_state, seed_multiplier)
        
        if candidates is None:
            candidate_indices = unlabeled_indices(
                y=y, 
                missing_label=self.missing_label
            )
            new_candidates = random_state.choice(
                a=candidate_indices, 
                size=self.max_candidates,
                replace=False
            )
        else:
            if candidates.ndim==1: 
                new_candidates = random_state.choice(
                    a=candidates, 
                    size=self.max_candidates,
                    replace=False
                )
            else:
                candidate_indices = range(len(candidates))
                new_candidate_indices = random_state.choice(
                    a=candidate_indices, 
                    size=self.max_candidates,
                    replace=False
                )
                new_candidates = candidates[new_candidate_indices]

        qs_output = self.query_strategy.query(
            X=X, 
            y=y, 
            candidates=new_candidates, 
            batch_size=batch_size, 
            return_utilities=return_utilities, 
            **query_kwargs
        )

        if not return_utilities:
            if candidates is not None and candidates.ndim > 1:
                return new_candidate_indices[qs_output]
            return qs_output

        queried_indices, utilities = qs_output

        if candidates is None or candidates.ndim == 1:
            new_utilities = utilities
        else:
            new_utilities = np.full(shape=(batch_size, len(candidates)), fill_value=np.nan)
            new_utilities[:, new_candidate_indices] = utilities
            queried_indices = new_candidate_indices[queried_indices]

        return queried_indices, new_utilities