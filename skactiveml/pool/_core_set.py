"""
Module implementing coreset query strategies.
"""

import numpy as np

from ..base import SingleAnnotatorPoolQueryStrategy
from ..utils import MISSING_LABEL, simple_batch

class CoreSet(SingleAnnotatorPoolQueryStrategy):
    
    def __init__(
          self,
          missing_label=MISSING_LABEL,
          random_state=None  
    ):
        super().__init__(
            missing_label=missing_label, random_state=random_state
        )

    def query(
        self, X, y, candidates=None, batch_size=1, return_utilities=False
    ):
        X, y, candidates, batch_size, return_utilities = self._validate_data(
            X, y, candidates, batch_size, return_utilities, reset=True
        )

        return X
    
    