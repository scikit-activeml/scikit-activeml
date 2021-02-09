import numpy as np

import skactiveml.pool._probal as probal

from sklearn.base import is_classifier, clone
from sklearn.utils import check_array

from ..base import SingleAnnotStreamBasedQueryStrategy
from ..classifier import PWC

from .budget_manager import BIQF


class PAL(SingleAnnotStreamBasedQueryStrategy):
    def __init__(self, clf=None, budget_manager=BIQF(),
                 random_state=None, prior=1.e-3, m_max=2):
        self.clf = clf
        self.budget_manager = budget_manager
        self.random_state = random_state
        self.prior = prior
        self.m_max = m_max
        
    def query(
        self, X_cand, X, y, return_utilities=False, simulate=False, **kwargs
    ):
        # check the shape of data
        X_cand = check_array(X_cand, force_all_finite=False)
        # check if a random state is set
        self._validate_random_state()
        # check if a budget_manager is set
        self._validate_budget_manager()
        # check if prior is set
        if not isinstance(self.prior, float) and not None:
            raise TypeError("{} is not a valid type for prior")
        if self.prior <= 0:
            raise ValueError(
                "The value of prior is incorrect." 
                + " prior must be greater than 0"
            )
        # check if m_max is set
        if not isinstance(self.m_max, int):
            raise TypeError("{} is not a valid type for m_max")
        if self.m_max <= 0:
            raise ValueError(
                "The value of m_max is incorrect." 
                + " m_max must be greater than 0"
            )
        # check if clf is a classifier
        if X is not None and y is not None:
            if self.clf is None:
                clf = PWC(random_state=self.random_state_.randint(2 ** 31 - 1))
            elif is_classifier(self.clf):
                clf = clone(self.clf)
            else:
                raise TypeError(
                    "clf is not a classifier. Please refer to "
                    + "sklearn.base.is_classifier"
                )
            clf.fit(X, y)
            # check if y is not multi dimensinal
            if isinstance(y, np.ndarray):
                if y.ndim > 1:
                    raise ValueError("{} is not a valid Value for y")
        else:
            clf = self.clf

        k_vec = clf.predict_freq(X_cand)
        # n = np.sum(k_vec)
        utilities = probal._cost_reduction(
            k_vec, prior=self.prior, m_max=self.m_max
        )
        sampled_indices = self.budget_manager_.sample(utilities)

        if return_utilities:
            return sampled_indices, utilities
        else:
            return sampled_indices

    def update(self, sampled, **kwargs):
        # check if a budget_manager is set
        self._validate_budget_manager()
        self.budget_manager_.update(sampled)
        return self
