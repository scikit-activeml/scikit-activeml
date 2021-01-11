from abc import ABC, abstractmethod

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.utils import check_scalar


class BudgetManager(ABC, BaseEstimator):
    """Base class for all budget managers for stream-based active learning
       in scikit-activeml to model budgeting constraints.

    Parameters
    ----------
    budget : float
        Specifies the ratio of instances which are allowed to be sampled, with
        0 <= budget <= 1.
    """
    def __init__(self, budget, theta, s):
        self.budget = budget
        self.theta = theta
        self.s = s
        

    @abstractmethod
    def is_budget_left(self):
        """Check whether there is any utility given to sample(...), which may
        lead to sampling the corresponding instance, i.e., check if sampling
        another instance is currently possible under the specified budgeting
        constraint. This function is useful to determine, whether a provided
        utility is not sufficient, or the budgeting constraint was simply
        exhausted.

        Returns
        -------
        budget_left : bool
            True, if there is a utility which leads to sampling another
            instance.
        """
        return NotImplemented

    @abstractmethod
    def sample(self, utilities, return_budget_left=True, simulate=False,
               return_utilities=False, use_theta=True, **kwargs):
        """Ask the budget manager which utilities are sufficient to sample the
        corresponding instance.

        Parameters
        ----------
        utilities : ndarray of shape (n_samples,)
            The utilities provided by the stream-based active learning
            strategy, which are used to determine whether sampling an instance
            is worth it given the budgeting constraint.

        return_utilities : bool, optional
            If true, also return whether there was budget left for each
            assessed utility. The default is False.

        simulate : bool, optional
            If True, the internal state of the budget manager before and after
            the query is the same. This should only be used to prevent the
            budget manager from adapting itself. The default is False.

        Returns
        -------
        sampled_indices : ndarray of shape (n_sampled_instances,)
            The indices of instances represented by utilities which should be
            sampled, with 0 <= n_sampled_instances <= n_samples.

        budget_left: ndarray of shape (n_samples,), optional
            Shows whether there was budget left for each assessed utility. Only
            provided if return_utilities is True.
        """
        return NotImplemented

    @abstractmethod
    def update(self, sampled, **kwargs):
        """Updates the BudgetManager.

        Parameters
        ----------
        sampled : array-like
            Indicates which instances from X_cand have been sampled.

        Returns
        -------
        self : BudgetManager
            The BudgetManager returns itself, after it is updated.
        """
        return NotImplemented

    def _validate_budget(self, default=None):
        """check the assigned budget and set a default value, when none is set
        prior.

        Parameters
        ----------
        default : float, optional
            the budget which should be assigned, when none is set.
        """
        if self.budget is not None:
            self.budget_ = self.budget
        else:
            self.budget_ = default
        check_scalar(self.budget_, 'budget',
                     np.float, min_val=0.0, max_val=1.0)
                    
    def calculate_fixed_theta(self, num_classes, budget):
        """calculate theta for Fixeduncertainty and returns theta
        
        Parameters
        ----------
        num_classes : int
            the number of classes is used to calculate theta
            
        budget : float
            
        """
        self.theta = 1/num_classes + budget*(1-1/num_classes)
        
        


def get_default_budget():
    return 0.1