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

    def __init__(self, budget):
        self.budget = budget

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
    def sample(
        self, utilities, return_budget_left=True, simulate=False, **kwargs
    ):
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
        check_scalar(
            self.budget_, "budget", np.float, min_val=0.0, max_val=1.0
        )

    def _validate_data(self, utilities, return_budget_left, simulate):
        """Validate input data and set or check the `n_features_in_` attribute.

        Parameters
        ----------
        X_cand: array-like, shape (n_candidates, n_features)
            Candidate samples.
        return_utilities : bool,
            If true, also return the utilities based on the query strategy.
        reset : bool, default=True
            Whether to reset the `n_features_in_` attribute.
            If False, the input will be checked for consistency with data
            provided when reset was last True.
        **check_X_cand_params : kwargs
            Parameters passed to :func:`sklearn.utils.check_array`.

        Returns
        -------
        X_cand: np.ndarray, shape (n_candidates, n_features)
            Checked candidate samples
        batch_size : int
            Checked number of samples to be selected in one AL cycle.
        return_utilities : bool,
            Checked boolean value of `return_utilities`.
        random_state : np.random.RandomState,
            Checked random state to use.
        """
        # check if utilities is set
        if not isinstance(utilities, np.ndarray):
            raise TypeError("{} is not a valid type for utilities")
        # Check return_utilities.
        check_scalar(return_budget_left, 'return_budget_left', bool)
        # Check return_utilities.
        check_scalar(simulate, 'simulate', bool)

        self._validate_budget()
        return utilities, return_budget_left, simulate


def get_default_budget():
    return 0.1
