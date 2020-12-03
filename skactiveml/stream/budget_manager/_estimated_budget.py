import numpy as np

from .base import BudgetManager, get_default_budget


class EstimatedBudget(BudgetManager):
    """

    Parameters
    ----------
    budget : float
        Specifies the ratio of instances which are allowed to be sampled, with
        0 <= budget <= 1.
    """
    def __init__(self, budget=None):
        super().__init__(budget)
        #size of the memory/step window
        self.w = 100
        self.cnt = 0
        self.u_t = 0

    def is_budget_left(self):
        """Check whether there is any utility given to sample(...), which may
        lead to sampling the corresponding instance, i.e., check if sampling
        another instance is currently possible under the estimated budgeting
        constraint. This function is useful to determine, whether a provided
        utility is not sufficient, or the budgeting constraint was simply
        exhausted. For this budget manager this function returns True, when
        budget > estimated_spending
        
        Returns
        -------
        budget_left : bool
            True, if there is a utility which leads to sampling another
            instance.
        """
        #TODO
        return self.budget_ > self.u_t/self.w

    def sample(self, utilities, simulate=False, return_budget_left=False,
               **kwargs):
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
        self._validate_budget(get_default_budget())
        qs_decisions = np.array(utilities) >= self.budget_
        sampled_indices = []
        budget_left = []
        # check after 100 steps if budget is left 
        for i , d in enumerate(qs_decisions) :
            budget_left.append(self.budget_ > self.u_t/self.w)
            
            #u_t = u_t-1 * (w-1)/w + labeling_t
            self.u_t = self.u_t * ((self.w-1)/self.w) + d
            if self.u_t/self.w < self.budget_:
                sampled_indices.append(i)
                       
        # check if budget_left should be returned
        if return_budget_left:
            return sampled_indices, budget_left
        else:
            return sampled_indices

    def update(self, sampled, **kwargs):
        """Updates the budget manager.

        Parameters
        ----------
        sampled : array-like
            Indicates which instances from X_cand have been sampled.

        Returns
        -------
        self : FixedBudget
            The FixedBudget returns itself, after it is updated.
        """
        # check if budget has been set
        self._validate_budget(get_default_budget())
        # check if counting of instances has begun
        if not hasattr(self, "observed_instances_"):
            self.observed_instances_ = 0
        if not hasattr(self, "queried_instances_"):
            self.queried_instances_ = 0
        self.observed_instances_ += sampled.shape[0]
        self.queried_instances_ += np.sum(sampled)
        return self
