import numpy as np

from .base import BudgetManager, get_default_budget


class EstimatedBudget(BudgetManager):
    """Budget manager which checks, whether the specified budget has been
    exhausted already. If not, an instance is sampled, when the utility is
    higher than the specified budget.

    This budget manager calculates the estimated budget spent in the last 
    w steps and compares that to the budget. If the ratio is smaller
    than the specified budget, i.e.,
    budget - u_t / w > 0 , the budget
    manager samples an instance when its utility is higher than the budget.
    u is the estimate of how many true lables were queried within the last 
    w steps. The recursive funktion, 
    u_t = u_t-1 * (w-1) / w + labeling_t , is used to calculate u at time t.

    Parameters
    ----------
    budget : float
        Specifies the ratio of instances which are allowed to be sampled, with
        0 <= budget <= 1.
    
    w : int
        Specifies the size of the memory window. Controlles the budget in the 
        last w steps taken. Default = 100
    """
    def __init__(self, budget=None, theta=None, s=None, w=100):
        super().__init__(budget, theta, s)
        self.w = w

    def is_budget_left(self):
        """Check whether there is any utility given to sample(...), which may
        lead to sampling the corresponding instance, i.e., check if sampling
        another instance is currently possible under the budgeting constraint. 
        This function is useful to determine, whether a provided
        utility is not sufficient, or the budgeting constraint was simply
        exhausted. For this budget manager this function returns True, when
        budget > estimated_spending
        
        Returns
        -------
        budget_left : bool
            True, if there is a utility which leads to sampling another
            instance.
        """
        
        return self.budget_ > self.u_t_/self.w

    def sample(self, y_hat, simulate=False, return_budget_left=False,
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
        # check if budget has been set
        self._validate_budget(get_default_budget())
        if not hasattr(self, "theta_"):
            self.theta_ = self.theta
        # check if theta is set
        if not isinstance(self.theta, float) and self.theta is not None:
            raise TypeError("{} is not a valid type for theta")
        # ckeck if s a float and in range (0,1]
        if self.s is not None:
                if not isinstance(self.s, float):
                    raise TypeError("{} is not a valid type for s")
                if self.s <= 0 or self.s > 1.0:
                    raise ValueError("The value of s is incorrect." +
                                     " s must be defined in range (0,1]")
        
        # check if calculation of estimate bought/true lables has begun
        if not hasattr(self, 'u_t_'):
            self.u_t_ = 0
        # intialise return parameters
        utilities = []
        sampled_indices = []
        budget_left = []
        # keep the internal state to reset it later if simulate is true
        tmp_u_t = self.u_t_
        tmp_theta = self.theta
        # get utilities
        for y_ in y_hat:
                # the original inequation is:
                # sample_instance: True if y < theta_t
                # to scale this inequation to the desired range, i.e., utilities
                # higher than 1-budget should lead to sampling the instance, we use
                # sample_instance: True if 1-budget < theta_t + (1-budget) - y
                if tmp_theta is not None and use_theta:
                    utilities.append(y_ <= tmp_theta)
                else:
                    utilities.append(y_)

        # check for each sample separately if budget is left and the utility is
        # high enough
        for i , d in enumerate(utilities) :
            budget_left.append(tmp_u_t/self.w < self.budget_)
            if not budget_left[-1]:
                d = False
            # u_t = u_t-1 * (w-1)/w + labeling_t
            tmp_u_t = tmp_u_t * ((self.w-1)/self.w) + d
            # get the indices instances that should be sampled
            if d:
                sampled_indices.append(i)
        
        # calculate theta for Varuncertainty
        for y_ in y_hat:
                if self.s is not None:
                    #tmp_theata = self.calculate_var_theta(budget_left, sampled_indices)
                    if budget_left[-1] and use_theta:
                        if len(sampled_indices):
                            tmp_theta = tmp_theta * (1-self.s)
                        else:
                            tmp_theta = tmp_theta * (1+self.s)
        
        # set the internal state to the previous values
        if not simulate:
            self.u_t_ = tmp_u_t
            self.theta_ = tmp_theta
            
        # check if budget_left and utilities should be returned
        if return_utilities:
            if return_budget_left:
                return sampled_indices, budget_left, utilities
            else:
                return sampled_indices, utilities
        else:
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
        self : EstimatedBudget
            The EstimatedBudget returns itself, after it is updated.
        """
        # check if budget has been set
        self._validate_budget(get_default_budget())
        # check if calculation of estimate bought/true lables has begun
        if not hasattr(self, 'u_t_'):
            self.u_t_ = 0
        # update u_t for sampled X_cand
        for s in sampled:
            self.u_t_ = self.u_t_ * ((self.w-1)/self.w) + s
        
        return self
