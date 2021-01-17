import numpy as np

from .base import BudgetManager, get_default_budget

#TODO add classes estimadetBudget for split var and Fixed. Return to standart
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
    def __init__(self, budget=None, w=100):
        super().__init__(budget)
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

    
    
#     def sample(self, utilities, simulate=False, return_budget_left=False,
#                **kwargs):
#         """Ask the budget manager which utilities are sufficient to sample the
#         corresponding instance.

#         Parameters
#         ----------
#         utilities : ndarray of shape (n_samples,)
#             The utilities provided by the stream-based active learning
#             strategy, which are used to determine whether sampling an instance
#             is worth it given the budgeting constraint.

#         return_utilities : bool, optional
#             If true, also return whether there was budget left for each
#             assessed utility. The default is False.

#         simulate : bool, optional
#             If True, the internal state of the budget manager before and after
#             the query is the same. This should only be used to prevent the
#             budget manager from adapting itself. The default is False.

#         Returns
#         -------
#         sampled_indices : ndarray of shape (n_sampled_instances,)
#             The indices of instances represented by utilities which should be
#             sampled, with 0 <= n_sampled_instances <= n_samples.

#         budget_left: ndarray of shape (n_samples,), optional
#             Shows whether there was budget left for each assessed utility. Only
#             provided if return_utilities is True.
#         """
#         # check if budget has been set
#         self._validate_budget(get_default_budget())
        
#         # check if calculation of estimate bought/true lables has begun
#         if not hasattr(self, 'u_t_'):
#             self.u_t_ = 0
#         # intialise return parameters
#         sampled_indices = []
#         budget_left = []
#         # keep the internal state to reset it later if simulate is true
#         tmp_u_t = self.u_t_
#         tmp_theta = self.theta_
        

#         # check for each sample separately if budget is left and the utility is
#         # high enough
#         for i , d in enumerate(utilities) :
#             budget_left.append(tmp_u_t/self.w < self.budget_)
#             if not budget_left[-1]:
#                 d = False
#             # u_t = u_t-1 * (w-1)/w + labeling_t
#             tmp_u_t = tmp_u_t * ((self.w-1)/self.w) + d
#             # get the indices instances that should be sampled
#             if d:
#                 sampled_indices.append(i)
        
#         # set the internal state to the previous values
#         if not simulate:
#             self.u_t_ = tmp_u_t
#             self.theta_ = tmp_theta
            
#         # check if budget_left should be returned
#         if return_budget_left:
#             return sampled_indices, budget_left
#         else:
#             return sampled_indices
        

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

class FixedUncertaintyBudget(EstimatedBudget):
    """
    
    """
    def __init__(self,  budget=None, w=100):
        super().__init__(budget, w)
        
        
    def sample(self, utilities, num_classes, return_budget_left=True, simulate=False, **kwargs):
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
        # check if theta exists
        if not hasattr(self, "theta_"):
            self.theta_ = 0.0
        # check if theta is set
        if not isinstance(self.theta_, float):
            raise TypeError("{} is not a valid type for theta")
        # check if num_classes is set
        if not hasattr(self, "num_classes_"):
            self.num_classes_ = 0.0
        # check if calculation of estimate bought/true lables has begun
        if not hasattr(self, 'u_t_'):
            self.u_t_ = 0
        
        

        # intialise return parameters
        sampled_indices = []
        budget_left = []
        # calculate theta with num_classes
        self.num_classes_ = num_classes
        self.theta_ = 1/self.num_classes_ + self.budget_ * (1-1/self.num_classes_)
        
        # keep the internal state to reset it later if simulate is true
        tmp_u_t = self.u_t_
        tmp_theta = self.theta_
        
        samples = np.array(utilities <= tmp_theta)
        # check for each sample separately if budget is left and the utility is
        # high enough
        for i , d in enumerate(samples) :
            budget_left.append(tmp_u_t/self.w < self.budget_)
            if not budget_left[-1]:
                d = False
            # u_t = u_t-1 * (w-1)/w + labeling_t
            tmp_u_t = tmp_u_t * ((self.w-1)/self.w) + d
            # get the indices instances that should be sampled
            if d:
                sampled_indices.append(i)
        
        # set the internal state to the previous values
        if not simulate:
            self.u_t_ = tmp_u_t
            self.theta_ = tmp_theta
        
        # check if budget_left should be returned
        if return_budget_left:
            return sampled_indices, budget_left
        else:
            return sampled_indices
        
        
        
    
class VarUncertaintyBudget(EstimatedBudget):
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
    def __init__(self, budget=None, w=100, theta=1.0, s=0.01):
        super().__init__(budget, w)
        self.theta = theta
        self.s = s
        
    def sample(self, utilities, return_budget_left=True, simulate=False, **kwargs):
        """
        
        """
        # check if budget has been set
        self._validate_budget(get_default_budget())
        # check if theta exists
        if not hasattr(self, "theta_"):
            self.theta_ = self.theta
        # check if theta is set
        if not isinstance(self.theta_, float):
            raise TypeError("{} is not a valid type for theta")
        # check if calculation of estimate bought/true lables has begun
        if not hasattr(self, 'u_t_'):
            self.u_t_ = 0
        # ckeck if s a float and in range (0,1]
        if self.s is not None:
                if not isinstance(self.s, float):
                    raise TypeError("{} is not a valid type for s")
                if self.s <= 0 or self.s > 1.0:
                    raise ValueError("The value of s is incorrect." +
                                     " s must be defined in range (0,1]")
        
        # intialise return parameters
        sampled_indices = []
        budget_left = []
        # keep the internal state to reset it later if simulate is true
        tmp_u_t = self.u_t_
        tmp_theta = self.theta_
        
        # get utilities
        for i, u in enumerate(utilities):  
            budget_left.append(tmp_u_t/self.w < self.budget_)
            
            if not budget_left[-1]:
                sample = False
            else:
                # the original inequation is:
                # sample_instance: True if y < theta_t
                # to scale this inequation to the desired range, i.e., utilities
                # higher than 1-budget should lead to sampling the instance, we use
                # sample_instance: True if 1-budget < theta_t + (1-budget) - y
                sample = u < tmp_theta
                # get the indices instances that should be sampled
                if sample:
                    tmp_theta *= (1-self.s)
                    sampled_indices.append(i)
                else:
                    tmp_theta *= (1+self.s) 
            # u_t = u_t-1 * (w-1)/w + labeling_t
            tmp_u_t = tmp_u_t * ((self.w-1)/self.w) + sample
            
                
        # set the internal state to the previous values
        if not simulate:
            self.u_t_ = tmp_u_t
            self.theta_ = tmp_theta
            
        # check if budget_left should be returned
        if return_budget_left:
            return sampled_indices, budget_left
        else:
            return sampled_indices
    
class SplitBudget(EstimatedBudget):
    """    
    """
    def __init__(self, budget=None, w=100, theta=1.0, s=0.01, v=0.1):
        super().__init__(budget, w)
        self.v = v
        self.theta = theta
        self.s = s
        
    def sample(self, utilities, return_budget_left=True, simulate=False, **kwargs):
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
        # check if theta exists
        if not hasattr(self, "theta_"):
            self.theta_ = self.theta
        # check if theta is set
        if not isinstance(self.theta_, float):
            raise TypeError("{} is not a valid type for theta")
        # check if calculation of estimate bought/true lables has begun
        if not hasattr(self, 'u_t_'):
            self.u_t_ = 0
        if not hasattr(self, 'rand_'):
            self.rand_ = np.random
        # ckeck if s a float and in range (0,1]
        if self.s is not None:
                if not isinstance(self.s, float):
                    raise TypeError("{} is not a valid type for s")
                if self.s <= 0 or self.s > 1.0:
                    raise ValueError("The value of s is incorrect." +
                                     " s must be defined in range (0,1]")
        # ckeck if v is a float and in range (0,1]
        if not isinstance(self.v, float):
            raise TypeError("{} is not a valid type for v")
        if self.v <= 0 or self.v >= 1:
            raise ValueError("The value of v is incorrect." +
                             " v must be defined in range (0,1)")
        # intialise return parameters
        sampled_indices = []
        budget_left = []
        # keep the internal state to reset it later if simulate is true
        tmp_u_t = self.u_t_
        tmp_theta = self.theta_
        
        # check for each sample separately if budget is left and the utility is
        # high enough
        for i, u in enumerate(utilities):
            budget_left.append(tmp_u_t/self.w < self.budget_)
            if not budget_left[-1]:
                sample = False
            else:
                # changed self.v < self.rand_.random_sample()
                if self.v > self.rand_.random_sample():
                    sample = self.rand_.random_sample() <= self.budget_
                else: 
                    # the original inequation is:
                    # sample_instance: True if y < theta_t
                    # to scale this inequation to the desired range, i.e., utilities
                    # higher than 1-budget should lead to sampling the instance, we use
                    # sample_instance: True if 1-budget < theta_t + (1-budget) - y
                    sample = u < tmp_theta
                    # get the indices instances that should be sampled 
                    if sample:
                        tmp_theta *= (1-self.s)
                        sampled_indices.append(i)
                    else:
                        tmp_theta *= (1+self.s)
                    
            # u_t = u_t-1 * (w-1)/w + labeling_t
            tmp_u_t = tmp_u_t * ((self.w-1)/self.w) + sample
            
        # set the internal state to the previous values
        if not simulate:
            self.u_t_ = tmp_u_t
            self.theta_ = tmp_theta
            
        # check if budget_left should be returned
        if return_budget_left:
            return sampled_indices, budget_left
        else:
            return sampled_indices
