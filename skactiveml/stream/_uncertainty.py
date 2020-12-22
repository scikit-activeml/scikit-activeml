import numpy as np

from sklearn.utils import check_array

from sklearn.base import is_classifier, clone

from ..base import SingleAnnotStreamBasedQueryStrategy

from ..classifier import PWC

from ._random import RandomSampler

import copy

from .budget_manager import FixedBudget


class FixedUncertainty(SingleAnnotStreamBasedQueryStrategy):
    """The FixedUncertainty (Fixed-Uncertainty in [1]) query strategy samples
    instances based on the classifiers uncertainty assessed based on the
    classifier's predictions. The instance is sampled when the probability of
    the most likely class exceeds a threshold calculated based on the budget
    and the number of classes.

    Parameters
    ----------
    budget_manager : BudgetManager
        The BudgetManager which models the budgeting constraint used in
        the stream-based active learning setting. The budget attribute set for
        the budget_manager will be used to determine the interval between
        sampling instances

    clf : BaseEstimator
        The classifier which is trained using this query startegy.

    random_state : int, RandomState instance, default=None
        Controls the randomness of the estimator.

    References
    ----------
    [1] Zliobaite, Indre & Bifet, Albert & Pfahringer, Bernhard & Holmes,
        Geoffrey. (2014). Active Learning With Drifting Streaming Data. Neural
        Networks and Learning Systems, IEEE Transactions on. 25. 27-39.

    """
    def __init__(self, clf=None, budget_manager=FixedBudget(),
                 random_state=None):
        super().__init__(budget_manager=budget_manager,
                         random_state=random_state)
        self.clf = clf

    def query(self, X_cand, X, y, return_utilities=False, simulate=False,
              **kwargs):
        """Ask the query strategy which instances in X_cand to acquire.

        Please note that, when the decisions from this function may differ from
        the final sampling, simulate=True can set, so that the query strategy
        can be updated later with update(...) with the final sampling. This is
        especially helpful, when developing wrapper query strategies.

        Parameters
        ----------
        X_cand : {array-like, sparse matrix} of shape (n_samples, n_features)
            The instances which may be sampled. Sparse matrices are accepted
            only if they are supported by the base query strategy.

        return_utilities : bool, optional
            If true, also return the utilities based on the query strategy.
            The default is False.

        simulate : bool, optional
            If True, the internal state of the query strategy before and after
            the query is the same. This should only be used to prevent the
            query strategy from adapting itself. Note, that this is propagated
            to the budget_manager, as well. The default is False.

        Returns
        -------
        sampled_indices : ndarray of shape (n_sampled_instances,)
            The indices of instances in X_cand which should be sampled, with
            0 <= n_sampled_instances <= n_samples.

        utilities: ndarray of shape (n_samples,), optional
            The utilities based on the query strategy. Only provided if
            return_utilities is True.
        """
        # check the shape of data
        X_cand = check_array(X_cand, force_all_finite=False)
        # check if a budget_manager is set
        self._validate_budget_manager()
        # check if a random state is set
        self._validate_random_state()
        # check if clf is a classifier
        if X is not None and y is not None:
            if self.clf is None:
                clf = PWC(
                    random_state=self.random_state_.randint(2**31-1))
            elif is_classifier(self.clf):
                clf = clone(self.clf)
            else:
                raise TypeError("clf is not a classifier. Please refer to " +
                                 "sklearn.base.is_classifier")
            clf.fit(X, y)
            # check if y is not multi dimensinal
            if isinstance(y, np.ndarray):
                if y.ndim > 1:
                    raise ValueError("{} is not a valid Value for y")
        else:
            clf = self.clf
        predict_proba = clf.predict_proba(X_cand)
        y_hat = np.max(predict_proba, axis=1)
        num_classes = predict_proba.shape[1]
        budget = getattr(self.budget_manager_, "budget_", 0)
        theta = 1/num_classes + budget*(1-1/num_classes)
        # the original inequation is:
        # sample_instance: True if y < theta_t
        # to scale this inequation to the desired range, i.e., utilities
        # higher than 1-budget should lead to sampling the instance, we use
        # sample_instance: True if 1-budget < theta_t + (1-budget) - y
        utilities = theta + (1 - budget) - y_hat

        sampled_indices = self.budget_manager_.sample(utilities,
                                                      simulate=simulate)

        if return_utilities:
            return sampled_indices, utilities
        else:
            return sampled_indices

    def update(self, X_cand, sampled, **kwargs):
        """Updates the budget manager and the count for seen and sampled
        instances

        Parameters
        ----------
        X_cand : {array-like, sparse matrix} of shape (n_samples, n_features)
            The instances which could be sampled. Sparse matrices are accepted
            only if they are supported by the base query strategy.

        sampled : array-like
            Indicates which instances from X_cand have been sampled.

        Returns
        -------
        self : FixedUncertainty
            The FixedUncertainty returns itself, after it is updated.
        """
        # check if a budget_manager is set
        self._validate_budget_manager()
        self.budget_manager_.update(sampled)
        return self


class VariableUncertainty(SingleAnnotStreamBasedQueryStrategy):
    """The VariableUncertainty (Var-Uncertainty in [1]) query strategy samples
    instances based on the classifiers uncertainty assessed based on the
    classifier's predictions. The instance is sampled when the probability of
    the most likely class exceeds a time-dependent threshold calculated based
    on the budget, the number of classes and the number of observed and
    acquired samples.

    Parameters
    ----------
    budget_manager : BudgetManager
        The BudgetManager which models the budgeting constraint used in
        the stream-based active learning setting. The budget attribute set for
        the budget_manager will be used to determine the interval between
        sampling instances

    clf : BaseEstimator
        The classifier which is trained using this query startegy.

    random_state : int, RandomState instance, default=None
        Controls the randomness of the estimator.

    References
    ----------
    [1] Zliobaite, Indre & Bifet, Albert & Pfahringer, Bernhard & Holmes,
        Geoffrey. (2014). Active Learning With Drifting Streaming Data. Neural
        Networks and Learning Systems, IEEE Transactions on. 25. 27-39.

    """
    def __init__(self, clf=None, budget_manager=FixedBudget(),
                 theta=1.0, s=0.01, random_state=None):
        super().__init__(budget_manager=budget_manager,
                         random_state=random_state)
        self.clf = clf
        self.theta = theta
        self.s = s
    
    def query(self, X_cand, X, y, return_utilities=False, simulate=False,
              **kwargs):
        """Ask the query strategy which instances in X_cand to acquire.

        Please note that, when the decisions from this function may differ from
        the final sampling, simulate=True can set, so that the query strategy
        can be updated later with update(...) with the final sampling. This is
        especially helpful, when developing wrapper query strategies.

        Parameters
        ----------
        X_cand : {array-like, sparse matrix} of shape (n_samples, n_features)
            The instances which may be sampled. Sparse matrices are accepted
            only if they are supported by the base query strategy.

        return_utilities : bool, optional
            If true, also return the utilities based on the query strategy.
            The default is False.

        simulate : bool, optional
            If True, the internal state of the query strategy before and after
            the query is the same. This should only be used to prevent the
            query strategy from adapting itself. Note, that this is propagated
            to the budget_manager, as well. The default is False.

        Returns
        -------
        sampled_indices : ndarray of shape (n_sampled_instances,)
            The indices of instances in X_cand which should be sampled, with
            0 <= n_sampled_instances <= n_samples.

        utilities: ndarray of shape (n_samples,), optional
            The utilities based on the query strategy. Only provided if
            return_utilities is True.
        """
        # ckeck if s a float and in range (0,1]
        if not isinstance(self.s, float):
            raise TypeError("{} is not a valid type for s")
        if self.s <= 0 or self.s > 1.0:
            raise ValueError("The value of s is incorrect." +
                             " s must be defined in range (0,1]")
        # check the shape of data
        X_cand = check_array(X_cand, force_all_finite=False)
        # check if a budget_manager is set
        self._validate_budget_manager()
        # check if a random state is set
        self._validate_random_state()
        # check if clf is a classifier
        if X is not None and y is not None:
            if self.clf is None:
                clf = PWC(
                    random_state=self.random_state_.randint(2**31-1))
            elif is_classifier(self.clf):
                clf = clone(self.clf)
            else:
                raise TypeError("clf is not a classifier. Please refer to " +
                                 "sklearn.base.is_classifier")
            clf.fit(X, y)
            # check if y is not multi dimensinal
            if isinstance(y, np.ndarray):
                if y.ndim > 1:
                    raise ValueError("{} is not a valid Value for y")
        else:
            clf = self.clf
        if not hasattr(self, "theta_"):
            self.theta_ = self.theta
        # check if theta is set
        if not isinstance(self.theta, float):
            raise TypeError("{} is not a valid type for theta")
        predict_proba = clf.predict_proba(X_cand)
        y_hat = np.max(predict_proba, axis=1)
        budget = getattr(self.budget_manager_, "budget_", 0)

        tmp_theta = self.theta_

        utilities = []
        sampled_indices = []

        for y_ in y_hat:
            # the original inequation is:
            # sample_instance: True if y < theta_t
            # to scale this inequation to the desired range, i.e., utilities
            # higher than 1-budget should lead to sampling the instance, we use
            # sample_instance: True if 1-budget < theta_t + (1-budget) - y
            utilities.append(tmp_theta + (1 - budget) - y_)
            sampled, budget_left = self.budget_manager_.sample(
                utilities,
                simulate=True,
                return_budget_left=True
            )
            sampled_indices.append(sampled)
            if budget_left[-1]:
                if len(sampled):
                    tmp_theta = tmp_theta * (1-self.s)
                else:
                    tmp_theta = tmp_theta * (1+self.s)

        if not simulate:
            self.theta_ = tmp_theta

        sampled_indices = self.budget_manager_.sample(utilities,
                                                      simulate=simulate)

        if return_utilities:
            return sampled_indices, utilities
        else:
            return sampled_indices

    def update(self, X_cand, sampled, **kwargs):
        """Updates the budget manager and the count for seen and sampled
        instances

        Parameters
        ----------
        X_cand : {array-like, sparse matrix} of shape (n_samples, n_features)
            The instances which could be sampled. Sparse matrices are accepted
            only if they are supported by the base query strategy.

        sampled : array-like
            Indicates which instances from X_cand have been sampled.

        Returns
        -------
        self : VariableUncertainty
            The VariableUncertainty returns itself, after it is updated.
        """
        # check if a budget_manager is set
        self._validate_budget_manager()
        if not hasattr(self, "theta_"):
            self.theta_ = self.theta
        self.budget_manager_.update(sampled)
        budget_left = kwargs.get('budget_left', None)
        for i, s in enumerate(sampled):
            if budget_left is None:
                if sampled[-1]:
                    self.theta_ *= (1-self.s)
                else:
                    self.theta_ *= (1+self.s)
        return self


class Split(SingleAnnotStreamBasedQueryStrategy):
    """The Split [1] query strategy samples in 100*s% of instances randomly and
    in 100*(1-s)% of cases according to VarUncertainty.

    Parameters
    ----------
    budget_manager : BudgetManager
        The BudgetManager which models the budgeting constraint used in
        the stream-based active learning setting. The budget attribute set for
        the budget_manager will be used to determine the interval between
        sampling instances

    clf : BaseEstimator
        The classifier which is trained using this query startegy.

    random_state : int, RandomState instance, default=None
        Controls the randomness of the estimator.

    References
    ----------
    [1] Zliobaite, Indre & Bifet, Albert & Pfahringer, Bernhard & Holmes,
        Geoffrey. (2014). Active Learning With Drifting Streaming Data. Neural
        Networks and Learning Systems, IEEE Transactions on. 25. 27-39.

    """
    def __init__(self, clf=None, budget_manager=FixedBudget(), v=0.1,
                 theta=1.0, s=0.01, random_state=None):
        super().__init__(budget_manager=budget_manager,
                         random_state=random_state)
        self.clf = clf
        self.s = s
        self.v = v
        self.theta = theta
    
    def query(self, X_cand, X, y, return_utilities=False, simulate=False,
              **kwargs):
        """Ask the query strategy which instances in X_cand to acquire.

        Please note that, when the decisions from this function may differ from
        the final sampling, simulate=True can set, so that the query strategy
        can be updated later with update(...) with the final sampling. This is
        especially helpful, when developing wrapper query strategies.

        Parameters
        ----------
        X_cand : {array-like, sparse matrix} of shape (n_samples, n_features)
            The instances which may be sampled. Sparse matrices are accepted
            only if they are supported by the base query strategy.

        return_utilities : bool, optional
            If true, also return the utilities based on the query strategy.
            The default is False.

        simulate : bool, optional
            If True, the internal state of the query strategy before and after
            the query is the same. This should only be used to prevent the
            query strategy from adapting itself. Note, that this is propagated
            to the budget_manager, as well. The default is False.

        Returns
        -------
        sampled_indices : ndarray of shape (n_sampled_instances,)
            The indices of instances in X_cand which should be sampled, with
            0 <= n_sampled_instances <= n_samples.

        utilities: ndarray of shape (n_samples,), optional
            The utilities based on the query strategy. Only provided if
            return_utilities is True.
        """
        # ckeck if v is a float and in range (0,1]
        if not isinstance(self.v, float):
            raise TypeError("{} is not a valid type for s")
        if self.v <= 0 or self.v >= 1:
            raise ValueError("The value of v is incorrect." +
                             " v must be defined in range (0,1)")
        # check the shape of data
        X_cand = check_array(X_cand, force_all_finite=False)
        # check if a budget_manager is set
        self._validate_budget_manager()
        # check if a random state is set
        self._validate_random_state()
        # check if clf is a classifier
        if X is not None and y is not None:
            if self.clf is None:
                clf = PWC(
                    random_state=self.random_state_.randint(2**31-1))
            elif is_classifier(self.clf):
                clf = clone(self.clf)
            else:
                raise TypeError("clf is not a classifier. Please refer to " +
                                 "sklearn.base.is_classifier")
            clf.fit(X, y)
            # check if y is not multi dimensinal
            if isinstance(y, np.ndarray):
                if y.ndim > 1:
                    raise ValueError("{} is not a valid Value for y.")
        else:
            clf = self.clf

        if not hasattr(self, 'random_sampler_'):
            self.random_sampler_ = RandomSampler(
                self.budget_manager_,
                random_state=self.random_state_.randint(2**31-1))
        if not hasattr(self, 'variable_uncertainty_'):
            self.variable_uncertainty_ = VariableUncertainty(
                clf,
                self.budget_manager_,
                theta=self.theta,
                s=self.s,
                random_state=self.random_state_.randint(2**31-1))
        # copy random state in case of simulating the query
        prior_random_state_state = self.random_state_.get_state()

        utilities = []
        sampled_indices = []

        use_random_sampler = self.random_state_.choice([0, 1], X_cand.shape[0],
                                                       p=[(1-self.v), self.v])

        tmp_budget_manager = copy.deepcopy(self.budget_manager_)
        tmp_random_sampler = copy.deepcopy(self.random_sampler_)
        tmp_random_sampler.budget_manager_ = tmp_budget_manager
        tmp_random_sampler.clf = self.clf
        tmp_var_uncertainty = copy.deepcopy(self.variable_uncertainty_)
        tmp_var_uncertainty.budget_manager_ = tmp_budget_manager
        tmp_var_uncertainty.clf = self.clf

        merged_sampled_indices = []
        merged_utilities = []

        for x, use_rand in zip(X_cand, use_random_sampler):
            if use_rand:
                sampled_indices, utilities = tmp_random_sampler.query(
                    x.reshape([1, -1]),
                    X=X,
                    y=y,
                    return_utilities=True,
                    simulate=False)
            else:
                sampled_indices, utilities = tmp_var_uncertainty.query(
                    x.reshape([1, -1]),
                    X=X,
                    y=y,
                    return_utilities=True,
                    simulate=False)
            merged_sampled_indices.extend(sampled_indices)
            merged_utilities.extend(utilities)

        if not simulate:
            self.budget_manager_ = tmp_budget_manager
            self.random_sampler_ = tmp_random_sampler
            self.variable_uncertainty_ = tmp_var_uncertainty
        else:
            self.random_state_.set_state(prior_random_state_state)

        if return_utilities:
            return sampled_indices, utilities
        else:
            return sampled_indices

    def update(self, X_cand, sampled, X, y, **kwargs):
        """Updates the budget manager and the count for seen and sampled
        instances

        Parameters
        ----------
        X_cand : {array-like, sparse matrix} of shape (n_samples, n_features)
            The instances which could be sampled. Sparse matrices are accepted
            only if they are supported by the base query strategy.

        sampled : array-like
            Indicates which instances from X_cand have been sampled.

        Returns
        -------
        self : VariableUncertainty
            The VariableUncertainty returns itself, after it is updated.
        """
        # check the shape of data
        X_cand = check_array(X_cand, force_all_finite=False)
        # check if a budget_manager is set
        self._validate_budget_manager()
        # check if a random state is set
        self._validate_random_state()

        use_random_sampler = self.random_state_.choice([0, 1], X_cand.shape[0],
                                                       p=[(1-self.v), self.v])

        for x, s, use_rand in zip(X_cand, sampled, use_random_sampler):
            if use_rand:
                self.random_sampler_.update(
                    x.reshape([1, -1]),
                    np.array([s]))
            else:
                self.variable_uncertainty_.update(
                    x.reshape([1, -1]),
                    np.array([s]))
        self.budget_manager_.update(sampled)
        return self
