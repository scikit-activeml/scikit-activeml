import numpy as np

from sklearn.utils import check_array

from ..base import StreamBasedQueryStrategy

from ._random import RandomSampler

import copy


class FixedUncertainty(StreamBasedQueryStrategy):
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
    def __init__(self, clf, budget_manager, random_state=None):
        super().__init__(budget_manager=budget_manager,
                         random_state=random_state)
        self.clf = clf

    def query(self, X_cand, return_utilities=False, simulate=False, **kwargs):
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
        X_cand = check_array(X_cand, force_all_finite=False)

        predict_proba = self.clf.predict_proba(X_cand)
        y_hat = np.max(predict_proba, axis=1)
        num_classes = predict_proba.shape[1]
        theta = 1/num_classes + self.budget_manager.budget*(1-1/num_classes)
        utilities = y_hat <= theta

        sampled_indices = self.budget_manager.sample(utilities,
                                                     simulate=simulate)

        if return_utilities:
            return sampled_indices, utilities
        else:
            return sampled_indices

    def update(self, X_cand, sampled, *args, **kwargs):
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
        self.budget_manager.update(sampled)
        return self


class VariableUncertainty(StreamBasedQueryStrategy):
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
    def __init__(self, clf, budget_manager, theta=1.0, s=0.01,
                 random_state=None):
        super().__init__(budget_manager=budget_manager,
                         random_state=random_state)
        self.clf = clf
        self.theta = theta
        self.s = s

    def query(self, X_cand, return_utilities=False, simulate=False, **kwargs):
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
        X_cand = check_array(X_cand, force_all_finite=False)

        predict_proba = self.clf.predict_proba(X_cand)
        y_hat = np.max(predict_proba, axis=1)

        tmp_theta = self.theta

        utilities = []
        sampled_indices = []

        for y in y_hat:
            utilities.append(y < tmp_theta)
            sampled, budget_left = self.budget_manager.sample(
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
            self.theta = tmp_theta

        sampled_indices = self.budget_manager.sample(utilities,
                                                     simulate=simulate)

        if return_utilities:
            return sampled_indices, utilities
        else:
            return sampled_indices

    def update(self, X_cand, sampled, *args, **kwargs):
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
        self.budget_manager.update(sampled)
        budget_left = kwargs.get('budget_left', None)
        for i, s in enumerate(sampled):
            if budget_left is None:
                if sampled[-1]:
                    self.theta = self.theta * (1-self.s)
                else:
                    self.theta = self.theta * (1+self.s)
        return self


class Split(StreamBasedQueryStrategy):
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
    def __init__(self, clf, budget_manager, v=0.1, theta=1.0, s=0.01,
                 random_state=None):
        super().__init__(budget_manager=budget_manager,
                         random_state=random_state)
        self.clf = clf
        self.s = s
        self.random_sampler = RandomSampler(
            self.budget_manager,
            random_state=self.random_state.tomaxint()
        )
        self.variable_uncertainty = VariableUncertainty(
            clf,
            self.budget_manager,
            theta=theta,
            s=s,
            random_state=self.random_state.tomaxint()
        )

    def query(self, X_cand, return_utilities=False, simulate=False, **kwargs):
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
        X_cand = check_array(X_cand, force_all_finite=False)

        utilities = []
        sampled_indices = []

        use_random_sampler = self.random_state.choice([0, 1], X_cand.shape[0],
                                                      p=[(1-self.s), self.s])

        tmp_budget_manager = copy.copy(self.budget_manager)
        tmp_random_sampler = copy.copy(self.random_sampler)
        tmp_random_sampler.budget_manager = tmp_budget_manager
        tmp_var_uncertainty = copy.copy(self.variable_uncertainty)
        tmp_var_uncertainty.budget_manager = tmp_budget_manager

        merged_sampled_indices = []
        merged_utilities = []

        for x, use_rand in zip(X_cand, use_random_sampler):
            if use_rand:
                sampled_indices, utilities = tmp_random_sampler.query(
                    x.reshape([1, -1]),
                    return_utilities=True,
                    simulate=False
                )
            else:
                sampled_indices, utilities = tmp_var_uncertainty.query(
                    x.reshape([1, -1]),
                    return_utilities=True,
                    simulate=False
                )
            merged_sampled_indices.extend(sampled_indices)
            merged_utilities.extend(utilities)

        if not simulate:
            self.budget_manager = tmp_budget_manager
            self.random_sampler = tmp_random_sampler
            self.variable_uncertainty = tmp_var_uncertainty

        if return_utilities:
            return sampled_indices, utilities
        else:
            return sampled_indices

    def update(self, X_cand, sampled, *args, **kwargs):
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
        use_random_sampler = self.random_state.choice([0, 1], X_cand.shape[0],
                                                      p=[(1-self.s), self.s])

        for x, s, use_rand in zip(X_cand, sampled, use_random_sampler):
            if use_rand:
                sampled_indices, utilities = self.random_sampler.update(
                    x.reshape([1, -1]),
                    [s]
                )
            else:
                sampled_indices, utilities = self.variable_uncertainty.update(
                    x.reshape([1, -1]),
                    [s]
                )

        return self
