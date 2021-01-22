import numpy as np

from sklearn.utils import check_array

from sklearn.base import is_classifier, clone

from ..base import SingleAnnotStreamBasedQueryStrategy

from ..classifier import PWC

from ._random import RandomSampler

import copy

from .budget_manager import EstimatedBudget, FixedUncertaintyBudget, VarUncertaintyBudget, SplitBudget


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
    def __init__(self, clf=None, budget_manager=FixedUncertaintyBudget(),
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
        # check if a random state is set
        self._validate_random_state()
        # check if a budget_manager is set
        self._validate_budget_manager()
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
        utilities = np.max(predict_proba, axis=1)
        num_classes = predict_proba.shape[1]
        
        sampled_indices = self.budget_manager_.sample(utilities, num_classes,
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
    def __init__(self, clf=None, budget_manager=VarUncertaintyBudget(),
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
        # check if a random state is set
        self._validate_random_state()
        # check if a budget_manager is set
        self._validate_budget_manager()
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
        utilities = np.max(predict_proba, axis=1)
        
        sampled_indices = []
        
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
        self.budget_manager_.update(sampled)
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
    def __init__(self, clf=None, 
                 budget_manager=SplitBudget(),
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
        # check if a random state is set
        self._validate_random_state()
        # check if a budget_manager is set
        self._validate_budget_manager()
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

        predict_proba = clf.predict_proba(X_cand)
        utilities = np.max(predict_proba, axis=1)
        sampled_indices = []
        
        sampled_indices = self.budget_manager_.sample(utilities, 
                                                      simulate=simulate)

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

        self.budget_manager_.update(sampled)
        return self
