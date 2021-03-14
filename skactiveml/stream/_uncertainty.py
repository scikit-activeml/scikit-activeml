import numpy as np

from sklearn.utils import check_array

from sklearn.base import is_classifier, clone

from ..base import SingleAnnotStreamBasedQueryStrategy

from ..classifier import PWC

from .budget_manager import (
    FixedUncertaintyBudget,
    VarUncertaintyBudget,
    SplitBudget,
)


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

    def __init__(
        self,
        clf=None,
        budget_manager=FixedUncertaintyBudget(),
        random_state=None,
    ):
        super().__init__(
            budget_manager=budget_manager, random_state=random_state
        )
        self.clf = clf

    def query(
        self, X_cand, X, y, return_utilities=False, simulate=False, **kwargs
    ):
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
        X : array-like, shape (n_samples, n_features)
            Input samples used to fit the classifier.
        y : array-like, shape (n_samples)
            Labels of the input samples 'X'. There may be missing labels.
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
        self._validate_data(X_cand, return_utilities, X, y, simulate)

        predict_proba = self.clf_.predict_proba(X_cand)
        utilities = np.max(predict_proba, axis=1)

        sampled_indices = self.budget_manager_.sample(
            utilities, simulate=simulate
        )

        if return_utilities:
            return sampled_indices, utilities
        else:
            return sampled_indices

    def update(self, X_cand, sampled, budget_manager_kwargs={}, **kwargs):
        """Updates the budget manager and the count for seen and sampled
        instances

        Parameters
        ----------
        X_cand : {array-like, sparse matrix} of shape (n_samples, n_features)
            The instances which could be sampled. Sparse matrices are accepted
            only if they are supported by the base query strategy.

        sampled : array-like
            Indicates which instances from X_cand have been sampled.

        budget_manager_kwargs : kwargs
            optional data-dependent parameters for budget_manager

        Returns
        -------
        self : FixedUncertainty
            The FixedUncertainty returns itself, after it is updated.
        """
        # check if a budget_manager is set
        self._validate_budget_manager()
        self.budget_manager_.update(sampled, **budget_manager_kwargs)
        return self

    def _validate_data(
        self,
        X_cand,
        return_utilities,
        X,
        y,
        simulate,
        reset=True,
        **check_X_cand_params
    ):
        """Validate input data and set or check the `n_features_in_` attribute.

        Parameters
        ----------
        X_cand: array-like, shape (n_candidates, n_features)
            Candidate samples.
        return_utilities : bool,
            If true, also return the utilities based on the query strategy.
        X : array-like, shape (n_samples, n_features)
            Input samples used to fit the classifier.
        y : array-like, shape (n_samples)
            Labels of the input samples 'X'. There may be missing labels.
        reset : bool, default=True
            Whether to reset the `n_features_in_` attribute.
            If False, the input will be checked for consistency with data
            provided when reset was last True.
        **check_X_cand_params : kwargs
            Parameters passed to :func:`sklearn.utils.check_array`.
        simulate : bool,
            If True, the internal state of the query strategy before and after
            the query is the same.

        Returns
        -------
        X_cand: np.ndarray, shape (n_candidates, n_features)
            Checked candidate samples
        return_utilities : bool,
            Checked boolean value of `return_utilities`.
        X : array-like, shape (n_samples, n_features)
            Checked Input samples.
        y : array-like, shape (n_samples)
            Checked Labels of the input samples 'X'.  
        simulate : bool,
            Checked boolean value of `simulate`.    
        """
        X_cand, return_utilities, simulate = super()._validate_data(
            X_cand,
            return_utilities,
            simulate,
            reset=reset,
            **check_X_cand_params
        )

        self._validate_clf(X, y)
        self._validate_random_state()

        return X_cand, return_utilities, X, y, simulate

    def _validate_clf(self, X, y):
        """Validate if clf is a classifier or create a new clf and fit X and y.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input samples used to fit the classifier.

        y : array-like, shape (n_samples)
            Labels of the input samples 'X'. There may be missing labels. 
        """
        # check if clf is a classifier
        if X is not None and y is not None:
            if self.clf is None:
                self.clf_ = PWC(
                    random_state=self.random_state_.randint(2 ** 31 - 1)
                )
            elif is_classifier(self.clf):
                self.clf_ = clone(self.clf)
            else:
                raise TypeError(
                    "clf is not a classifier. Please refer to "
                    + "sklearn.base.is_classifier"
                )
            self.clf_.fit(X, y)
            # check if y is not multi dimensinal
            if isinstance(y, np.ndarray):
                if y.ndim > 1:
                    raise ValueError("{} is not a valid Value for y")
        else:
            self.clf_ = self.clf


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

    def __init__(
        self,
        clf=None,
        budget_manager=VarUncertaintyBudget(),
        random_state=None,
    ):
        super().__init__(
            budget_manager=budget_manager, random_state=random_state
        )
        self.clf = clf

    def query(
        self, X_cand, X, y, return_utilities=False, simulate=False, **kwargs
    ):
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

        X : array-like, shape (n_samples, n_features)
            Input samples used to fit the classifier.

        y : array-like, shape (n_samples)
            Labels of the input samples 'X'. There may be missing labels.

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
        self._validate_data(X_cand, return_utilities, X, y, simulate)

        predict_proba = self.clf_.predict_proba(X_cand)
        utilities = np.max(predict_proba, axis=1)

        sampled_indices = []

        sampled_indices = self.budget_manager_.sample(
            utilities, simulate=simulate
        )

        if return_utilities:
            return sampled_indices, utilities
        else:
            return sampled_indices

    def update(self, X_cand, sampled, budget_manager_kwargs={}, **kwargs):
        """Updates the budget manager and the count for seen and sampled
        instances

        Parameters
        ----------
        X_cand : {array-like, sparse matrix} of shape (n_samples, n_features)
            The instances which could be sampled. Sparse matrices are accepted
            only if they are supported by the base query strategy.
    
        sampled : array-like
            Indicates which instances from X_cand have been sampled.

        budget_manager_kwargs : kwargs
            optional data-dependent parameters for budget_manager

        Returns
        -------
        self : VariableUncertainty
            The VariableUncertainty returns itself, after it is updated.
        """
        # check if a budget_manager is set
        self._validate_budget_manager()
        self.budget_manager_.update(sampled, **budget_manager_kwargs)
        return self

    def _validate_data(
        self,
        X_cand,
        return_utilities,
        X,
        y,
        simulate,
        reset=True,
        **check_X_cand_params
    ):
        """Validate input data and set or check the `n_features_in_` attribute.

        Parameters
        ----------
        X_cand: array-like, shape (n_candidates, n_features)
            Candidate samples.
        return_utilities : bool,
            If true, also return the utilities based on the query strategy.
        X : array-like, shape (n_samples, n_features)
            Input samples used to fit the classifier.
        y : array-like, shape (n_samples)
            Labels of the input samples 'X'. There may be missing labels.
        reset : bool, default=True
            Whether to reset the `n_features_in_` attribute.
            If False, the input will be checked for consistency with data
            provided when reset was last True.
        **check_X_cand_params : kwargs
            Parameters passed to :func:`sklearn.utils.check_array`.
        simulate : bool,
            If True, the internal state of the query strategy before and after
            the query is the same.

        Returns
        -------
        X_cand: np.ndarray, shape (n_candidates, n_features)
            Checked candidate samples
        return_utilities : bool,
            Checked boolean value of `return_utilities`.
        X : array-like, shape (n_samples, n_features)
            Checked Input samples.
        y : array-like, shape (n_samples)
            Checked Labels of the input samples 'X'.  
        simulate : bool,
            Checked boolean value of `simulate`.
        """
        X_cand, return_utilities, simulate = super()._validate_data(
            X_cand,
            return_utilities,
            simulate,
            reset=reset,
            **check_X_cand_params
        )

        self._validate_clf(X, y)
        self._validate_random_state()

        return X_cand, return_utilities, X, y, simulate

    def _validate_clf(self, X, y):
        """Validate if clf is a classifier or create a new clf and fit X and y.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input samples used to fit the classifier.

        y : array-like, shape (n_samples)
            Labels of the input samples 'X'. There may be missing labels. 
        """
        # check if clf is a classifier
        if X is not None and y is not None:
            if self.clf is None:
                self.clf_ = PWC(
                    random_state=self.random_state_.randint(2 ** 31 - 1)
                )
            elif is_classifier(self.clf):
                self.clf_ = clone(self.clf)
            else:
                raise TypeError(
                    "clf is not a classifier. Please refer to "
                    + "sklearn.base.is_classifier"
                )
            self.clf_.fit(X, y)
            # check if y is not multi dimensinal
            if isinstance(y, np.ndarray):
                if y.ndim > 1:
                    raise ValueError("{} is not a valid Value for y")
        else:
            self.clf_ = self.clf


class Split(SingleAnnotStreamBasedQueryStrategy):
    """The Split [1] query strategy samples in 100*v% of instances randomly and
    in 100*(1-v)% of cases according to VarUncertainty.

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

    def __init__(
        self, clf=None, budget_manager=SplitBudget(), random_state=None
    ):
        super().__init__(
            budget_manager=budget_manager, random_state=random_state
        )
        self.clf = clf

    def query(
        self, X_cand, X, y, return_utilities=False, simulate=False, **kwargs
    ):
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

        X : array-like, shape (n_samples, n_features)
            Input samples used to fit the classifier.

        y : array-like, shape (n_samples)
            Labels of the input samples 'X'. There may be missing labels.

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
        self._validate_data(X_cand, return_utilities, X, y, simulate)

        predict_proba = self.clf_.predict_proba(X_cand)
        utilities = np.max(predict_proba, axis=1)
        sampled_indices = []

        sampled_indices = self.budget_manager_.sample(
            utilities, simulate=simulate
        )

        if return_utilities:
            return sampled_indices, utilities
        else:
            return sampled_indices

    def update(
        self, X_cand, sampled, X, y, budget_manager_kwargs={}, **kwargs
    ):
        """Updates the budget manager and the count for seen and sampled
        instances

        Parameters
        ----------
        X_cand : {array-like, sparse matrix} of shape (n_samples, n_features)
            The instances which could be sampled. Sparse matrices are accepted
            only if they are supported by the base query strategy.

        X : array-like, shape (n_samples, n_features)
            Input samples used to fit the classifier.

        y : array-like, shape (n_samples)
            Labels of the input samples 'X'. There may be missing labels.

        sampled : array-like
            Indicates which instances from X_cand have been sampled.
        
        budget_manager_kwargs : kwargs
            optional data-dependent parameters for budget_manager

        Returns
        -------
        self : VariableUncertainty
            The VariableUncertainty returns itself, after it is updated.
        """
        # Check the shape of data
        X_cand = check_array(X_cand, force_all_finite=False)
        # Check if a budget_manager is set
        self._validate_budget_manager()
        # Check if a random state is set
        self._validate_random_state()

        self.budget_manager_.update(sampled, **budget_manager_kwargs)
        return self

    def _validate_data(
        self,
        X_cand,
        return_utilities,
        X,
        y,
        simulate,
        reset=True,
        **check_X_cand_params
    ):
        """Validate input data and set or check the `n_features_in_` attribute.

        Parameters
        ----------
        X_cand: array-like, shape (n_candidates, n_features)
            Candidate samples.
        return_utilities : bool,
            If true, also return the utilities based on the query strategy.
        X : array-like, shape (n_samples, n_features)
            Input samples used to fit the classifier.
        y : array-like, shape (n_samples)
            Labels of the input samples 'X'. There may be missing labels.
        reset : bool, default=True
            Whether to reset the `n_features_in_` attribute.
            If False, the input will be checked for consistency with data
            provided when reset was last True.
        **check_X_cand_params : kwargs
            Parameters passed to :func:`sklearn.utils.check_array`.
        simulate : bool,
            If True, the internal state of the query strategy before and after
            the query is the same. This should only be used to prevent the
            query strategy from adapting itself. Note, that this is propagated
            to the budget_manager, as well.

        Returns
        -------
        X_cand: np.ndarray, shape (n_candidates, n_features)
            Checked candidate samples
        return_utilities : bool,
            Checked boolean value of `return_utilities`.
        X : array-like, shape (n_samples, n_features)
            Checked Input samples.
        y : array-like, shape (n_samples)
            Checked Labels of the input samples 'X'.  
        simulate : bool,
            Checked boolean value of `simulate`.
        """
        X_cand, return_utilities, simulate = super()._validate_data(
            X_cand,
            return_utilities,
            simulate,
            reset=reset,
            **check_X_cand_params
        )

        self._validate_clf(X, y)
        self._validate_random_state()
        self._validate_budget_manager()

        return X_cand, return_utilities, X, y, simulate

    def _validate_clf(self, X, y):
        """Validate if clf is a classifier or create a new clf and fit X and y.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input samples used to fit the classifier.

        y : array-like, shape (n_samples)
            Labels of the input samples 'X'. There may be missing labels. 
        """
        # check if clf is a classifier
        if X is not None and y is not None:
            if self.clf is None:
                self.clf_ = PWC(
                    random_state=self.random_state_.randint(2 ** 31 - 1)
                )
            elif is_classifier(self.clf):
                self.clf_ = clone(self.clf)
            else:
                raise TypeError(
                    "clf is not a classifier. Please refer to "
                    + "sklearn.base.is_classifier"
                )
            self.clf_.fit(X, y)
            # check if y is not multi dimensinal
            if isinstance(y, np.ndarray):
                if y.ndim > 1:
                    raise ValueError("{} is not a valid Value for y")
        else:
            self.clf_ = self.clf
