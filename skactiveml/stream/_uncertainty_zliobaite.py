import numpy as np
from sklearn.base import clone
from sklearn.utils import check_array, check_consistent_length

from .budgetmanager import (
    FixedUncertaintyBudgetManager,
    VariableUncertaintyBudgetManager,
    SplitBudgetManager,
    RandomVariableUncertaintyBudgetManager,
)
from ..base import (
    BudgetManager,
    SingleAnnotatorStreamQueryStrategy,
    SkactivemlClassifier,
)
from ..utils import (
    check_type,
    call_func,
    check_budget_manager,
)


class UncertaintyZliobaite(SingleAnnotatorStreamQueryStrategy):
    """The UncertaintyZliobaite (Utility calculation in [1]) query strategy
    samples instances based on the classifiers uncertainty assessed based on
    the classifier's predictions. The instance is queried when the probability
    of the most likely class exceeds a threshold calculated based on the budget
    and the number of classes. It is used as the base class for uncertainty
    strategies provided by Zliobaite in [1].

    Parameters
    ----------
    budget : float, default=None
        The budget which models the budgeting constraint used in
        the stream-based active learning setting.

    budget_manager : BudgetManager, default=None
        The BudgetManager which models the budgeting constraint used in
        the stream-based active learning setting. if set to None,
        FixedUncertaintyBudgetManager will be used by default. The
        budget manager will be initialized based on the following conditions:
            If only a budget is given the default budget manager is initialized
            with the given budget.
            If only a budget manager is given use the budget manager.
            If both are not given the default budget manager with the
            default budget.
            If both are given and the budget differs from budgetmanager.budget
            a warning is thrown.

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
            budget_manager=None,
            budget=None,
            random_state=None,
    ):
        super().__init__(budget=budget, random_state=random_state)
        self.budget_manager = budget_manager

    def query(
        self,
        candidates,
        clf,
        X=None,
        y=None,
        sample_weight=None,
        fit_clf=False,
        return_utilities=False,
    ):
        """Ask the query strategy which instances in candidates to acquire.

        Please note that, when the decisions from this function may differ from
        the final sampling, simulate=True can be set, so that the query
        strategy can be updated later with update(...) with the final sampling.
        This is especially helpful when developing wrapper query strategies.

        Parameters
        ----------
        candidates : {array-like, sparse matrix} of shape
        (n_samples, n_features)
            The instances which may be queried. Sparse matrices are accepted
            only if they are supported by the base query strategy.
        clf : SkactivemlClassifier
            Model implementing the methods `fit` and `predict_freq`.
        X : array-like of shape (n_samples, n_features), optional
            Input samples used to fit the classifier.
        y : array-like of shape (n_samples), optional
            Labels of the input samples 'X'. There may be missing labels.
        sample_weight : array-like of shape (n_samples,), optional
            Sample weights for X, used to fit the clf.
        fit_clf : bool,
            If true, refit the classifier also requires X and y to be given.
        return_utilities : bool, optional
            If true, also return the utilities based on the query strategy.
            The default is False.

        Returns
        -------
        queried_indices : ndarray of shape (n_queried_instances,)
            The indices of instances in candidates which should be queried,
            with 0 <= n_queried_instances <= n_samples.

        utilities: ndarray of shape (n_samples,), optional
            The utilities based on the query strategy. Only provided if
            return_utilities is True.
        """
        (
            candidates,
            clf,
            X,
            y,
            sample_weight,
            fit_clf,
            return_utilities,
        ) = self._validate_data(
            candidates,
            clf=clf,
            X=X,
            y=y,
            sample_weight=sample_weight,
            fit_clf=fit_clf,
            return_utilities=return_utilities,
        )

        predict_proba = clf.predict_proba(candidates)
        utilities = np.max(predict_proba, axis=1)

        queried_indices = self.budget_manager_.query_by_utility(utilities)

        if return_utilities:
            return queried_indices, utilities
        else:
            return queried_indices

    def update(
            self, candidates, queried_indices, budget_manager_param_dict=None
    ):
        """Updates the budget manager and the count for seen and queried
        instances

        Parameters
        ----------
        candidates : {array-like, sparse matrix} of shape
        (n_samples, n_features)
            The instances which could be queried. Sparse matrices are accepted
            only if they are supported by the base query strategy.

        queried_indices : array-like of shape (n_samples,)
            Indicates which instances from candidates have been queried.

        budget_manager_param_dict : kwargs
            Optional kwargs for budget manager.

        Returns
        -------
        self : UncertaintyZliobaite
            The UncertaintyZliobaite returns itself, after it is updated.
        """
        # check if a budgetmanager is set
        if not hasattr(self, "budget_manager_"):
            check_type(
                self.budget_manager,
                "budget_manager_",
                BudgetManager,
                type(None),
            )
            self.budget_manager_ = check_budget_manager(
                self.budget,
                self.budget_manager,
                self._get_default_budget_manager(),
            )

        budget_manager_param_dict = (
            {}
            if budget_manager_param_dict is None
            else budget_manager_param_dict
        )

        call_func(
            self.budget_manager_.update,
            candidates=candidates,
            queried_indices=queried_indices,
            **budget_manager_param_dict
        )
        return self

    def _validate_data(
        self,
        candidates,
        clf,
        X,
        y,
        sample_weight,
        fit_clf,
        return_utilities,
        reset=True,
        **check_candidates_params
    ):
        """Validate input data and set or check the `n_features_in_` attribute.

        Parameters
        ----------
        candidates: array-like of shape (n_candidates, n_features)
            The instances which may be queried. Sparse matrices are accepted
            only if they are supported by the base query strategy.
        clf : SkactivemlClassifier
            Model implementing the methods `fit` and `predict_freq`.
        X : array-like of shape (n_samples, n_features)
            Input samples used to fit the classifier.
        y : array-like of shape (n_samples)
            Labels of the input samples 'X'. There may be missing labels.
        sample_weight : array-like of shape (n_samples,) (default=None)
            Sample weights for X, used to fit the clf.
        return_utilities : bool,
            If true, also return the utilities based on the query strategy.
        fit_clf : bool,
            If true, refit the classifier also requires X and y to be given.
        reset : bool, default=True
            Whether to reset the `n_features_in_` attribute.
            If False, the input will be checked for consistency with data
            provided when reset was last True.
        **check_candidates_params : kwargs
            Parameters passed to :func:`sklearn.utils.check_array`.

        Returns
        -------
        candidates: np.ndarray, shape (n_candidates, n_features)
            Checked candidate samples
        clf : SkactivemlClassifier
            Checked model implementing the methods `fit` and `predict_freq`.
        X: np.ndarray, shape (n_samples, n_features)
            Checked training samples
        y: np.ndarray, shape (n_candidates)
            Checked training labels
        sampling_weight: np.ndarray, shape (n_candidates)
            Checked training sample weight
        fit_clf : bool,
            Checked boolean value of `fit_clf`
        candidates: np.ndarray, shape (n_candidates, n_features)
            Checked candidate samples
        return_utilities : bool,
            Checked boolean value of `return_utilities`.
        """
        candidates, return_utilities = super()._validate_data(
            candidates,
            return_utilities,
            reset=reset,
            **check_candidates_params
        )
        self._validate_random_state()
        X, y, sample_weight = self._validate_X_y_sample_weight(
            X=X, y=y, sample_weight=sample_weight
        )
        clf = self._validate_clf(clf, X, y, sample_weight, fit_clf)

        # check if a budgetmanager is set
        if not hasattr(self, "budget_manager_"):
            check_type(
                self.budget_manager,
                "budget_manager_",
                BudgetManager,
                type(None),
            )
            self.budget_manager_ = check_budget_manager(
                self.budget,
                self.budget_manager,
                self._get_default_budget_manager(),
            )

        return candidates, clf, X, y, sample_weight, fit_clf, return_utilities

    def _validate_clf(self, clf, X, y, sample_weight, fit_clf):
        """Validate if clf is a valid SkactivemlClassifier. If clf is
        untrained, clf is trained using X, y and sample_weight.

        Parameters
        ----------
        clf : SkactivemlClassifier
            Model implementing the methods `fit` and `predict_freq`.
        X : array-like of shape (n_samples, n_features)
            Input samples used to fit the classifier.
        y : array-like of shape (n_samples)
            Labels of the input samples 'X'. There may be missing labels.
        sample_weight : array-like of shape (n_samples,) (default=None)
            Sample weights for X, used to fit the clf.
        fit_clf : bool,
            If true, refit the classifier also requires X and y to be given.
        Returns
        -------
        clf : SkactivemlClassifier
            Checked model implementing the methods `fit` and `predict_freq`.
        """
        # Check if the classifier and its arguments are valid.
        check_type(clf, "clf", SkactivemlClassifier)
        check_type(fit_clf, "fit_clf", bool)
        if fit_clf:
            clf = clone(clf).fit(X, y, sample_weight)
        return clf

    def _validate_X_y_sample_weight(self, X, y, sample_weight):
        """Validate if X, y and sample_weight are numeric and of equal length.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples used to fit the classifier.

        y : array-like of shape (n_samples)
            Labels of the input samples 'X'. There may be missing labels.

        sample_weight : array-like of shape (n_samples,) (default=None)
            Sample weights for X, used to fit the clf.

        Returns
        -------
        X : array-like of shape (n_samples, n_features)
            Checked Input samples.
        y : array-like of shape (n_samples)
            Checked Labels of the input samples 'X'. Converts y to a numpy
            array
        """
        if sample_weight is not None:
            sample_weight = np.array(sample_weight)
            check_consistent_length(sample_weight, y)
        if X is not None and y is not None:
            X = check_array(X)
            y = np.array(y)
            check_consistent_length(X, y)
        return X, y, sample_weight


class FixedUncertainty(UncertaintyZliobaite):
    """The FixedUncertainty (Fixed-Uncertainty in [1]) query strategy samples
    instances based on the classifiers uncertainty assessed based on the
    classifier's predictions. The instance is queried when the probability of
    the most likely class exceeds a threshold calculated based on the budget
    and the number of classes.

    Parameters
    ----------
    budget : float, default=None
        The budget which models the budgeting constraint used in
        the stream-based active learning setting.

    budgetmanager : BudgetManager, default=None
        The BudgetManager which models the budgeting constraint used in
        the stream-based active learning setting. if set to None,
        FixedUncertaintyBudgetManager will be used by default. The budget
        manager will be initialized based on the following conditions:
            If only a budget is given the default budget manager is initialized
            with the given budget.
            If only a budget manager is given use the budget manager.
            If both are not given the default budget manager with the
            default budget.
            If both are given and the budget differs from budget manager.budget
            a warning is thrown.

    random_state : int, RandomState instance, default=None
        Controls the randomness of the estimator.

    References
    ----------
    [1] Zliobaite, Indre & Bifet, Albert & Pfahringer, Bernhard & Holmes,
        Geoffrey. (2014). Active Learning With Drifting Streaming Data. Neural
        Networks and Learning Systems, IEEE Transactions on. 25. 27-39.

    """

    def _get_default_budget_manager(self):
        """Provide the budget manager that will be used as default.

        Returns
        -------
        budgetmanager : BudgetManager
            The BudgetManager that should be used by default.
        """
        return FixedUncertaintyBudgetManager


class VariableUncertainty(UncertaintyZliobaite):
    """The VariableUncertainty (Var-Uncertainty in [1]) query strategy samples
    instances based on the classifiers uncertainty assessed based on the
    classifier's predictions. The instance is queried when the probability of
    the most likely class exceeds a time-dependent threshold calculated based
    on the budget, the number of classes and the number of observed and
    acquired samples.

    Parameters
    ----------
    budget : float, default=None
        The budget which models the budgeting constraint used in
        the stream-based active learning setting.

    budgetmanager : BudgetManager, default=None
        The BudgetManager which models the budgeting constraint used in
        the stream-based active learning setting. if set to None,
        FixedUncertaintyBudgetManager will be used by default. The budget
        manager will be initialized based on the following conditions:
            If only a budget is given the default budgetmanager is initialized
            with the given budget.
            If only a budgetmanager is given use the budgetmanager.
            If both are not given the default budgetmanager with the
            default budget.
            If both are given and the budget differs from budgetmanager.budget
            a warning is thrown.

    random_state : int, RandomState instance, default=None
        Controls the randomness of the estimator.

    References
    ----------
    [1] Zliobaite, Indre & Bifet, Albert & Pfahringer, Bernhard & Holmes,
        Geoffrey. (2014). Active Learning With Drifting Streaming Data. Neural
        Networks and Learning Systems, IEEE Transactions on. 25. 27-39.

    """

    def _get_default_budget_manager(self):
        """Provide the budget manager that will be used as default.

        Returns
        -------
        budgetmanager : BudgetManager
            The BudgetManager that should be used by default.
        """
        return VariableUncertaintyBudgetManager


class RandomVariableUncertainty(UncertaintyZliobaite):
    """The RandomVariableUncertainty (Ran-Var-Uncertainty in [1]) query
    strategy samples instances based on the classifier's uncertainty assessed
    based on the classifier's predictions. The instance is queried when the
    probability of the most likely class exceeds a time-dependent threshold
    calculated based on the budget, the number of classes and the number of
    observed and acquired samples. To better adapt at change detection the
    threshold is multiplied by a random number generator with N(1,delta).

    Parameters
    ----------
    budget : float, default=None
        The budget which models the budgeting constraint used in
        the stream-based active learning setting.

    budgetmanager : BudgetManager, default=None
        The BudgetManager which models the budgeting constraint used in
        the stream-based active learning setting. if set to None,
        FixedUncertaintyBudgetManager will be used by default. The budget
        manager will be initialized based on the following conditions:
            If only a budget is given the default budgetmanager is initialized
            with the given budget.
            If only a budgetmanager is given use the budgetmanager.
            If both are not given the default budgetmanager with the
            default budget.
            If both are given and the budget differs from budgetmanager.budget
            a warning is thrown.

    random_state : int, RandomState instance, default=None
        Controls the randomness of the estimator.

    References
    ----------
    [1] Zliobaite, Indre & Bifet, Albert & Pfahringer, Bernhard & Holmes,
        Geoffrey. (2014). Active Learning With Drifting Streaming Data. Neural
        Networks and Learning Systems, IEEE Transactions on. 25. 27-39.

    """

    def _get_default_budget_manager(self):
        """Provide the budget manager that will be used as default.

        Returns
        -------
        budgetmanager : BudgetManager
            The BudgetManager that should be used by default.
        """
        return RandomVariableUncertaintyBudgetManager


class Split(UncertaintyZliobaite):
    """The Split [1] query strategy samples in 100*v% of instances randomly and
    in 100*(1-v)% of cases according to VariableUncertainty.

    Parameters
    ----------
    budget : float, default=None
        The budget which models the budgeting constraint used in
        the stream-based active learning setting.

    budgetmanager : BudgetManager, default=None
        The BudgetManager which models the budgeting constraint used in
        the stream-based active learning setting. if set to None,
        FixedUncertaintyBudgetManager will be used by default. The budget
        manager will
        be initialized based on the following conditions:
            If only a budget is given the default budget manager is initialized
            with the given budget.
            If only a budgetmanager is given use the budgetmanager.
            If both are not given the default budgetmanager with the
            default budget.
            If both are given and the budget differs from budgetmanager.budget
            a warning is thrown.

    random_state : int, RandomState instance, default=None
        Controls the randomness of the estimator.

    References
    ----------
    [1] Zliobaite, Indre & Bifet, Albert & Pfahringer, Bernhard & Holmes,
        Geoffrey. (2014). Active Learning With Drifting Streaming Data. Neural
        Networks and Learning Systems, IEEE Transactions on. 25. 27-39.

    """

    def _get_default_budget_manager(self):
        return SplitBudgetManager
