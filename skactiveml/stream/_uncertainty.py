import numpy as np

from sklearn.utils import check_array, check_consistent_length

from ..base import SingleAnnotStreamBasedQueryStrategy, SkactivemlClassifier
from ..utils import fit_if_not_fitted, check_type, call_func

from .budget_manager import (
    FixedUncertaintyBudget,
    VariableUncertaintyBudget,
    SplitBudget,
)


class FixedUncertainty(SingleAnnotStreamBasedQueryStrategy):
    """The FixedUncertainty (Fixed-Uncertainty in [1]) query strategy samples
    instances based on the classifiers uncertainty assessed based on the
    classifier's predictions. The instance is queried when the probability of
    the most likely class exceeds a threshold calculated based on the budget
    and the number of classes.

    Parameters
    ----------
    budget_manager : BudgetManager
        The BudgetManager which models the budgeting constraint used in
        the stream-based active learning setting. if set to None,
        FixedUncertaintyBudget will be used by default.

    random_state : int, RandomState instance, default=None
        Controls the randomness of the estimator.

    References
    ----------
    [1] Zliobaite, Indre & Bifet, Albert & Pfahringer, Bernhard & Holmes,
        Geoffrey. (2014). Active Learning With Drifting Streaming Data. Neural
        Networks and Learning Systems, IEEE Transactions on. 25. 27-39.

    """

    def __init__(
        self, budget_manager=None, random_state=None,
    ):
        super().__init__(
            budget_manager=budget_manager, random_state=random_state
        )

    def query(
        self,
        X_cand,
        clf,
        X=None,
        y=None,
        sample_weight=None,
        return_utilities=False,
    ):
        """Ask the query strategy which instances in X_cand to acquire.

        Please note that, when the decisions from this function may differ from
        the final sampling, simulate=True can be set, so that the query
        strategy can be updated later with update(...) with the final sampling.
        This is especially helpful when developing wrapper query strategies.

        Parameters
        ----------
        X_cand : {array-like, sparse matrix} of shape (n_samples, n_features)
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
        return_utilities : bool, optional
            If true, also return the utilities based on the query strategy.
            The default is False.

        Returns
        -------
        queried_indices : ndarray of shape (n_queried_instances,)
            The indices of instances in X_cand which should be queried, with
            0 <= n_queried_instances <= n_samples.

        utilities: ndarray of shape (n_samples,), optional
            The utilities based on the query strategy. Only provided if
            return_utilities is True.
        """
        (
            X_cand,
            clf,
            X,
            y,
            sample_weight,
            return_utilities,
        ) = self._validate_data(
            X_cand,
            clf=clf,
            X=X,
            y=y,
            sample_weight=sample_weight,
            return_utilities=return_utilities,
        )

        predict_proba = clf.predict_proba(X_cand)
        utilities = np.max(predict_proba, axis=1)

        queried_indices = self.budget_manager_.query_by_utility(utilities)

        if return_utilities:
            return queried_indices, utilities
        else:
            return queried_indices

    def update(self, X_cand, queried_indices, budget_manager_param_dict=None):
        """Updates the budget manager and the count for seen and queried
        instances

        Parameters
        ----------
        X_cand : {array-like, sparse matrix} of shape (n_samples, n_features)
            The instances which could be queried. Sparse matrices are accepted
            only if they are supported by the base query strategy.

        queried_indices : array-like of shape (n_samples,)
            Indicates which instances from X_cand have been queried.

        budget_manager_param_dict : kwargs
            Optional kwargs for budget_manager.

        Returns
        -------
        self : FixedUncertainty
            The FixedUncertainty returns itself, after it is updated.
        """
        # check if a budget_manager is set
        self._validate_budget_manager()
        budget_manager_param_dict = ({} if budget_manager_param_dict is None
                                     else budget_manager_param_dict)
        call_func(
            self.budget_manager_.update,
            X_cand=X_cand,
            queried_indices=queried_indices,
            **budget_manager_param_dict
        )
        return self

    def get_default_budget_manager(self):
        """Provide the budget manager that will be used as default.

        Returns
        -------
        budget_manager : BudgetManager
            The BudgetManager that should be used by default.
        """
        return FixedUncertaintyBudget()

    def _validate_data(
        self,
        X_cand,
        clf,
        X,
        y,
        sample_weight,
        return_utilities,
        reset=True,
        **check_X_cand_params
    ):
        """Validate input data and set or check the `n_features_in_` attribute.

        Parameters
        ----------
        X_cand: array-like of shape (n_candidates, n_features)
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
        clf : SkactivemlClassifier
            Checked model implementing the methods `fit` and `predict_freq`.
        X: np.ndarray, shape (n_samples, n_features)
            Checked training samples
        y: np.ndarray, shape (n_candidates)
            Checked training labels
        sampling_weight: np.ndarray, shape (n_candidates)
            Checked training sample weight
        X_cand: np.ndarray, shape (n_candidates, n_features)
            Checked candidate samples
        return_utilities : bool,
            Checked boolean value of `return_utilities`.
        """
        X_cand, return_utilities = super()._validate_data(
            X_cand, return_utilities, reset=reset, **check_X_cand_params
        )
        self._validate_random_state()
        X, y, sample_weight = _validate_X_y_sample_weight(
            X=X, y=y, sample_weight=sample_weight
        )
        clf = self._validate_clf(clf, X, y, sample_weight)

        return X_cand, clf, X, y, sample_weight, return_utilities

    def _validate_clf(self, clf, X, y, sample_weight):
        """Validate if clf is a valid SkactivemlClassifier. If clf is
        untrained, clf is trained using X, y and sample_weight.

        Parameters
        ----------
        clf : SkactivemlClassifier
            Model implementing the methods `fit` and `predict_freq`.
        Returns
        -------
        clf : SkactivemlClassifier
            Checked model implementing the methods `fit` and `predict_freq`.
        """
        # Check if the classifier and its arguments are valid.
        check_type(clf, SkactivemlClassifier, "clf")
        return fit_if_not_fitted(clf, X, y, sample_weight)


class VariableUncertainty(SingleAnnotStreamBasedQueryStrategy):
    """The VariableUncertainty (Var-Uncertainty in [1]) query strategy samples
    instances based on the classifiers uncertainty assessed based on the
    classifier's predictions. The instance is queried when the probability of
    the most likely class exceeds a time-dependent threshold calculated based
    on the budget, the number of classes and the number of observed and
    acquired samples.

    Parameters
    ----------
    budget_manager : BudgetManager
        The BudgetManager which models the budgeting constraint used in
        the stream-based active learning setting. if set to None,
        VariableUncertaintyBudget will be used by default.

    random_state : int, RandomState instance, default=None
        Controls the randomness of the estimator.

    References
    ----------
    [1] Zliobaite, Indre & Bifet, Albert & Pfahringer, Bernhard & Holmes,
        Geoffrey. (2014). Active Learning With Drifting Streaming Data. Neural
        Networks and Learning Systems, IEEE Transactions on. 25. 27-39.

    """

    def __init__(
        self, budget_manager=None, random_state=None,
    ):
        super().__init__(
            budget_manager=budget_manager, random_state=random_state
        )

    def query(
        self,
        X_cand,
        clf,
        X=None,
        y=None,
        sample_weight=None,
        return_utilities=False,
    ):
        """Ask the query strategy which instances in X_cand to acquire.

        Please note that, when the decisions from this function may differ from
        the final sampling, simulate=True can be set, so that the query
        strategy can be updated later with update(...) with the final sampling.
        This is especially helpful when developing wrapper query strategies.

        Parameters
        ----------
        X_cand : {array-like, sparse matrix} of shape (n_samples, n_features)
            The instances which may be queried. Sparse matrices are accepted
            only if they are supported by the base query strategy.

        clf : SkactivemlClassifier
            Model implementing the methods `fit` and `predict_freq`.

        X : array-like of shape (n_samples, n_features), optional
            Input samples used to fit the classifier.

        y : array-like of shape (n_samples), optional
            Labels of the input samples 'X'. There may be missing labels.

        sample_weight : array-like of shape (n_samples,)
            Sample weights for X, used to fit the clf.

        return_utilities : bool, optional
            If true, also return the utilities based on the query strategy.
            The default is False.

        Returns
        -------
        queried_indices : ndarray of shape (n_queried_instances,)
            The indices of instances in X_cand which should be queried, with
            0 <= n_queried_instances <= n_samples.

        utilities: ndarray of shape (n_samples,), optional
            The utilities based on the query strategy. Only provided if
            return_utilities is True.
        """
        (
            X_cand,
            clf,
            X,
            y,
            sample_weight,
            return_utilities,
        ) = self._validate_data(
            X_cand,
            clf=clf,
            X=X,
            y=y,
            sample_weight=sample_weight,
            return_utilities=return_utilities,
        )

        predict_proba = clf.predict_proba(X_cand)
        utilities = np.max(predict_proba, axis=1)

        queried_indices = self.budget_manager_.query_by_utility(utilities)

        if return_utilities:
            return queried_indices, utilities
        else:
            return queried_indices

    def update(self, X_cand, queried_indices, budget_manager_param_dict=None):
        """Updates the budget manager and the count for seen and queried
        instances

        Parameters
        ----------
        X_cand : {array-like, sparse matrix} of shape (n_samples, n_features)
            The instances which could be queried. Sparse matrices are accepted
            only if they are supported by the base query strategy.

        queried_indices : array-like of shape (n_samples,)
            Indicates which instances from X_cand have been queried.

        budget_manager_param_dict : kwargs
            Optional kwargs for budget_manager.

        Returns
        -------
        self : VariableUncertainty
            The VariableUncertainty returns itself, after it is updated.
        """
        # check if a budget_manager is set
        self._validate_budget_manager()
        budget_manager_param_dict = ({} if budget_manager_param_dict is None
                                     else budget_manager_param_dict)
        call_func(
            self.budget_manager_.update,
            X_cand=X_cand,
            queried_indices=queried_indices,
            **budget_manager_param_dict
        )
        return self

    def get_default_budget_manager(self):
        """Provide the budget manager that will be used as default.

        Returns
        -------
        budget_manager : BudgetManager
            The BudgetManager that should be used by default.
        """
        return VariableUncertaintyBudget()

    def _validate_data(
        self,
        X_cand,
        clf,
        X,
        y,
        sample_weight,
        return_utilities,
        reset=True,
        **check_X_cand_params
    ):
        """Validate input data and set or check the `n_features_in_` attribute.

        Parameters
        ----------
        X_cand: array-like of shape (n_candidates, n_features)
            The instances which may be queried. Sparse matrices are accepted
            only if they are supported by the base query strategy.
        X : array-like of shape (n_samples, n_features)
            Input samples used to fit the classifier.
        clf : SkactivemlClassifier
            Model implementing the methods `fit` and `predict_freq`.
        y : array-like of shape (n_samples)
            Labels of the input samples 'X'. There may be missing labels.
        sample_weight : array-like of shape (n_samples,) (default=None)
            Sample weights for X, used to fit the clf.
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
        clf : SkactivemlClassifier
            Checked model implementing the methods `fit` and `predict_freq`.
        X: np.ndarray, shape (n_samples, n_features)
            Checked training samples
        y: np.ndarray, shape (n_candidates)
            Checked training labels
        sampling_weight: np.ndarray, shape (n_candidates)
            Checked training sample weight
        X_cand: np.ndarray, shape (n_candidates, n_features)
            Checked candidate samples
        return_utilities : bool,
            Checked boolean value of `return_utilities`.
        """
        X_cand, return_utilities = super()._validate_data(
            X_cand, return_utilities, reset=reset, **check_X_cand_params
        )

        X, y, sample_weight = _validate_X_y_sample_weight(
            X=X, y=y, sample_weight=sample_weight
        )
        clf = self._validate_clf(clf, X, y, sample_weight)
        self._validate_random_state()

        return X_cand, clf, X, y, sample_weight, return_utilities

    def _validate_clf(self, clf, X, y, sample_weight):
        """Validate if clf is a valid SkactivemlClassifier. If clf is
        untrained, clf is trained using X, y and sample_weight.

        Parameters
        ----------
        clf : SkactivemlClassifier
            Model implementing the methods `fit` and `predict_freq`.
        Returns
        -------
        clf : SkactivemlClassifier
            Checked model implementing the methods `fit` and `predict_freq`.
        """
        # Check if the classifier and its arguments are valid.
        check_type(clf, SkactivemlClassifier, "clf")
        return fit_if_not_fitted(clf, X, y, sample_weight)


class Split(SingleAnnotStreamBasedQueryStrategy):
    """The Split [1] query strategy samples in 100*v% of instances randomly and
    in 100*(1-v)% of cases according to VariableUncertainty.

    Parameters
    ----------
    budget_manager : BudgetManager
        The BudgetManager which models the budgeting constraint used in
        the stream-based active learning setting. if set to None, SplitBudget
        will be used by default.

    random_state : int, RandomState instance, default=None
        Controls the randomness of the estimator.

    References
    ----------
    [1] Zliobaite, Indre & Bifet, Albert & Pfahringer, Bernhard & Holmes,
        Geoffrey. (2014). Active Learning With Drifting Streaming Data. Neural
        Networks and Learning Systems, IEEE Transactions on. 25. 27-39.

    """

    def __init__(self, budget_manager=None, random_state=None):
        super().__init__(
            budget_manager=budget_manager, random_state=random_state
        )

    def query(
        self,
        X_cand,
        clf,
        X=None,
        y=None,
        sample_weight=None,
        return_utilities=False,
    ):
        """Ask the query strategy which instances in X_cand to acquire.

        Please note that, when the decisions from this function may differ from
        the final sampling, simulate=True can be set, so that the query
        strategy can be updated later with update(...) with the final sampling.
        This is especially helpful when developing wrapper query strategies.

        Parameters
        ----------
        X_cand : {array-like, sparse matrix} of shape (n_samples, n_features)
            The instances which may be queried. Sparse matrices are accepted
            only if they are supported by the base query strategy.

        clf : SkactivemlClassifier
            Model implementing the methods `fit` and `predict_freq`.

        X : array-like of shape (n_samples, n_features), optional
            Input samples used to fit the classifier.

        y : array-like of shape (n_samples), optional
            Labels of the input samples 'X'. There may be missing labels.


        sample_weight : array-like of shape (n_samples,)
            Sample weights for X, used to fit the clf.

        return_utilities : bool, optional
            If true, also return the utilities based on the query strategy.
            The default is False.

        Returns
        -------
        queried_indices : ndarray of shape (n_queried_instances,)
            The indices of instances in X_cand which should be queried, with
            0 <= n_queried_instances <= n_samples.

        utilities: ndarray of shape (n_samples,), optional
            The utilities based on the query strategy. Only provided if
            return_utilities is True.
        """
        (
            X_cand,
            clf,
            X,
            y,
            sample_weight,
            return_utilities,
        ) = self._validate_data(
            X_cand,
            clf=clf,
            X=X,
            y=y,
            sample_weight=sample_weight,
            return_utilities=return_utilities,
        )

        predict_proba = clf.predict_proba(X_cand)
        utilities = np.max(predict_proba, axis=1)

        queried_indices = self.budget_manager_.query_by_utility(utilities)

        if return_utilities:
            return queried_indices, utilities
        else:
            return queried_indices

    def update(self, X_cand, queried_indices, budget_manager_param_dict=None):
        """Updates the budget manager and the count for seen and queried
        instances

        Parameters
        ----------
        X_cand : {array-like, sparse matrix} of shape (n_samples, n_features)
            The instances which could be queried. Sparse matrices are accepted
            only if they are supported by the base query strategy.

        queried_indices : array-like of shape (n_samples,)
            Indicates which instances from X_cand have been queried.

        budget_manager_param_dict : kwargs
            Optional kwargs for budget_manager.

        Returns
        -------
        self : VariableUncertainty
            The VariableUncertainty returns itself, after it is updated.
        """
        # Check if a budget_manager is set
        self._validate_budget_manager()
        # Check if a random state is set
        self._validate_random_state()
        budget_manager_param_dict = ({} if budget_manager_param_dict is None
                                     else budget_manager_param_dict)
        call_func(
            self.budget_manager_.update,
            X_cand=X_cand,
            queried_indices=queried_indices,
            **budget_manager_param_dict
        )
        return self

    def get_default_budget_manager(self):
        """Provide the budget manager that will be used as default.

        Returns
        -------
        budget_manager : BudgetManager
            The BudgetManager that should be used by default.
        """
        random_state = self.random_state_.randint(2**31)
        return SplitBudget(random_state=random_state)

    def _validate_data(
        self,
        X_cand,
        clf,
        X,
        y,
        sample_weight,
        return_utilities,
        reset=True,
        **check_X_cand_params
    ):
        """Validate input data and set or check the `n_features_in_` attribute.

        Parameters
        ----------
        X_cand: array-like of shape (n_candidates, n_features)
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
        X: np.ndarray, shape (n_samples, n_features)
            Checked training samples
        clf : SkactivemlClassifier
            Checked model implementing the methods `fit` and `predict_freq`.
        y: np.ndarray, shape (n_candidates)
            Checked training labels
        sampling_weight: np.ndarray, shape (n_candidates)
            Checked training sample weight
        X_cand: np.ndarray, shape (n_candidates, n_features)
            Checked candidate samples
        return_utilities : bool,
            Checked boolean value of `return_utilities`.
        """
        X_cand, return_utilities = super()._validate_data(
            X_cand, return_utilities, reset=reset, **check_X_cand_params
        )

        X, y, sample_weight = _validate_X_y_sample_weight(X, y, sample_weight)
        self._validate_random_state()
        self._validate_budget_manager()
        clf = self._validate_clf(clf, X, y, sample_weight)

        return X_cand, clf, X, y, sample_weight, return_utilities

    def _validate_clf(self, clf, X, y, sample_weight):
        """Validate if clf is a valid SkactivemlClassifier. If clf is
        untrained, clf is trained using X, y and sample_weight.

        Parameters
        ----------
        clf : SkactivemlClassifier
            Model implementing the methods `fit` and `predict_freq`.
        Returns
        -------
        clf : SkactivemlClassifier
            Checked model implementing the methods `fit` and `predict_freq`.
        """
        # Check if the classifier and its arguments are valid.
        check_type(clf, SkactivemlClassifier, "clf")
        return fit_if_not_fitted(clf, X, y, sample_weight)


def _validate_X_y_sample_weight(X, y, sample_weight):
    """Validate if X, y and sample_weight are numeric and of equal lenght.

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
        Checked Labels of the input samples 'X'. Converts y to a numpy array
    """
    if sample_weight is not None:
        sample_weight = np.array(sample_weight)
        check_consistent_length(sample_weight, y)
    if X is not None and y is not None:
        X = check_array(X)
        y = np.array(y)
        check_consistent_length(X, y)
    return X, y, sample_weight
