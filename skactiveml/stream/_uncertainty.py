import numpy as np

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
    classifier's predictions. The instance is queried when the probability of
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
        self,
        X_cand,
        X,
        y,
        return_utilities=False,
        sample_weight=None,
        **kwargs
    ):
        """Ask the query strategy which instances in X_cand to acquire.

        Please note that, when the decisions from this function may differ from
        the final sampling, simulate=True can be set, so that the query strategy
        can be updated later with update(...) with the final sampling. This is
        especially helpful when developing wrapper query strategies.

        Parameters
        ----------
        X_cand : {array-like, sparse matrix} of shape (n_samples, n_features)
            The instances which may be queried. Sparse matrices are accepted
            only if they are supported by the base query strategy.
        X : array-like of shape (n_samples, n_features)
            Input samples used to fit the classifier.
        y : array-like of shape (n_samples)
            Labels of the input samples 'X'. There may be missing labels.
        return_utilities : bool, optional
            If true, also return the utilities based on the query strategy.
            The default is False.
        sample_weight : array-like of shape (n_samples,) (default=None)
            Sample weights for X, used to fit the clf.

        Returns
        -------
        queried_indices : ndarray of shape (n_queried_instances,)
            The indices of instances in X_cand which should be queried, with
            0 <= n_queried_instances <= n_samples.

        utilities: ndarray of shape (n_samples,), optional
            The utilities based on the query strategy. Only provided if
            return_utilities is True.
        """
        self._validate_data(
            X_cand,
            X=X,
            y=y,
            sample_weight=sample_weight,
            return_utilities=return_utilities,
        )

        predict_proba = self.clf_.predict_proba(X_cand)
        utilities = np.max(predict_proba, axis=1)

        queried_indices = self.budget_manager_.query(utilities)

        if return_utilities:
            return queried_indices, utilities
        else:
            return queried_indices

    def update(
        self, X_cand, queried_indices, budget_manager_kwargs={}, **kwargs
    ):
        """Updates the budget manager and the count for seen and queried
        instances

        Parameters
        ----------
        X_cand : {array-like, sparse matrix} of shape (n_samples, n_features)
            The instances which could be queried. Sparse matrices are accepted
            only if they are supported by the base query strategy.

        queried_indices : array-like of shape (n_samples,)
            Indicates which instances from X_cand have been queried.

        budget_manager_kwargs : kwargs
            Optional kwargs for budget_manager.

        Returns
        -------
        self : FixedUncertainty
            The FixedUncertainty returns itself, after it is updated.
        """
        # check if a budget_manager is set
        self._validate_budget_manager()
        self.budget_manager_.update(
            X_cand, queried_indices, **budget_manager_kwargs
        )
        return self

    def _validate_data(
        self,
        X_cand,
        X,
        y,
        return_utilities,
        sample_weight,
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

        self._validate_clf(X, y, sample_weight)
        self._validate_random_state()

        return X_cand, X, y, sample_weight, return_utilities

    def _validate_clf(self, X, y, sample_weight):
        """Validate if clf is a classifier or create a new clf and fit X and y.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples used to fit the classifier.

        y : array-like of shape (n_samples)
            Labels of the input samples 'X'. There may be missing labels.

        sample_weight : array-like of shape (n_samples,) (default=None)
            Sample weights for X, used to fit the clf.
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
            self.clf_.fit(X, y, sample_weight=sample_weight)
            # check if y is not multi dimensinal
            if isinstance(y, np.ndarray):
                if y.ndim > 1:
                    raise ValueError(
                        "{} is not a valid Value for y".format(type(y))
                    )
        else:
            self.clf_ = self.clf


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
        self,
        X_cand,
        X,
        y,
        return_utilities=False,
        sample_weight=None,
        **kwargs
    ):
        """Ask the query strategy which instances in X_cand to acquire.

        Please note that, when the decisions from this function may differ from
        the final sampling, simulate=True can be set, so that the query strategy
        can be updated later with update(...) with the final sampling. This is
        especially helpful when developing wrapper query strategies.

        Parameters
        ----------
        X_cand : {array-like, sparse matrix} of shape (n_samples, n_features)
            The instances which may be queried. Sparse matrices are accepted
            only if they are supported by the base query strategy.

        X : array-like of shape (n_samples, n_features)
            Input samples used to fit the classifier.

        y : array-like of shape (n_samples)
            Labels of the input samples 'X'. There may be missing labels.

        return_utilities : bool, optional
            If true, also return the utilities based on the query strategy.
            The default is False.

        sample_weight : array-like of shape (n_samples,) (default=None)
            Sample weights for X, used to fit the clf.

        Returns
        -------
        queried_indices : ndarray of shape (n_queried_instances,)
            The indices of instances in X_cand which should be queried, with
            0 <= n_queried_instances <= n_samples.

        utilities: ndarray of shape (n_samples,), optional
            The utilities based on the query strategy. Only provided if
            return_utilities is True.
        """
        self._validate_data(
            X_cand, return_utilities, X, y, sample_weight=sample_weight,
        )

        predict_proba = self.clf_.predict_proba(X_cand)
        utilities = np.max(predict_proba, axis=1)

        queried_indices = self.budget_manager_.query(utilities)

        if return_utilities:
            return queried_indices, utilities
        else:
            return queried_indices

    def update(
        self, X_cand, queried_indices, budget_manager_kwargs={}, **kwargs
    ):
        """Updates the budget manager and the count for seen and queried
        instances

        Parameters
        ----------
        X_cand : {array-like, sparse matrix} of shape (n_samples, n_features)
            The instances which could be queried. Sparse matrices are accepted
            only if they are supported by the base query strategy.

        queried_indices : array-like of shape (n_samples,)
            Indicates which instances from X_cand have been queried.

        budget_manager_kwargs : kwargs
            Optional kwargs for budget_manager.

        Returns
        -------
        self : VariableUncertainty
            The VariableUncertainty returns itself, after it is updated.
        """
        # check if a budget_manager is set
        self._validate_budget_manager()
        self.budget_manager_.update(
            X_cand, queried_indices, **budget_manager_kwargs
        )
        return self

    def _validate_data(
        self,
        X_cand,
        return_utilities,
        X,
        y,
        sample_weight,
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

        self._validate_clf(X, y, sample_weight=sample_weight)
        self._validate_random_state()

        return X_cand, return_utilities, X, y

    def _validate_clf(self, X, y, sample_weight):
        """Validate if clf is a classifier or create a new clf and fit X and y.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples used to fit the classifier.

        y : array-like of shape (n_samples)
            Labels of the input samples 'X'. There may be missing labels.

        sample_weight : array-like of shape (n_samples,) (default=None)
            Sample weights for X, used to fit the clf.
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
            self.clf_.fit(X, y, sample_weight=sample_weight)
            # check if y is not multi dimensinal
            if isinstance(y, np.ndarray):
                if y.ndim > 1:
                    raise ValueError(
                        "{} is not a valid Value for y".format(type(y))
                    )
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
        self,
        X_cand,
        X,
        y,
        return_utilities=False,
        sample_weight=None,
        **kwargs
    ):
        """Ask the query strategy which instances in X_cand to acquire.

        Please note that, when the decisions from this function may differ from
        the final sampling, simulate=True can be set, so that the query strategy
        can be updated later with update(...) with the final sampling. This is
        especially helpful when developing wrapper query strategies.

        Parameters
        ----------
        X_cand : {array-like, sparse matrix} of shape (n_samples, n_features)
            The instances which may be queried. Sparse matrices are accepted
            only if they are supported by the base query strategy.

        X : array-like of shape (n_samples, n_features)
            Input samples used to fit the classifier.

        y : array-like of shape (n_samples)
            Labels of the input samples 'X'. There may be missing labels.

        return_utilities : bool, optional
            If true, also return the utilities based on the query strategy.
            The default is False.

        sample_weight : array-like of shape (n_samples,) (default=None)
            Sample weights for X, used to fit the clf.

        Returns
        -------
        queried_indices : ndarray of shape (n_queried_instances,)
            The indices of instances in X_cand which should be queried, with
            0 <= n_queried_instances <= n_samples.

        utilities: ndarray of shape (n_samples,), optional
            The utilities based on the query strategy. Only provided if
            return_utilities is True.
        """
        self._validate_data(
            X_cand, return_utilities, X, y, sample_weight=sample_weight,
        )

        predict_proba = self.clf_.predict_proba(X_cand)
        utilities = np.max(predict_proba, axis=1)

        queried_indices = self.budget_manager_.query(utilities)

        if return_utilities:
            return queried_indices, utilities
        else:
            return queried_indices

    def update(
        self, X_cand, queried_indices, budget_manager_kwargs={}, **kwargs
    ):
        """Updates the budget manager and the count for seen and queried
        instances

        Parameters
        ----------
        X_cand : {array-like, sparse matrix} of shape (n_samples, n_features)
            The instances which could be queried. Sparse matrices are accepted
            only if they are supported by the base query strategy.

        queried_indices : array-like of shape (n_samples,)
            Indicates which instances from X_cand have been queried.

        budget_manager_kwargs : kwargs
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

        self.budget_manager_.update(
            X_cand, queried_indices, **budget_manager_kwargs
        )
        return self

    def _validate_data(
        self,
        X_cand,
        return_utilities,
        X,
        y,
        sample_weight,
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

        self._validate_clf(X, y, sample_weight)
        self._validate_random_state()
        self._validate_budget_manager()

        return X_cand, return_utilities, X, y

    def _validate_clf(self, X, y, sample_weight):
        """Validate if clf is a classifier or create a new clf and fit X and y.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples used to fit the classifier.

        y : array-like of shape (n_samples)
            Labels of the input samples 'X'. There may be missing labels.

        sample_weight : array-like of shape (n_samples,) (default=None)
            Sample weights for X, used to fit the clf.
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
            self.clf_.fit(X, y, sample_weight=sample_weight)
            # check if y is not multi dimensinal
            if isinstance(y, np.ndarray):
                if y.ndim > 1:
                    raise ValueError(
                        "{} is not a valid Value for y".format(type(y))
                    )
        else:
            self.clf_ = self.clf
