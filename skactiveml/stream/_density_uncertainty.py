from collections import deque

from copy import copy
import numpy as np
from sklearn.utils import check_array, check_consistent_length, check_scalar
from sklearn.base import clone
from sklearn.metrics.pairwise import euclidean_distances

from skactiveml.base import (
    BudgetManager,
    SingleAnnotatorStreamQueryStrategy,
    SkactivemlClassifier,
)
from skactiveml.utils import (
    check_type,
    call_func,
    check_budget_manager,
)
from skactiveml.stream.budgetmanager import (
    FixedUncertaintyBudgetManager,
    DensityBasedBudgetManager,
    VariableUncertaintyBudgetManager,
    RandomBudgetManager,
    RandomVariableUncertaintyBudgetManager,
)


class DBStream(SingleAnnotatorStreamQueryStrategy):
    """The DBStream [1] query strategy
    samples instances based on the classifiers minimum margin between posterior
    probabilities assessed based on the classifier's predictions.The instance
    is queried when the probability of the most likely class exceeds a
    threshold calculated based on the budget and the number of classes as well
    as the instance is a new nearest neighbor of the local density sliding
    window.
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

    window_size : int, default=1000
        Determines the sliding window size of the local density window

    random_state : int, RandomState instance, default=None
        Controls the randomness of the estimator.

    dist_func : callable, default=None
        The distance function used to calculate the distances within the local
        density window. If None use `sklearn.metrics.pairwise`
        euclidean_distances.

    force_full_budget : bool, default=False
            If true, tries to utilize the full budget. The paper doesn't update
            the budget manager if the locale density factor is 0

    References
    ----------
    [1]

    """
    def __init__(
        self,
        budget_manager=None,
        budget=None,
        random_state=None,
        window_size=1000,
        dist_func=None,
        force_full_budget=False,
    ):
        super().__init__(budget=budget, random_state=random_state)
        self.budget_manager = budget_manager
        self.window_size = window_size
        self.dist_func = dist_func
        self.force_full_budget = force_full_budget

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
        the final sampling, so the query strategy can be updated later with
        update(...) with the final sampling.

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

        # calculate the margin used as utillities
        predict_proba = clf.predict_proba(candidates)
        utilities_index = np.argpartition(predict_proba, -2)[:, -2:]
        utilities = (
            predict_proba[:, utilities_index[:, 1]]
            - predict_proba[:, utilities_index[:, 0]]
        )
        tmp_min_dist = copy(self.min_dist_)
        tmp_window = copy(self.window_)
        queried_indices = []
        for t, (u, x_cand) in enumerate(zip(utilities, candidates)):
            local_density_factor = self._calculate_ldf([x_cand])
            if local_density_factor > 0:
                queried_indice = self.budget_manager_.query_by_utility(u)
                if len(queried_indice) > 0:
                    queried_indices.append(t)
            elif self.force_full_budget:
                self.budget_manager_.query_by_utility(np.array([np.nan]))
            self.window_.append(x_cand)

        self.min_dist_ = tmp_min_dist
        self.window_ = tmp_window

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
        candidates : {array-like, sparse matrix} of shape (n_samples, n_features)
            The instances which could be queried. Sparse matrices are accepted
            only if they are supported by the base query strategy.

        queried_indices : array-like of shape (n_samples,)
            Indicates which instances from candidates have been queried.

        budget_manager_param_dict : kwargs
            Optional kwargs for budget_manager.

        Returns
        -------
        self : UncertaintyZliobaite
            The UncertaintyZliobaite returns itself, after it is updated.
        """
        # check if a budget_manager is set
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

        if not hasattr(self, "window_"):
            self.window_ = deque(maxlen=self.window_size)
        if not hasattr(self, "min_dist_"):
            self.min_dist_ = deque(maxlen=self.window_size)
        if self.dist_func is None:
            self.dist_func_ = euclidean_distances
        else:
            self.dist_func_ = self.dist_func
        if not callable(self.dist_func_):
            raise TypeError("frequency_estimation needs to be a callable")

        budget_manager_param_dict = (
            {}
            if budget_manager_param_dict is None
            else budget_manager_param_dict
        )
        new_candidates = []
        for x_cand in candidates:
            local_density_factor = self._calculate_ldf([x_cand])
            if local_density_factor > 0:
                new_candidates.append(x_cand)
            else:
                new_candidates.append(np.nan)
            self.window_.append(x_cand)
        call_func(
            self.budget_manager_.update,
            candidates=new_candidates,
            queried_indices=queried_indices,
            **budget_manager_param_dict
        )
        return self

    def _calculate_ldf(self, candidates):
        """Calculate the number of new nearest neighbor for candiates in the
        sliding window.

        Parameters
        ----------
        candidates: array-like of shape (n_candidates, n_features)
            The instances which may be queried. Sparse matrices are accepted
            only if they are supported by the base query strategy.

        Returns
        -------
        ldf: array-like of shape (n_candiates)
            Numbers of new nearest neighbor for candidates
        """
        ldf = 0
        if len(self.window_) >= 1:
            distances = self.dist_func_(self.window_, candidates).ravel()
            is_new_nn = distances < np.array(self.min_dist_)
            ldf = np.sum(is_new_nn)
            for i in np.where(is_new_nn)[0]:
                self.min_dist_[i] = distances[i]
            self.min_dist_.append(np.min(distances))
        else:
            self.min_dist_.append(np.inf)

        return ldf

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

        # check if a budget_manager is set
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
        if self.dist_func is None:
            self.dist_func_ = euclidean_distances
        else:
            self.dist_func_ = self.dist_func
        if not callable(self.dist_func_):
            raise TypeError("frequency_estimation needs to be a callable")

        # check density_threshold
        check_scalar(self.window_size, "window_size", int, min_val=1)

        # check force_full_budget
        check_type(self.force_full_budget, "force_full_budget", bool)

        if not hasattr(self, "window_"):
            self.window_ = deque(maxlen=self.window_size)

        if not hasattr(self, "min_dist_"):
            self.min_dist_ = deque(maxlen=self.window_size)

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

    def _get_default_budget_manager(self):
        """Provide the budget manager that will be used as default.

        Returns
        -------
        budget_manager : BudgetManager
            The BudgetManager that should be used by default.
        """
        return DensityBasedBudgetManager


class CogDQS(SingleAnnotatorStreamQueryStrategy):
    """The CogDQS [1] query strategy
    samples instances based on the classifiers uncertainty assessed based on
    the classifier's predictions. The instance is queried when the probability
    of the most likely class exceeds a threshold calculated based on the budget
    and the number of classes as well as the instance is at least the new
    nearest neighbor of density_threshold instances in the cognition window.

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

    density_threshold : int, default=1
        Determines the local density factor size that needs to be reached
        in order to sample the candidate.

    cognition_window_size : int, default=10
        Determines the size of the cognition window

    random_state : int, RandomState instance, default=None
        Controls the randomness of the estimator.

    dist_func : callable, default=None
        The distance function used to calculate the distances within the local
        density window. If None use `sklearn.metrics.pairwise`
        euclidean_distances.

    force_full_budget : bool, default=False
            If true, tries to utilize the full budget. The paper doesn't update
            the budget manager if the locale density factor is 0

    References
    ----------
    [1]

    """
    def __init__(
        self,
        budget_manager=None,
        budget=None,
        density_threshold=1,
        cognition_window_size=10,
        dist_func=None,
        random_state=None,
        force_full_budget=False,
    ):
        super().__init__(budget=budget, random_state=random_state)
        self.budget_manager = budget_manager
        self.density_threshold = density_threshold
        self.dist_func = dist_func
        self.cognition_window_size = cognition_window_size
        self.force_full_budget = force_full_budget

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
        the final sampling, so the query strategy can be updated later with
        update(...) with the final sampling.

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

        # its the margin but used as utillities
        predict_proba = clf.predict_proba(candidates)
        utilities = np.max(predict_proba, axis=1)

        # copy variables
        tmp_cognition_window = copy(self.cognition_window_)
        tmp_theta = copy(self.theta_)
        tmp_s = copy(self.s_)
        tmp_t_x = copy(self.t_x_)
        f = (self.f_)
        min_dist = copy(self.min_dist_)
        t = copy(self.t_)
        queried_indices = []

        for i, (u, x_cand) in enumerate(zip(utilities, candidates)):
            local_density_factor = self._calculate_ldf([x_cand])
            if local_density_factor >= self.density_threshold:
                queried_indice = self.budget_manager_.query_by_utility(np.array([u]))
                if len(queried_indice) > 0:
                    queried_indices.append(i)
            elif self.force_full_budget:
                self.budget_manager_.query_by_utility(np.array([np.nan]))
            self.t_ += 1

        # overwrite changes
        self.cognition_window_ = tmp_cognition_window
        self.theta_ = tmp_theta
        self.s_ = tmp_s
        self.t_x_ = tmp_t_x
        self.f_ = f
        self.min_dist_ = min_dist
        self.t_ = t

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
        candidates : {array-like, sparse matrix} of shape (n_samples, n_features)
            The instances which could be queried. Sparse matrices are accepted
            only if they are supported by the base query strategy.

        queried_indices : array-like of shape (n_samples,)
            Indicates which instances from candidates have been queried.

        budget_manager_param_dict : kwargs
            Optional kwargs for budget_manager.

        Returns
        -------
        self : UncertaintyZliobaite
            The UncertaintyZliobaite returns itself, after it is updated.
        """
        # check if a budget_manager is set
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
        # _init_members
        if self.dist_func is None:
            self.dist_func_ = euclidean_distances
        else:
            self.dist_func_ = self.dist_func
        if not callable(self.dist_func_):
            raise TypeError("frequency_estimation needs to be a callable")
        if not hasattr(self, "min_dist_"):
            self.min_dist_ = []
        if not hasattr(self, "t_"):
            self.t_ = 0
        if not hasattr(self, "cognition_window_"):
            self.cognition_window_ = []
        if not hasattr(self, "f_"):
            self.f_ = []
        if not hasattr(self, "theta_"):
            self.theta_ = []
        if not hasattr(self, "s_"):
            self.s_ = []
        if not hasattr(self, "t_x_"):
            self.t_x_ = []

        budget_manager_param_dict = (
            {}
            if budget_manager_param_dict is None
            else budget_manager_param_dict
        )
        new_candidates = []
        for x_cand in candidates:
            local_density_factor = self._calculate_ldf([x_cand])
            if local_density_factor >= self.density_threshold:
                new_candidates.append(x_cand)
            else:
                new_candidates.append(np.nan)
            self.t_ += 1
        call_func(
            self.budget_manager_.update,
            candidates=new_candidates,
            queried_indices=queried_indices,
            **budget_manager_param_dict
        )
        return self

    def _calculate_ldf(self, candidates):
        """Calculate the number of new nearest neighbor for candiates in the
        cognition_window.

        Parameters
        ----------
        candidates: array-like of shape (n_candidates, n_features)
            The instances which may be queried. Sparse matrices are accepted
            only if they are supported by the base query strategy.

        Returns
        -------
        ldf: array-like of shape (n_candiates)
            Numbers of new nearest neighbor for candidates

        """
        ldf = 0
        f = 1
        t_x = self.t_
        s = 1
        theta = 0
        if len(self.cognition_window_) >= 1:
            distances = self.dist_func_(
                self.cognition_window_, candidates
            ).ravel()
            is_new_nn = distances < np.array(self.min_dist_)
            ldf = np.sum(is_new_nn)
            for i in np.where(is_new_nn)[0]:
                self.t_x_[i] = t_x
                self.theta_[i] += 1
                self.min_dist_[i] = distances[i]
            self.min_dist_.append(np.min(distances))
        else:
            self.min_dist_.append(np.inf)
        for t, _ in enumerate(self.cognition_window_):
            self.f_[t] = 1 / (self.theta_[t] + 1)
            tmp = -self.f_[t] * (t_x - self.t_x_[t])
            self.s_[t] = np.exp(tmp)
        if len(self.cognition_window_) > self.cognition_window_size:
            # remove element with the smallest memory strength
            remove_index = np.argmin(self.s_)
            self.cognition_window_.pop(remove_index)
            self.theta_.pop(remove_index)
            self.s_.pop(remove_index)
            self.t_x_.pop(remove_index)
            self.f_.pop(remove_index)
            self.min_dist_.pop(remove_index)
        self.cognition_window_.extend(candidates)
        self.theta_.append(theta)
        self.s_.append(s)
        self.t_x_.append(t_x)
        self.f_.append(f)

        return ldf

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

        # check density_threshold
        check_scalar(self.density_threshold, "density_threshold", int, min_val=0)
        check_scalar(self.cognition_window_size, "cognition_window_size", int, min_val=1)

        # check force_full_budget
        check_type(self.force_full_budget, "force_full_budget", bool)

        # check if a budget_manager is set
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

        if self.dist_func is None:
            self.dist_func_ = euclidean_distances
        else:
            self.dist_func_ = self.dist_func
        if not callable(self.dist_func_):
            raise TypeError("frequency_estimation needs to be a callable")

        if not hasattr(self, "min_dist_"):
            self.min_dist_ = []
        if not hasattr(self, "t_"):
            self.t_ = 0
        if not hasattr(self, "cognition_window_"):
            self.cognition_window_ = []
        if not hasattr(self, "f_"):
            self.f_ = []
        if not hasattr(self, "theta_"):
            self.theta_ = []
        if not hasattr(self, "s_"):
            self.s_ = []
        if not hasattr(self, "t_x_"):
            self.t_x_ = []

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


class CogDQSRan(CogDQS):
    """The CogDQS [1] query strategy
    samples instances based on the classifiers uncertainty assessed based on
    the classifier's predictions. The instance is queried when the probability
    of the most likely class exceeds a threshold calculated based on the budget
    and the number of classes as well as the instance is at least the new
    nearest neighbor of density_threshold instances in the cognition window.

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

    density_threshold : int, default=1
        Determines the local density factor size that needs to be reached
        in order to sample the candidate.

    cognition_window_size : int, default=10
        Determines the size of the cognition window

    random_state : int, RandomState instance, default=None
        Controls the randomness of the estimator.

    dist_func : callable, default=None
        The distance function used to calculate the distances within the local
        density window. If None use `sklearn.metrics.pairwise`
        euclidean_distances.

    force_full_budget : bool, default=False
            If true, tries to utilize the full budget. The paper doesn't update
            the budget manager if the locale density factor is 0

    References
    ----------
    [1]

    """
    def _get_default_budget_manager(self):
        """Provide the budget manager that will be used as default.

        Returns
        -------
        budget_manager : BudgetManager
            The BudgetManager that should be used by default.
        """
        return RandomBudgetManager


class CogDQSRanVarUn(CogDQS):
    """The CogDQS [1] query strategy
    samples instances based on the classifiers uncertainty assessed based on
    the classifier's predictions. The instance is queried when the probability
    of the most likely class exceeds a threshold calculated based on the budget
    and the number of classes as well as the instance is at least the new
    nearest neighbor of density_threshold instances in the cognition window.

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

    density_threshold : int, default=1
        Determines the local density factor size that needs to be reached
        in order to sample the candidate.

    cognition_window_size : int, default=10
        Determines the size of the cognition window

    random_state : int, RandomState instance, default=None
        Controls the randomness of the estimator.

    dist_func : callable, default=None
        The distance function used to calculate the distances within the local
        density window. If None use `sklearn.metrics.pairwise`
        euclidean_distances.

    force_full_budget : bool, default=False
            If true, tries to utilize the full budget. The paper doesn't update
            the budget manager if the locale density factor is 0

    References
    ----------
    [1]

    """
    def _get_default_budget_manager(self):
        """Provide the budget manager that will be used as default.

        Returns
        -------
        budget_manager : BudgetManager
            The BudgetManager that should be used by default.
        """
        return RandomVariableUncertaintyBudgetManager


class CogDQSVarUn(CogDQS):
    """The CogDQS [1] query strategy
    samples instances based on the classifiers uncertainty assessed based on
    the classifier's predictions. The instance is queried when the probability
    of the most likely class exceeds a threshold calculated based on the budget
    and the number of classes as well as the instance is at least the new
    nearest neighbor of density_threshold instances in the cognition window.

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

    density_threshold : int, default=1
        Determines the local density factor size that needs to be reached
        in order to sample the candidate.

    cognition_window_size : int, default=10
        Determines the size of the cognition window

    random_state : int, RandomState instance, default=None
        Controls the randomness of the estimator.

    dist_func : callable, default=None
        The distance function used to calculate the distances within the local
        density window. If None use `sklearn.metrics.pairwise`
        euclidean_distances.

    force_full_budget : bool, default=False
            If true, tries to utilize the full budget. The paper doesn't update
            the budget manager if the locale density factor is 0

    References
    ----------
    [1]

    """
    def _get_default_budget_manager(self):
        """Provide the budget manager that will be used as default.

        Returns
        -------
        budget_manager : BudgetManager
            The BudgetManager that should be used by default.
        """
        return VariableUncertaintyBudgetManager


class CogDQSFixUn(CogDQS):
    """The CogDQS [1] query strategy
    samples instances based on the classifiers uncertainty assessed based on
    the classifier's predictions. The instance is queried when the probability
    of the most likely class exceeds a threshold calculated based on the budget
    and the number of classes as well as the instance is at least the new
    nearest neighbor of density_threshold instances in the cognition window.

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

    density_threshold : int, default=1
        Determines the local density factor size that needs to be reached
        in order to sample the candidate.

    cognition_window_size : int, default=10
        Determines the size of the cognition window

    random_state : int, RandomState instance, default=None
        Controls the randomness of the estimator.

    dist_func : callable, default=None
        The distance function used to calculate the distances within the local
        density window. If None use `sklearn.metrics.pairwise`
        euclidean_distances.

    force_full_budget : bool, default=False
            If true, tries to utilize the full budget. The paper doesn't update
            the budget manager if the locale density factor is 0

    References
    ----------
    [1]

    """
    def _get_default_budget_manager(self):
        """Provide the budget manager that will be used as default.

        Returns
        -------
        budget_manager : BudgetManager
            The BudgetManager that should be used by default.
        """
        return FixedUncertaintyBudgetManager