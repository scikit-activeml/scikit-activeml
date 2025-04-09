import numpy as np
from sklearn.base import clone
from sklearn.utils import check_array, check_consistent_length
from copy import deepcopy

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
    """Base class for the uncertainty sampling strategies proposed by Žliobaitė
    et al. in [1]_.

    The UncertaintyZliobaite class provides the base for query strategies
    proposed by Žliobaitė et al. in [1]_. The strategies evaluate the
    classifier's uncertainty based on its predictions and samples' labels are
    queried when the uncertainty exceeds a specific threshold. Žliobaitė et al.
    propose various techniques to calculate such a threshold.

    Parameters
    ----------
    budget_manager : BudgetManager, default=None
        The BudgetManager which models the budgeting constraint used in the
        stream-based active learning setting. if set to `None`, a default
        budger manager will be used.  The budget manager will be initialized
        based on the following conditions:

        - If only a `budget` is given, the default budget manager is
          initialized with the given budget.
        - If only a budget manager is given, use the budget manager.
        - If both are not given, the default budget manager with the default
          budget.
        - If both are given, and the budget differs from
          `budgetmanager.budget`, throw a warning and the budget manager is
          used as is.
    budget : float, default=None
        Specifies the ratio of samples which are allowed to be sampled, with
        `0 <= budget <= 1`. If `budget` is `None`, it is replaced with the
        default budget 0.1.
    random_state : int or RandomState instance, default=None
        Controls the randomness of the estimator.

    References
    ----------
    .. [1] I. Žliobaitė, A. Bifet, B. Pfahringer, and G. Holmes. Active
        Learning With Drifting Streaming Data. IEEE Trans. Neural Netw. Learn.
        Syst., 25(1):27–39, 2014.
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
        """Determines for which candidate samples labels are to be queried.

        The query startegy determines the most useful samples in candidates,
        which can be acquired within the budgeting constraint specified by
        `budget`. Please note that, this method does not change the internal
        state of the query strategy. To adapt the query strategy to the
        selected candidates, use `update(...)`.

        Parameters
        ----------
        candidates : {array-like, sparse matrix} of shape\
                (n_candidates, n_features)
            The samples which may be queried. Sparse matrices are accepted
            only if they are supported by the base query strategy.
        clf : skactiveml.base.SkactivemlClassifier
            Model implementing the methods `fit` and `predict_proba`.
        X : array-like of shape (n_samples, n_features), default=None
            Training data set used to fit the classifier.
        y : array-like of shape (n_samples,)
            Labels of the training data set (possibly including unlabeled ones
            indicated by `self.missing_label`).
        sample_weight : array-like of shape (n_samples,), default=None
            Weights of training samples in `X`.
        fit_clf : bool, default=False
            Defines whether the classifier should be fitted on `X`, `y`, and
            `sample_weight`.
        return_utilities : bool, default=False
            If `True`, also return the `utilities` based on the query strategy.

        Returns
        -------
        queried_indices : np.ndarray of shape (n_queried_indices,)
            The indices of samples in candidates whose labels are queried,
            with `0 <= queried_indices <= n_candidates`.
        utilities: np.ndarray of shape (n_candidates,),
            The utilities based on the query strategy. Only provided if
            `return_utilities` is `True`.
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
        confidence = np.max(predict_proba, axis=1)
        utilities = 1 - confidence

        queried_indices = self.budget_manager_.query_by_utility(utilities)

        if return_utilities:
            return queried_indices, utilities
        else:
            return queried_indices

    def update(
        self, candidates, queried_indices, budget_manager_param_dict=None
    ):
        """Updates the budget manager and the count for seen and queried
        labels. This function should be used in conjunction with the `query`
        function.

        Parameters
        ----------
        candidates : {array-like, sparse matrix} of shape\
                (n_candidates, n_features)
            The samples which may be queried. Sparse matrices are accepted
            only if they are supported by the base query strategy.
        queried_indices : np.ndarray of shape (n_queried_indices,)
            The indices of samples in candidates whose labels are queried,
            with `0 <= queried_indices <= n_candidates`.
        budget_manager_param_dict : dict, default=None
            Optional kwargs for `budget_manager`.

        Returns
        -------
        self : SingleAnnotatorStreamQueryStrategy
            The query strategy returns itself, after it is updated.
        """
        # check if a budgetmanager is set
        if not hasattr(self, "budget_manager_"):
            self._validate_random_state()
            random_seed = deepcopy(self.random_state_).randint(2**31 - 1)
            check_type(
                self.budget_manager,
                "budget_manager_",
                BudgetManager,
                type(None),
            )
            default_budget_manager_kwargs = (
                self._get_default_budget_manager_kwargs()
            )
            default_budget_manager_kwargs["random_state"] = random_seed
            self.budget_manager_ = check_budget_manager(
                self.budget,
                self.budget_manager,
                self._get_default_budget_manager(),
                default_budget_manager_kwargs,
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
        candidates : {array-like, sparse matrix} of shape\
                (n_candidates, n_features)
            The samples which may be queried. Sparse matrices are accepted
            only if they are supported by the base query strategy.
        clf : skactiveml.base.SkactivemlClassifier
            Model implementing the methods `fit` and `predict_proba`.
        X : array-like of shape (n_samples, n_features), default=None
            Training data set used to fit the classifier.
        y : array-like of shape (n_samples,)
            Labels of the training data set (possibly including unlabeled ones
            indicated by `self.missing_label`).
        sample_weight : array-like of shape (n_samples,), default=None
            Weights of training samples in `X`.
        fit_clf : bool, default=False
            Defines whether the classifier should be fitted on `X`, `y`, and
            `sample_weight`.
        return_utilities : bool, default=False
            If `True`, also return the utilities based on the query strategy.
        reset : bool, default=True
            Whether to reset the `n_features_in_` attribute. If False, the
            input will be checked for consistency with data provided when reset
            was last True.
        **check_candidates_params : kwargs
            Parameters passed to :func:`sklearn.utils.check_array`.

        Returns
        -------
        candidates: np.ndarray, shape (n_candidates, n_features)
            Checked candidate samples.
        clf : SkactivemlClassifier
            Checked model implementing the methods `fit` and `predict_freq`.
        X: np.ndarray, shape (n_samples, n_features)
            Checked training data set.
        y: np.ndarray, shape (n_samples)
            Checked training labels.
        sampling_weight: np.ndarray, shape (n_candidates)
            Checked training sample weight.
        fit_clf : bool,
            Checked boolean value of `fit_clf`.
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
            random_seed = deepcopy(self.random_state_).randint(2**31 - 1)
            check_type(
                self.budget_manager,
                "budget_manager_",
                BudgetManager,
                type(None),
            )
            default_budget_manager_kwargs = (
                self._get_default_budget_manager_kwargs()
            )
            default_budget_manager_kwargs["random_state"] = random_seed
            self.budget_manager_ = check_budget_manager(
                self.budget,
                self.budget_manager,
                self._get_default_budget_manager(),
                default_budget_manager_kwargs,
            )

        return candidates, clf, X, y, sample_weight, fit_clf, return_utilities

    def _get_default_budget_manager_kwargs(self):
        """Provide the kwargs for the budget manager that will be used as
        default.

        Returns
        -------
        default_budget_manager_kwargs : dict
            The arguments necessary to initialize the budget manager.
        """
        return {}

    def _validate_clf(self, clf, X, y, sample_weight, fit_clf):
        """Validate if `clf` is a valid `SkactivemlClassifier`. If `clf` is
        untrained and `fit_clf`=`True`, `clf` is trained using X, y and
        sample_weight.

        Parameters
        ----------
        clf : skactiveml.base.SkactivemlClassifier
            Model implementing the methods `fit` and `predict_proba`.
        X : array-like of shape (n_samples, n_features), default=None
            Training data set used to fit the classifier.
        y : array-like of shape (n_samples,)
            Labels of the training data set (possibly including unlabeled ones
            indicated by `self.missing_label`).
        sample_weight : array-like of shape (n_samples,), default=None
            Weights of training samples in `X`.
        fit_clf : bool, default=False
            Defines whether the classifier should be fitted on `X`, `y`, and
            `sample_weight`.
        Returns
        -------
        clf : skactiveml.base.SkactivemlClassifier
            Checked model implementing the methods `fit` and `predict_freq`.
        """
        # Check if the classifier and its arguments are valid.
        check_type(clf, "clf", SkactivemlClassifier)
        check_type(fit_clf, "fit_clf", bool)
        if fit_clf:
            if sample_weight is None:
                clf = clone(clf).fit(X, y)
            else:
                clf = clone(clf).fit(X, y, sample_weight)
        return clf

    def _validate_X_y_sample_weight(self, X, y, sample_weight):
        """Validate if X, y and sample_weight are numeric and of equal length.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data set used to fit the classifier.
        y : array-like of shape (n_samples,)
            Labels of the training data set (possibly including unlabeled ones
            indicated by `self.missing_label`).
        sample_weight : array-like of shape (n_samples,)
            Weights of training samples in `X`.
        Returns
        -------
        X : array-like of shape (n_samples, n_features)
            Checked training data set.
        y : array-like of shape (n_samples)
            Checked labels of the input samples `X`. Converts `y` to a numpy
            array.
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
    """Fixed Uncertainty Strategy

    The FixedUncertainty (Fixed Uncertainty Strategy in [1]_) query strategy
    queries samples based on the classifiers uncertainty that is assessed
    based on the classifier's predictions. The sample is queried when the
    probability of the most likely class exceeds a threshold calculated based
    on the budget and the number of classes. See also
    :class:`.FixedUncertaintyBudgetManager`

    Parameters
    ----------
    classes : array-like of shape (n_classes,)
        Holds the label for each class.
    budget_manager : BudgetManager, default=None
        The BudgetManager which models the budgeting constraint used in the
        stream-based active learning setting. if set to `None`,
        `FixedUncertaintyBudgetManager` will be used by default.  The budget
        manager will be initialized based on the following conditions:

        - If only a `budget` is given, the default budget manager is
          initialized with the given budget.
        - If only a budget manager is given, use the budget manager.
        - If both are not given, the default budget manager with the default
          budget.
        - If both are given, and the budget differs from
          `budgetmanager.budget`, throw a warning and the budget manager is
          used as is.
    budget : float, default=None
        Specifies the ratio of samples which are allowed to be sampled, with
        `0 <= budget <= 1`. If `budget` is `None`, it is replaced with the
        default budget 0.1.
    random_state : int or RandomState instance, default=None
        Controls the randomness of the estimator.

    References
    ----------
    .. [1] I. Žliobaitė, A. Bifet, B. Pfahringer, and G. Holmes. Active
        Learning With Drifting Streaming Data. IEEE Trans. Neural Netw. Learn.
        Syst., 25(1):27–39, 2014.
    """

    def __init__(
        self,
        classes,
        budget_manager=None,
        budget=None,
        random_state=None,
    ):
        super().__init__(
            budget_manager=budget_manager,
            budget=budget,
            random_state=random_state,
        )
        self.classes = classes

    def _get_default_budget_manager(self):
        """Provide the budget manager that will be used as default.

        Returns
        -------
        budget_manager : BudgetManager
            The BudgetManager that should be used by default.
        """
        return FixedUncertaintyBudgetManager

    def _get_default_budget_manager_kwargs(self):
        """Provide the kwargs for the budget manager that will be used as
        default.

        Returns
        -------
        default_budget_manager_kwargs : dict
            The arguments necessary to initialize the budget manager.
        """
        return {"classes": self.classes}


class VariableUncertainty(UncertaintyZliobaite):
    """Variable Uncertainty Strategy

    The VariableUncertainty query strategy (Variable Uncertainty Strategy in
    [1]_) queries labels based on the classifiers uncertainty assessed based on
    the classifier's predictions. The sample is queried when the probability
    of the most likely class exceeds a time-dependent threshold calculated
    based on the budget, number of observed and acquired samples. See also
    :class:`.VariableUncertaintyBudgetManager`

    Parameters
    ----------
    budget_manager : BudgetManager, default=None
        The BudgetManager which models the budgeting constraint used in the
        stream-based active learning setting. if set to `None`,
        `VariableUncertaintyBudgetManager` will be used by default. The budget
        manager will be initialized based on the following conditions:

        - If only a `budget` is given, the default budget manager is
          initialized with the given budget.
        - If only a budget manager is given, use the budget manager.
        - If both are not given, the default budget manager with the default
          budget.
        - If both are given, and the budget differs from
          `budgetmanager.budget`, throw a warning and the budget manager is
          used as is.
    budget : float, default=None
        Specifies the ratio of samples which are allowed to be sampled, with
        `0 <= budget <= 1`. If `budget` is `None`, it is replaced with the
        default budget 0.1.
    random_state : int or RandomState instance, default=None
        Controls the randomness of the estimator.

    References
    ----------
    .. [1] I. Žliobaitė, A. Bifet, B. Pfahringer, and G. Holmes. Active
        Learning With Drifting Streaming Data. IEEE Trans. Neural Netw. Learn.
        Syst., 25(1):27–39, 2014.
    """

    def _get_default_budget_manager(self):
        """Provide the budget manager that will be used as default.

        Returns
        -------
        budget_manager : BudgetManager
            The BudgetManager that should be used by default.
        """
        return VariableUncertaintyBudgetManager


class RandomVariableUncertainty(UncertaintyZliobaite):
    """RandomVariableUncertainty

    The RandomVariableUncertainty (Uncertainty Strategy With Randomization in
    [1]_) query strategy samples samples based on the classifier's
    uncertainty assessed based on the classifier's predictions. The sample is
    queried when the probability of the most likely class exceeds a
    time-dependent threshold calculated based on the budget, and the number of
    observed and acquired samples. The threshold is randomized by being
    multiplied with a random number sampled from N(1,delta). See also
    :class:`.RandomVariableUncertaintyBudgetManager`

    Parameters
    ----------
    budget_manager : BudgetManager, default=None
        The BudgetManager which models the budgeting constraint used in the
        stream-based active learning setting. if set to `None`,
        `RandomVariableUncertaintyBudgetManager` will be used by default.  The
        budget manager will be initialized based on the following conditions:

        - If only a `budget` is given, the default budget manager is
          initialized with the given budget.
        - If only a budget manager is given, use the budget manager.
        - If both are not given, the default budget manager with the default
          budget.
        - If both are given, and the budget differs from
          `budgetmanager.budget`, throw a warning and the budget manager is
          used as is.
    budget : float, default=None
        Specifies the ratio of samples which are allowed to be sampled, with
        `0 <= budget <= 1`. If `budget` is `None`, it is replaced with the
        default budget 0.1.
    random_state : int or RandomState instance, default=None
        Controls the randomness of the estimator.

    References
    ----------
    .. [1] I. Žliobaitė, A. Bifet, B. Pfahringer, and G. Holmes. Active
        Learning With Drifting Streaming Data. IEEE Trans. Neural Netw. Learn.
        Syst., 25(1):27–39, 2014.
    """

    def _get_default_budget_manager(self):
        """Provide the budget manager that will be used as default.

        Returns
        -------
        budget_manager : BudgetManager
            The BudgetManager that should be used by default.
        """
        return RandomVariableUncertaintyBudgetManager


class Split(UncertaintyZliobaite):
    """Split

    The Split query strategy (Split Strategy in [1]_) queries labels based on
    the classifiers uncertainty assessed based on the classifier's predictions.
    The sample is queried when the probability of the most likely class
    exceeds a time-dependent threshold calculated based on the budget, number
    of observed and acquired samples. It is a hybrid strategy that combines
    `VariableUncertainty` with randomly sampling samples with a given
    probability. See also :class:`.SplitBudgetManager`

    Parameters
    ----------
    budget_manager : BudgetManager, default=None
        The BudgetManager which models the budgeting constraint used in the
        stream-based active learning setting. if set to `None`,
        `SplitBudgetManager` will be used by default. The budget manager will
        be initialized based on the following conditions:

        - If only a `budget` is given, the default budget manager is
          initialized with the given budget.
        - If only a budget manager is given, use the budget manager.
        - If both are not given, the default budget manager with the default
          budget.
        - If both are given, and the budget differs from
          `budgetmanager.budget`, throw a warning and the budget manager is
          used as is.
    budget : float, default=None
        Specifies the ratio of samples which are allowed to be sampled, with
        `0 <= budget <= 1`. If `budget` is `None`, it is replaced with the
        default budget 0.1.
    random_state : int or RandomState instance, default=None
        Controls the randomness of the estimator.

    References
    ----------
    .. [1] I. Žliobaitė, A. Bifet, B. Pfahringer, and G. Holmes. Active
        Learning With Drifting Streaming Data. IEEE Trans. Neural Netw. Learn.
        Syst., 25(1):27–39, 2014
    """

    def _get_default_budget_manager(self):
        """Provide the budget manager that will be used as default.

        Returns
        -------
        budget_manager : BudgetManager
            The BudgetManager that should be used by default.
        """
        return SplitBudgetManager
