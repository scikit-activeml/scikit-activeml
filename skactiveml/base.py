import warnings
from abc import ABC, abstractmethod

import numpy as np
from copy import deepcopy
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import accuracy_score
from sklearn.utils import check_array, check_consistent_length
from sklearn.utils.multiclass import type_of_target

from skactiveml.utils import MISSING_LABEL, check_classifier_params, \
    check_random_state, rand_argmin, ExtLabelEncoder, check_cost_matrix, \
    is_labeled, check_scalar, check_class_prior


class QueryStrategy(ABC, BaseEstimator):
    """Base class for all query strategies in scikit-activeml.

    Parameters
    ----------
    random_state : int, RandomState instance, default=None
        Controls the randomness of the estimator.
    """

    def __init__(self, random_state=None):
        self.random_state = random_state

    @abstractmethod
    def query(self, *args, **kwargs):
        """Determines the query for active learning based on input arguments.
        """
        return NotImplemented


class SingleAnnotPoolBasedQueryStrategy(QueryStrategy):
    """Base class for all pool-based active learning query strategies with a
    single annotator in scikit-activeml.

    Parameters
    ----------
    random_state : int, RandomState instance, default=None
        Controls the randomness of the estimator.
    """

    def __init__(self, random_state=None):
        super().__init__(random_state=random_state)

    @abstractmethod
    def query(self, X_cand, *args, batch_size=1, return_utilities=False,
              **kwargs):
        """Determines which for which candidate samples labels are to be
        queried.

        Parameters
        ----------
        X_cand : array-like, shape (n_samples, n_features)
            Candidate samples from which the strategy can select.
        batch_size : int, optional (default=1)
            The number of samples to be selected in one AL cycle.
        return_utilities : bool, optional (default=False)
            If true, also return the utilities based on the query strategy.

        Returns
        -------
        query_indices : numpy.ndarray, shape (batch_size)
            The query_indices indicate for which candidate sample a label is
            to queried, e.g., `query_indices[0]` indicates the first selected
            sample.
        utilities : numpy.ndarray, shape (batch_size, n_samples)
            The utilities of all candidate samples after each selected
            sample of the batch, e.g., `utilities[0]` indicates the utilities
            used for selecting the first sample (with index `query_indices[0]`)
            of the batch.
        """
        return NotImplemented

    def _validate_data(self, X_cand, return_utilities, batch_size,
                       random_state, reset=True, **check_X_cand_params):
        """Validate input data and set or check the `n_features_in_` attribute.

        Parameters
        ----------
        X_cand: array-like, shape (n_candidates, n_features)
            Candidate samples.
        batch_size : int,
            The number of samples to be selected in one AL cycle.
        return_utilities : bool,
            If true, also return the utilities based on the query strategy.
        random_state : numeric | np.random.RandomState, optional
            The random state to use.
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
        batch_size : int
            Checked number of samples to be selected in one AL cycle.
        return_utilities : bool,
            Checked boolean value of `return_utilities`.
        random_state : np.random.RandomState,
            Checked random state to use.
        """
        # Check candidate instances.
        X_cand = check_array(X_cand, **check_X_cand_params)

        # Check number of features.
        self._check_n_features(X_cand, reset=reset)

        # Check return_utilities.
        check_scalar(return_utilities, 'return_utilities', bool)

        # Check batch size.
        check_scalar(batch_size, target_type=int, name='batch_size',
                     min_val=1)
        batch_size = batch_size
        if len(X_cand) < batch_size:
            warnings.warn(
                "'batch_size={}' is larger than number of candidate samples "
                "in 'X_cand'. Instead, 'batch_size={}' was set ".format(
                    batch_size, len(X_cand)))
            batch_size = len(X_cand)

        # Check random state.
        random_state = check_random_state(random_state=self.random_state,
                                          seed_multiplier=len(X_cand))

        return X_cand, return_utilities, batch_size, random_state


class MultiAnnotPoolBasedQueryStrategy(QueryStrategy):
    """Base class for all pool-based active learning query strategies with
    multiple annotators in scikit-activeml.

    Parameters
    ----------
    random_state : int, RandomState instance, default=None
        Controls the randomness of the estimator.
    """
    def __init__(self, random_state=None):
        super().__init__(random_state=random_state)

    @abstractmethod
    def query(self, X_cand, *args, A_cand=None, batch_size=1,
              return_utilities=False, **kwargs):
        """Determines which candidate sample is to be annotated by which
        annotator.

        Parameters
        ----------
        X_cand : array-like, shape (n_samples, n_features)
            Candidate samples from which the strategy can select.
        A_cand : array-like, shape (n_samples, n_features), optional
        (default=None)
            Boolean matrix where `A_cand[i,j] = True` indicates that
            annotator `j` can be selected for annotating sample `X_cand[i]`,
            while `A_cand[i,j] = False` indicates that annotator `j` cannot be
            selected for annotating sample `X_cand[i]`. If A_cand=None, each
            annotator is assumed to be available for labeling each sample.
        batch_size : int, optional (default=1)
            The number of samples to be selected in one AL cycle.
        return_utilities : bool, optional (default=False)
            If true, also return the utilities based on the query strategy.

        Returns
        -------
        query_indices : numpy.ndarray, shape (batch_size, 2)
            The query_indices indicate which candidate sample is to be
            annotated by which annotator, e.g., `query_indices[:, 0]`
            indicates the selected candidate samples and `query_indices[:, 1]`
            indicates the respectively selected annotators.
        utilities: numpy.ndarray, shape (batch_size, n_samples, n_annotators)
            The utilities of all candidate samples w.r.t. to the available
            annotators after each selected sample of the batch, e.g.,
            `utilities[0, :, j]` indicates the utilities used for selecting
            the first sample-annotator pair (with indices `query_indices[0]`).
        """
        return NotImplemented


class BudgetManager(ABC, BaseEstimator):
    """Base class for all budget managers for stream-based active learning
       in scikit-activeml to model budgeting constraints.

    Parameters
    ----------
    budget : float (default=None)
        Specifies the ratio of instances which are allowed to be sampled, with
        0 <= budget <= 1. If budget is None, it is replaced with the default
        budget
    """

    def __init__(self, budget=None):
        self.budget = budget

    @abstractmethod
    def is_budget_left(self):
        """Check whether there is any utility given to query(...), which may
        lead to sampling the corresponding instance, i.e., check if sampling
        another instance is currently possible under the specified budgeting
        constraint. This function is useful to determine, whether a provided
        utility is not sufficient, or the budgeting constraint was simply
        exhausted.

        Returns
        -------
        budget_left : bool
            True, if there is a utility which leads to sampling another
            instance.
        """
        return NotImplemented

    @abstractmethod
    def query(
        self, utilities, return_budget_left=True, **kwargs
    ):
        """Ask the budget manager which utilities are sufficient to query the
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
        
        Returns
        -------
        queried_indices : ndarray of shape (n_queried_instances,)
            The indices of instances represented by utilities which should be
            queried, with 0 <= n_queried_instances <= n_samples.
        
        budget_left: ndarray of shape (n_samples,), optional
            Shows whether there was budget left for each assessed utility. Only
            provided if return_utilities is True.
        """
        return NotImplemented

    @abstractmethod
    def update(self, queried, **kwargs):
        """Updates the BudgetManager.

        Parameters
        ----------
        queried : array-like
            Indicates which instances from X_cand have been queried.

        Returns
        -------
        self : BudgetManager
            The BudgetManager returns itself, after it is updated.
        """
        return NotImplemented

    def _validate_budget(self, default=None):
        """check the assigned budget and set a default value, when none is set
        prior.

        Parameters
        ----------
        default : float, optional
            the budget which should be assigned, when none is set.
        """
        if self.budget is not None:
            self.budget_ = self.budget
        else:
            if default is None:
                default = self.get_default_stream_budget()
            self.budget_ = default
        check_scalar(
            self.budget_, "budget", float, min_val=0.0, max_val=1.0
        )

    def _validate_data(self, utilities, return_budget_left):
        """Validate input data.

        Parameters
        ----------
        utilities: ndarray of shape (n_samples,)
            The utilities provided by the stream-based active learning
            strategy.
        return_budget_left : bool,
            If true, also return whether there was budget left for each
            assessed utility.

        Returns
        -------
        utilities: ndarray of shape (n_samples,)
            Checked utilities
        return_budget_left : bool,
            Checked boolean value of `return_budget_left`.
        """
        # Check if utilities is set
        if not isinstance(utilities, np.ndarray):
            raise TypeError(
                "{} is not a valid type for utilities".format(type(utilities))
            )
        # Check return_utilities.
        check_scalar(return_budget_left, "return_budget_left", bool)
        # Check budget
        self._validate_budget(self.get_default_stream_budget())
        return utilities, return_budget_left

    def get_default_stream_budget(self):
        """This function defines the default budget which should be used when no
        budget is provided by the user.

            Returns
            -------
            default_budget: float
                The default budget used by the user.
            """
        return 0.1


class SingleAnnotStreamBasedQueryStrategy(QueryStrategy):
    """Base class for all stream-based active learning query strategies in
       scikit-activeml.

    Parameters
    ----------
    budget_manager : BudgetManager
        The BudgetManager which models the budgeting constraint used in
        the stream-based active learning setting.

    random_state : int, RandomState instance, default=None
        Controls the randomness of the estimator.
    """

    def __init__(self, budget_manager, random_state=None):
        super().__init__(random_state=random_state)
        self.budget_manager = budget_manager

    @abstractmethod
    def query(
        self, X_cand, *args, return_utilities=False, **kwargs
    ):
        """Ask the query strategy which instances in X_cand to acquire.

        The query startegy determines the most useful instances in X_cand,
        which can be acquired within the budgeting constraint specified by the
        budget_manager.
        Please note that, when the decisions from this function
        may differ from the final sampling, simulate=True can set, so that the
        query strategy can be updated later with update(...) with the final
        sampling. This is especially helpful, when developing wrapper query
        strategies.

        Parameters
        ----------
        X_cand : {array-like, sparse matrix} of shape (n_samples, n_features)
            The instances which may be queried. Sparse matrices are accepted
            only if they are supported by the base query strategy.

        return_utilities : bool, optional
            If true, also return the utilities based on the query strategy.
            The default is False.

        Returns
        -------
        queried_indices : ndarray of shape (n_sampled_instances,)
            The indices of instances in X_cand which should be sampled, with
            0 <= n_sampled_instances <= n_samples.

        utilities: ndarray of shape (n_samples,), optional
            The utilities based on the query strategy. Only provided if
            return_utilities is True.
        """
        return NotImplemented

    @abstractmethod
    def update(self, X_cand, queried, *args, **kwargs):
        """Update the query strategy with the decisions taken.

        This function should be used in conjunction with the query function,
        when the instances queried from query(...) may differ from the
        instances queried in the end. In this case use query(...) with
        simulate=true and provide the final decisions via update(...).
        This is especially helpful, when developing wrapper query strategies.

        Parameters
        ----------
        X_cand : {array-like, sparse matrix} of shape (n_samples, n_features)
            The instances which could be queried. Sparse matrices are accepted
            only if they are supported by the base query strategy.

        queried : array-like
            Indicates which instances from X_cand have been queried.

        Returns
        -------
        self : StreamBasedQueryStrategy
            The StreamBasedQueryStrategy returns itself, after it is updated.
        """
        return NotImplemented

    def _validate_random_state(self):
        """Creates a copy 'random_state_' if random_state is an instance of
        np.random_state. If not create a new random state. See also
        :func:`~sklearn.utils.check_random_state`
        """
        if not hasattr(self, "random_state_"):
            self.random_state_ = deepcopy(self.random_state)
        self.random_state_ = check_random_state(self.random_state_)

    def _validate_budget_manager(self):
        """Validate if budget manager is a budget_manager class and create a
        copy 'budget_manager_'.
        """
        if not hasattr(self, "budget_manager_"):
            self.budget_manager_ = clone(self.budget_manager)
        if not isinstance(self.budget_manager_, BudgetManager):
            raise TypeError(
                "{} is not a valid Type for budget_manager".format(
                    type(self.budget_manager_)
                )
            )

    def _validate_data(
        self,
        X_cand,
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
        return_utilities : bool,
            Checked boolean value of `return_utilities`.
        random_state : np.random.RandomState,
            Checked random state to use.
        """
        # Check candidate instances.
        X_cand = check_array(X_cand, **check_X_cand_params)

        # Check number of features.
        self._check_n_features(X_cand, reset=reset)

        # Check return_utilities.
        check_scalar(return_utilities, "return_utilities", bool)


        # Check random state.
        self._validate_random_state()

        # Check budget_manager.
        self._validate_budget_manager()

        return X_cand, return_utilities


class SingleAnnotStreamBasedQueryStrategyWrapper(QueryStrategy):
    """Base class for wrappers that modify the behavior of another pre-existing
    query strategy (base_query_strategy). Ultimately, the difference between
    SingleAnnotStreamBasedQueryStrategy and this class is that no budget is 
    needed as it is predefined in base_query_strategy and the forwarding of
    accesses.
    

    Parameters
    ----------
    base_query_strategy : SingleAnnotStreamBasedQueryStrategy
        The query strategy that should be wrapped. All function calls and
        variable accesses are forwarded to the base_query_strategy if the
        function or variable is not present within the wrapper itself.

    random_state : int, RandomState instance, default=None
        Controls the randomness of the estimator.
    """

    def __init__(self, base_query_strategy, random_state=None):
        super().__init__(random_state=random_state)
        self.base_query_strategy = base_query_strategy

    @abstractmethod
    def query(
        self, X_cand, *args, return_utilities=False, **kwargs
    ):
        """Ask the query strategy which instances in X_cand to acquire.

        The query startegy determines the most useful instances in X_cand,
        which can be acquired within the budgeting constraint specified by the
        budget_manager.
        Please note that, when the decisions from this function
        may differ from the final sampling, simulate=True can set, so that the
        query strategy can be updated later with update(...) with the final
        sampling. This is especially helpful, when developing wrapper query
        strategies.

        Parameters
        ----------
        X_cand : {array-like, sparse matrix} of shape (n_samples, n_features)
            The instances which may be queried. Sparse matrices are accepted
            only if they are supported by the base query strategy.

        return_utilities : bool, optional
            If true, also return the utilities based on the query strategy.
            The default is False.

        Returns
        -------
        queried_indices : ndarray of shape (n_queried_instances,)
            The indices of instances in X_cand which should be queried, with
            0 <= n_sampled_instances <= n_samples.

        utilities: ndarray of shape (n_samples,), optional
            The utilities based on the query strategy. Only provided if
            return_utilities is True.
        """
        return NotImplemented

    @abstractmethod
    def update(self, X_cand, queried, *args, **kwargs):
        """Update the query strategy with the decisions taken.

        This function should be used in conjunction with the query function,
        when the instances queried from query(...) may differ from the
        instances queried in the end. In this case use query(...) with
        simulate=true and provide the final decisions via update(...).
        This is especially helpful, when developing wrapper query strategies.

        Parameters
        ----------
        X_cand : {array-like, sparse matrix} of shape (n_samples, n_features)
            The instances which could be queried. Sparse matrices are accepted
            only if they are supported by the base query strategy.

        queried : array-like
            Indicates which instances from X_cand have been queried.

        Returns
        -------
        self : StreamBasedQueryStrategy
            The StreamBasedQueryStrategy returns itself, after it is updated.
        """
        return NotImplemented

    def __getattr__(self, item):
        if item in self.__dict__ or item == "base_query_strategy_":
            if "base_query_strategy_" not in self.__dict__:
                return getattr(self.base_query_strategy, item)
            return self.__dict__[item]
        elif "base_query_strategy_" in self.__dict__:
            return getattr(self.base_query_strategy_, item)
        else:
            self._validate_base_query_strategy()
            return getattr(self.base_query_strategy_, item)

    def _validate_random_state(self):
        """Creates a copy 'random_state_' if random_state is an instance of
        np.random_state. If not create a new random state. See also
        :func:`~sklearn.utils.check_random_state`
        """
        if not hasattr(self, "random_state_"):
            self.random_state_ = deepcopy(self.random_state)
        self.random_state_ = check_random_state(self.random_state_)

    def _validate_base_query_strategy(self):
        """Validate if query strategy is a query_strategy class and create a
        copy 'base_query_strategy_'.
        """
        if not hasattr(self, "base_query_strategy_"):
            self.base_query_strategy_ = clone(self.base_query_strategy)
        if not (isinstance(
            self.base_query_strategy_, SingleAnnotStreamBasedQueryStrategy
        ) or isinstance(
            self.base_query_strategy_,
            SingleAnnotStreamBasedQueryStrategyWrapper
        )):
            raise TypeError(
                "{} is not a valid Type for query_strategy".format(
                    type(self.base_query_strategy_)
                )
            )

    def _validate_data(
        self,
        X_cand,
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
        return_utilities : bool,
            Checked boolean value of `return_utilities`.
        random_state : np.random.RandomState,
            Checked random state to use.
        """
        # Check candidate instances.
        X_cand = check_array(X_cand, **check_X_cand_params)

        # Check number of features.
        self._check_n_features(X_cand, reset=reset)

        # Check return_utilities.
        check_scalar(return_utilities, "return_utilities", bool)

        # Check random state.
        self._validate_random_state()

        # Check base_query_strategy.
        self._validate_base_query_strategy()

        return X_cand, return_utilities


class SkactivemlClassifier(BaseEstimator, ClassifierMixin, ABC):
    """SkactivemlClassifier

    Base class for scikit-activeml classifiers such that missing labels,
    user-defined classes, cost-sensitive classification (i.e., cost matrix),
    and multiple labels per sample can be handled.

    Parameters
    ----------
    classes : array-like, shape (n_classes), default=None
        Holds the label for each class. If none, the classes are determined
        during the fit.
    missing_label : {scalar, string, np.nan, None}, default=np.nan
        Value to represent a missing label.
    cost_matrix : array-like, shape (n_classes, n_classes)
        Cost matrix with cost_matrix[i,j] indicating cost of predicting class
        classes[j]  for a sample of class classes[i]. Can be only set, if
        classes is not none.
    random_state : int, RandomState instance or None, optional (default=None)
        Determines random number for 'predict' method. Pass an int for
        reproducible results across multiple method calls.

    Attributes
    ----------
    classes_ : array-like, shape (n_classes)
        Holds the label for each class after fitting.
    cost_matrix_ : array-like, shape (classes, classes)
        Cost matrix with C[i,j] indicating cost of predicting class classes_[j]
        for a sample of class classes_[i].
    """
    def __init__(self, classes, missing_label=MISSING_LABEL, cost_matrix=None,
                 random_state=None):
        self.classes = classes
        self.missing_label = missing_label
        self.cost_matrix = cost_matrix
        self.random_state = random_state

    @abstractmethod
    def fit(self, X, y, sample_weight=None):
        """Fit the model using X as training data and y as class labels.

        Parameters
        ----------
        X : matrix-like, shape (n_samples, n_features)
            The sample matrix X is the feature matrix representing the samples.
        y : array-like, shape (n_samples) or (n_samples, n_outputs)
            It contains the class labels of the training samples.
            The number of class labels may be variable for the samples, where
            missing labels are represented the attribute 'missing_label'.
        sample_weight : array-like, shape (n_samples) or (n_samples, n_outputs)
            It contains the weights of the training samples' class labels.
            It must have the same shape as y.

        Returns
        -------
        self: SkactivemlClassifier,
            The SkactivemlClassifier is fitted on the training data.
        """
        return NotImplemented

    def predict_proba(self, X):
        """Return probability estimates for the test data X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Test samples.

        Returns
        -------
        P : numpy.ndarray, shape (n_samples, classes)
            The class probabilities of the test samples. Classes are ordered
            according to 'classes_'.
        """
        return NotImplemented

    def predict(self, X):
        """Return class label predictions for the test samples X.

        Parameters
        ----------
        X :  array-like, shape (n_samples, n_features) or
        shape (n_samples, m_samples) if metric == 'precomputed'
            Input samples.

        Returns
        -------
        y : numpy.ndarray, shape (n_samples)
            Predicted class labels of the test samples 'X'. Classes are ordered
            according to 'classes_'.
        """
        P = self.predict_proba(X)
        costs = np.dot(P, self.cost_matrix_)
        y_pred = rand_argmin(costs, random_state=self._random_state, axis=1)
        y_pred = self._le.inverse_transform(y_pred)
        y_pred = np.asarray(y_pred, dtype=self.classes_.dtype)
        return y_pred

    def score(self, X, y, sample_weight=None):
        """Return the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True labels for X.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) wrt. y.
        """
        y = self._le.transform(y)
        y_pred = self._le.transform(self.predict(X))
        return accuracy_score(y, y_pred, sample_weight=sample_weight)

    def _validate_data(self, X, y, sample_weight):
        # Check common classifier parameters.
        check_classifier_params(self.classes, self.missing_label,
                                self.cost_matrix)
        # Store and check random state.
        self._random_state = check_random_state(self.random_state)

        # Create label encoder.
        self._le = ExtLabelEncoder(classes=self.classes,
                                   missing_label=self.missing_label)

        # Check input parameters.
        X = np.array(X)
        y = np.array(y)
        check_consistent_length(X, y)
        if len(X) > 0:
            X = check_array(X)
            is_lbdl = is_labeled(y, self.missing_label)
            if len(y[is_lbdl]) > 0:
                y_type = type_of_target(y[is_lbdl])
                if y_type not in [
                    'binary', 'multiclass', 'multiclass-multioutput',
                    'multilabel-indicator', 'multilabel-sequences', 'unknown'
                ]:
                    raise ValueError("Unknown label type: %r" % y_type)

            y = self._le.fit_transform(y)
            if len(self._le.classes_) == 0:
                raise ValueError(
                    "No class label is known because 'y' contains no actual "
                    "class labels and 'classes' is not defined. Change at "
                    "least on of both to overcome this error."
                )
        else:
            self._le.fit_transform(self.classes)

        # Update detected classes.
        self.classes_ = self._le.classes_

        # Check classes.
        if sample_weight is not None:
            sample_weight = np.array(sample_weight)
            if not np.array_equal(y.shape, sample_weight.shape):
                raise ValueError(
                    f'`y` has the shape {y.shape} and `sample_weight` has the '
                    f'shape {sample_weight.shape}. Both need to have identical'
                    f' shapes.'
                )

        # Update cost matrix.
        self.cost_matrix_ = 1 - np.eye(len(self.classes_)) \
            if self.cost_matrix is None else self.cost_matrix
        self.cost_matrix_ = check_cost_matrix(self.cost_matrix_,
                                              len(self.classes_))
        if self.classes is not None:
            class_indices = np.argsort(self.classes)
            self.cost_matrix_ = self.cost_matrix_[class_indices]
            self.cost_matrix_ = self.cost_matrix_[:, class_indices]
        return X, y, sample_weight

    def _check_n_features(self, X, reset):
        if reset:
            self.n_features_in_ = X.shape[1] if len(X) > 0 else None
        elif not reset:
            if self.n_features_in_ is not None:
                super()._check_n_features(X, reset=reset)


class ClassFrequencyEstimator(SkactivemlClassifier):
    """ClassFrequencyEstimator

    Extends scikit-activeml classifiers to estimators that are able to estimate
    class frequencies for given samples (by calling 'predict_freq').

    Parameters
    ----------
    classes : array-like, shape (n_classes), default=None
        Holds the label for each class. If none, the classes are determined
        during the fit.
    missing_label : scalar | string | np.nan | None|, default=np.nan
        Value to represent a missing label.
    cost_matrix : array-like, shape (n_classes, n_classes)
        Cost matrix with `cost_matrix[i,j]` indicating cost of predicting class
        `classes[j]`  for a sample of class `classes[i]`. Can be only set, if
        classes is not none.
    class_prior : float | array-like, shape (n_classes), optional (default=0)
        Prior observations of the class frequency estimates. If `class_prior`
        is an array, the entry `class_prior[i]` indicates the non-negative
        prior number of samples belonging to class `classes_[i]`. If
        `class_prior` is a float, `class_prior` indicates the non-negative
        prior number of samples per class.
    random_state : int | np.RandomState | None, optional (default=None)
        Determines random number for 'predict' method. Pass an int for
        reproducible results across multiple method calls.

    Attributes
    ----------
    classes_ : np.ndarray, shape (n_classes)
        Holds the label for each class after fitting.
    class_prior_ : np.ndarray, shape (n_classes)
        Prior observations of the class frequency estimates. The entry
        `class_prior_[i]` indicates the non-negative prior number of samples
        belonging to class `classes_[i]`.
    cost_matrix_ : np.ndarray, shape (classes, classes)
        Cost matrix with `cost_matrix_[i,j]` indicating cost of predicting
        class `classes_[j]` for a sample of class `classes_[i]`.
    """
    @abstractmethod
    def predict_freq(self, X):
        """Return class frequency estimates for the test samples X.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Test samples whose class frequencies are to be estimated.

        Returns
        -------
        F: array-like, shape (n_samples, classes)
            The class frequency estimates of the test samples 'X'. Classes are
            ordered according to attribute 'classes_'.
        """
        return NotImplemented

    def predict_proba(self, X):
        """Return probability estimates for the test data X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features) or
        shape (n_samples, m_samples) if metric == 'precomputed'
            Input samples.

        Returns
        -------
        P : array-like, shape (n_samples, classes)
            The class probabilities of the test samples. Classes are ordered
            according to classes_.
        """
        # Normalize probabilities of each sample.
        P = self.predict_freq(X) + self.class_prior_
        normalizer = np.sum(P, axis=1)
        P[normalizer > 0] /= normalizer[normalizer > 0, np.newaxis]
        P[normalizer == 0, :] = [1 / len(self.classes_)] * len(self.classes_)
        return P

    def _validate_data(self, X, y, sample_weight):
        X, y, sample_weight = super()._validate_data(X, y, sample_weight)
        # Check class prior.
        self.class_prior_ = check_class_prior(self.class_prior,
                                              len(self.classes_))
        return X, y, sample_weight


class AnnotModelMixing(ABC):
    """AnnotModelMixing

    Base class of all annotator models estimating the performances of
    annotators for given samples.
    """
    @abstractmethod
    def predict_annot_proba(self, X):
        """Calculates the probability that an annotator provides the true label
        for a given sample.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Test samples.

        Returns
        -------
        P_annot : numpy.ndarray, shape (n_samples, n_annotators)
            `P_annot[i,l]` is the probability, that annotator `l` provides the
            correct class label for sample `X[i]`.
        """
        return NotImplemented
