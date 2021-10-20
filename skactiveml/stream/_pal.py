import numpy as np

from sklearn.utils import check_array, check_consistent_length

import skactiveml.pool._probal as probal
from ..base import SingleAnnotStreamBasedQueryStrategy, SkactivemlClassifier

# from ..classifier import PWC
from ..utils import fit_if_not_fitted, check_type, check_random_state

from .budget_manager import BIQF


class PAL(SingleAnnotStreamBasedQueryStrategy):
    """Probabilistic Active Learning (PAL) in Datastreams is an extension to
    Multi-Class Probabilistic Active Learning (see pool.McPAL). It assesses
    MCPAL spatial to assess the spatial utility. The Balanced Incremental
    Quantile Filter (BIQF), that is implemented within the default
    budget_manager, is used to evaluate the temporal utility
    (see stream.budget_manager.BIQF).

    Parameters
    ----------
    budget_manager : BudgetManager
        The BudgetManager which models the budgeting constraint used in
        the stream-based active learning setting. The budget attribute set for
        the budget_manager will be used to determine the probability to sample
        instances
    random_state : int, RandomState instance, default=None
        Controls the randomness of the estimator.
    prior : float
        The prior value that is passed onto McPAL (see pool.McPAL).
    m_max : float
        The m_max value that is passed onto McPAL (see pool.McPAL).

    References
    ----------
    [1] Kottke D., Krempl G., Spiliopoulou M. (2015) Probabilistic Active
        Learning in Datastreams. In: Fromont E., De Bie T., van Leeuwen M.
        (eds) Advances in Intelligent Data Analysis XIV. IDA 2015. Lecture
        Notes in Computer Science, vol 9385. Springer, Cham.
    """

    def __init__(
        self, budget_manager=BIQF(), random_state=None, prior=1.0e-3, m_max=2,
    ):
        self.budget_manager = budget_manager
        self.random_state = random_state
        self.prior = prior
        self.m_max = m_max

    def query(
        self,
        X_cand,
        clf,
        X=None,
        y=None,
        sample_weight=None,
        return_utilities=False,
        **kwargs
    ):
        """Ask the query strategy which instances in X_cand to acquire.

        Please note that, when the decisions from this function may differ from
        the final sampling, simulate=True can be set, so that the query
        strategy can be updated later with update(...) with the final sampling.
        This is especially helpful, when developing wrapper query strategies.

        Parameters
        ----------
        X_cand : {array-like, sparse matrix} of shape (n_samples, n_features)
            The instances which may be queried. Sparse matrices are accepted
            only if they are supported by the base query strategy.
        clf : SkactivemlClassifier
            Model implementing the methods `fit` and `predict_freq`.
        X : array-like of shape (n_samples, n_features), optional (default=None)
            Input samples used to fit the classifier.
        y : array-like of shape (n_samples), optional (default=None)
            Labels of the input samples 'X'. There may be missing labels.
        sample_weight : array-like of shape (n_samples,) (default=None)
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
            X_cand=X_cand,
            clf=clf,
            X=X,
            y=y,
            sample_weight=sample_weight,
            return_utilities=return_utilities,
        )

        k_vec = clf.predict_freq(X_cand)
        utilities = probal.cost_reduction(
            k_vec, prior=self.prior, m_max=self.m_max
        )
        queried_indices = self.budget_manager_.query(utilities)

        if return_utilities:
            return queried_indices, utilities
        else:
            return queried_indices

    def update(
        self, X_cand, queried_indices, budget_manager_kwargs={}, **kwargs
    ):
        """Updates the budget manager

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
        self : PAL
            PAL returns itself, after it is updated.
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
        X_cand: array-like, shape (n_candidates, n_features)
            Candidate samples.
        X : array-like of shape (n_samples, n_features)
            Input samples used to fit the classifier.
        y : array-like of shape (n_samples)
            Labels of the input samples 'X'. There may be missing labels.
        clf : SkactivemlClassifier
            Model implementing the methods `fit` and `predict_freq`.
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
        self._validate_X_y_sample_weight(X, y, sample_weight)
        self._validate_clf(clf, X, y, sample_weight)
        self._validate_prior()
        self._validate_m_max()
        self._validate_random_state()

        return X_cand, clf, X, y, sample_weight, return_utilities

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

    def _validate_clf(self, clf, X, y, sample_weight):
        """Validate if clf is a valid SkactivemlClassifier. If clf is
        untrained, clf is trained using X, y and sample_weight.

        Parameters
        ----------
        clf : SkactivemlClassifier
            Model implementing the methods `fit` and `predict_freq`.
        """
        # Check if the classifier and its arguments are valid.
        check_type(clf, SkactivemlClassifier, "clf")
        return fit_if_not_fitted(clf, X, y, sample_weight)

    def _validate_prior(self):
        """Validate if the prior is a float and greater than 0.
        """
        if not isinstance(self.prior, float) and not None:
            raise TypeError(
                "{} is not a valid type for prior".format(type(self.prior))
            )
        if self.prior <= 0:
            raise ValueError(
                "The value of prior is incorrect."
                + " prior must be greater than 0"
            )

    def _validate_m_max(self):
        """Validate if the m_max is an integer and greater than 0.
        """
        # check if m_max is set
        if not isinstance(self.m_max, int):
            raise TypeError(
                "{} is not a valid type for m_max".format(type(self.m_max))
            )
        if self.m_max <= 0:
            raise ValueError(
                "The value of m_max is incorrect."
                + " m_max must be greater than 0"
            )

    def _validate_random_state(self):
        """Creates a copy 'random_state_' if random_state is an instance of
        np.random_state. If not create a new random state. See also
        :func:`~sklearn.utils.check_random_state`
        """
        if not hasattr(self, "random_state_"):
            self.random_state_ = self.random_state
        self.random_state_ = check_random_state(self.random_state_)
