import numpy as np

from abc import abstractmethod
from copy import deepcopy
from ...base import SingleAnnotStreamBasedQueryStrategy, SkactivemlClassifier
from ...utils import fit_if_not_fitted, check_type, check_random_state
from sklearn.base import is_classifier, clone
from ...classifier import PWC

from sklearn.utils import check_array, check_scalar, check_consistent_length
from skactiveml.base import SingleAnnotStreamBasedQueryStrategyWrapper
from .._uncertainty import Split

from skactiveml.utils import check_random_state


class SingleAnnotStreamBasedQueryStrategyDelayWrapper(
    SingleAnnotStreamBasedQueryStrategyWrapper
):
    """Base class for stream-based active learning query strategies that can
    incorparate known verification_latency in scikit-activeml.

    Parameters
    ----------
    base_query_strategy : QuaryStrategy
        The QuaryStrategy which evaluates the utility of given instances used
        in the stream-based active learning setting.

    random_state : int, RandomState instance, default=None
        Controls the randomness of the estimator.
    """

    def __init__(self, base_query_strategy, random_state=None):
        super().__init__(
            base_query_strategy=base_query_strategy, random_state=random_state
        )
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
            0 <= n_queried_instances <= n_samples.

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

    def _validate_base_query_strategy(self):
        """Validate if query strategy is a query_strategy class and create a
        copy 'base_query_strategy_'.
        """

        if not hasattr(self, "base_query_strategy_"):
            if self.base_query_strategy is None:
                self.base_query_strategy_ = Split()
            else:
                self.base_query_strategy_ = clone(self.base_query_strategy)
        if not (
            isinstance(
                self.base_query_strategy_, SingleAnnotStreamBasedQueryStrategy
            )
            or isinstance(
                self.base_query_strategy_,
                SingleAnnotStreamBasedQueryStrategyWrapper,
            )
        ):
            raise TypeError(
                "{} is not a valid Type for query_strategy".format(
                    type(self.base_query_strategy_)
                )
            )

    def _validate_X_y_sample_weight(self, X, y, sample_weight):
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
        X = check_array(X)
        y = np.array(y)
        check_consistent_length(X, y)
        return X, y, sample_weight

    def _validate_acquisitions(self, acquisitions):
        """Validate if acquisitions is an boolean and convert it into a
        numpy array.

        Parameters
        ----------
        acquisitions : array-like of shape (n_samples, n_features)
            boolean array of acquired instances.

        Returns
        -------
        acquisitions : array-like of shape (n_samples, n_features)
            Checked boolean array value of `acquisitions`.
        """
        acquisitions = np.array(acquisitions)
        if not acquisitions.dtype == bool:
            raise TypeError(
                "{} is not a valid type for acquisitions".format(
                    acquisitions.dtype
                )
            )
        return acquisitions

    def _validate_tX_ty(self, tX, ty):
        """Validate if tX and ty are numeric and of equal lenght.

        Parameters
        ----------
        tX : array-like of shape (n_samples)
            Arrival time of the input samples 'X'
        ty : array-like of shape (n_samples)
            Arrival time of the Labels 'y'

        Returns
        -------
        tX : array-like of shape (n_samples)
            Checked arrival time List
        ty : array-like of shape (n_samples)
            Checked arrival time of the Labels List
        """
        tX = check_array(tX, ensure_2d=False)
        ty = check_array(ty, ensure_2d=False)
        check_consistent_length(tX, ty)
        return tX, ty

    def _validate_tX_cand_ty_cand(self, tX_cand, ty_cand):
        """Validate if tX_cand and ty_cand are numeric and of equal lenght.

        Parameters
        ----------
        tX_cand : array-like of shape (n_samples)
            Arrival time of the input samples 'X_cand'
        ty_cand : array-like of shape (n_samples)
            Arrival time of the Labels 'y_cand'

        Returns
        -------
        tX_cand : array-like of shape (n_samples)
            Checked arrival time List
        ty_cand : array-like of shape (n_samples)
            Checked arrival time of the Labels List
        """
        tX_cand = check_array(tX_cand, ensure_2d=False)
        ty_cand = check_array(ty_cand, ensure_2d=False)
        check_consistent_length(tX_cand, ty_cand)
        return tX_cand, ty_cand

    def _validate_data(
        self,
        X_cand,
        X,
        y,
        tX,
        ty,
        tX_cand,
        ty_cand,
        acquisitions,
        sample_weight,
        return_utilities,
        reset=True,
        **check_X_cand_params
    ):
        """Validate input data and set or check the `n_features_in_` attribute.

        Parameters
        ----------
        X_cand : {array-like, sparse matrix} of shape (n_samples, n_features)
            The instances which may be queried. Sparse matrices are accepted
            only if they are supported by the base query strategy.
        X : array-like of shape (n_samples, n_features)
            Input samples used to fit the classifier.
        y : array-like of shape (n_samples)
            Labels of the input samples 'X'. There may be missing labels.
        tX : array-like of shape (n_samples)
            Arrival time of the input samples 'X'
        ty : array-like of shape (n_samples)
            Arrival time of the Labels 'y'
        tX_cand : array-like of shape (n_samples)
            Arrival time of the input samples 'X_cand'
        ty_cand : array-like of shape (n_samples)
            Arrival time of the Labels 'y_cand'
        acquisitions : array-like of shape (n_samples)
            List of arrived labels. True if Label arrived otherwise False
        sample_weight : array-like of shape (n_samples,) (default=None)
            Sample weights for X, used to fit the clf.
        return_utilities : bool, optional
            If true, also return the utilities based on the query strategy.
            The default is False.
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
        self._validate_base_query_strategy()
        X_cand, return_utilities = super()._validate_data(
            X_cand,
            return_utilities,
            reset=reset,
            **check_X_cand_params
        )
        X, y, sample_weight = self._validate_X_y_sample_weight(
            X, y, sample_weight
        )
        acquisitions = self._validate_acquisitions(acquisitions)
        tX, ty = self._validate_tX_ty(tX, ty)
        tX_cand, ty_cand = self._validate_tX_cand_ty_cand(tX_cand, ty_cand)

        return (
            X_cand,
            X,
            y,
            tX,
            ty,
            tX_cand,
            ty_cand,
            acquisitions,
            sample_weight,
            return_utilities
        )


class ForgettingWrapper(SingleAnnotStreamBasedQueryStrategyDelayWrapper):
    """This query strategy hides instances that would be obsolete from the
    base_query_strategy that would be unavailable once the label for X_cand
    arrives. The ForgettingWrapper strategy assumes a sliding window with a
    length of w_train to forget obsolete data. Each instance for which
    ty_cand - w_train + verification_latency <= tX does not hold, are
    discarded.

    Parameters
    ----------
    base_query_strategy : QuaryStrategy
        The QuaryStrategy which evaluates the utility of given instances used
        in the stream-based active learning setting.

    w_train : int, default=500
        Size of the forgetting window

    random_state : int, RandomState instance, default=None
        Controls the randomness of the estimator.
    """

    def __init__(
        self, base_query_strategy=None, w_train=500, random_state=None
    ):
        super().__init__(base_query_strategy, random_state)
        self.w_train = w_train

    def query(
        self,
        X_cand,
        clf,
        X,
        y,
        tX,
        ty,
        tX_cand,
        ty_cand,
        acquisitions,
        sample_weight=None,
        return_utilities=False,
        al_kwargs={},
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
        tX : array-like of shape (n_samples)
            Arrival time of the input samples 'X'
        ty : array-like of shape (n_samples)
            Arrival time of the Labels 'y'
        tX_cand : array-like of shape (n_samples)
            Arrival time of the input samples 'X_cand'
        ty_cand : array-like of shape (n_samples)
            Arrival time of the Labels 'y_cand'
        acquisitions : array-like of shape (n_samples)
            List of arrived labels. True if Label arrived otherwise False
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
            X,
            y,
            tX,
            ty,
            tX_cand,
            ty_cand,
            acquisitions,
            sample_weight,
            return_utilities,
        ) = self._validate_data(
            X_cand=X_cand,
            X=X,
            y=y,
            tX=tX,
            ty=ty,
            tX_cand=tX_cand,
            ty_cand=ty_cand,
            acquisitions=acquisitions,
            sample_weight=sample_weight,
            return_utilities=return_utilities,
        )

        utilities = []
        queried_indices = []
        for i, (tX_cand_current, ty_cand_current, X_cand_current) in enumerate(
            zip(tX_cand, ty_cand, X_cand)
        ):
            tX_in_A_n = tX >= (ty_cand_current - self.w_train)
            A_n_X = X[tX_in_A_n, :]
            A_n_tX = tX[tX_in_A_n]
            A_n_y = y[tX_in_A_n]
            A_n_ty = ty[tX_in_A_n]
            A_n_acquisitions = acquisitions[tX_in_A_n]
            A_n_sample_weight = (
                None if sample_weight is None else sample_weight[tX_in_A_n]
            )
            sample, utility = self.base_query_strategy_.query(
                X_cand=X_cand_current.reshape([1, -1]),
                clf=clone(clf),
                X=A_n_X,
                y=A_n_y,
                tX=A_n_tX,
                ty=A_n_ty,
                tX_cand=[tX_cand_current],
                ty_cand=[ty_cand_current],
                acquisitions=A_n_acquisitions,
                sample_weight=A_n_sample_weight,
                return_utilities=True,
                **al_kwargs,
                **kwargs
            )
            if len(sample):
                queried_indices.append(i)
            utilities.append(utility)

        if return_utilities:
            return queried_indices, utilities
        else:
            return queried_indices

    def update(self, X_cand, queried, **kwargs):
        """Updates the budget manager and the count for seen and queried
        instances

        Parameters
        ----------
        X_cand : {array-like, sparse matrix} of shape (n_samples, n_features)
            The instances which could be queried. Sparse matrices are accepted
            only if they are supported by the base query strategy.

        queried : array-like of shape (n_samples,)
            Indicates which instances from X_cand have been queried.

        kwargs : kwargs
            Optional kwargs for budget_manager and query_strategy.

        Returns
        -------
        self : ForgettingWrapper
            The ForgettingWrapper returns itself, after it is updated.
        """
        self._validate_base_query_strategy()
        self.base_query_strategy_.update(X_cand, queried, **kwargs)
        return self

    def _validate_data(
        self,
        X_cand,
        X,
        y,
        tX,
        ty,
        tX_cand,
        ty_cand,
        acquisitions,
        sample_weight,
        return_utilities,
        reset=True,
        **check_X_cand_params
    ):
        """Validate input data and set or check the `n_features_in_` attribute.

        Parameters
        ----------
        X_cand : {array-like, sparse matrix} of shape (n_samples, n_features)
            The instances which may be queried. Sparse matrices are accepted
            only if they are supported by the base query strategy.
        X : array-like of shape (n_samples, n_features)
            Input samples used to fit the classifier.
        y : array-like of shape (n_samples)
            Labels of the input samples 'X'. There may be missing labels.
        tX : array-like of shape (n_samples)
            Arrival time of the input samples 'X'
        ty : array-like of shape (n_samples)
            Arrival time of the Labels 'y'
        tX_cand : array-like of shape (n_samples)
            Arrival time of the input samples 'X_cand'
        ty_cand : array-like of shape (n_samples)
            Arrival time of the Labels 'y_cand'
        acquisitions : array-like of shape (n_samples)
            List of arrived labels. True if Label arrived otherwise False
        sample_weight : array-like of shape (n_samples,) (default=None)
            Sample weights for X, used to fit the clf.
        return_utilities : bool, optional
            If true, also return the utilities based on the query strategy.
            The default is False.
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
        X : array-like of shape (n_samples, n_features)
            Checked input samples used to fit the classifier.
        y : array-like of shape (n_samples)
            Checked Labels of the input samples 'X'. There may be missing labels.
        tX : array-like of shape (n_samples)
            Checked arrival time of the input samples 'X'
        ty : array-like of shape (n_samples)
            Checked arrival time of the Labels 'y'
        tX_cand : array-like of shape (n_samples)
            Checked arrival time of the input samples 'X_cand'
        ty_cand : array-like of shape (n_samples)
            Checked arrival time of the Labels 'y_cand'
        acquisitions : array-like of shape (n_samples)
            List of arrived labels. True if Label arrived otherwise False
        sample_weight : array-like of shape (n_samples,)
            Checked sample weights for X
        return_utilities : bool,
            Checked boolean value of `return_utilities`.
        """
        (
            X_cand,
            X,
            y,
            tX,
            ty,
            tX_cand,
            ty_cand,
            acquisitions,
            sample_weight,
            return_utilities,
        ) = super()._validate_data(
            X_cand=X_cand,
            X=X,
            y=y,
            tX=tX,
            ty=ty,
            tX_cand=tX_cand,
            ty_cand=ty_cand,
            acquisitions=acquisitions,
            sample_weight=sample_weight,
            return_utilities=return_utilities,
            reset=reset,
            **check_X_cand_params
        )

        self._validate_w_train()

        return (
            X_cand,
            X,
            y,
            tX,
            ty,
            tX_cand,
            ty_cand,
            acquisitions,
            sample_weight,
            return_utilities,
        )

    def _validate_w_train(self):
        """Validate if w_train is a positive float.
        """
        if self.w_train is not None:
            if not (
                isinstance(self.w_train, float)
                or isinstance(self.w_train, int)
            ):
                raise TypeError(
                    "{} is not a valid type for w_train".format(
                        type(self.w_train)
                    )
                )
            if self.w_train < 0:
                raise ValueError(
                    "The value of w_train is incorrect."
                    + " w_train must be positive"
                )


class BaggingDelaySimulationWrapper(
    SingleAnnotStreamBasedQueryStrategyDelayWrapper
):
    """The BaggingDelaySimulationWrapper takes already acquired instances
    without labeling into account by simulating 1 to K number of labels and
    estimates the average utility over all simulations.
    To simulate the labels we estimate the class probabilities for each label
    using bayesian estimation.

    Parameters
    ----------
    base_query_strategy : QuaryStrategy
        The QuaryStrategy which evaluates the utility of given instances used
        in the stream-based active learning setting.

    K : int, default=2
        Number of instances that will be Simulated

    delay_prior : float, default=0.001
        Value to correct the predicted frequancy

    clf : BaseEstimator
        The classifier which is trained using this query startegy.

    random_state : int, RandomState instance, default=None
        Controls the randomness of the estimator.
    """
    def __init__(
        self,
        base_query_strategy=None,
        K=2,
        delay_prior=0.001,
        random_state=None,
    ):
        super().__init__(base_query_strategy, random_state)
        self.K = K
        self.delay_prior = delay_prior

    def query(
        self,
        X_cand,
        clf,
        X,
        y,
        tX,
        ty,
        tX_cand,
        ty_cand,
        acquisitions,
        sample_weight=None,
        return_utilities=False,
        al_kwargs={},
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
        tX : array-like of shape (n_samples)
            Arrival time of the input samples 'X'
        ty : array-like of shape (n_samples)
            Arrival time of the Labels 'y'
        tX_cand : array-like of shape (n_samples)
            Arrival time of the input samples 'X_cand'
        ty_cand : array-like of shape (n_samples)
            Arrival time of the Labels 'y_cand'
        acquisitions : array-like of shape (n_samples)
            List of arrived labels. True if Label arrived otherwise False
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
        (
            X_cand,
            X,
            y,
            tX,
            ty,
            tX_cand,
            ty_cand,
            acquisitions,
            sample_weight,
            return_utilities,
        ) = self._validate_data(
            X_cand=X_cand,
            X=X,
            y=y,
            tX=tX,
            ty=ty,
            tX_cand=tX_cand,
            ty_cand=ty_cand,
            acquisitions=acquisitions,
            sample_weight=sample_weight,
            return_utilities=return_utilities,
        )

        # Check if the classifier and its arguments are valid.
        check_type(clf, SkactivemlClassifier, 'clf')

        clf_fitted = fit_if_not_fitted(clf, X, y, sample_weight, print_warning=False)

        sum_utilities = np.zeros(len(X))
        queried_indices = []
        avg_utilities = []

        # A_geq_n_tx = get_selected_A_geq_n_tx(X, y, X_cand, ty, TY_dict)
        for i, (tX_cand_current, ty_cand_current) in enumerate(
            zip(tX_cand, ty_cand)
        ):
            tX_in_A_geq_n = ty >= tX_cand_current
            ty_in_A_geq_n = ty < ty_cand_current

            map_B_n = np.logical_and(
                acquisitions, np.logical_and(tX_in_A_geq_n, ty_in_A_geq_n)
            )
            if np.sum(map_B_n):

                X_B_n = X[map_B_n, :]
                # calculate p^L
                probabilities = self.get_class_probabilities(
                    X_B_n, clf_fitted, X, y, sample_weight
                )
                tmp_queried_indices = []
                tmp_avg_utilities = []
                sum_utilities = np.zeros(self.K)
                # simulate randomly sampleing future instances
                for _ in range(self.K):
                    # list of 0 exept instance to be aquired
                    y_B_n = np.argmax(
                        [
                            self.random_state_.multinomial(1, p_d)
                            for p_d in probabilities
                        ],
                        axis=1,
                    )
                    new_y = np.copy(y)
                    new_y[map_B_n] = y_B_n
                    new_ty = np.copy(ty)
                    new_ty[map_B_n] = (np.max([ty, tX]) + tX_cand_current) / 2
                    _, utilities = self.base_query_strategy_.query(
                        X_cand=X_cand[[i], :],
                        X=X,
                        clf=clone(clf_fitted),
                        y=new_y,
                        tX=tX,
                        ty=new_ty,
                        tX_cand=[tX_cand[i]],
                        ty_cand=[ty_cand[i]],
                        sample_weight=sample_weight,
                        acquisitions=acquisitions,
                        return_utilities=True,
                        **al_kwargs,
                        **kwargs
                    )
                    sum_utilities += utilities[0]
                tmp_avg_utilities = sum_utilities / self.K
                avg_utilities.append(tmp_avg_utilities[0])
            else:
                (
                    tmp_queried_indices,
                    tmp_utilities,
                ) = self.base_query_strategy_.query(
                    X_cand=X_cand[[i], :],
                    clf=clone(clf_fitted),
                    X=X,
                    y=y,
                    tX=tX,
                    ty=ty,
                    tX_cand=[tX_cand[i]],
                    ty_cand=[ty_cand[i]],
                    sample_weight=sample_weight,
                    acquisitions=acquisitions,
                    return_utilities=True,
                    **al_kwargs,
                    **kwargs
                )
                avg_utilities.append(tmp_utilities[0])

        avg_utilities = np.array(avg_utilities)
        queried_indices = self.base_query_strategy_.budget_manager_.query(
            avg_utilities
        )
        queried = np.zeros(len(X_cand))
        queried[queried_indices] = 1
        kwargs = dict(utilities=avg_utilities)

        if return_utilities:
            return queried_indices, avg_utilities
        else:
            return queried_indices

    def get_class_probabilities(self, X_B_n, clf, X, y, sample_weight):
        """Calculate the probabilities for the simulating 'X_B_n' window.

        Parameters
        ----------
        X_B_n : {array-like, sparse matrix} of shape (n_samples, n_features)
            The instances which may be queried. Sparse matrices are accepted
            only if they are supported by the base query strategy.
        X : array-like of shape (n_samples, n_features)
            Input samples used to fit the classifier.
        y : array-like of shape (n_samples)
            Labels of the input samples 'X'. There may be missing labels.
        sample_weight : array-like of shape (n_samples,) (default=None)
            Sample weights for X, used to fit the clf.

        Returns
        -------
        probabilities : ndarray of shape (n_probabilities,)
            List of probabilities for 'X_B_n'
        """
        pwc = clf
        pwc.fit(X=X, y=y, sample_weight=sample_weight)
        frequencies = pwc.predict_freq(X_B_n)
        frequencies_w_prior = frequencies + self.delay_prior
        probabilities = frequencies_w_prior / np.sum(
            frequencies_w_prior, axis=1, keepdims=True
        )
        return probabilities

    def update(self, X_cand, queried, **kwargs):
        """Updates the budget manager and the count for seen and queried
        instances

        Parameters
        ----------
        X_cand : {array-like, sparse matrix} of shape (n_samples, n_features)
            The instances which could be queried. Sparse matrices are accepted
            only if they are supported by the base query strategy.

        queried : array-like of shape (n_samples,)
            Indicates which instances from X_cand have been queried.

        kwargs : kwargs
            Optional kwargs for budget_manager and query_strategy.

        Returns
        -------
        self : BaggingDelaySimulationWrapper
            The BaggingDelaySimulationWrapper returns itself, after it is updated.
        """
        self._validate_base_query_strategy()
        self.base_query_strategy_.update(X_cand, queried, **kwargs)
        return self

    def _validate_data(
        self,
        X_cand,
        X,
        y,
        tX,
        ty,
        tX_cand,
        ty_cand,
        acquisitions,
        sample_weight,
        return_utilities,
        reset=True,
        **check_X_cand_params
    ):
        """Validate input data and set or check the `n_features_in_` attribute.

        Parameters
        ----------
        X_cand : {array-like, sparse matrix} of shape (n_samples, n_features)
            The instances which may be queried. Sparse matrices are accepted
            only if they are supported by the base query strategy.
        X : array-like of shape (n_samples, n_features)
            Input samples used to fit the classifier.
        y : array-like of shape (n_samples)
            Labels of the input samples 'X'. There may be missing labels.
        tX : array-like of shape (n_samples)
            Arrival time of the input samples 'X'
        ty : array-like of shape (n_samples)
            Arrival time of the Labels 'y'
        tX_cand : array-like of shape (n_samples)
            Arrival time of the input samples 'X_cand'
        ty_cand : array-like of shape (n_samples)
            Arrival time of the Labels 'y_cand'
        acquisitions : array-like of shape (n_samples)
            List of arrived labels. True if Label arrived otherwise False
        sample_weight : array-like of shape (n_samples,) (default=None)
            Sample weights for X, used to fit the clf.
        return_utilities : bool, optional
            If true, also return the utilities based on the query strategy.
            The default is False.
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
        X : array-like of shape (n_samples, n_features)
            Checked input samples used to fit the classifier.
        y : array-like of shape (n_samples)
            Checked Labels of the input samples 'X'. There may be missing labels.
        tX : array-like of shape (n_samples)
            Checked arrival time of the input samples 'X'
        ty : array-like of shape (n_samples)
            Checked arrival time of the Labels 'y'
        tX_cand : array-like of shape (n_samples)
            Checked arrival time of the input samples 'X_cand'
        ty_cand : array-like of shape (n_samples)
            Checked arrival time of the Labels 'y_cand'
        acquisitions : array-like of shape (n_samples)
            List of arrived labels. True if Label arrived otherwise False
        sample_weight : array-like of shape (n_samples,)
            Checked sample weights for X
        return_utilities : bool,
            Checked boolean value of `return_utilities`.
        """
        (
            X_cand,
            X,
            y,
            tX,
            ty,
            tX_cand,
            ty_cand,
            acquisitions,
            sample_weight,
            return_utilities,
        ) = super()._validate_data(
            X_cand=X_cand,
            X=X,
            y=y,
            tX=tX,
            ty=ty,
            tX_cand=tX_cand,
            ty_cand=ty_cand,
            acquisitions=acquisitions,
            sample_weight=sample_weight,
            return_utilities=return_utilities,
            reset=reset,
            **check_X_cand_params
        )
        
        self._validate_delay_prior()
        self._validate_K()

        return (
            X_cand,
            X,
            y,
            tX,
            ty,
            tX_cand,
            ty_cand,
            acquisitions,
            sample_weight,
            return_utilities,
        )

    def _validate_K(self):
        """Validate if K is an integer and greater than 0.
        """
        if self.K is not None:
            if not isinstance(self.K, int):
                raise TypeError(
                    "{} is not a valid type for K".format(type(self.K))
                )
            if self.K <= 0:
                raise ValueError(
                    "The value of K is incorrect."
                    + " K must be greater than 0."
                )

    def _validate_delay_prior(self):
        """Validate if delay_prior a float and greater than 0.
        """
        if self.delay_prior is not None:
            check_scalar(
                self.delay_prior, "delay_prior", (float, int), min_val=0.0
            )


class FuzzyDelaySimulationWrapper(
    SingleAnnotStreamBasedQueryStrategyDelayWrapper
):
    """The FuzzyDelaySimulationWrapper takes already acquired instances
    without labeling into account. Those labels are simulated by using fuzzy
    labels via using the sample weight. The class probabilities for each label
    are estimated using bayesian estimation.

    Parameters
    ----------
    base_query_strategy : QuaryStrategy
        The QuaryStrategy which evaluates the utility of given instances used
        in the stream-based active learning setting.

    delay_prior : float, default=0.001
        Value to correct the predicted frequancy

    clf : BaseEstimator
        The classifier which is trained using this query startegy.

    random_state : int, RandomState instance, default=None
        Controls the randomness of the estimator.
    """
    def __init__(
        self,
        base_query_strategy=None,
        delay_prior=0.001,
        random_state=None,
    ):
        super().__init__(base_query_strategy, random_state)
        self.delay_prior = delay_prior

    def query(
        self,
        X_cand,
        clf,
        X,
        y,
        tX,
        ty,
        tX_cand,
        ty_cand,
        acquisitions,
        sample_weight=None,
        return_utilities=False,
        al_kwargs={},
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
        tX : array-like of shape (n_samples)
            Arrival time of the input samples 'X'
        ty : array-like of shape (n_samples)
            Arrival time of the Labels 'y'
        tX_cand : array-like of shape (n_samples)
            Arrival time of the input samples 'X_cand'
        ty_cand : array-like of shape (n_samples)
            Arrival time of the Labels 'y_cand'
        acquisitions : array-like of shape (n_samples)
            List of arrived labels. True if Label arrived otherwise False
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
            X,
            y,
            tX,
            ty,
            tX_cand,
            ty_cand,
            acquisitions,
            sample_weight,
            return_utilities,
        ) = self._validate_data(
            X_cand=X_cand,
            X=X,
            y=y,
            tX=tX,
            ty=ty,
            tX_cand=tX_cand,
            ty_cand=ty_cand,
            acquisitions=acquisitions,
            sample_weight=sample_weight,
            return_utilities=return_utilities,
        )

        # Check if the classifier and its arguments are valid.
        check_type(clf, SkactivemlClassifier, 'clf')

        clf_fitted = fit_if_not_fitted(clf, X, y, sample_weight, print_warning=False)

        queried_indices = []
        utilities = []

        for i, (tX_cand_current, ty_cand_current) in enumerate(
            zip(tX_cand, ty_cand)
        ):
            tX_in_A_geq_n = ty >= tX_cand_current
            ty_in_A_geq_n = ty < ty_cand_current
            map_B_n = np.logical_and(
                acquisitions, np.logical_and(tX_in_A_geq_n, ty_in_A_geq_n)
            )

            if np.sum(map_B_n):

                X_B_n = X[map_B_n, :]

                probabilities = self.get_class_probabilities(
                    X_B_n, clf_fitted, X, y, sample_weight
                )

                if sample_weight is None:
                    sample_weight = np.ones(len(y))

                add_X = []
                add_y = []
                add_tX = []
                add_ty = []
                add_sample_weight = []
                simulate_ty = (np.max([ty, tX]) + tX_cand_current) / 2
                tmp_queried_indices = []
                tmp_utilities = []
                add_acquisitions = []
                for count in range(len(X_B_n)):
                    for c in range(probabilities.shape[1]):
                        add_X.append(X_B_n[count])
                        add_y.append(c)
                        add_tX.append(tX[map_B_n][count])
                        add_sample_weight.append(probabilities[count, c])
                        add_ty.append(simulate_ty)
                        add_acquisitions.append(True)
                new_acquisitions = np.concatenate(
                    [acquisitions, add_acquisitions]
                )
                new_X = np.concatenate([X, add_X])
                new_y = np.concatenate([y, add_y])
                new_tX = np.concatenate([tX, add_tX])
                new_ty = np.concatenate([ty, add_ty])
                new_sample_weight = np.concatenate(
                    [sample_weight, add_sample_weight]
                )
                (
                    tmp_queried_indices,
                    tmp_utilities,
                ) = self.base_query_strategy_.query(
                    X_cand=X_cand[[i], :],
                    clf=clone(clf_fitted),
                    X=new_X,
                    y=new_y,
                    tX=new_tX,
                    ty=new_ty,
                    tX_cand=[tX_cand[i]],
                    ty_cand=[ty_cand[i]],
                    sample_weight=new_sample_weight,
                    aquisitions=new_acquisitions,
                    return_utilities=True,
                    **al_kwargs,
                    **kwargs
                )
                if len(tmp_queried_indices):
                    queried_indices.append(i)
                utilities.append(tmp_utilities[0])
            else:
                (
                    tmp_queried_indices,
                    tmp_utilities,
                ) = self.base_query_strategy_.query(
                    X_cand=X_cand[[i], :],
                    clf=clone(clf_fitted),
                    X=X,
                    y=y,
                    tX=tX,
                    ty=ty,
                    tX_cand=[tX_cand[i]],
                    ty_cand=[ty_cand[i]],
                    aquisitions=acquisitions,
                    sample_weight=sample_weight,
                    return_utilities=True,
                    **al_kwargs,
                    **kwargs
                )
                if len(tmp_queried_indices):
                    queried_indices.append(i)
                utilities.append(tmp_utilities[0])

        # update base_query_strategy
        queried = np.zeros(len(X_cand))
        queried[tmp_queried_indices] = 1
        kwargs = dict(utilities=utilities)

        if return_utilities:
            return queried_indices, utilities
        else:
            return queried_indices

    def get_class_probabilities(self, X_B_n, clf, X, y, sample_weight):
        """Calculate the probabilities for the simulating 'X_B_n' window.

        Parameters
        ----------
        X_B_n : {array-like, sparse matrix} of shape (n_samples, n_features)
            The instances which may be queried. Sparse matrices are accepted
            only if they are supported by the base query strategy.
        X : array-like of shape (n_samples, n_features)
            Input samples used to fit the classifier.
        y : array-like of shape (n_samples)
            Labels of the input samples 'X'. There may be missing labels.
        sample_weight : array-like of shape (n_samples,) (default=None)
            Sample weights for X, used to fit the clf.

        Returns
        -------
        probabilities : ndarray of shape (n_probabilities,)
            List of probabilities for 'X_B_n'
        """
        pwc = clf
        pwc.fit(X=X, y=y, sample_weight=sample_weight)
        frequencies = pwc.predict_freq(X_B_n)
        frequencies_w_prior = frequencies + self.delay_prior
        probabilities = frequencies_w_prior / np.sum(
            frequencies_w_prior, axis=1, keepdims=True
        )
        return probabilities

    def update(self, X_cand, queried, **kwargs):
        """Updates the budget manager and the count for seen and queried
        instances

        Parameters
        ----------
        X_cand : {array-like, sparse matrix} of shape (n_samples, n_features)
            The instances which could be queried. Sparse matrices are accepted
            only if they are supported by the base query strategy.

        queried : array-like of shape (n_samples,)
            Indicates which instances from X_cand have been queried.

        kwargs : kwargs
            Optional kwargs for budget_manager and query_strategy.

        Returns
        -------
        self : FuzzyDelaySimulationWrapper
            The FuzzyDelaySimulationWrapper returns itself, after it is updated.
        """
        self._validate_base_query_strategy()
        self.base_query_strategy_.update(X_cand, queried, **kwargs)
        return self

    def _validate_data(
        self,
        X_cand,
        return_utilities,
        X,
        y,
        tX,
        ty,
        tX_cand,
        ty_cand,
        acquisitions,
        sample_weight,
        reset=True,
        **check_X_cand_params
    ):
        """Validate input data and set or check the `n_features_in_` attribute.

        Parameters
        ----------
        X_cand : {array-like, sparse matrix} of shape (n_samples, n_features)
            The instances which may be queried. Sparse matrices are accepted
            only if they are supported by the base query strategy.
        X : array-like of shape (n_samples, n_features)
            Input samples used to fit the classifier.
        y : array-like of shape (n_samples)
            Labels of the input samples 'X'. There may be missing labels.
        tX : array-like of shape (n_samples)
            Arrival time of the input samples 'X'
        ty : array-like of shape (n_samples)
            Arrival time of the Labels 'y'
        tX_cand : array-like of shape (n_samples)
            Arrival time of the input samples 'X_cand'
        ty_cand : array-like of shape (n_samples)
            Arrival time of the Labels 'y_cand'
        acquisitions : array-like of shape (n_samples)
            List of arrived labels. True if Label arrived otherwise False
        sample_weight : array-like of shape (n_samples,) (default=None)
            Sample weights for X, used to fit the clf.
        return_utilities : bool, optional
            If true, also return the utilities based on the query strategy.
            The default is False.
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
        X : array-like of shape (n_samples, n_features)
            Checked input samples used to fit the classifier.
        y : array-like of shape (n_samples)
            Checked Labels of the input samples 'X'. There may be missing labels.
        tX : array-like of shape (n_samples)
            Checked arrival time of the input samples 'X'
        ty : array-like of shape (n_samples)
            Checked arrival time of the Labels 'y'
        tX_cand : array-like of shape (n_samples)
            Checked arrival time of the input samples 'X_cand'
        ty_cand : array-like of shape (n_samples)
            Checked arrival time of the Labels 'y_cand'
        acquisitions : array-like of shape (n_samples)
            List of arrived labels. True if Label arrived otherwise False
        sample_weight : array-like of shape (n_samples,)
            Checked sample weights for X
        return_utilities : bool,
            Checked boolean value of `return_utilities`.
        """
        (
            X_cand,
            X,
            y,
            tX,
            ty,
            tX_cand,
            ty_cand,
            acquisitions,
            sample_weight,
            return_utilities,
        ) = super()._validate_data(
            X_cand=X_cand,
            X=X,
            y=y,
            tX=tX,
            ty=ty,
            tX_cand=tX_cand,
            ty_cand=ty_cand,
            acquisitions=acquisitions,
            sample_weight=sample_weight,
            return_utilities=return_utilities,
            reset=reset,
            **check_X_cand_params
        )
        self._validate_delay_prior()

        return (
            X_cand,
            X,
            y,
            tX,
            ty,
            tX_cand,
            ty_cand,
            acquisitions,
            sample_weight,
            return_utilities,
        )

    def _validate_delay_prior(self):
        """Validate if delay_prior a float and greater than 0.
        """
        if self.delay_prior is not None:
            check_scalar(
                self.delay_prior, "delay_prior", (float, int), min_val=0.0
            )