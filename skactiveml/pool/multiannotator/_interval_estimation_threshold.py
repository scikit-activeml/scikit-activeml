import numpy as np
from scipy.stats import t, rankdata
from sklearn.base import BaseEstimator, clone
from sklearn.utils.validation import check_array, check_is_fitted

from ...base import (
    MultiAnnotatorPoolQueryStrategy,
    SkactivemlClassifier,
    AnnotatorModelMixin,
)
from ...pool._uncertainty_sampling import uncertainty_scores
from ...utils import (
    check_scalar,
    MISSING_LABEL,
    is_labeled,
    check_type,
    simple_batch,
    majority_vote,
)


class IntervalEstimationAnnotModel(BaseEstimator, AnnotatorModelMixin):
    """IntervalEstimationAnnotModel

    This annotator model relies on 'Interval Estimation Learning' (IELearning)
    for estimating the annotation performances, i.e., labeling accuracies,
    of multiple annotators [1]. Therefore, it computes the mean accuracy and
    the lower as well as the upper bound of the labeling accuracy per
    annotator. (Weighted) majority vote is used as estimated ground truth.

    Parameters
    ----------
    classes : array-like, shape (n_classes), optional (default=None)
        Holds the label for each class.
    missing_label : scalar or string or np.nan or None, optional
    (default=np.nan)
        Value to represent a missing label.
    alpha : float, interval=(0, 1), optional (default=0.05)
        Half of the confidence level for student's t-distribution.
    mode : 'lower' or 'mean' or 'upper', optional (default='upper')
        Mode of the estimated annotation performance.
    random_state : None|int|numpy.random.RandomState, optional (default=None)
        The random state used for deciding on majority vote labels in case of
        ties.

    Attributes
    ----------
    classes_: array-like, shape (n_classes)
        Holds the label for each class.
    n_annotators_: int
        Number of annotators.
    A_perf_ : ndarray, shape (n_annotators, 3)
        Estimated annotation performances (i.e., labeling accuracies), where
        `A_cand[i, 0]` indicates the lower bound, `A_cand[i, 1]` indicates the
        mean, and `A_cand[i, 2]` indicates the upper bound of the estimation
        labeling accuracy.

    References
    ----------
    [1] Donmez, Pinar, Jaime G. Carbonell, and Jeff Schneider.
        "Efficiently learning the accuracy of labeling sources for selective
        sampling." 15th ACM SIGKDD International Conference on Knowledge
        Discovery and Data Mining, pp. 259-268. 2009.
    """

    def __init__(
            self,
            classes=None,
            missing_label=MISSING_LABEL,
            alpha=0.05,
            mode="upper",
            random_state=None,
    ):
        self.classes = classes
        self.missing_label = missing_label
        self.alpha = alpha
        self.mode = mode
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        """Fit annotator model for given samples.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Test samples.
        y : array-like, shape (n_samples, n_annotators)
            Class labels of annotators.
        sample_weight : array-like, shape (n_samples, n_annotators),
        optional (default=None)
            Sample weight for each label and annotator.

        Returns
        -------
        self : IntervalEstimationAnnotModel object
            The fitted annotator model.
        """

        # Check whether alpha is float in (0, 1).
        check_scalar(
            x=self.alpha,
            target_type=float,
            name="alpha",
            min_val=0,
            max_val=1,
            min_inclusive=False,
            max_inclusive=False,
        )

        # Check mode.
        if self.mode not in ["lower", "mean", "upper"]:
            raise ValueError("`mode` must be in `['lower', 'mean', `upper`].`")

        # Check shape of labels.
        if y.ndim != 2:
            raise ValueError(
                "`y` but must be a 2d array with shape "
                "`(n_samples, n_annotators)`."
            )

        # Compute majority vote labels.
        y_mv = majority_vote(
            y=y,
            w=sample_weight,
            classes=self.classes,
            random_state=self.random_state,
            missing_label=self.missing_label,
        )

        # Number of annotators.
        self.n_annotators_ = y.shape[1]
        is_lbld = is_labeled(y, missing_label=self.missing_label)
        self.A_perf_ = np.zeros((self.n_annotators_, 3))
        for a_idx in range(self.n_annotators_):
            is_correct = np.equal(
                y_mv[is_lbld[:, a_idx]], y[is_lbld[:, a_idx], a_idx]
            )
            is_correct = np.concatenate((is_correct, [0, 1]))
            mean = np.mean(is_correct)
            std = np.std(is_correct)
            t_value = t.isf([self.alpha / 2], len(is_correct) - 1)[0]
            t_value *= std / np.sqrt(len(is_correct))
            self.A_perf_[a_idx, 0] = mean - t_value
            self.A_perf_[a_idx, 1] = mean
            self.A_perf_[a_idx, 2] = mean + t_value

        return self

    def predict_annotator_perf(self, X):
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
        check_is_fitted(self)
        X = check_array(X)
        if self.mode == "lower":
            mode = 0
        elif self.mode == "mean":
            mode = 1
        else:
            mode = 2
        return np.tile(self.A_perf_[:, mode], (len(X), 1))


class IntervalEstimationThreshold(MultiAnnotatorPoolQueryStrategy):
    """IntervalEstimationThreshold

    The strategy 'Interval Estimation Threshold' (IEThresh) [1] is useful for
    addressing the exploration vs. exploitation trade-off when dealing with
    multiple error-prone annotators in active learning. This class relies on
    `IntervalEstimationAnnotModel` for estimating the annotation
    performances, i.e., label accuracies, of multiple annotators. Samples are
    selected based on 'Uncertainty Sampling' (US). The selected samples are
    labeled by the annotators whose estimated annotation performances are equal
    or greater than an adaptive threshold.
    The strategy assumes all annotators to be available and is not defined
    otherwise. To deal with this case nonetheless value-annotator pairs are
    first ranked according to the amount of annotators available for the given
    value in `candidates` and are than ranked according to
    `IntervalEstimationThreshold`.

    Parameters
    ----------
    epsilon : float, interval=[0, 1], optional (default=0.9)
        Parameter for specifying the adaptive threshold used for annotator
        selection.
    alpha : float, interval=(0, 1), optional (default=0.05)
        Half of the confidence level for student's t-distribution.
    random_state : None or int or numpy.random.RandomState, optional
    (default=None)
        The random state used for deciding on majority vote labels in case of
        ties.

    References
    ----------
    [1] Donmez, Pinar, Jaime G. Carbonell, and Jeff Schneider.
        "Efficiently learning the accuracy of labeling sources for selective
        sampling." 15th ACM SIGKDD International Conference on Knowledge
        Discovery and Data Mining, pp. 259-268. 2009.
    """

    def __init__(
            self,
            epsilon=0.9,
            alpha=0.05,
            random_state=None,
            missing_label=MISSING_LABEL,
    ):
        super().__init__(
            random_state=random_state, missing_label=missing_label
        )
        self.epsilon = epsilon
        self.alpha = alpha

    def query(
            self,
            X,
            y,
            clf,
            fit_clf=True,
            candidates=None,
            annotators=None,
            sample_weight=None,
            batch_size="adaptive",
            return_utilities=False,
    ):
        """Determines which candidate sample is to be annotated by which
        annotator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data set, usually complete, i.e., including the labeled
            and unlabeled samples.
        y : array-like of shape (n_samples, n_annotators)
            Labels of the training data set for each annotator (possibly
            including unlabeled ones indicated by self.MISSING_LABEL), meaning
            that `y[i, j]` contains the label annotated by annotator `i` for
            sample `j`.
        clf : skactiveml.base.SkactivemlClassifier
            Model implementing the methods `fit` and `predict_proba`.
        fit_clf : bool, default=True
            Defines whether the classifier should be fitted on `X`, `y`, and
            `sample_weight`.
        candidates : None or array-like of shape (n_candidates), dtype=int or
            array-like of shape (n_candidates, n_features),
            optional (default=None)
            If `candidates` is None, the samples from (X,y), for which an
            annotator exists such that the annotator sample pairs is
            unlabeled are considered as sample candidates.
            If `candidates` is of shape (n_candidates,) and of type int,
            `candidates` is considered as the indices of the sample candidates
            in (X,y).
            If `candidates` is of shape (n_candidates, n_features), the
            sample candidates are directly given in `candidates` (not
            necessarily contained in `X`).
        annotators : array-like, shape (n_candidates, n_annotators), optional
        (default=None)
            If `annotators` is None, all annotators are considered as available
            annotators.
            If `annotators` is of shape (n_avl_annotators) and of type int,
            `annotators` is considered as the indices of the available
            annotators.
            If candidate samples and available annotators are specified:
            The annotator sample pairs, for which the sample is a candidate
            sample and the annotator is an available annotator are considered
            as candidate annotator sample pairs.
            If `annotators` is None and `candidates` is of shape
            (n_candidates,), all annotator sample pairs, for which the sample
            is indexed by `candidates` are considered as candidate annotator
            sample pairs. If `annotators` is a boolean array of shape
            (n_candidates, n_avl_annotators) the annotator sample pairs, for
            which the sample is a candidate sample and the boolean matrix has
            entry `True` are considered as candidate sample pairs.
        sample_weight : array-like, (n_samples, n_annotators), optional
        (default=None)
            It contains the weights of the training samples' class labels.
            It must have the same shape as y.
        batch_size : 'adaptive' or int, optional (default=1)
            The number of samples to be selected in one AL cycle. If 'adaptive'
            is set, the `batch_size` is determined based on the annotation
            performances and the parameter `epsilon`.
        return_utilities : bool, optional (default=False)
            If true, also return the utilities based on the query strategy.

        Returns
        -------
        query_indices : numpy.ndarray of shape (batch_size, 2)
            The query_indices indicate which candidate sample is to be
            annotated by which annotator, e.g., `query_indices[:, 0]`
            indicates the selected candidate samples and `query_indices[:, 1]`
            indicates the respectively selected annotators.
        utilities: numpy.ndarray of shape (batch_size, n_cand_samples,
         n_annotators)
            The utilities of all candidate samples w.r.t. to the available
            annotators after each selected sample of the batch, e.g.,
            `utilities[0, :, j]` indicates the utilities used for selecting
            the first sample-annotator pair (with indices `query_indices[0]`).
        """

        # base check
        (
            X,
            y,
            candidates,
            annotators,
            _,
            return_utilities,
        ) = super()._validate_data(
            X, y, candidates, annotators, 1, return_utilities, reset=True
        )

        X_cand, mapping, A_cand = self._transform_cand_annot(
            candidates, annotators, X, y
        )

        # Validate classifier type.
        check_type(clf, "clf", SkactivemlClassifier)

        # Check whether epsilon is float in [0, 1].
        check_scalar(
            x=self.epsilon,
            target_type=float,
            name="epsilon",
            min_val=0,
            max_val=1,
        )

        # Check whether alpha is float in (0, 1).
        check_scalar(
            x=self.alpha,
            target_type=float,
            name="alpha",
            min_val=0,
            max_val=1,
            min_inclusive=False,
            max_inclusive=False,
        )

        n_annotators = y.shape[1]
        # Check whether unlabeled data exists
        A_cand = np.repeat(
            np.all(A_cand, axis=1).reshape(-1, 1), n_annotators, axis=1
        )

        # Fit classifier and compute uncertainties on candidate samples.
        if fit_clf:
            clf = clone(clf).fit(X, y, sample_weight)

        P = clf.predict_proba(X_cand)
        uncertainties = uncertainty_scores(probas=P, method="least_confident")

        # Fit annotator model and compute performance estimates.
        ie_model = IntervalEstimationAnnotModel(
            classes=clf.classes_,
            missing_label=clf.missing_label,
            alpha=self.alpha,
            mode="upper",
        )

        ie_model.fit(X=X, y=y, sample_weight=sample_weight)
        A_perf = ie_model.A_perf_

        # Compute utilities.

        # combine the values of A_perf and uncertainties
        A_perf = A_perf[:, 2] + 1
        A_perf = A_perf[np.newaxis]
        max_range = np.max(A_perf) + 1
        uncertainties = rankdata(uncertainties, method="ordinal") * max_range
        uncertainties = np.tile(uncertainties, (n_annotators, 1)).T
        utilities = uncertainties + A_perf

        # exclude not available annotators
        utilities[~A_cand] = np.nan

        # Determine actual batch size.
        if isinstance(batch_size, str) and batch_size != "adaptive":
            raise ValueError(
                f"If `batch_size` is of type `string`, "
                f"it must equal `'adaptive'`."
            )
        elif batch_size == "adaptive":
            required_perf = self.epsilon * np.max(A_perf)
            actl_batch_size = int(np.sum(A_perf >= required_perf))
        elif isinstance(batch_size, int):
            actl_batch_size = batch_size
        else:
            raise TypeError(
                f"`batch_size` is of type `{type(batch_size)}` "
                f"but must equal `'adaptive'` or be of type "
                f"`int`."
            )

        if mapping is not None:
            w_utilities = utilities
            utilities = np.full((len(X), n_annotators), np.nan)
            utilities[mapping, :] = w_utilities

        # Perform selection based on previously computed utilities.
        return simple_batch(
            utilities,
            self.random_state_,
            batch_size=actl_batch_size,
            return_utilities=return_utilities,
        )
