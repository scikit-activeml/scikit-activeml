import numpy as np
import warnings

from copy import deepcopy

from scipy.stats import t, rankdata

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array, check_consistent_length, \
    check_random_state, check_is_fitted

from skactiveml.base import MultiAnnotPoolBasedQueryStrategy, \
    SkactivemlClassifier, AnnotModelMixing
from skactiveml.utils import rand_argmax, check_scalar, compute_vote_vectors, \
    MISSING_LABEL, ExtLabelEncoder, is_labeled
from skactiveml.pool._uncertainty import uncertainty_scores


class IEAnnotModel(BaseEstimator, AnnotModelMixing):
    """IEAnnotModel

    This annotator model relies on 'Interval Estimation Learning' (IELearning)
    for estimating the annotation performances, i.e., labeling accuracies,
    of multiple annotators [1]. Therefor, it computes the mean accuracy and the
    lower as well as the upper bound of the labeling accuracy per annotator.
    (Weighted) majority vote is used as as estimated ground truth.

    Parameters
    ----------
    classes : array-like, shape (n_classes), default=None
        Holds the label for each class.
    missing_label : scalar|string|np.nan|None, default=np.nan
        Value to represent a missing label.
    alpha : float, interval=(0, 1), optional (default=0.05)
        Half of the confidence level for student's t-distribution.
    mode : {'lower', 'mean', 'upper'}, optional (default='upper')
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
    def __init__(self, classes=None, missing_label=MISSING_LABEL, alpha=0.05,
                 mode='upper', random_state=None):
        self.classes = classes
        self.missing_label = missing_label
        self.alpha = alpha
        self.mode = mode
        self.random_state = random_state

    def fit(self, y, sample_weight=None):
        """Fit annotator model for given samples.

        Parameters
        ----------
        y : array-like, shape (n_samples, n_annotators)
            Class labels of annotators.
        sample_weight : array-like, shape (n_samples, n_annotators), optional (default=None)
            Sample weight for each label and annotator.

        Returns
        -------
        self : IEAnnotModel object
            The fitted annotator model.
        """
        # Check whether alpha is float in (0, 1).
        check_scalar(x=self.alpha, target_type=float, name='alpha', min_val=0,
                     max_val=1, min_inclusive=False, max_inclusive=False)

        # Check mode.
        if self.mode not in ['lower', 'mean', 'upper']:
            raise ValueError("`mode` must be in `['lower', 'mean', `upper`].`")

        # Check random state.
        random_state = check_random_state(self.random_state)

        # Encode class labels from `0` to `n_classes-1`.
        label_encoder = ExtLabelEncoder(missing_label=self.missing_label,
                                        classes=self.classes).fit(y)
        self.classes_ = label_encoder.classes_
        y = label_encoder.transform(y)

        # Check shape of labels.
        if y.ndim != 2:
            raise ValueError("`y` but must be a 2d array with shape "
                             "`(n_samples, n_annotators)`.")

        # Compute majority vote labels.
        V = compute_vote_vectors(y=y, w=sample_weight, classes=self.classes_)
        y_mv = rand_argmax(V, axis=1, random_state=random_state)

        # Number of annotators.
        self.n_annotators_ = y.shape[1]
        is_lbld = is_labeled(y, missing_label=np.nan)
        self.A_perf_ = np.zeros((self.n_annotators_, 3))
        for a_idx in range(self.n_annotators_):
            is_correct = np.equal(y_mv[is_lbld[:, a_idx]],
                                  y[is_lbld[:, a_idx], a_idx])
            is_correct = np.concatenate((is_correct, [0, 1]))
            mean = np.mean(is_correct)
            std = np.std(is_correct)
            t_value = t.isf([self.alpha / 2], len(is_correct) - 1)[0]
            t_value *= std / np.sqrt(len(is_correct))
            self.A_perf_[a_idx, 0] = mean - t_value
            self.A_perf_[a_idx, 1] = mean
            self.A_perf_[a_idx, 2] = mean + t_value

        return self

    def predict_annot_perf(self, X):
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
        if self.mode == 'lower':
            mode = 0
        elif self.mode == 'mean':
            mode = 1
        else:
            mode = 2
        return np.tile(self.A_perf_[:, mode], (len(X), 1))


class IEThresh(MultiAnnotPoolBasedQueryStrategy):
    """IEThresh

    The strategy 'Interval Estimation Threshold' (IEThresh) [1] is useful for
    addressing the exploration vs. exploitation trade-off when dealing with
    multiple error-prone annotators in active learning. This class relies on
    'Interval Estimation Learning' (IELearning) for estimating the annotation
    performances, i.e., label accuracies, of multiple annotators. Samples are
    selected based on 'Uncertainty Sampling' (US). The selected samples are
    labeled by the annotators whose estimated annotation performances are equal
    or greater than an adaptive threshold.

    Parameters
    ----------
    clf : classifier with 'fit' and 'predict_proba' method
        Classifier whose expected error reduction is measured.
    epsilon : float, interval=[0, 1], optional (default=0.9)
        Parameter for specifying the adaptive threshold used for annotator
        selection.
    alpha : float, interval=(0, 1), optional (default=0.05)
        Half of the confidence level for student's t-distribution.
    random_state : None|int|numpy.random.RandomState, optional (default=None)
        The random state used for deciding on majority vote labels in case of
        ties.

    References
    ----------
    [1] Donmez, Pinar, Jaime G. Carbonell, and Jeff Schneider.
        "Efficiently learning the accuracy of labeling sources for selective
        sampling." 15th ACM SIGKDD International Conference on Knowledge
        Discovery and Data Mining, pp. 259-268. 2009.
    """
    def __init__(self, clf, epsilon=0.9, alpha=0.05, random_state=None):
        super().__init__(random_state=random_state)
        self.clf = clf
        self.epsilon = epsilon
        self.alpha = alpha

    def query(self, X_cand, A_cand, X, y, sample_weight=None,
              batch_size='adaptive', return_utilities=False, **kwargs):
        """Determines which candidate sample is to be annotated by which
        annotator.

        Parameters
        ----------
        X_cand : array-like, shape (n_cand_samples, n_features)
            Candidate samples from which the strategy can select.
        A_cand : array-like, shape (n_cand_samples, n_features)
            Boolean matrix where `A_cand[i,j] = True` indicates that
            annotator `j` can be selected for annotating sample `X_cand[i]`,
            while `A_cand[i,j] = False` indicates that annotator `j` cannot be
            selected for annotating sample `X_cand[i]`. If A_cand=None, each
            annotator is assumed to be available for labeling each sample.
        X : matrix-like, shape (n_samples, n_features)
            The sample matrix X is the feature matrix representing the samples.
        y : array-like, shape (n_samples, n_annotators)
            It contains the class labels of the training samples.
            The number of class labels may be variable for the samples, where
            missing labels are represented the attribute 'missing_label'.
        sample_weight : array-like, (n_samples, n_annotators)
            It contains the weights of the training samples' class labels.
            It must have the same shape as y.
        batch_size : 'adaptive'|int, optional (default=1)
            The number of samples to be selected in one AL cycle. If 'adaptive'
            is set, the `batch_size` is determined based on the annotation
            performances and the parameter `epsilon`.
        return_utilities : bool, optional (default=False)
            If true, also return the utilities based on the query strategy.

        Returns
        -------
        query_indices : numpy.ndarray, shape (batch_size, 2)
            The query_indices indicate which candidate sample is to be
            annotated by which annotator, e.g., `query_indices[:, 0]`
            indicates the selected candidate samples and `query_indices[:, 1]`
            indicates the respectively selected annotators.
        utilities: numpy.ndarray, shape (batch_size, n_cand_samples,
         n_annotators)
            The utilities of all candidate samples w.r.t. to the available
            annotators after each selected sample of the batch, e.g.,
            `utilities[0, :, j]` indicates the utilities used for selecting
            the first sample-annotator pair (with indices `query_indices[0]`).
        """
        # Check classifier type.
        if not isinstance(self.clf, SkactivemlClassifier):
            raise TypeError("`clf` must be of the type "
                            "`skactiveml._base.SkactivemlClassifier`.")
        self.clf_ = deepcopy(self.clf)

        # Check whether epsilon is float in [0, 1].
        check_scalar(x=self.epsilon, target_type=float, name='epsilon',
                     min_val=0, max_val=1)

        # Check whether alpha is float in (0, 1).
        check_scalar(x=self.alpha, target_type=float, name='alpha', min_val=0,
                     max_val=1, min_inclusive=False, max_inclusive=False)

        # Set and check random state.
        random_state = check_random_state(self.random_state)

        # Check candidate samples.
        X_cand = check_array(X_cand)
        n_cand_samples = len(X_cand)

        # Check training data.
        X = check_array(X)
        y = check_array(y, force_all_finite=False)
        check_consistent_length(X, y)
        check_consistent_length(X.T, X_cand.T)

        # Check boolean matrix.
        A_cand = check_array(A_cand, dtype=bool)
        n_annotators = A_cand.shape[1]
        check_consistent_length(X_cand, A_cand)
        sample_is_labeled = np.sum(A_cand, axis=1) < n_annotators

        # Check whether y and A_cand have the same shape.
        if y.shape[1] != A_cand.shape[1]:
            raise ValueError("The 2d arrays `y` and `A_cand` must have the "
                             "the same number of columns; got instead "
                             "`y.shape[1]={}` and `A_cand[1].shape={}`"
                             .format(y.shape[1], A_cand.shape[1]))

        # Check batch_size.
        if isinstance(batch_size, str):
            if batch_size != 'adaptive':
                raise ValueError('If `batch_size` is a string, it '
                                 'must be set to `adaptive`.')
        elif isinstance(batch_size, int):
            check_scalar(batch_size, target_type=int, name='batch_size',
                         min_val=1)
            batch_size = batch_size
            batch_size_max = np.sum(~sample_is_labeled) * n_annotators
            if batch_size_max < batch_size:
                warnings.warn("'batch_size={}' is larger than number of "
                              "candidate samples in 'X_cand'. Instead, "
                              "'batch_size={}' was set ".format(
                    batch_size, np.sum(A_cand)))
                batch_size = batch_size_max
        else:
            raise TypeError('`batch_size` must be either a string or an '
                            'integer.')

        # Check return utilities.
        check_scalar(return_utilities, target_type=bool,
                     name='return_utilties')

        # Fit classifier and compute uncertainties on candidate samples.
        self.clf_.fit(X, y, sample_weight=sample_weight)
        P = self.clf_.predict_proba(X_cand)
        uncertainties = uncertainty_scores(P=P, method='least_confident')

        # Check whether there exists a sample with no annotation at all.
        if np.sum(~sample_is_labeled) == 0:
            warnings.warn(
                "'X_cand' contains no sample being fully unlabeled, i.e., "
                "in each row of `A_cand` is at least one zero/False entry. "
                "Therefor, no further sample-annotator pairs are selected."
            )
            if return_utilities:
                return np.array([]), np.array([])
            else:
                return np.array([])

        # Fit annotator model and compute performance estimates.
        self.ie_model_ = IEAnnotModel(classes=self.clf_.classes_,
                                      missing_label=self.clf_.missing_label,
                                      alpha=self.alpha, mode='upper')
        self.ie_model_.fit(y=y, sample_weight=sample_weight)
        A_perf = self.ie_model_.A_perf_

        # Compute utilities.
        A_perf = A_perf[:, 2] + 1
        A_perf = A_perf[np.newaxis]
        uncertainties[sample_is_labeled] = np.nan
        ranks = rankdata(uncertainties[~sample_is_labeled],
                         method='ordinal') * 5
        uncertainties[~sample_is_labeled] = ranks
        uncertainties = np.tile(uncertainties, (n_annotators, 1)).T
        init_utilities = uncertainties + A_perf

        # Determine actual batch size.
        if batch_size == 'adaptive':
            required_perf = self.epsilon * np.max(A_perf)
            actual_batch_size = np.sum(A_perf >= required_perf)
        else:
            actual_batch_size = batch_size

        # Perform selection based on previously computed utilities.
        query_indices = np.zeros((actual_batch_size, 2), dtype=int)
        utilities = np.empty((actual_batch_size, n_cand_samples, n_annotators))
        utilities[0] = init_utilities
        for b in range(actual_batch_size):
            query_indices[b] = rand_argmax(utilities[b],
                                           random_state=random_state)
            if b < actual_batch_size - 1:
                init_utilities[query_indices[b, 0], query_indices[b, 1]] \
                    = np.nan
                utilities[b+1] = init_utilities

        # Check whether utilities are to be returned.
        if return_utilities:
            return query_indices, utilities
        else:
            return query_indices
