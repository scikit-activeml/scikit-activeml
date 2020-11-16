import itertools
import warnings
from copy import deepcopy

import numpy as np
from scipy.special import factorial, gammaln
from sklearn import clone
from sklearn.metrics import pairwise_kernels
from sklearn.utils import check_array, check_random_state

from skactiveml.base import SingleAnnotPoolBasedQueryStrategy, \
    ClassFrequencyEstimator
from skactiveml.classifier import PWC
from skactiveml.utils import check_classifier_params
from skactiveml.utils import rand_argmax, MISSING_LABEL, check_cost_matrix, \
    check_scalar, is_labeled


class McPAL(SingleAnnotPoolBasedQueryStrategy):
    """Multi-class Probabilistic Active Learning

    This class implements multi-class probabilistic active learning (McPAL) [1]
    strategy.

    Parameters
    ----------
    clf: BaseEstimator
        Probabilistic classifier for gain calculation.
    prior: float, optional (default=1)
        Prior probabilities for the Dirichlet distribution of the samples.
    m_max: int, optional (default=1)
        Maximum number of hypothetically acquired labels.
    random_state: numeric | np.random.RandomState, optional
        Random state for candidate selection.

    References
    ----------
    [1] Daniel Kottke, Georg Krempl, Dominik Lang, Johannes Teschner, and Myra
    Spiliopoulou.
        Multi-Class Probabilistic Active Learning,
        vol. 285 of Frontiers in Artificial Intelligence and Applications,
        pages 586-594. IOS Press, 2016
    """

    def __init__(self, clf, prior=1, m_max=1, random_state=None):
        super().__init__(random_state=random_state)
        self.clf = clf
        self.prior = prior
        self.m_max = m_max
        self.random_state = random_state

    def query(self, X_cand, X, y, weights, return_utilities=False, **kwargs):
        """Query the next instance to be labeled.

        Parameters
        ----------
        X_cand: array-like, shape(n_candidates, n_features)
            Unlabeled candidate samples
        X: array-like (n_training_samples, n_features)
            Complete data set
        y: array-like (n_training_samples)
            Labels of the data set
        weights: array-like (n_training_samples)
            Densities for each instance in X
        return_utilities: bool (default=False)
            If True, the utilities are additionally returned.

        Returns
        -------
        query_indices: np.ndarray, shape (1)
            The index of the queried instance.
        utilities: np.ndarray, shape (1, n_candidates)
            The utilities of all instances in X_cand
            (only returned if return_utilities is True).
        """
        # Check class attributes
        if not isinstance(self.clf, ClassFrequencyEstimator):
            raise TypeError("'clf' must implement methods according to "
                            "'ClassFrequencyEstimator'.")
        if not isinstance(self.prior, (int, float)):
            raise TypeError("'prior' must be an int or float.")
        if self.prior <= 0:
            raise ValueError("'prior' must be greater than zero.")
        if self.m_max < 1 or not float(self.m_max).is_integer():
            raise ValueError("'m_max' must be a positive integer.")
        check_random_state(self.random_state)
        self.clf = clone(self.clf)
        check_classifier_params(self.clf.classes, self.clf.missing_label)

        X_cand = check_array(X_cand, force_all_finite=False)
        X = check_array(X, force_all_finite=False)
        y = check_array(y, force_all_finite=False, ensure_2d=False)

        # Calculate gains
        self.clf.fit(X, y)
        k_vec = self.clf.predict_freq(X_cand)
        utilities = weights * cost_reduction(k_vec, prior=self.prior,
                                             m_max=self.m_max)
        query_indices = rand_argmax(utilities, random_state=self.random_state)

        if return_utilities:
            return query_indices, np.array([utilities])
        else:
            return query_indices

class BayesianPriorLearner(PWC):
    def __init__(self, n_neighbors=None, metric='rbf', metric_dict=None,
                 classes=None, missing_label=MISSING_LABEL, cost_matrix=None,
                 prior=None, random_state=None):
        super().__init__(n_neighbors=n_neighbors, metric=metric,
                       metric_dict=metric_dict, classes=classes,
                       missing_label=missing_label, cost_matrix=cost_matrix,
                       random_state=random_state)
        self.prior = prior

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
        self: PWC,
            The PWC is fitted on the training data.
        """
        super().fit(X=X, y=y, sample_weight=sample_weight)
        n_classes = len(self.classes_)
        if self.prior is None:
            self._prior = np.zeros([1,n_classes])
        else:
            if self.prior.shape != (n_classes,):
                raise ValueError(
                    "Shape mismatch for 'prior': It is '{}' instead of '({"
                    "})'".format(self.prior.shape, n_classes)
                )
            else:
                self._prior = self.prior.reshape(1, -1)


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
        P = self.predict_freq(X) + self.prior
        normalizer = np.sum(P, axis=1)
        P[normalizer > 0] /= normalizer[normalizer > 0, np.newaxis]
        P[normalizer == 0, :] = [1 / len(self.classes_)] * len(self.classes_)
        return P

class XPAL(SingleAnnotPoolBasedQueryStrategy):

    def __init__(self, clf, metric='error', mode='default',
                 max_nonmyopic_size=1, cost_vector=None, cost_matrix=None,
                 custom_perf_func=None, prior_cand=1.e-3, prior_eval=1.e-3,
                 estimator_metric='rbf', estimator_metric_dict={},
                 random_state=None):
        # TODO @DK: clean up
        """ XPAL
        The cost-sensitive expected probabilistic active learning (CsXPAL) strategy is a generalization of the
        multi-class probabilistic active learning (McPAL) [1], the optimised probabilistic active learning (OPAL) [2]
        strategy, and the cost-sensitive probabilistic active learning (CsPAL) strategy due to consideration of a cost
        matrix for multi-class classification problems and the computation of the expected error on an evaluation set.

        Attributes
        ----------
        prob_estimator: BaseEstimator
            The method that estimates the ground truth. The method should be
            Bayesian, e.g. local learning with Bayesian Prior.
        prob_estimator_eval: BaseEstimator
            Similar to prob_estimator but used for calculating the
            probabilities of evaluation instances. The default is to use the
            prob_estimator if not given.
        metric: string
            "error"
                equivalent to accuracy
            "cost-vector"
                requires cost-vector
            "misclassification-loss"
                requires cost-matrix
            "mean-abs-error"
            "macro-accuracy"
                optional: TODO: prior_class_probability
            "f1-score"

        cost_matrix: array-like (n_classes, n_classes)
        cost_vector: array-like (n_classes)

        mode: string
            The mode to select candidates:
                "default" (default)
                    Instances are selected sequentially using the non-myopic
                    selection with max_nonmyopic_size into a batch.
                "density-qpprox"
                    no X_eval needed, but weights

        max_nonmyopic_size: int

        random_state

        missing_label

        References
        ----------
        [1] TODO: xpal arxiv, diss
        """
        super().__init__(random_state=random_state)

        self.clf = clf
        self.metric = metric
        self.mode = mode
        self.max_nonmyopic_size = max_nonmyopic_size
        self.cost_vector = cost_vector
        self.cost_matrix = cost_matrix
        self.custom_perf_func = custom_perf_func
        self.prior_cand = prior_cand
        self.prior_eval = prior_eval
        self.estimator_metric = estimator_metric
        self.estimator_metric_dict = estimator_metric_dict

    def _validate_input(self, X_cand, X, y, X_eval, batch_size, sample_weight,
                    return_utilities):

        # init parameters
        # TODO: implement and adapt, what happens if X_eval=self
        # random_state = check_random_state(self.random_state, len(X_cand),
        #                                   len(X_eval))
        self._random_state = check_random_state(self.random_state)

        check_classifier_params(self._classes, self._missing_label,
                                self.cost_matrix)
        check_scalar(self.prior_cand, target_type=float, name='prior_cand',
                     min_val=0, min_inclusive=False)
        check_scalar(self.prior_eval, target_type=float, name='prior_eval',
                     min_val=0, min_inclusive=False)

        self._batch_size = batch_size
        check_scalar(self._batch_size, target_type=int, name='batch_size',
                     min_val=1)
        if len(X_cand) < self._batch_size:
            warnings.warn(
                "'batch_size={}' is larger than number of candidate samples "
                "in 'X_cand'. Instead, 'batch_size={}' was set ".format(
                    self._batch_size, len(X_cand)))
            self._batch_size = len(X_cand)

        # TODO: check if X_cand, X, X_eval have similar num_features
        # TODO: check if X, y match
        # TODO: check if weight match with X_cand
        if sample_weight is None:
            self.sample_weight_ = np.ones(len(X))
        # TODO: check if return_utilities is bool

    def _validate_and_transform_metric(self):
        self._metric = None
        self._cost_matrix = None
        self._perf_func = None
        if self.metric == 'error':
            self._metric = 'misclassification-loss'
            self._cost_matrix = 1 - np.eye(self._n_classes)
        elif self.metric == 'cost-vector':
            if self.cost_vector is None or \
                    self.cost_vector.shape != (self._n_classes):
                raise ValueError(
                    "For metric='cost-vector', the argument "
                    "'cost_vector' must be given when initialized and must "
                    "have shape (n_classes)"
                )
            self._metric = 'misclassification-loss'
            self._cost_matrix = self.cost_vector.reshape(-1, 1) \
                                @ np.ones([1, self._n_classes])
            np.fill_diagonal(self._cost_matrix, 0)

        elif self.metric == 'misclassification-loss':
            if self.cost_matrix is None:
                raise ValueError(
                    "'cost_matrix' cannot be None for "
                    "metric='misclasification-loss'"
                )
            check_cost_matrix(self.cost_matrix, self._n_classes)
            self._metric = 'misclassification-loss'
            self._cost_matrix = self.cost_matrix
        elif self.metric == 'mean-abs-error':
            self._metric = 'misclassification-loss'
            row_matrix = np.arange(self._n_classes).reshape(-1, 1) @ np.ones(
                [1, self._n_classes])
            self._cost_matrix = abs(row_matrix - row_matrix.T)
        elif self.metric == 'macro-accuracy':
            self._metric = 'custom'
            self._perf_func = macro_accuracy_func
        elif self.metric == 'f1-score':
            self._metric = 'custom'
            self._perf_func = f1_score_func
        elif self.metric == 'cohens-kappa':
            # TODO: implement
            self._metric = 'custom'
            self._perf_func = self.custom_perf_func
        else:
            raise ValueError("Metric '{}' not implemented. Use "
                             "metric='custom' instead.".format(
                self.metric))

    def query(self, X_cand, X, y, X_eval, batch_size=1, sample_weight=None,
              return_utilities=False, **kwargs):
        """

        Attributes
        ----------
        X: array-like (n_training_samples, n_features)
            Labeled samples
        y: array-like (n_training_samples)
            Labels of labeled samples
        X_cand: array-like (n_samples, n_features)
            Unlabeled candidate samples
        X_eval: array-like (n_samples, n_features) or string
            Unlabeled evaluation samples
        """

        # fit and initialize classifier to check if dimensions for classes
        # are correct
        self._cur_clf = deepcopy(self.clf).fit(X, y, sample_weight)
        self._classes = self._cur_clf._le.classes_
        self._n_classes = len(self._classes)
        self._missing_label = self._cur_clf.missing_label

        self._validate_and_transform_metric()

        self._validate_input(X_cand, X, y, X_eval, batch_size, sample_weight,
              return_utilities)

        # TODO: was passiert, wenn alpha nicht berechenbar ist oder das
        #  perf basierend auf der confmat berechnet werden soll?
        prior = calculate_optimal_prior(self._cost_matrix)
        self._prior_cand = self.prior_cand * prior
        self._prior_eval = self.prior_eval * prior


        prob_kernel = lambda X, X_ : pairwise_kernels(X, X_,
                                                metric=self.estimator_metric,
                                                **self.estimator_metric_dict)

        # filter out unlabeled for Bayesian Local Learner
        mask_labeled = is_labeled(y, self._missing_label)
        if np.sum(mask_labeled) == 0:
            # TODO: how to cope with pairwise kernels, when X_ is empty?
            mask_labeled[0] = True
        X_ = X[mask_labeled]
        y_ = y[mask_labeled]
        self.sample_weight_ = self.sample_weight_[mask_labeled]
        K_c_x = prob_kernel(X_cand, X_)
        # TODO: K_c_c only necessary when non-independent non-myopic OR
        #  partially for batch selection
        K_c_c = prob_kernel(X_cand, X_cand)
        K_e_x = prob_kernel(X_eval, X_)
        K_e_c = prob_kernel(X_eval, X_cand)

        prob_estimator_cand = BayesianPriorLearner(prior=self._prior_cand,
                                    classes=self._classes,
                                    missing_label=self._missing_label,
                                    metric="precomputed")
        prob_estimator_eval = BayesianPriorLearner(prior=self._prior_cand,
                                    classes=self._classes,
                                    missing_label=self._missing_label,
                                    metric="precomputed")

        prob_estimator_cand.fit(X_, y_, self.sample_weight_)
        prob_cand = prob_estimator_cand.predict_proba(K_c_x)

        pred_old = self._cur_clf.predict(X_eval)

        # TODO: neighbor_mode: 'nearest', 'same'
        #cand_idx_set = np.arange(len(X_cand)).reshape(-1,1)
        cand_idx_set = np.tile(np.arange(len(X_cand)), [self.max_nonmyopic_size,
                                                        1]).T
        utilities = np.empty([len(X_cand)])
        dperf_mat = np.empty_like(prob_cand)
        for i_c in range(len(X_cand)):
            utilities[i_c] = nonmyopic_gain(clf=self._cur_clf,
                                            X_c=X_cand[cand_idx_set[i_c]],
                                            X=X_,
                                            y=y_,
                                            sample_weight=self.sample_weight_,
                                            prob_estimator_cand=prob_estimator_cand,
                                            K_c_x=K_c_x[cand_idx_set[i_c],:],
                                            K_c_c=K_c_c[cand_idx_set[i_c],:],
                                            prob_estimator_eval=prob_estimator_eval,
                                            K_e_x=K_e_x,
                                            K_e_c=K_e_c[:, cand_idx_set[i_c]],
                                            pred_old=pred_old, X_eval=X_eval,
                                            classes=self._classes,
                                            metric=self._metric,
                                            cost_matrix=self._cost_matrix,
                                            perf_func=self._perf_func)


        best_indices = rand_argmax([utilities], axis=1, random_state=self.random_state)
        if return_utilities:
            return best_indices, np.array([utilities])
        else:
            return best_indices


def nonmyopic_gain(clf, X_c, X, y, sample_weight,
                   prob_estimator_cand, K_c_x, K_c_c,
                   prob_estimator_eval, K_e_x, K_e_c,
                   pred_old, X_eval, classes, metric, cost_matrix,
                   perf_func, prob_mode='exact', label_mode='all'):
    """
    prob_mode: 'exact', 'approx' (instances are considered independent)
    label_mode: 'all', 'single'
    """
    if sample_weight is None:
        sample_weight = np.ones(len(X))

    dperf_mat = np.full(len(X_c), np.nan)

    for i_c in range(len(X_c)):
        y_sim_list = _get_y_sim_list(n_classes=len(classes), n_instances=i_c+1,
                                    label_mode=label_mode)

        if prob_mode == 'approx':
            prob_estimator_cand.fit(X, y, sample_weight)
            prob_cand_y = prob_estimator_cand.predict_proba(K_c_x[0:i_c+1,:])
            prob_y_sim = np.prod(prob_cand_y[range(i_c+1), y_sim_list],axis=1)
        elif prob_mode == 'exact':
            # TODO: Speed Up
            prob_y_sim = np.ones(len(y_sim_list))
            for i_y_sim, y_sim in enumerate(y_sim_list):
                for i_y in range(len(y_sim)):
                    X_new = np.concatenate([X, X_c[0:i_y]], axis=0)
                    y_new = np.concatenate([y, y_sim[0:i_y]], axis=0)
                    sample_weight_new = np.concatenate([sample_weight, np.ones(
                        i_y)], axis=0)
                    K_c_cx = np.concatenate([K_c_x[i_y:i_y+1,:],
                                             K_c_c[i_y:i_y+1, 0:i_y]], axis=1)
                    prob_estimator_cand.fit(X_new, y_new, sample_weight_new)
                    prob_cand_y = prob_estimator_cand.predict_proba(K_c_cx)
                    prob_y_sim[i_y_sim] *= prob_cand_y[0][y_sim[i_y]]

        X_new = np.concatenate([X, X_c[0:i_c+1]], axis=0)
        K_e_cx = np.concatenate([K_e_x, K_e_c[:,0:i_c+1]], axis=1)
        sample_weight_new = np.concatenate([sample_weight, np.ones(
            i_c+1)], axis=0)

        dperf_mat[i_c] = 0
        for i_y_sim, y_sim in enumerate(y_sim_list):

            y_new = np.concatenate([y, y_sim], axis=0)
            prob_estimator_eval.fit(X_new, y_new, sample_weight_new)
            prob_eval = prob_estimator_eval.predict_proba(K_e_cx)
            new_clf = clf.fit(X_new, y_new, sample_weight_new)
            pred_new = new_clf.predict(X_eval)

            dperf_mat[i_c] += dperf(prob_eval, pred_old, pred_new,
                                    metric=metric, cost_matrix=cost_matrix,
                                    perf_func=perf_func) * prob_y_sim[i_y_sim]

    avg_dperf_mat = dperf_mat / np.arange(1, len(X_c)+1)
    return np.max(avg_dperf_mat)

def _get_y_sim_list(n_classes, n_instances, label_mode):
    if label_mode=='all':
        return list(itertools.product(*([range(n_classes)]*n_instances)))
    if label_mode=='single':
        return np.tile(np.arange(n_classes), [n_instances,1]).T

def dperf(probs, pred_old, pred_new, metric, cost_matrix=None, perf_func=None):
    if metric == 'misclassification-loss':
        # TODO: check if cost_matrix is correct
        pred_changed = (pred_new != pred_old)
        # TODO: check why cost_matrix needs to be transposed here
        return np.sum(probs[pred_changed, :] * (cost_matrix.T[pred_old[
            pred_changed]] - cost_matrix.T[pred_new[pred_changed]])) / len(
            probs)
    elif metric == 'custom':
        # TODO: check if perf_func is correct
        n_classes = probs.shape[1]
        conf_mat_old = np.zeros([n_classes, n_classes])
        conf_mat_new = np.zeros([n_classes, n_classes])
        for y_pred in range(n_classes):
            conf_mat_old[:, y_pred] += np.sum(probs[pred_old == y_pred], axis=0)
            conf_mat_new[:, y_pred] += np.sum(probs[pred_new == y_pred], axis=0)
        return perf_func(conf_mat_new) - perf_func(conf_mat_old)
    else:
        raise ValueError("metric unknown")


"""
def compute_scores_sequential(freq_cand, freq_eval_mat, pred_eval_mat,
                              freq_eval_new_mat, pred_eval_new_mat, classes,
                              alpha_cand, alpha_eval, risk, **kwargs):
    prob_cand = get_prior_prob(freq_cand, alpha_cand)
    prob_eval_new_mat = get_prior_prob(freq_eval_new_mat, alpha_eval)

    risk_diff_mat = np.full(prob_cand.shape, np.nan)
    for i_x_c in range(prob_cand.shape[0]):
        for i_y_c in range(prob_cand.shape[1]):
            # risk difference for one data model (trained with new label) - only predictions should vary
            risk_diff_mat[i_x_c, i_y_c] = risk_difference(prob_eval_new_mat[i_x_c, i_y_c, :, :],
                                                          freq_eval_mat[i_x_c], pred_eval_mat[i_x_c],
                                                          freq_eval_new_mat[i_x_c, i_y_c], pred_eval_new_mat[i_x_c, i_y_c],
                                                          risk=risk, classes=classes, **kwargs)

    return -np.sum(risk_diff_mat * prob_cand, axis=1)
"""

def score_recall(conf_matrix):
    return conf_matrix[-1, -1] / conf_matrix[-1, :].sum()


def macro_accuracy_func(conf_matrix):
    return np.mean(conf_matrix.diagonal() / conf_matrix.sum(axis=1))


def score_accuracy(conf_matrix):
    return conf_matrix.diagonal().sum() / conf_matrix.sum()


def score_precision(conf_matrix):
    pos_pred = conf_matrix[:, -1].sum()
    return conf_matrix[-1, -1] / pos_pred if pos_pred > 0 else 0


def f1_score_func(conf_matrix):
    recall = score_recall(conf_matrix)
    precision = score_precision(conf_matrix)
    norm = recall + precision
    return 2 * recall * precision / norm if norm > 0 else 0


def calculate_optimal_prior(CM=None):
    n_classes = len(CM)
    if CM is None:
        return np.full([n_classes], 1./n_classes)
    else:
        M = np.ones([1, len(CM)]) @ np.linalg.inv(CM)
        if np.all(M[0] >= 0):
            return M[0] / np.sum(M)
        else:
            return np.full([n_classes], 1. / n_classes)


def cost_reduction(k_vec_list, C=None, m_max=2, prior=1.e-3):
    """Calculate the expected cost reduction.

    Calculate the expected cost reduction for given maximum number of
    hypothetically acquired labels, observed labels and cost matrix.

    Parameters
    ----------
    k_vec_list: array-like, shape (n_classes)
        Observed class labels.
    C: array-like, shape = (n_classes, n_classes)
        Cost matrix.
    m_max: int
        Maximal number of hypothetically acquired labels.
    prior : float | array-like, shape (n_classes)
       Prior value for each class.

    Returns
    -------
    expected_cost_reduction: array-like, shape (n_samples)
        Expected cost reduction for given parameters.
    """
    n_classes = len(k_vec_list[0])
    n_samples = len(k_vec_list)

    # check cost matrix
    C = 1 - np.eye(n_classes) if C is None else np.asarray(C)

    # generate labelling vectors for all possible m values
    l_vec_list = np.vstack([gen_l_vec_list(m, n_classes)
                            for m in range(m_max + 1)])
    m_list = np.sum(l_vec_list, axis=1)
    n_l_vecs = len(l_vec_list)

    # compute optimal cost-sensitive decision for all combination of k-vectors
    # and l-vectors
    k_l_vec_list = np.swapaxes(np.tile(k_vec_list, (n_l_vecs, 1, 1)), 0, 1)\
                   + l_vec_list
    y_hats = np.argmin(k_l_vec_list @ C, axis=2)

    # add prior to k-vectors
    prior = prior * np.ones(n_classes)
    k_vec_list = np.asarray(k_vec_list) + prior

    # all combination of k-, l-, and prediction indicator vectors
    combs = [k_vec_list, l_vec_list, np.eye(n_classes)]
    combs = np.asarray([list(elem)
                        for elem in list(itertools.product(*combs))])

    # three factors of the closed form solution
    factor_1 = 1 / euler_beta(k_vec_list)
    factor_2 = multinomial(l_vec_list)
    factor_3 = euler_beta(np.sum(combs, axis=1)).reshape(n_samples, n_l_vecs,
                                                         n_classes)

    # expected classification cost for each m
    m_sums = np.asarray(
        [factor_1[k_idx]
         * np.bincount(m_list, factor_2 * [C[:, y_hats[k_idx, l_idx]]
                                           @ factor_3[k_idx, l_idx]
                                           for l_idx in range(n_l_vecs)])
         for k_idx in range(n_samples)]
    )

    # compute classification cost reduction as difference
    gains = np.zeros((n_samples, m_max)) + m_sums[:, 0].reshape(-1, 1)
    gains -= m_sums[:, 1:]

    # normalize  cost reduction by number of hypothetical label acquisitions
    gains /= np.arange(1, m_max + 1)

    return np.max(gains, axis=1)


def gen_l_vec_list(m_approx, n_classes):
    """
    Creates all possible class labeling vectors for given number of
    hypothetically acquired labels and given number of classes.

    Parameters
    ----------
    m_approx: int
        Number of hypothetically acquired labels..
    n_classes: int,
        Number of classes

    Returns
    -------
    label_vec_list: array-like, shape = [n_labelings, n_classes]
        All possible class labelings for given parameters.
    """

    label_vec_list = [[]]
    label_vec_res = np.arange(m_approx + 1)
    for i in range(n_classes - 1):
        new_label_vec_list = []
        for labelVec in label_vec_list:
            for newLabel in label_vec_res[label_vec_res
                                          - (m_approx - sum(labelVec))
                                          <= 1.e-10]:
                new_label_vec_list.append(labelVec + [newLabel])
        label_vec_list = new_label_vec_list

    new_label_vec_list = []
    for labelVec in label_vec_list:
        new_label_vec_list.append(labelVec + [m_approx - sum(labelVec)])
    label_vec_list = np.array(new_label_vec_list, int)

    return label_vec_list


def euler_beta(a):
    """
    Represents Euler beta function:
    B(a(i)) = Gamma(a(i,1))*...*Gamma(a_n)/Gamma(a(i,1)+...+a(i,n))

    Parameters
    ----------
    a: array-like, shape (m, n)
        Vectors to evaluated.

    Returns
    -------
    result: array-like, shape (m)
        Euler beta function results [B(a(0)), ..., B(a(m))
    """
    return np.exp(np.sum(gammaln(a), axis=1)-gammaln(np.sum(a, axis=1)))


def multinomial(a):
    """
    Computes Multinomial coefficient:
    Mult(a(i)) = (a(i,1)+...+a(i,n))!/(a(i,1)!...a(i,n)!)

    Parameters
    ----------
    a: array-like, shape (m, n)
        Vectors to evaluated.

    Returns
    -------
    result: array-like, shape (m)
        Multinomial coefficients [Mult(a(0)), ..., Mult(a(m))
    """
    return factorial(np.sum(a, axis=1))/np.prod(factorial(a), axis=1)
