import itertools
import numpy as np

from scipy.special import factorial, gammaln
from sklearn.utils import check_array, check_random_state

from skactiveml.base import PoolBasedQueryStrategy, ClassFrequencyEstimator
from skactiveml.utils import rand_argmax, is_labeled, MISSING_LABEL


class McPAL(PoolBasedQueryStrategy):
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

    Attributes
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

        X_cand = check_array(X_cand, force_all_finite=False)
        labeled_idx = is_labeled(y, missing_label=self.clf.missing_label)
        X = np.array(X)
        y = np.array(y)
        X_labeled = X[labeled_idx]
        y_labeled = y[labeled_idx]

        # Calculate gains
        self.clf.fit(X_labeled, y_labeled)
        k_vec = self.clf.predict_freq(X_cand)
        utilities = weights * cost_reduction(k_vec, prior=self.prior,
                                             m_max=self.m_max)
        query_indices = rand_argmax(utilities, random_state=self.random_state)

        if return_utilities:
            return query_indices, np.array([utilities])
        else:
            return query_indices


class XPAL(PoolBasedQueryStrategy):

    def __init__(self, clf, classes, missing_label=MISSING_LABEL, perf_est=None, risk='error', mode='sequential', prior_cand=0.001, prior_eval=0.001, random_state=None, **kwargs):
        # TODO @DK: clean up
        """ XPAL
        The cost-sensitive expected probabilistic active learning (CsXPAL) strategy is a generalization of the
        multi-class probabilistic active learning (McPAL) [1], the optimised probabilistic active learning (OPAL) [2]
        strategy, and the cost-sensitive probabilistic active learning (CsPAL) strategy due to consideration of a cost
        matrix for multi-class classification problems and the computation of the expected error on an evaluation set.

        Attributes
        ----------
        classes: list (n_classes)
            List of all classes
        risk: string
            "error"
            "accuracy"
            "misclassification-loss"
                requires cost-matrix or cost-vector
            "mean-abs-error"
            "macro-accuracy"
                optional: prior_class_probability
            "f1-score"

        cost_matrix: array-like (n_classes, n_classes)
        cost_vector: array-like (n_classes)
        prior_class_probability: array-like (n_classes)

        mode: string
            The mode to select candidates:
                "sequential" (default)
                "batch"
                    requires batch-size
                "non-myopic"
                    requires max_nonmyopic_size

        batch_size: int
        max_nonmyopic_size: int

        prior_cand: float (default 10^-3)
        prior_eval: float (default 10^-3)

        References
        ----------
        [1] Daniel Kottke, Georg Krempl, Dominik Lang, Johannes Teschner, and Myra Spiliopoulou.
            Multi-Class Probabilistic Active Learning,
            vol. 285 of Frontiers in Artificial Intelligence and Applications, pages 586-594. IOS Press, 2016
        [2] Georg Krempl, Daniel Kottke, Vincent Lemaire.
            Optimised probabilistic active learning (OPAL),
            vol. 100 oof Machine Learning, pages 449-476. Springer, 2015
        """
        super().__init__(random_state=random_state)

        # TODO perf_est needs to implement predict_freq
        self.clf = clf
        self.perf_est = perf_est
        self.mode = mode
        self.missing_label = missing_label

        # TODO remove self.classes
        self.classes = classes

        if risk == 'error' or risk == 'accuracy':
            self.risk = 'misclassification-loss'
            self.cost_matrix = 1 - np.eye(len(self.classes))
        elif risk == 'misclassification-loss':
            self.risk = 'misclassification-loss'
            self.cost_matrix = kwargs.pop('cost_matrix', None)
            if self.cost_matrix is None or self.cost_matrix.shape[0] != len(self.classes) or self.cost_matrix.shape[
                1] != len(self.classes):
                raise ValueError(
                    "cost-matrix must be given and must have shape (n_classes x n_classes)"
                )
        elif risk == 'mean-abs-error':
            self.risk = 'misclassification-loss'
            row_matrix = np.arange(len(self.classes)).reshape(-1, 1) * np.ones([1, len(self.classes)])
            self.cost_matrix = abs(row_matrix - row_matrix.T)
        elif risk == 'macro-accuracy':
            self.risk = risk
            # the cost matrix for macro accuracy is overwritten in the risk difference step.
            # it can be used to set the prior if you have prior knowledge about the data
            prior_class_probability = kwargs.pop('prior_class_probability',
                                                 np.ones(len(self.classes)) / len(self.classes))
            self.cost_matrix = cost_vector_to_cost_matrix(1 / prior_class_probability)

        # set the priors accordingly
        alpha = get_alpha(self.cost_matrix)
        # TODO: check alpha

        self.alpha_cand = prior_cand * alpha
        self.alpha_eval = prior_eval * alpha

    def query(self, X_cand, X, y, X_eval, return_utilities=False, **kwargs):
        """

        Attributes
        ----------
        X: array-like (n_training_samples, n_features)
            Labeled samples
        y: array-like (n_training_samples)
            Labels of labeled samples
        X_cand: array-like (n_samples, n_features)
            Unlabeled candidate samples
        X_eval: array-like (n_samples, n_features)
            Unlabeled evaluation samples
        """
        #labeled_idx = is_labeled(y, missing_label=self.missing_label)
        #X = X[labeled_idx]
        #y = y[labeled_idx]

        if self.mode == 'sequential':

            if False: # hasattr(self.perf_est, 'predict_freq_seqal'):
                # freq_cand          (n_cand, n_classes)
                # pred_eval          (n_eval)
                # freq_eval_new_mat  (n_cand, n_classes, n_eval, n_classes),
                # pred_eval_new_mat  (n_cand, n_classes, n_eval)

                freq_cand, freq_eval, freq_eval_new_mat = self.perf_est.predict_freq_seqal(X, y, X_cand, self.classes, X_eval)
            else:
                self.clf.fit(X, y)
                pred_eval = self.clf.predict(X_eval).astype(int)
                if self.perf_est is None:
                    freq_cand = self.clf.predict_freq(X_cand)
                    freq_eval = self.clf.predict_freq(X_eval)
                else:
                    self.perf_est.fit(X, y)
                    freq_cand = self.perf_est.predict_freq(X_cand)
                    freq_eval = self.perf_est.predict_freq(X_eval)

                freq_eval_new_mat = np.full([len(X_cand), len(self.classes), len(X_eval), len(self.classes)], np.nan)
                pred_eval_new_mat = np.full([len(X_cand), len(self.classes), len(X_eval)], np.nan, dtype=int)
                for i_x_c, x_c in enumerate(X_cand):
                    for i_y_c, y_c in enumerate(self.classes):
                        X_new = np.vstack([X, [x_c]])
                        y_new = np.hstack([y, [y_c]])

                        self.clf.fit(X_new, y_new)
                        pred_eval_new_mat[i_x_c, i_y_c, :] = self.clf.predict(X_eval).astype(int)
                        if self.perf_est is None:
                            freq_eval_new_mat[i_x_c, i_y_c, :, :] = self.clf.predict_freq(X_eval)
                        else:
                            self.perf_est.fit(X_new, y_new)
                            freq_eval_new_mat[i_x_c, i_y_c, :, :] = self.perf_est.predict_freq(X_eval)

            # TODO: np.broadcast_to (different old predictions for pred_eval => pred_eval_mat)
            freq_eval_mat = np.tile(freq_eval, [len(X_cand), 1, 1])
            pred_eval_mat = np.tile(pred_eval, [len(X_cand), 1]).astype(int)

            # freq_cand          (n_cand, n_classes)
            # freq_eval_mat      (n_cand, n_eval, n_classes)
            # freq_eval_new_mat  (n_cand, n_classes, n_eval, n_classes)

            utilities = compute_scores_sequential(freq_cand, freq_eval_mat, pred_eval_mat,
                                                  freq_eval_new_mat, pred_eval_new_mat,
                                                  classes=self.classes,
                                                  alpha_cand=self.alpha_cand, alpha_eval=self.alpha_eval,
                                                  risk=self.risk, cost_matrix=self.cost_matrix)

            best_indices = rand_argmax([utilities], axis=1, random_state=self.random_state)
            if return_utilities:
                return best_indices, np.array([utilities])
            else:
                return best_indices


def cost_vector_to_cost_matrix(cost_vector):
    cost_matrix = np.array(cost_vector).reshape(-1, 1) @ np.ones((1, len(cost_vector)))
    np.fill_diagonal(cost_matrix, 0)
    return cost_matrix


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


def risk_difference(prob_eval_new, freq_eval, pred_eval, freq_eval_new,
                    pred_eval_new, risk, classes, **kwargs):
    # prob_eval_new (n_eval, n_classes)
    # freq_eval     (n_eval, n_classes)
    # freq_eval_new  (n_eval, n_classes)
    if risk == 'error':
        #pred_eval = np.argmax(freq_eval, axis=1)
        #pred_eval_new = np.argmax(freq_eval_new, axis=1)
        loss_diffs = np.array([np.array(y != pred_eval_new, int) - np.array(y != pred_eval, int) for y in classes]).T
        return np.mean(np.sum(prob_eval_new * loss_diffs, axis=-1))
    elif risk == 'misclassification-loss':
        cost_matrix = kwargs.pop('cost_matrix', None)
        #pred_eval = np.argmin(freq_eval @ cost_matrix, axis=1)
        #pred_eval_new = np.argmin(freq_eval_new @ cost_matrix, axis=1)
        loss_diffs = np.array([cost_matrix[y, pred_eval_new] - cost_matrix[y, pred_eval] for y in classes]).T
        return np.mean(np.sum(prob_eval_new * loss_diffs, axis=-1))
    elif risk == 'f1-score':
        C = cost_vector_to_cost_matrix(1 / np.sum(prob_eval_new, axis=0))
        #pred_eval = np.argmin(freq_eval @ C, axis=1)
        #pred_eval_new = np.argmin(freq_eval_new @ C, axis=1)
        conf_matrix, conf_matrix_new = get_conf_matrices(prob_eval_new, pred_eval, pred_eval_new, classes)
        return score_f1(conf_matrix) - score_f1(conf_matrix_new)
    elif risk == 'macro-accuracy':
        C = cost_vector_to_cost_matrix(1 / np.sum(prob_eval_new, axis=0))
        #pred_eval = np.argmin(freq_eval @ C, axis=1)
        #pred_eval_new = np.argmin(freq_eval_new @ C, axis=1)
        conf_matrix, conf_matrix_new = get_conf_matrices(prob_eval_new, pred_eval, pred_eval_new, classes)
        # if score_macro_accuracy(conf_matrix) - score_macro_accuracy(conf_matrix_new) > 0:
        # print(conf_matrix)
        # print(conf_matrix_new)
        return score_macro_accuracy(conf_matrix) - score_macro_accuracy(conf_matrix_new)
    elif risk == 'accuracy':
        #pred_eval = np.argmax(freq_eval, axis=1)
        #pred_eval_new = np.argmax(freq_eval_new, axis=1)
        conf_matrix, conf_matrix_new = get_conf_matrices(prob_eval_new, pred_eval, pred_eval_new, classes)
        return score_accuracy(conf_matrix) - score_accuracy(conf_matrix_new)


def get_conf_matrices(prob_eval_new, pred_eval, pred_eval_new, classes):
    conf_matrix = np.full([len(classes), len(classes)], np.nan)
    conf_matrix_new = np.full([len(classes), len(classes)], np.nan)
    for i_y, y in enumerate(classes):
        for i_y_hat, y_hat in enumerate(classes):
            conf_matrix[i_y, i_y_hat] = np.sum(prob_eval_new[np.array([y_hat == pred_eval]).flatten(), i_y])
            conf_matrix_new[i_y, i_y_hat] = np.sum(prob_eval_new[np.array([y_hat == pred_eval_new]).flatten(), i_y])
    return conf_matrix, conf_matrix_new


def score_recall(conf_matrix):
    return conf_matrix[-1, -1] / conf_matrix[-1, :].sum()


def score_macro_accuracy(conf_matrix):
    return np.mean(conf_matrix.diagonal() / conf_matrix.sum(axis=1))


def score_accuracy(conf_matrix):
    return conf_matrix.diagonal().sum() / conf_matrix.sum()


def score_precision(conf_matrix):
    pos_pred = conf_matrix[:, -1].sum()
    return conf_matrix[-1, -1] / pos_pred if pos_pred > 0 else 0


def score_f1(conf_matrix):
    recall = score_recall(conf_matrix)
    precision = score_precision(conf_matrix)
    norm = recall + precision
    return 2 * recall * precision / norm if norm > 0 else 0


def get_prior_prob(freq, alpha):
    freq = freq + alpha
    return freq / np.sum(freq, axis=-1, keepdims=True)


def get_alpha(CM):
    M = np.ones([1, len(CM)]) @ np.linalg.inv(CM)
    return M[0] / np.sum(M)


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
