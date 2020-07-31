import numpy as np

from sklearn.utils import check_array

from ..base import PoolBasedQueryStrategy

# TODO: @CS: you can decide if you want to implement OPAL or McPAL
class McPAL(PoolBasedQueryStrategy):
    # TODO: @CS: add comments and doc_string (incl paper reference) as in sklearn

    def __init__(self, clf, random_state=None):
        super().__init__(random_state=random_state)

        # TODO: @CS: add all necessary parameters
        self.clf = clf

    def query(self, X_cand, X, y, return_utilities=False, **kwargs):
        X_cand = check_array(X_cand, force_all_finite=False)

        # TODO: @CS: check if 'clf' has attr predict_freq()

        # TODO: @CS: complete
        # TODO: @CS: please use functions outside this class if appropriate

        # best_indices is a np.array (batch_size=1)
        # utilities is a np.array (batch_size=1 x len(X_cand)
        if return_utilities:
            return best_indices, utilities
        else:
            return best_indices


class XPAL(PoolBasedQueryStrategy):

    def __init__(self, clf, perf_est, classes, risk='error', mode='sequential', prior_cand=0.001, prior_eval=0.001, random_state=None, **kwargs):
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

        if self.mode == 'sequential':

            if hasattr(self.perf_est, 'predict_freq_seqal'):
                # freq_cand          (n_cand, n_classes)
                # pred_eval          (n_eval)
                # freq_eval_new_mat  (n_cand, n_classes, n_eval, n_classes),
                # pred_eval_new_mat  (n_cand, n_classes, n_eval)

                freq_cand, freq_eval, freq_eval_new_mat = self.perf_est.predict_freq_seqal(X, y, X_cand, self.classes, X_eval)
            else:
                self.perf_est.fit(X, y)
                freq_cand = self.perf_est.predict_freq(X_cand)
                freq_eval = self.perf_est.predict_freq(X_eval)

                freq_eval_new_mat = np.full([len(X_cand), len(self.classes), len(X_eval), len(self.classes)], np.nan)
                for i_x_c, x_c in enumerate(X_cand):
                    for i_y_c, y_c in enumerate(self.classes):
                        X_new = np.vstack([X, [x_c]])
                        y_new = np.hstack([y, [y_c]])

                        self.perf_est.fit(X_new, y_new)
                        freq_eval_new_mat[i_x_c, i_y_c, :, :] = self.perf_est.predict_freq(X_eval)

            # TODO: np.broadcast_to (differnet old predictions for pred_eval => pred_eval_mat)
            freq_eval_mat = np.tile(freq_eval, [len(X_cand), 1, 1])

            # freq_cand          (n_cand, n_classes)
            # freq_eval_mat      (n_cand, n_eval, n_classes)
            # freq_eval_new_mat  (n_cand, n_classes, n_eval, n_classes)

            return compute_scores_sequential(freq_cand, freq_eval_mat, freq_eval_new_mat, classes=self.classes,
                                             alpha_cand=self.alpha_cand, alpha_eval=self.alpha_eval,
                                             risk=self.risk, cost_matrix=self.cost_matrix)


def cost_vector_to_cost_matrix(cost_vector):
    cost_matrix = np.array(cost_vector).reshape(-1, 1) @ np.ones((1, len(cost_vector)))
    np.fill_diagonal(cost_matrix, 0)
    return cost_matrix


def compute_scores_sequential(freq_cand, freq_eval_mat, freq_eval_new_mat, classes, alpha_cand,
                              alpha_eval, risk, **kwargs):
    prob_cand = get_prior_prob(freq_cand, alpha_cand)
    prob_eval_new_mat = get_prior_prob(freq_eval_new_mat, alpha_eval)

    risk_diff_mat = np.full(prob_cand.shape, np.nan)
    for i_x_c in range(prob_cand.shape[0]):
        for i_y_c in range(prob_cand.shape[1]):
            risk_diff_mat[i_x_c, i_y_c] = risk_difference(prob_eval_new_mat[i_x_c, i_y_c, :, :],
                                                          freq_eval_mat[i_x_c], freq_eval_new_mat[i_x_c, i_y_c],
                                                          risk=risk, classes=classes, **kwargs)

    return -np.sum(risk_diff_mat * prob_cand, axis=1)


def risk_difference(prob_eval_new, freq_eval, freq_eval_new, risk, classes, **kwargs):
    # prob_eval_new (n_eval, n_classes)
    # freq_eval     (n_eval, n_classes)
    # freq_eval_new  (n_eval, n_classes)
    if risk == 'error':
        pred_eval = np.argmax(freq_eval, axis=1)
        pred_eval_new = np.argmax(freq_eval_new, axis=1)
        loss_diffs = np.array([np.array(y != pred_eval_new, int) - np.array(y != pred_eval, int) for y in classes]).T
        return np.mean(np.sum(prob_eval_new * loss_diffs, axis=-1))
    elif risk == 'misclassification-loss':
        cost_matrix = kwargs.pop('cost_matrix', None)
        pred_eval = np.argmin(freq_eval @ cost_matrix, axis=1)
        pred_eval_new = np.argmin(freq_eval_new @ cost_matrix, axis=1)
        loss_diffs = np.array([cost_matrix[y, pred_eval_new] - cost_matrix[y, pred_eval] for y in classes]).T
        return np.mean(np.sum(prob_eval_new * loss_diffs, axis=-1))
    elif risk == 'f1-score':
        C = cost_vector_to_cost_matrix(1 / np.sum(prob_eval_new, axis=0))
        pred_eval = np.argmin(freq_eval @ C, axis=1)
        pred_eval_new = np.argmin(freq_eval_new @ C, axis=1)
        conf_matrix, conf_matrix_new = get_conf_matrices(prob_eval_new, pred_eval, pred_eval_new, classes)
        return score_f1(conf_matrix) - score_f1(conf_matrix_new)
    elif risk == 'macro-accuracy':
        C = cost_vector_to_cost_matrix(1 / np.sum(prob_eval_new, axis=0))
        pred_eval = np.argmin(freq_eval @ C, axis=1)
        pred_eval_new = np.argmin(freq_eval_new @ C, axis=1)
        conf_matrix, conf_matrix_new = get_conf_matrices(prob_eval_new, pred_eval, pred_eval_new, classes)
        # if score_macro_accuracy(conf_matrix) - score_macro_accuracy(conf_matrix_new) > 0:
        # print(conf_matrix)
        # print(conf_matrix_new)
        return score_macro_accuracy(conf_matrix) - score_macro_accuracy(conf_matrix_new)
    elif risk == 'accuracy':
        pred_eval = np.argmax(freq_eval, axis=1)
        pred_eval_new = np.argmax(freq_eval_new, axis=1)
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
