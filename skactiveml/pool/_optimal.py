import warnings

import itertools

import numpy as np
from sklearn import clone

from skactiveml.base import SingleAnnotPoolBasedQueryStrategy

from sklearn.metrics import accuracy_score, pairwise_kernels

from skactiveml.utils import check_random_state, ExtLabelEncoder, rand_argmax


class Optimal(SingleAnnotPoolBasedQueryStrategy):

    def __init__(self, clf, score=accuracy_score, maximize_score=True,
                 nonmyopic_look_ahead=2,
                 similarity_metric='rbf', similarity_metric_dict=None,
                 random_state=None):
        """ An optimal Al strategy
        """
        super().__init__(random_state=random_state)

        self.clf = clf
        self.score = score
        self.maximize_score = maximize_score
        self.nonmyopic_look_ahead = nonmyopic_look_ahead
        self.similarity_metric = similarity_metric
        self.similarity_metric_dict = similarity_metric_dict
        self.random_state = random_state

    def query(self, X_cand, y_cand, X, y, X_eval, y_eval, batch_size=1,
              sample_weight_cand=None, sample_weight=None,
              sample_weight_eval=None, return_utilities=False, **kwargs):
        """

        Attributes
        ----------
        """

        X_cand, return_utilities, batch_size, random_state = \
            self._validate_data(X_cand, return_utilities, batch_size,
                                self.random_state, reset=True)

        clf = clone(self.clf)

        if sample_weight is None:
            sample_weight = np.ones(len(X))
        if sample_weight_cand is None:
            sample_weight_cand = np.ones(len(X_cand))
        if sample_weight_eval is None:
            sample_weight_eval = np.ones(len(X_eval))

        if self.similarity_metric_dict is None:
            similarity_metric_dict = {}

        sim_cand = pairwise_kernels(X_cand, X_cand,
                                    metric=self.similarity_metric,
                                    **similarity_metric_dict)

        utilities = np.full([batch_size, len(X_cand)], np.nan, dtype=float)
        best_idx = np.full([batch_size], np.nan, dtype=int)
        for i_batch in range(batch_size):
            unlbld_cand_idx = np.setdiff1d(np.arange(len(X_cand)), best_idx)

            sim_unlbld_cand = sim_cand[unlbld_cand_idx][:, unlbld_cand_idx]
            # cand_idx_set = (-sim_unlbld_cand).argsort(axis=1)[:, :self.nonmyopic_look_ahead]

            cand_idx_set = np.array(list(itertools.permutations(range(len(
                X_cand)), self.nonmyopic_look_ahead)))

            batch_utilities = np.empty([len(cand_idx_set),
                                        self.nonmyopic_look_ahead])

            X_ = np.concatenate([X_cand, X_cand[best_idx[:i_batch]], X], axis=0)
            y_ = np.concatenate([y_cand, y_cand[best_idx[:i_batch]], y], axis=0)
            sample_weight_ = np.concatenate([sample_weight_cand,
                                             sample_weight_cand[best_idx[:i_batch]],
                                             sample_weight])

            lbld_idx_ = list(range(len(X_cand), len(X_)))
            append_lbld = lambda x: list(x) + lbld_idx_

            idx_new = append_lbld([])
            X_new = X_[idx_new]
            y_new = y_[idx_new]
            sample_weight_new = sample_weight_[idx_new]
            clf_new = clf.fit(X_new, y_new, sample_weight_new)
            pred_eval = clf_new.predict(X_eval)

            old_perf = self.score(y_eval, pred_eval)  # TODO, sample_weight_eval)

            for i_full_cand_idx, full_cand_idx in enumerate(cand_idx_set):
                full_cand_idx = list(full_cand_idx)
                cand_idx_subsets = [full_cand_idx[:(i + 1)] for i in
                                    range(len(full_cand_idx))]

                for i_cand_idx, cand_idx in enumerate(cand_idx_subsets):
                    idx_new = append_lbld(cand_idx)
                    X_new = X_[idx_new]
                    y_new = y_[idx_new]
                    sample_weight_new = sample_weight_[idx_new]

                    clf_new = clf.fit(X_new, y_new, sample_weight_new)

                    pred_eval = clf_new.predict(X_eval)

                    batch_utilities[i_full_cand_idx, i_cand_idx] = \
                        self.score(y_eval, pred_eval) - old_perf# TODO, sample_weight_eval)

                if not self.maximize_score:
                    batch_utilities *= -1

                signs = np.sign(batch_utilities)
                if not ((signs >= 0).all() or (signs <= 0).all()):
                    pass
                    # TODO warnings.warn("There exist positive and negative utilities")

            batch_utilities /= np.arange(1, self.nonmyopic_look_ahead + 1).\
                reshape(1, -1)

            look_ahead_maximums = np.nanmax(batch_utilities, axis=0)
            opt_look_ahead = np.argmax(look_ahead_maximums)

            cur_best_idx = rand_argmax([batch_utilities[:, opt_look_ahead]],
                                       axis=1, random_state=random_state)

            best_idx[i_batch] = cand_idx_set[cur_best_idx, 0]

            batch_utilities_ = np.nanmax(batch_utilities, axis=1)
            for c_idx in unlbld_cand_idx:
                subset = (cand_idx_set[:,0] == c_idx)
                utilities[i_batch, c_idx] = np.max(batch_utilities_[subset])

        if return_utilities:
            return best_idx, utilities
        else:
            return best_idx
