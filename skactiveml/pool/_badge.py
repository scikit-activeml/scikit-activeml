import numpy as np
from scipy import stats
from sklearn import clone
from sklearn.metrics import pairwise_distances_argmin_min

from ..base import SingleAnnotatorPoolQueryStrategy, SkactivemlClassifier
from ..utils import (
    MISSING_LABEL,
    check_type,
    check_equal_missing_label,
    unlabeled_indices
)


class Badge(SingleAnnotatorPoolQueryStrategy):
    def __init__(
            self,
            missing_label=MISSING_LABEL,
            random_state=None
    ):
        super().__init__(
            missing_label=missing_label, random_state=random_state
        )

    def query(
            self,
            X,
            y,
            clf,
            candidates=None,
            batch_size=1,
            return_utilities=False,
            return_embeddings=False
    ):
        # Validate input parameters
        X, y, candidates, batch_size, return_utilities = self._validate_data(
            X, y, candidates, batch_size, return_utilities, reset=True
        )

        X_cand, mapping = self._transform_candidates(candidates, X, y)

        # Validate classifier type
        check_type(clf, "clf", SkactivemlClassifier)
        check_equal_missing_label(clf.missing_label, self.missing_label_)
        if not isinstance(return_embeddings, bool):
            raise TypeError("'return_embeddings' must be a boolean.")

        # Fit the classifier
        clf = clone(clf).fit(X, y)

        # find the unlabeled dataset
        if candidates is None:
            X_unlabeled = X_cand
            unlabeled_mapping = mapping
        elif mapping is not None:
            unlabeled_mapping = unlabeled_indices(y[mapping], missing_label=self.missing_label)
            X_unlabeled = X_cand[unlabeled_mapping]
            unlabeled_mapping = mapping[unlabeled_mapping]
        else:
            X_unlabeled = X_cand
            unlabeled_mapping = np.arange(len(X_cand))

        # Gradient embedding, aka predict class membership probabilities
        if return_embeddings:
            probas, X_embedding_ulbd = clf.predict_proba(X_unlabeled, return_embeddings=True)
            p_max = np.max(probas)
            g_x = (p_max - 1) * X_embedding_ulbd
        else:
            probas = clf.predict_proba(X_unlabeled)
            p_max = np.max(probas)
            g_x = (p_max - 1) * X_unlabeled

        # 2. sampling with kmeans++


def k_means_plus_plus(g_x, batch_size, random_state_):

    query_indicies = np.array([], dtype=int)
    for i in range(batch_size):
        if i == 0:
            idx = random_state_.choice(len(g_x))
            query_indicies = np.append(query_indicies, [idx])
            continue
        D_probas = _d_probability(g_x, query_indicies)
        customDist = stats.rv_discrete(name="customDist", values=(len(D_probas), D_probas))
        idx = customDist.rvs(size=1, random_state=random_state_)

        query_indicies = np.append(query_indicies, [idx])

    # 1. Dt
    # 2. Dt^2
    # stats.rv_distance


def _d_probability(g_x, query_indicies, d_latest=None):
    g_query_indicies = g_x[query_indicies]
    argmin, D = pairwise_distances_argmin_min(X=g_x, Y=g_query_indicies)
    if d_latest is not None:
        D = np.minimum(d_latest, D)
    D2 = np.square(D)
    D2_sum = np.sum(D2)
    D_probas = D2 / D2_sum
    return D_probas
