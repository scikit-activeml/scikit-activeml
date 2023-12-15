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
    """
    Badge query strategy

    Parameters
    ----------
    missing_label
    random_state
    """
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
            return_embeddings=False,
    ):
        """
        Query strategy for BADGE

        Parameters
        ----------
        X
        y
        clf
        candidates
        batch_size
        return_utilities
        return_embeddings

        Returns
        -------

        """
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
            X_unlbld = X_cand
            unlbld_mapping = mapping
        elif mapping is not None:
            unlbld_mapping = unlabeled_indices(y[mapping], missing_label=self.missing_label)
            X_unlbld = X_cand[unlbld_mapping]
            unlbld_mapping = mapping[unlbld_mapping]
        else:
            X_unlbld = X_cand
            unlbld_mapping = np.arange(len(X_cand))

        # gradient embedding, aka predict class membership probabilities
        probas = clf.predict_proba(X_unlbld)
        print(probas)
        p_max = np.max(probas, axis=1).reshape(-1, 1)  # gaile
        g_x = (p_max - 1) * X_unlbld

        # init the utilities
        if mapping is not None:
            utilities = np.full(shape=(batch_size, X.shape[0]), fill_value=np.nan)
        else:
            utilities = np.full(shape=(batch_size, X_cand.shape[0]), fill_value=np.nan)

        # sampling with kmeans++
        query_indicies = np.array([], dtype=int)
        D_p = []
        for i in range(batch_size):
            if i == 0:
                d_probas, is_randomized = _d_probability(g_x, [])
            else:
                d_probas, is_randomized = _d_probability(g_x, [idx_in_unlbld], D_p[i - 1])
            D_p.append(d_probas)

            utilities[i, unlbld_mapping] = d_probas
            utilities[i, query_indicies] = np.nan

            if is_randomized:
                idx_in_unlbld = self.random_state_.choice(len(g_x))
            else:
                customDist = stats.rv_discrete(name="customDist", values=(np.arange(len(d_probas)), d_probas))  # gaile
                idx_in_unlbld_array = customDist.rvs(size=1, random_state=self.random_state_)
                idx_in_unlbld = idx_in_unlbld_array[0]

            idx = unlbld_mapping[idx_in_unlbld]
            query_indicies = np.append(query_indicies, [idx])

        if return_utilities:
            return query_indicies, utilities
        else:
            return query_indicies


def _d_probability(g_x, query_indicies, d_latest=None):
    """

    Parameters
    ----------
    g_x
    query_indicies
    d_latest

    Returns
    -------
    probability:
    is_randomized:
    """
    if len(query_indicies) == 0:
        return np.ones(shape=len(g_x)), True
    g_query_indicies = g_x[query_indicies]
    _, D = pairwise_distances_argmin_min(X=g_x, Y=g_query_indicies)
    if d_latest is not None:
        print("d_latest:", d_latest)
        D = np.minimum(d_latest, D)
    D2 = np.square(D)
    print("D2 ", D2)
    D2_sum = np.sum(D2)
    # for the case that clf only know a label aka class
    if D2_sum == 0:
        return np.ones(shape=len(g_x)), True
    D_probas = D2 / D2_sum
    return D_probas, False
