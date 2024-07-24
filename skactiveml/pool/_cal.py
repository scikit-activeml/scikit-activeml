"""
Module implementing `Cal`, which is a deep active learning strategy selecting
contrastive samples.
"""

import numpy as np

from sklearn.neighbors import NearestNeighbors
from sklearn.base import clone

from ..base import SingleAnnotatorPoolQueryStrategy
from ..utils import (
    MISSING_LABEL,
    is_labeled,
    simple_batch,
)


class Cal(SingleAnnotatorPoolQueryStrategy):
    """Contrastive Active Learning (Cal)

    This class implements the Contrastive Active Learning (Cal) query strategy
    [1], which  selects samples similar in the (classifier's learned) feature
    space, while the classifier outputs maximally different class-membership
    probabilities.

    Parameters
    ----------
    nearest_neighbors_dict : dict, default=None
        The parameters passed to the clustering algorithm `cluster_algo`,
        excluding the parameter for the number of clusters.
    clf_embedding_flag_name : str or None, default=None
        Name of the flag, which is passed to the `predict_proba` method for
        getting the (learned) sample representations. If
        `clf_embedding_flag_name=None`, the input samples `X` are used.
    missing_label : scalar or string or np.nan or None, default=np.nan
        Value to represent a missing label.
    random_state : None or int or np.random.RandomState, default=None
        The random state to use.

    References
    ----------
    [1] Margatina, Katerina, Giorgos Vernikos, Loïc Barrault, and Nikolaos
    Aletras. "Active Learning by Acquiring Contrastive Examples." In EMNLP,
    pp. 650-663. 2021.
    """

    def __init__(
        self,
        nearest_neighbors_dict=None,
        clf_embedding_flag_name=None,
        missing_label=MISSING_LABEL,
        random_state=None,
    ):
        super().__init__(
            missing_label=missing_label, random_state=random_state
        )
        self.nearest_neighbors_dict = nearest_neighbors_dict
        self.clf_embedding_flag_name = clf_embedding_flag_name

    def query(
        self,
        X,
        y,
        clf,
        fit_clf=True,
        sample_weight=None,
        candidates=None,
        batch_size=1,
        return_utilities=False,
        update=False,
    ):
        """Query the next samples to be labeled.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data set, usually complete, i.e. including the labeled and
            unlabeled samples.
        y : array-like of shape (n_samples,)
            Labels of the training data set (possibly including unlabeled ones
            indicated by self.missing_label).
        clf : skactiveml.base.SkactivemlClassifier
            Model implementing the methods `fit` and `predict_proba`.
        fit_clf : bool, default=True
            Defines whether the classifier should be fitted on `X`, `y`, and
            `sample_weight`.
        candidates : None or array-like of shape (n_candidates), dtype=int or
        array-like of shape (n_candidates, n_features), default=None
            If `candidates` is None, the unlabeled samples from (X, y)
            are considered as candidates.
            If `candidates` is of shape (n_candidates) and of type int,
            candidates is considered as a list of the indices of the samples in
            (X, y).
        batch_size : int, default=1
            The number of samples to be selected in one AL cycle.
        return_utilities : bool, default=False
            If True, also return the utilities based on the query strategy.
        update : bool, default=False
            This boolean flag determines whether the computed `delta_max_`
            and the `distances_` shall be updated in the `query`. For the first
            call of `query`, this parameter has no impact because both
            quantities are computed for the first time.

        Returns
        ----------
        query_indices : numpy.ndarray of shape (batch_size)
            The `query_indices` indicate for which candidate sample a label is
            to queried, e.g., `query_indices[0]` indicates the first selected
            sample.
            If `candidates` in None or of shape (n_candidates), the indexing
            refers to samples in X.
        utilities : numpy.ndarray of shape (batch_size, n_samples)
            The utilities of samples for selecting each sample of the batch.
            Here, utilities mean the out-degree of the candidate samples.
            If `candidates` is None or of shape (n_candidates), the indexing
            refers to samples in `X`.
        """
        # Check parameters.
        X, y, candidates, batch_size, return_utilities = self._validate_data(
            X, y, candidates, batch_size, return_utilities, reset=True
        )
        X_cand, mapping = self._transform_candidates(candidates, X, y)
        X_labeled = X[is_labeled(y, self.missing_label_)]
        if not (
            isinstance(self.nearest_neighbors_dict, dict)
            or self.nearest_neighbors_dict is None
        ):
            raise TypeError(
                "Pass a dictionary with corresponding parameter names and "
                "values according to the `init` function of "
                "`sklearn.neighbors.NearestNeighbors`."
            )
        nearest_neighbors_dict = (
            {}
            if self.nearest_neighbors_dict is None
            else self.nearest_neighbors_dict.copy()
        )

        if fit_clf:
            if sample_weight is None:
                clf = clone(clf).fit(X, y)
            else:
                clf = clone(clf).fit(X, y, sample_weight)

        if len(X_labeled) > 0:
            # Obtain classifier predictions and optionally learned feature
            # embeddings (cf. line 3 and 4 in [1]).
            if self.clf_embedding_flag_name is not None:
                P_labeled, X_labeled = clf.predict_proba(
                    X_labeled, **{self.clf_embedding_flag_name: True}
                )
                P_cand, X_cand = clf.predict_proba(
                    X_cand, **{self.clf_embedding_flag_name: True}
                )
            else:
                P_labeled = clf.predict_proba(X_labeled)
                P_cand = clf.predict_proba(X_cand)
                if isinstance(P_labeled, tuple):
                    P_labeled, X_labeled = P_labeled
                if isinstance(P_cand, tuple):
                    P_cand, X_cand = P_cand

            # Clip probabilities to avoid zeros.
            np.clip(P_labeled, a_min=1e-3, a_max=1, out=P_labeled)
            P_labeled /= P_labeled.sum(axis=1, keepdims=True)
            np.clip(P_cand, a_min=1e-3, a_max=1, out=P_cand)
            P_cand /= P_cand.sum(axis=1, keepdims=True)

            # Find nearest labeled samples of candidate samples
            # (cf. line 2 in [1]).
            nn = NearestNeighbors(**nearest_neighbors_dict).fit(X_labeled)
            max_n_neighbors = min(nn.n_neighbors, len(X_labeled))
            nn_indices = nn.kneighbors(
                X_cand, n_neighbors=max_n_neighbors, return_distance=False
            )

            # Compute KL divergences between class-membership probabilities of
            # candidates and their respective neighbors
            # (cf. line 5 and 6 in [1]).
            P_labeled = P_labeled[nn_indices]
            utilities_cand = P_labeled * np.log(P_labeled / P_cand[:, None, :])
            utilities_cand = utilities_cand.sum(axis=-1).mean(axis=-1)
        else:
            # Fallback to random sampling, if there are no labeled samples.
            utilities_cand = np.zeros(len(X_cand))

        if mapping is None:
            utilities = utilities_cand
        else:
            utilities = np.full(len(X), np.nan)
            utilities[mapping] = utilities_cand

        return simple_batch(
            utilities,
            self.random_state_,
            batch_size=batch_size,
            return_utilities=return_utilities,
        )
