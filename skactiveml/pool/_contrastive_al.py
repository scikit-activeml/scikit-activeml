"""
Module implementing `ContrastiveAL`, which is a deep active learning strategy
selecting contrastive samples.
"""

import numpy as np

from sklearn.neighbors import NearestNeighbors
from sklearn.base import clone

from ..base import SingleAnnotatorPoolQueryStrategy, SkactivemlClassifier
from ..utils import (
    MISSING_LABEL,
    is_labeled,
    simple_batch,
    check_scalar,
    check_type,
    check_equal_missing_label,
)


class ContrastiveAL(SingleAnnotatorPoolQueryStrategy):
    """Contrastive Active Learning (ContrastiveAL)

    This class implements the Contrastive Active Learning (ContrastiveAL) query
    strategy [1]_, which  selects samples similar in the (classifier's learned)
    feature space, while the classifier predicts maximally different
    class-membership probabilities.

    Parameters
    ----------
    nearest_neighbors_dict : dict, default=None
        The parameters passed to the nearest neighboring algorithm
        `sklearn.neighbors.NearestNeighbors`.
    clf_embedding_flag_name : str or None, default=None
        Name of the flag, which is passed to the `predict_proba` method for
        getting the (learned) sample representations.

        - If `clf_embedding_flag_name=None` and `predict_proba` returns
          only one output, the input samples `X` are used.
        - If `predict_proba` returns two outputs or `clf_embedding_name` is
          not `None`, `(proba, embeddings)` are expected as outputs.
    eps : float  > 0, default=1e-7
        Minimum probability threshold to compute log-probabilities.
    missing_label : scalar or string or np.nan or None, default=np.nan
        Value to represent a missing label.
    random_state : None or int or np.random.RandomState, default=None
        The random state to use.

    References
    ----------
    .. [1] K. Margatina, G. Vernikos, L. Barrault, and N. Aletras. Active
       Learning by Acquiring Contrastive Examples. In Conf. Empir. Methods Nat.
       Lang. Process., pages 650–663, 2021.
    """

    def __init__(
        self,
        nearest_neighbors_dict=None,
        clf_embedding_flag_name=None,
        eps=1e-7,
        missing_label=MISSING_LABEL,
        random_state=None,
    ):
        super().__init__(
            missing_label=missing_label, random_state=random_state
        )
        self.nearest_neighbors_dict = nearest_neighbors_dict
        self.clf_embedding_flag_name = clf_embedding_flag_name
        self.eps = eps

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
    ):
        """Determines for which candidate samples labels are to be queried.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data set, usually complete, i.e., including the labeled
            and unlabeled samples.
        y : array-like of shape (n_samples,)
            Labels of the training data set (possibly including unlabeled ones
            indicated by `self.missing_label`).
        clf : skactiveml.base.SkactivemlClassifier
            Classifier implementing the methods `fit` and `predict_proba`.
        fit_clf : bool, default=True
            Defines whether the classifier `clf` should be fitted on `X`, `y`,
            and `sample_weight`.
        sample_weight: array-like of shape (n_samples), default=None
            Weights of training samples in `X`.
        candidates : None or array-like of shape (n_candidates), dtype=int or \
                array-like of shape (n_candidates, n_features), default=None
            - If `candidates` is `None`, the unlabeled samples from
              `(X,y)` are considered as `candidates`.
            - If `candidates` is of shape `(n_candidates,)` and of type
              `int`, `candidates` is considered as the indices of the
              samples in `(X,y)`.
            - If `candidates` is of shape `(n_candidates, *)`, the
              candidate samples are directly given in `candidates` (not
              necessarily contained in `X`). This is not supported by all
              query strategies.
        batch_size : int, default=1
            The number of samples to be selected in one AL cycle.
        return_utilities : bool, default=False
            If `True`, also return the utilities based on the query strategy.

        Returns
        -------
        query_indices : numpy.ndarray of shape (batch_size,)
            The query indices indicate for which candidate sample a label is
            to be queried, e.g., `query_indices[0]` indicates the first
            selected sample.

            - If `candidates` is `None` or of shape
              `(n_candidates,)`, the indexing refers to the samples in
              `X`.
            - If `candidates` is of shape `(n_candidates, n_features)`,
              the indexing refers to the samples in `candidates`.
        utilities : numpy.ndarray of shape (batch_size, n_samples) or \
                numpy.ndarray of shape (batch_size, n_candidates)
            The utilities of samples after each selected sample of the batch,
            e.g., `utilities[0]` indicates the utilities used for selecting
            the first sample (with index `query_indices[0]`) of the batch.
            Utilities for labeled samples will be set to np.nan.

            - If `candidates` is `None` or of shape
              `(n_candidates,)`, the indexing refers to the samples in
              `X`.
            - If `candidates` is of shape `(n_candidates, n_features)`,
              the indexing refers to the samples in `candidates`.
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
        check_scalar(
            self.eps,
            "eps",
            min_val=0,
            max_val=0.1,
            target_type=(float, int),
            min_inclusive=False,
        )
        check_type(clf, "clf", SkactivemlClassifier)
        check_equal_missing_label(clf.missing_label, self.missing_label_)
        check_scalar(fit_clf, "fit_clf", bool)

        if fit_clf:
            if sample_weight is None:
                clf = clone(clf).fit(X, y)
            else:
                clf = clone(clf).fit(X, y, sample_weight)

        if len(X_labeled) > 0:
            # Obtain classifier predictions and optionally learned feature
            # embeddings (cf. line 3 and 4 in [1]).
            predict_proba_kwargs = {}
            if self.clf_embedding_flag_name is not None:
                predict_proba_kwargs = {self.clf_embedding_flag_name: True}
            P_labeled = clf.predict_proba(X_labeled, **predict_proba_kwargs)
            P_cand = clf.predict_proba(X_cand, **predict_proba_kwargs)
            if isinstance(P_labeled, tuple):
                P_labeled, X_labeled = P_labeled
            if isinstance(P_cand, tuple):
                P_cand, X_cand = P_cand

            # Clip probabilities to avoid zeros.
            np.clip(P_labeled, a_min=self.eps, a_max=1, out=P_labeled)
            P_labeled /= P_labeled.sum(axis=1, keepdims=True)
            np.clip(P_cand, a_min=self.eps, a_max=1, out=P_cand)
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
