"""
Module implementing evidential uncertainty sampling
Hoarau, A., Lemaire, V., Le Gall, Y. et al. 
Evidential uncertainty sampling strategies for active learning. 
Machine Learning 113 (2024). 
https://doi.org/10.1007/s10994-024-06567-2
"""

# Authors: Arthur Hoarau <arthur.hoarau@outlook.fr>

import numpy as np
import math

from ..base import SingleAnnotatorPoolQueryStrategy
from ..classifier import EKNN, SklearnClassifier
from ..utils import (
    MISSING_LABEL,
    simple_batch,
    is_labeled,
    check_type,
    check_equal_missing_label,
)

class EvidentialUncertaintySampling(SingleAnnotatorPoolQueryStrategy):
    """Evidential Uncertainty Sampling.

    This class implement Evidential Uncertainty Sampling, i.e.[1]

    Parameters
    ----------
    method : string, default='least_confident'
        The method to calculate the uncertainty, entropy, least_confident,
        margin_sampling, and expected_average_precision  are possible.
    missing_label : scalar or string or np.nan or None, default=np.nan
        Value to represent a missing label.
    random_state : int or np.random.RandomState
        The random state to use.

    References
    ----------
    [1] Hoarau, A., Lemaire, V., Le Gall, Y. et al. 
        Evidential uncertainty sampling strategies for active learning. 
        Machine Learning 113 (2024). 
        https://doi.org/10.1007/s10994-024-06567-2
    """

    def __init__(
        self,
        missing_label=MISSING_LABEL,
        random_state=None,
    ):
        if random_state is not None:
            np.random.seed(random_state)
        super().__init__(
            missing_label=missing_label, random_state=random_state
        )

    def query(
        self,
        X,
        y,
        clf,
        fit_clf=True,
        candidates=None,
        batch_size=1,
        return_utilities=False,
    ):
        """Determines for which candidate samples labels are to be queried.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data set, usually complete, i.e. including the labeled and
            unlabeled samples.
        y : array-like of shape (n_samples)
            Labels of the training data set (possibly including unlabeled ones
            indicated by self.MISSING_LABEL.
        clf : skactiveml.base.SkactivemlClassifier
            Model implementing the methods `fit` and `predict_proba`.
        fit_clf : bool, optional (default=True)
            Defines whether the classifier should be fitted on `X`, `y`, and
            `sample_weight`.
        candidates : None or array-like of shape (n_candidates), dtype=int or
            array-like of shape (n_candidates, n_features),
            optional (default=None)
            If candidates is None, the unlabeled samples from (X,y) are
            considered as candidates.
            If candidates is of shape (n_candidates) and of type int,
            candidates is considered as the indices of the samples in (X,y).
            If candidates is of shape (n_candidates, n_features), the
            candidates are directly given in candidates (not necessarily
            contained in X). This is not supported by all query strategies.
        batch_size : int, default=1
            The number of samples to be selected in one AL cycle.
        return_utilities : bool, default=False
            If true, also return the utilities based on the query strategy.

        Returns
        -------
        query_indices : numpy.ndarray of shape (batch_size)
            The query_indices indicate for which candidate sample a label is
            to queried, e.g., `query_indices[0]` indicates the first selected
            sample.
            If candidates is None or of shape (n_candidates), the indexing
            refers to samples in X.
            If candidates is of shape (n_candidates, n_features), the indexing
            refers to samples in candidates.
        utilities : numpy.ndarray of shape (batch_size, n_samples) or
            numpy.ndarray of shape (batch_size, n_candidates)
            The utilities of samples after each selected sample of the batch,
            e.g., `utilities[0]` indicates the utilities used for selecting
            the first sample (with index `query_indices[0]`) of the batch.
            Utilities for labeled samples will be set to np.nan.
            If candidates is None or of shape (n_candidates), the indexing
            refers to samples in X.
            If candidates is of shape (n_candidates, n_features), the indexing
            refers to samples in candidates.
        """
        # Validate input parameters.
        X, y, candidates, batch_size, return_utilities = self._validate_data(
            X, y, candidates, batch_size, return_utilities, reset=True
        )

        X_cand, mapping = self._transform_candidates(candidates, X, y)

        # Validate classifier type.
        check_type(clf, "clf", SklearnClassifier)
        check_type(clf.estimator, "estimator", EKNN)
        check_equal_missing_label(clf.missing_label, self.missing_label_)

        # Validate classifier type.
        check_type(fit_clf, "fit_clf", bool)

        # Fit the classifier.
        if fit_clf:
            is_lbld = is_labeled(y, missing_label="nan")
            X_lbld = X[is_lbld]
            y_lbld = y[is_lbld]

            # If not enough neighbors, random sampling
            if clf.estimator.n_neighbors > X_lbld.shape[0]:
                 fit_clf = False
            else:
                clf.estimator.fit(X_lbld, y_lbld)

        # Evidential uncertainty sampling
        if fit_clf:
            _, bbas = clf.estimator.predict(X_cand, return_bba=True)
            
            # Choose the method and calculate corresponding utilities.
            utilities_cand = evidential_uncertainty_scores(
                        bbas=bbas,
            )
        # Random sampling (not enough neighbors)
        else:
            utilities_cand = np.random.random(size=X_cand.shape[0])

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


def evidential_uncertainty_scores(bbas):
    """Computes evidential uncertainty scores [1]. 

    Parameters
    ----------
    bbas : array-like, shape (n_samples, n_classes**2)
        Basic Belief Assignments

    References
    ----------
    [1] Hoarau, A., Lemaire, V., Le Gall, Y. et al. 
        Evidential uncertainty sampling strategies for active learning. 
        Machine Learning 113 (2024). 
        https://doi.org/10.1007/s10994-024-06567-2
    """

    card = np.zeros(bbas.shape[1])
    for i in range(1, bbas.shape[1]):
        card[i] = math.log2(bin(i).count("1"))

    pign_prob = np.zeros((bbas.shape[0], bbas.shape[1]))
    for k in range(bbas.shape[0]): 
            betp_atoms = EKNN.decisionDST(bbas[k].T, 4, return_prob=True)[0]
            for i in range(1, bbas.shape[1]):
                for j in range(betp_atoms.shape[0]):
                        if ((2**j) & i) == (2**j):
                            pign_prob[k][i] += betp_atoms[j]
            
                pign_prob[k][i] = math.log2(pign_prob[k][i])

    lbda = 0.2

    return np.sum((lbda * bbas * card) + (-(1-lbda) * bbas * pign_prob), axis=1)
