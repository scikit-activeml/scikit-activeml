import numpy as np
from sklearn.base import clone
from sklearn.metrics import pairwise_distances
from ..base import SingleAnnotatorPoolQueryStrategy, SkactivemlClassifier
from ..utils import MISSING_LABEL, check_type, ExtLabelEncoder

class XPAL(SingleAnnotatorPoolQueryStrategy):
    """
    XPAL: eXpected Performance-based Active Learning.

    Parameters
    ----------
    prior : float, default=0.001
        Prior probability for each class.
    m_max : int, default=1
        Maximum sequence length for label sequences.
    kernel : str, default='rbf'
        Kernel type to be used ('rbf', 'linear', 'poly', 'sigmoid').
    kernel_params : dict, default=None
        Parameters for the specified kernel.
    missing_label : scalar, default=MISSING_LABEL
        Value to represent a missing label.
    random_state : int, RandomState instance or None, default=None
        Random state for reproducibility.
    """

    def __init__(self, prior=0.001, m_max=1, kernel='rbf', kernel_params=None, missing_label=MISSING_LABEL, random_state=None):
        super().__init__(missing_label=missing_label, random_state=random_state)
        self.prior = prior
        self.m_max = m_max
        self.kernel = kernel
        self.kernel_params = kernel_params

    def query(self, X, y, clf, fit_clf=True, sample_weight=None, candidates=None, batch_size=1, return_utilities=False):
        """
        Query the next batch of instances to be labeled.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The feature matrix.
        y : array-like, shape (n_samples,)
            The label vector.
        clf : SkactivemlClassifier
            The classifier to be used.
        fit_clf : bool, default=True
            Whether to fit the classifier.
        sample_weight : array-like, shape (n_samples,), default=None
            Sample weights.
        candidates : array-like, shape (n_candidates,), default=None
            Indices of candidate samples.
        batch_size : int, default=1
            Number of samples to query.
        return_utilities : bool, default=False
            Whether to return the utilities.

        Returns
        -------
        query_indices : array-like, shape (batch_size,)
            Indices of the queried samples.
        full_utilities : array-like, shape (n_samples,), optional
            Utilities for all samples, if return_utilities is True.
        """
        X, y, candidates, batch_size, return_utilities = self._validate_data(X, y, candidates, batch_size, return_utilities, reset=True)
        X_cand, mapping = self._transform_candidates(candidates, X, y)
        check_type(clf, 'clf', SkactivemlClassifier)
        if fit_clf:
            clf = clone(clf).fit(X, y, sample_weight)
        le = ExtLabelEncoder(classes=clf.classes_, missing_label=self.missing_label_)
        y_enc = le.fit_transform(y)
        utilities = self._xpal_score(X, y_enc, X_cand, clf)
        if mapping is None:
            labeled_mask = ~np.isnan(y) if np.isnan(self.missing_label_) else (y != self.missing_label_)
            utilities[labeled_mask] = np.min(utilities) - 1 if len(utilities) > 0 else -np.inf
        else:
            labeled_mask = (~np.isnan(y[mapping]) if np.isnan(self.missing_label_) else (y[mapping] != self.missing_label_))
            utilities[labeled_mask] = np.min(utilities) - 1 if len(utilities) > 0 else -np.inf
        sorted_indices = np.argsort(utilities)[::-1]
        query_indices = mapping[sorted_indices[:batch_size]] if mapping is not None else sorted_indices[:batch_size]
        if return_utilities:
            if mapping is None:
                full_utilities = utilities
            else:
                full_utilities = np.full(len(X), fill_value=np.min(utilities) - 1 if len(utilities) > 0 else -np.inf)
                full_utilities[mapping] = utilities
            return query_indices, full_utilities
        else:
            return query_indices

    def _xpal_score(self, X, y, X_cand, clf):
        """
        Compute the XPAL score for each candidate sample.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The feature matrix.
        y : array-like, shape (n_samples,)
            The label vector.
        X_cand : array-like, shape (n_candidates, n_features)
            The candidate samples.
        clf : SkactivemlClassifier
            The classifier to be used.

        Returns
        -------
        utilities : array-like, shape (n_candidates,)
            The utilities for each candidate sample.
        """
        n_classes = len(clf.classes_)
        prior = np.full(n_classes, self.prior)
        k_L = self._compute_kernel_frequency(X, y, clf)
        current_risk = self._estimate_risk(X, k_L, prior)
        if len(X_cand) == 0:
            return np.array([])
        K = self._kernel_similarity(X_cand, X, clf)
        utilities = np.zeros(len(X_cand))
        for m in range(1, self.m_max + 1):
            for y_sequence in self._generate_label_sequences(n_classes, m):
                y_sequence_expanded = y_sequence[:, np.newaxis, np.newaxis]
                k_L_expanded = k_L[:, np.newaxis, :]
                K_expanded = K[np.newaxis, :, :]
                try:
                    k_L_plus = k_L_expanded + (K_expanded * y_sequence_expanded)
                except ValueError:
                    continue
                risk = self._estimate_risk(X, k_L_plus, prior)
                utilities += (current_risk - risk).sum(axis=0)
        return utilities

    def _compute_kernel_frequency(self, X, y, clf):
        """
        Compute the kernel frequency for each class.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The feature matrix.
        y : array-like, shape (n_samples,)
            The label vector.
        clf : SkactivemlClassifier
            The classifier to be used.

        Returns
        -------
        k_L : array-like, shape (n_classes, n_samples)
            The kernel frequency for each class.
        """
        n_classes = len(clf.classes_)
        k_L = np.zeros((n_classes, len(X)))
        for c in range(n_classes):
            mask = y == c
            if np.any(mask):
                similarity = self._kernel_similarity(X[mask], X, clf)
                k_L[c] = similarity.sum(axis=0)
        return k_L

    def _kernel_similarity(self, X_subset, X, clf):
        """
        Compute the kernel similarity between subsets of samples.

        Parameters
        ----------
        X_subset : array-like, shape (n_subset_samples, n_features)
            The subset of samples.
        X : array-like, shape (n_samples, n_features)
            The full set of samples.
        clf : SkactivemlClassifier
            The classifier to be used.

        Returns
        -------
        similarities : array-like, shape (n_subset_samples, n_samples)
            The kernel similarity matrix.
        """
        if len(X_subset) == 0 or len(X) == 0:
            return np.array([])
        kernel_params = self.kernel_params or {}
        if self.kernel == 'rbf':
            gamma = kernel_params.get('gamma', 1.0 / X.shape[1])
            dists = pairwise_distances(X_subset, X, metric='sqeuclidean')
            similarities = np.exp(-gamma * dists)
        elif self.kernel == 'linear':
            similarities = np.dot(X_subset, X.T)
        elif self.kernel == 'poly':
            degree = kernel_params.get('degree', 3)
            coef0 = kernel_params.get('coef0', 1)
            gamma = kernel_params.get('gamma', 1)
            similarities = (gamma * np.dot(X_subset, X.T) + coef0) ** degree
        elif self.kernel == 'sigmoid':
            coef0 = kernel_params.get('coef0', 0)
            gamma = kernel_params.get('gamma', 1)
            similarities = np.tanh(gamma * np.dot(X_subset, X.T) + coef0)
        else:
            raise ValueError(f"Unsupported kernel: {self.kernel}")
        return similarities

    def _estimate_risk(self, X, k_L, prior):
        """
        Estimate the risk using the zero-one loss function.
    
        Parameters:
        X : array-like, shape (n_samples, n_features)
            The input samples.
        k_L : array-like, shape (n_classes, n_samples)
            The kernel frequency matrix.
        prior : array-like, shape (n_classes,)
            The prior probabilities for each class.
    
        Returns:
        risk : float
            The estimated risk.
        """
        prior = prior[:, np.newaxis, np.newaxis]
        p_y = (k_L + prior) / (np.sum(k_L, axis=0, keepdims=True) + np.sum(prior))
        p_y = np.clip(p_y, 1e-10, 1 - 1e-10)
        zero_one_loss = 1 - np.max(p_y, axis=0)
        return zero_one_loss.mean()


    def _generate_label_sequences(self, n_classes, m):
        """
        Generate all possible label sequences of length m.

        Parameters
        ----------
        n_classes : int
            The number of classes.
        m : int
            The length of the label sequences.

        Returns
        -------
        sequences : array-like, shape (n_sequences, m)
            All possible label sequences.
        """
        return np.array(np.meshgrid(*[range(n_classes) for _ in range(m)])).T.reshape(-1, m)

    def _compute_label_sequence_probability(self, y_sequence, k_L, prior):
        """
        Compute the probability of a label sequence.

        Parameters
        ----------
        y_sequence : array-like, shape (m,)
            The label sequence.
        k_L : array-like, shape (n_classes, n_samples)
            The kernel frequency for each class.
        prior : array-like, shape (n_classes,)
            The prior probabilities for each class.

        Returns
        -------
        probability : float
            The probability of the label sequence.
        """
        p_y = (np.sum(k_L, axis=1) + prior) / (np.sum(k_L) + np.sum(prior))
        return np.prod([p_y[y] for y in y_sequence])