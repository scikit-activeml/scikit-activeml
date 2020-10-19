import numpy as np
from sklearn import clone
from sklearn.manifold import MDS

from sklearn.utils import check_array, check_X_y, check_random_state

from skactiveml.base import PoolBasedQueryStrategy
from skactiveml.utils import rand_argmax


class ALCE(PoolBasedQueryStrategy):
    """Active Learning with Cost Embedding (ALCE)

    Parameters
    ----------
    clf: model to be trained
        Model implementing the methods 'fit' and and 'predict_proba'.
    C: array-like, shape (n_classes, n_classes), optional (default=None)
        Cost matrix with C[i,j] defining the cost of predicting class j for a
        sample with the actual class i.
        Only supported for least confident variant.
    target_dimension: int, optional (default=None)
        Target dimension of the embedding. If None, it is set to n_classes.
    random_state: numeric | np.random.RandomState, optional (default=None)
        Random state for annotator selection.

    Attributes
    ----------

    References
    ----------
    [1] Kuan-Hao, and Hsuan-Tien Lin. "A Novel Uncertainty Sampling Algorithm
        for Cost-sensitive Multiclass Active Learning", In Proceedings of the
        IEEE International Conference on Data Mining (ICDM), 2016
    """

    def __init__(self, clf, C, random_state=None, target_dimension=None,
                 **kwargs):
        super().__init__(random_state=random_state)
        self.clf = clf
        self.C = C
        self.random_state = random_state
        self.target_dimension = target_dimension

    def query(self, X_cand, X, y, return_utilities=False, **kwargs):
        """Query the next instance to be labeled.

        Parameters
        ----------

        Returns
        -------
        query_indices: np.ndarray, shape (1)
            The index of the queried instance.
        utilities: np.ndarray shape (1, n_candidates)
            The utilities of all instances in X_cand
            (only returned if return_utilities is True).
        """
        X, y = check_X_y(X, y, force_all_finite=False)
        X_cand = check_array(X_cand, force_all_finite=False)
        check_random_state(self.random_state)

        self.clf = clone(self.clf)
        self.clf.fit(X, y)

        if self.target_dimension is None:
            self.target_dimension = len(self.clf.classes)

        utilities = _get_utilities(self.clf, X_cand, X, y, self.C,
                                   self.random_state, self.target_dimension)
        best_indices = rand_argmax([utilities], axis=1,
                                   random_state=self.random_state)
        if return_utilities:
            return best_indices, np.array([utilities])
        else:
            return best_indices


def _get_utilities(clf, X_cand, X, y, C, random_state, target_dimension):
    clf = clone(clf)
    clf.fit(X, y)
    n_classes = len(clf.classes)

    # Embedding g
    dissimilarities = np.zeros((2 * n_classes, 2 * n_classes))
    dissimilarities[:n_classes, n_classes:] = C
    dissimilarities[n_classes:, :n_classes] = C.T

    W = np.zeros((2 * n_classes, 2 * n_classes))
    W[:n_classes, n_classes:] = 1
    W[n_classes:, :n_classes] = 1

    mds = MDS(n_components=target_dimension, metric=False,
              dissimilarity='precomputed', random_state=random_state)
    mds.fit(dissimilarities)
    embedding = mds.embedding_
    print()
    print(embedding.shape)
    print((n_classes, target_dimension))

    # Nearest neighbor function phi

    # labeled_indices = is_labeled(y, missing_label=clf.missing_label)
    # X_labeled = X[labeled_indices]
    # y_labeled = y[labeled_indices]

    return np.zeros(len(X_cand))
