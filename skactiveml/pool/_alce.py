import numpy as np
from sklearn import clone
from sklearn.base import RegressorMixin
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_array, check_random_state

from skactiveml.base import SingleAnnotPoolBasedQueryStrategy
from skactiveml.pool._mdsp import MDSP
from skactiveml.utils import rand_argmax, MISSING_LABEL
from skactiveml.utils import check_cost_matrix, check_missing_label


class ALCE(SingleAnnotPoolBasedQueryStrategy):
    """Active Learning with Cost Embedding (ALCE)

    Cost sensitive multi-class algorithm.
    Assume each class has at least one sample in the labeled pool.

    Parameters
    ----------
    base_regressor : sklearn regressor
    C: array-like, shape (n_classes, n_classes), optional (default=None)
        Cost matrix with C[i,j] defining the cost of predicting class j for a
        sample with the actual class i. Only supported for least confident
        variant.
    random_state: numeric | np.random.RandomState, optional (default=None)
        Random state for annotator selection.
    missing_label: str | numeric, optional (default=MISSING_LABEL)
        Specifies the symbol that represents a missing label
    embed_dim : int, optional (default=None)
        If is None, embed_dim = n_classes
    mds_params : dict, optional
        http://scikit-learn.org/stable/modules/generated/sklearn.manifold.MDS.html
    nn_params : dict, optional
        http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html

    Attributes
    ----------

    References
    ----------
    [1] Kuan-Hao, and Hsuan-Tien Lin. "A Novel Uncertainty Sampling Algorithm
        for Cost-sensitive Multiclass Active Learning", In Proceedings of the
        IEEE International Conference on Data Mining (ICDM), 2016
    """

    def __init__(self,
                 base_regressor,
                 C,
                 embed_dim=None,
                 missing_label=MISSING_LABEL,
                 classes=None,
                 mds_params=None,
                 nn_params=None,
                 random_state=None):
        super().__init__(random_state=random_state)
        self.base_regressor = base_regressor
        self.C = C
        self.embed_dim = embed_dim
        self.missing_label = missing_label
        self.classes = classes
        self.random_state = random_state
        self.mds_params = mds_params
        self.nn_params = nn_params

    def query(self, X_cand, X, y, batch_size=1, return_utilities=False,
              **kwargs):
        """Query the next instance to be labeled.

        Parameters
        ----------
        X_cand: array-like, shape (n_candidates, n_features)
            Unlabeled candidate samples
        X: array-like, shape (n_samples, n_features)
            Complete data set
        y: array-like, shape (n_samples)
            Labels of the data set
        batch_size: int, optional (default=1)
            The number of instances to be selected.
        return_utilities: bool, optional (default=False)
            If True, the utilities are additionally returned.

        Returns
        -------
        query_indices: np.ndarray, shape (batch_size)
            The index of the queried instance.
        utilities: np.ndarray, shape (batch_size, n_candidates)
            The utilities of all instances in X_cand
            (only returned if return_utilities is True).
        """
        X_cand = check_array(X_cand, force_all_finite=False)
        X = check_array(X, force_all_finite=False)
        y = check_array(y, force_all_finite=False, ensure_2d=False)

        if not isinstance(self.base_regressor, RegressorMixin):
            raise TypeError("'base_regressor' must be an sklearn regressor")
        if self.classes is None:
            self.classes = np.arange(len(self.C))
        check_cost_matrix(self.C, len(self.classes))
        if self.embed_dim is None:
            self.embed_dim = len(self.classes)
        if not float(self.embed_dim).is_integer():
            raise TypeError("'embed_dim' must be an integer.")
        if self.embed_dim < 1:
            raise ValueError("'embed_dim' must be strictly positive.")
        check_missing_label(self.missing_label)
        self.random_state = check_random_state(self.random_state)
        if not set(y) <= set(self.classes):
            raise ValueError("y has labels that are not contained in "
                             "'classes'")

        mds_params = {
            'metric': False,
            'n_components': self.embed_dim,
            'n_uq': len(self.classes),
            'max_iter': 300,
            'eps': 1e-6,
            'dissimilarity': "precomputed",
            'n_init': 8,
            'n_jobs': 1,
            'random_state': self.random_state
        }
        if self.mds_params is not None:
            mds_params.update(self.mds_params)
        self.mds_params = mds_params
        if self.nn_params is None:
            self.nn_params = {}

        regressors = [
            clone(self.base_regressor) for _ in range(self.embed_dim)
        ]
        n_classes = len(self.classes)

        dissimilarities = np.zeros((2 * n_classes, 2 * n_classes))
        dissimilarities[:n_classes, n_classes:] = self.C
        dissimilarities[n_classes:, :n_classes] = self.C.T

        W = np.zeros((2 * n_classes, 2 * n_classes))
        W[:n_classes, n_classes:] = 1
        W[n_classes:, :n_classes] = 1

        mds = MDSP(**self.mds_params)
        embedding = mds.fit(dissimilarities).embedding_
        class_embed = embedding[:n_classes, :]

        nn = NearestNeighbors(n_neighbors=1, **self.nn_params)
        nn.fit(embedding[n_classes:, :])

        pred_embed = np.zeros((len(X_cand), self.embed_dim))
        for i in range(self.embed_dim):
            regressors[i].fit(X, class_embed[y, i])
            pred_embed[:, i] = regressors[i].predict(X_cand)

        dist, _ = nn.kneighbors(pred_embed)

        utilities = dist[:, 0]
        query_indices = rand_argmax([utilities], self.random_state, axis=1)
        if return_utilities:
            return query_indices, np.array([utilities])
        else:
            return query_indices
