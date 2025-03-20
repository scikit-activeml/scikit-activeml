"""
Implementation of Active Learning with Cost Embedding (CostEmbeddingAL), which
is modified of:

https://github.com/ntucllab/libact/blob/master/libact.
Copyright (c) 2014, National Taiwan University All rights reserved.
"""

import warnings

import numpy as np
from joblib import Parallel, delayed
from sklearn import clone
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import euclidean_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVR
from sklearn.utils import check_array, check_symmetric

from ..base import SingleAnnotatorPoolQueryStrategy
from ..utils import (
    simple_batch,
    check_classifier_params,
    MISSING_LABEL,
    check_scalar,
    check_random_state,
    check_X_y,
    is_labeled,
    ExtLabelEncoder,
)


class CostEmbeddingAL(SingleAnnotatorPoolQueryStrategy):
    """Active Learning with Cost Embedding (ALCE)

    The Active Learning with Cost Embeddings (ALCE) [1]_ query strategy uses
    a cost-sensitive uncertainty measure based on a distance measure
    embedding space reflecting a given cost matrix.

    This implementation is based on libact [2]_.

    Parameters
    ----------
    classes: array-like of shape (n_classes,)
        List of all classes that can occur.
    base_regressor : sklearn.base.RegressorMixin, default=None
        An sklearn regression model implementing the methods `fit` and
        `predict`.
    cost_matrix: array-like of shape (n_classes, n_classes), default=None
        Cost matrix with `cost_matrix[i,j]` defining the cost of predicting
        class `j` for a sample with the actual class `i`.
    embed_dim : int, optional (default=None)
        If it is `None`, `embed_dim=n_classes` is used.
    mds_params : dict, default=None
        Parameters passed to `sklearn.manifold.MDS`.
    nn_params : dict, default=None
        Parameters passed to `sklearn.neighbors.NearestNeighbors`.
    missing_label : scalar or string or np.nan or None, default=np.nan
        Value to represent a missing label.
    random_state : int or np.random.RandomState, default=None
        Random state for annotator selection.

    References
    ----------
    .. [1] K.-H. Huang and H.-T. Lin. A Novel Uncertainty Sampling Algorithm
       for Cost-Sensitive Multiclass Active Learning. In IEEE Int. Conf. Data
       Min., pages 925–930, 2016.
    .. [2] Y.-Y. Yang, S.-C. Lee, Y.-A. Chung, T.-E. Wu, S.-A. Chen, and H.-T.
       Lin. libact: Pool-based Active Learning in Python. arXiv:1710.00379,
       2017.
    """

    def __init__(
        self,
        classes,
        base_regressor=None,
        cost_matrix=None,
        embed_dim=None,
        mds_params=None,
        nn_params=None,
        missing_label=MISSING_LABEL,
        random_state=None,
    ):
        super().__init__(
            missing_label=missing_label, random_state=random_state
        )
        self.classes = classes
        self.base_regressor = base_regressor
        self.cost_matrix = cost_matrix
        self.embed_dim = embed_dim
        self.missing_label = missing_label
        self.random_state = random_state
        self.mds_params = mds_params
        self.nn_params = nn_params

    def query(
        self,
        X,
        y,
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
        sample_weight: array-like of shape (n_samples,), default=None
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
        # Check standard parameters.
        (
            X,
            y,
            candidates,
            batch_size,
            return_utilities,
        ) = super()._validate_data(
            X=X,
            y=y,
            candidates=candidates,
            batch_size=batch_size,
            return_utilities=return_utilities,
            reset=True,
        )

        # Obtain candidates plus mapping.
        X_cand, mapping = self._transform_candidates(candidates, X, y)

        util_cand = _alce(
            X_cand,
            X,
            y,
            self.base_regressor,
            self.cost_matrix,
            self.classes,
            self.embed_dim,
            sample_weight,
            self.missing_label,
            self.random_state_,
            self.mds_params,
            self.nn_params,
        )

        if mapping is None:
            utilities = util_cand
        else:
            utilities = np.full(len(X), np.nan)
            utilities[mapping] = util_cand

        return simple_batch(
            utilities,
            self.random_state_,
            batch_size=batch_size,
            return_utilities=return_utilities,
        )


def _alce(
    X_cand,
    X,
    y,
    base_regressor,
    cost_matrix,
    classes,
    embed_dim,
    sample_weight,
    missing_label,
    random_state,
    mds_params,
    nn_params,
):
    """Compute the ALCE score for the candidate samples.

    Parameters
    ----------
    X_cand : array-like of shape (n_candidates, n_features)
        Unlabeled candidate samples.
    X : array-like of shape (n_samples, n_features)
        Complete data set.
    y : array-like of shape (n_samples)
        Labels of the data set.
    base_regressor : RegressorMixin
        Regressor used for the embedding.
    cost_matrix : array-like of shape (n_classes, n_classes), default=None
        Cost matrix with `cost_matrix[i,j]` defining the cost of predicting
        class `j` for a sample with the actual class `i`.
    classes : array-like of shape (n_classes,)
        List of all classes that can occur.
    sample_weight : array-like, shape (n_samples)
        Weights for uncertain annotators.
    missing_label : scalar or string or np.nan or None, default=np.nan
        Value to represent a missing label.
    random_state : int or np.random.RandomState, default=None
        Random state for annotator selection.
    base_regressor : sklearn.base.RegressorMixin, default=None
        An sklearn regression model implementing the methods `fit` and
        `predict`.
    embed_dim : int, optional (default=None)
        If it is `None`, `embed_dim=n_classes` is used.
    mds_params : dict, default=None
        Parameters passed to `sklearn.manifold.MDS`.
    nn_params : dict, default=None
        Parameters passed to `sklearn.neighbors.NearestNeighbors`.

    Returns
    -------
    utilities : np.ndarray of shape (n_candidates,)
        The utilities of all candidate samples.
    """
    # Check base regressor
    if base_regressor is None:
        base_regressor = SVR()
    if not isinstance(base_regressor, RegressorMixin):
        raise TypeError("'base_regressor' must be an sklearn regressor")
    check_classifier_params(classes, missing_label, cost_matrix)
    if cost_matrix is None:
        cost_matrix = 1 - np.eye(len(classes))

    if np.count_nonzero(cost_matrix) == 0:
        raise ValueError(
            "The cost matrix must contain at least one positive " "number."
        )

    # Check the given data
    X, y, X_cand, sample_weight, sample_weight_cand = check_X_y(
        X,
        y,
        X_cand,
        sample_weight,
        ensure_all_finite=False,
        missing_label=missing_label,
    )

    labeled = is_labeled(y, missing_label=missing_label)
    y = ExtLabelEncoder(classes, missing_label).fit_transform(y)
    X = X[labeled]
    y = y[labeled].astype(int)
    sample_weight = sample_weight[labeled]

    # If all samples are unlabeled, the strategy randomly selects a sample
    if len(X) == 0:
        warnings.warn(
            "There are no labeled samples. The strategy selects "
            "one random sample."
        )
        return np.ones(len(X_cand))

    # Check embedding dimension
    embed_dim = len(classes) if embed_dim is None else embed_dim
    check_scalar(embed_dim, "embed_dim", int, min_val=1)

    # Update mds parameters
    mds_params_default = {
        "metric": False,
        "n_components": embed_dim,
        "n_uq": len(classes),
        "max_iter": 300,
        "eps": 1e-6,
        "dissimilarity": "precomputed",
        "n_init": 8,
        "n_jobs": 1,
        "random_state": random_state,
    }
    if mds_params is not None:
        if type(mds_params) is not dict:
            raise TypeError("'mds_params' must be a dictionary or None")
        mds_params_default.update(mds_params)
    mds_params = mds_params_default

    # Update nearest neighbor parameters
    nn_params = {} if nn_params is None else nn_params
    if type(nn_params) is not dict:
        raise TypeError("'nn_params' must be a dictionary or None")

    regressors = [clone(base_regressor) for _ in range(embed_dim)]
    n_classes = len(classes)

    dissimilarities = np.zeros((2 * n_classes, 2 * n_classes))
    dissimilarities[:n_classes, n_classes:] = cost_matrix
    dissimilarities[n_classes:, :n_classes] = cost_matrix.T

    W = np.zeros((2 * n_classes, 2 * n_classes))
    W[:n_classes, n_classes:] = 1
    W[n_classes:, :n_classes] = 1

    mds = MDSP(**mds_params)
    embedding = mds.fit(dissimilarities).embedding_
    class_embed = embedding[:n_classes, :]

    nn = NearestNeighbors(n_neighbors=1, **nn_params)
    nn.fit(embedding[n_classes:, :])

    pred_embed = np.zeros((len(X_cand), embed_dim))
    for i in range(embed_dim):
        regressors[i].fit(X, class_embed[y, i], sample_weight)
        pred_embed[:, i] = regressors[i].predict(X_cand)

    dist, _ = nn.kneighbors(pred_embed)

    utilities = dist[:, 0]
    return utilities


"""
Implementation of Multi-dimensional Scaling Partial (MDSP), which
is a modification of:

https://github.com/ntucllab/libact/blob/master/libact/query_strategies/multiclass/mdsp.py
written by Kuan-Hao Huang (BSD license. Copyright (c) 2014, National Taiwan
University All rights reserved.)

and

https://github.com/scikit-learn/scikit-learn/blob/14031f6/sklearn/manifold/mds.py.
written by Nelle Varoquaux <nelle.varoquaux@gmail.com> (BSD license.
Copyright (c) 2007–2016 The scikit-learn developers. All rights reserved.).
"""


def _smacof_single_p(
    similarities,
    n_uq,
    metric=True,
    n_components=2,
    init=None,
    max_iter=300,
    verbose=0,
    eps=1e-3,
    random_state=None,
):
    """Computes multidimensional scaling using the SMACOF algorithm.

    Parameters
    ----------
    similarities : symmetric np.ndarray of shape (n * n,)
        Similarities between the points.
    n_uq : int
        Number of unique values.
    metric : boolean, default=True
        Compute metric or nonmetric SMACOF algorithm.
    n_components : int default=2
        Number of dimension in which to immerse the similarities overwritten
        if initial array is provided.
    init : None or np.ndarray, default=None
        - If `init=None`, randomly chooses the initial configuration.
        - if `init` is np.ndarray, initialize SMACOF algorithm with this array.
    max_iter : int, default=300
        Maximum number of iterations of the SMACOF algorithm for a single run.
    verbose : int, default=0
        Level of verbosity.
    eps : float, default=1e-6
        Relative tolerance w.r.t stress to declare converge.
    random_state : integer or np.RandomState, default=None
        The generator used to initialize the centers. If an integer is
        given, it fixes the seed. Defaults to the global numpy random
        number generator.

    Returns
    -------
    X : np.ndarray of (n_samples, n_components)
        Coordinates of the n_samples points in a n_components-space.
    _stress : float
        The final value of the stress (sum of squared distance of the
        disparities and the distances for all constrained points).
    it : int
        Number of iterations run.
    """
    similarities = check_symmetric(similarities, raise_exception=True)

    n_samples = similarities.shape[0]
    random_state = check_random_state(random_state)

    W = np.ones((n_samples, n_samples))
    W[:n_uq, :n_uq] = 0.0
    W[n_uq:, n_uq:] = 0.0

    V = -W
    V[np.arange(len(V)), np.arange(len(V))] = W.sum(axis=1)
    e = np.ones((n_samples, 1))

    Vp = (
        np.linalg.inv(V + np.dot(e, e.T) / n_samples)
        - np.dot(e, e.T) / n_samples
    )

    sim_flat = similarities.ravel()
    sim_flat_w = sim_flat[sim_flat != 0]
    if init is None:
        # Randomly choose initial configuration.
        X = random_state.rand(n_samples * n_components)
        X = X.reshape((n_samples, n_components))
    else:
        # Overrides the parameter `p`.
        n_components = init.shape[1]
        if n_samples != init.shape[0]:
            raise ValueError(
                "init matrix should be of shape (%d, %d)"
                % (n_samples, n_components)
            )
        X = init

    old_stress = None
    ir = IsotonicRegression()
    for it in range(max_iter):
        # Compute distance and monotonic regression.
        dis = euclidean_distances(X)

        if metric:
            disparities = similarities
        else:
            dis_flat = dis.ravel()
            # Similarities with 0 are considered as missing values.
            dis_flat_w = dis_flat[sim_flat != 0]

            # Compute the disparities using a monotonic regression.
            disparities_flat = ir.fit_transform(sim_flat_w, dis_flat_w)
            disparities = dis_flat.copy()
            disparities[sim_flat != 0] = disparities_flat
            disparities = disparities.reshape((n_samples, n_samples))
            disparities *= np.sqrt(
                (n_samples * (n_samples - 1) / 2) / (disparities**2).sum()
            )
            disparities[similarities == 0] = 0

        # Compute stress.
        _stress = (
            W.ravel() * ((dis.ravel() - disparities.ravel()) ** 2)
        ).sum()
        _stress /= 2

        # Update X using the Guttman transform.
        dis[dis == 0] = 1e-5
        ratio = disparities / dis
        _B = -W * ratio
        _B[np.arange(len(_B)), np.arange(len(_B))] += (W * ratio).sum(axis=1)

        X = np.dot(Vp, np.dot(_B, X))

        dis = np.sqrt((X**2).sum(axis=1)).sum()

        if verbose >= 2:
            print("it: %d, stress %s" % (it, _stress))
        if old_stress is not None:
            if (old_stress - _stress / dis) < eps:
                if verbose:
                    print(f"breaking at iteration {it} with stress {_stress}")
                break
        old_stress = _stress / dis

    return X, _stress, it + 1


def smacof_p(
    similarities,
    n_uq,
    metric=True,
    n_components=2,
    init=None,
    n_init=8,
    n_jobs=1,
    max_iter=300,
    verbose=0,
    eps=1e-3,
    random_state=None,
    return_n_iter=False,
):
    """Computes multidimensional scaling using SMACOF (Scaling by Majorizing a
    Complicated Function) algorithm.

    The SMACOF algorithm is a multidimensional scaling algorithm: it minimizes
    a objective function, the *stress*, using a majorization technique. The
    Stress Majorization, also known as the Guttman Transform, guarantees a
    monotone convergence of Stress, and is more powerful than traditional
    techniques such as gradient descent.

    The SMACOF algorithm for metric MDS can summarized by the following steps:

    1. Set an initial start configuration, randomly or not.
    2. Compute the stress
    3. Compute the Guttman Transform
    4. Iterate 2 and 3 until convergence.

    The nonmetric algorithm adds a monotonic regression steps before computing
    the stress.

    Parameters
    ----------
    similarities : symmetric np.ndarray of shape (n_samples, n_samples)
        Similarities between the points.
    n_uq : int
        Number of unique values.
    metric : boolean, default=True
        Compute metric or nonmetric SMACOF algorithm.
    n_components : int, default=2
        Number of dimension in which to immerse the similarities overridden if
        initial array is provided.
    init : None or ndarray of shape (n_samples, n_components), default=None
        - If `init=None`, randomly chooses the initial configuration.
        - if `init` is np.ndarray, initialize SMACOF algorithm with this array.
    n_init : int, default=8
        Number of times the smacof_p algorithm will be run with different
        initialisations. The final results will be the best output of the
        n_init consecutive runs in terms of stress.
    n_jobs : int, default=1
        The number of jobs to use for the computation. This works by breaking
        down the pairwise matrix into n_jobs even slices and computing them in
        parallel.

        - If -1 all CPUs are used.
        - If 1 is given, no parallel computing code is used at all, which is
        useful for debugging.
        - For `n_jobs` below -1, `(n_cpus + 1 + n_jobs)` are used. Thus, for
        `n_jobs=-2, all CPUs but one are used.
    max_iter : int, default=300
        Maximum number of iterations of the SMACOF algorithm for a single run.
    verbose : int, default=0
        Level of verbosity.
    eps : float, default=1e-6
        Relative tolerance w.r.t stress to declare converge.
    random_state : integer or numpy.RandomState, default=None
        The generator used to initialize the centers. If an integer is
        given, it fixes the seed. Defaults to the global numpy random
        number generator.
    return_n_iter : bool
        Whether or not to return the number of iterations.

    Returns
    -------
    X : np.ndarray of shape (n_samples, n_components)
        Coordinates of the n_samples points in a n_components-space.
    stress : float
        The final value of the stress (sum of squared distance of the
        disparities and the distances for all constrained points).
    n_iter : int
        The number of iterations corresponding to the best stress.
        Returned only if `return_n_iter` is set to True.

    References
    ----------
    .. [1] "Modern Multidimensional Scaling - Theory and Applications" Borg,
       I.; Groenen P. Springer Series in Statistics (1997)
    .. [2] "Nonmetric multidimensional scaling: a numerical method" Kruskal, J.
       Psychometrika, 29 (1964)
    .. [3] "Multidimensional scaling by optimizing goodness of fit to a
       nonmetric hypothesis" Kruskal, J. Psychometrika, 29, (1964)
    """

    similarities = check_array(similarities)
    random_state = check_random_state(random_state)

    if hasattr(init, "__array__"):
        init = np.asarray(init).copy()
        if not n_init == 1:
            warnings.warn(
                "Explicit initial positions passed: "
                "performing only one init of the MDS instead of %d" % n_init
            )
            n_init = 1

    best_pos, best_stress = None, None

    if n_jobs == 1:
        for it in range(n_init):
            pos, stress, n_iter_ = _smacof_single_p(
                similarities,
                n_uq,
                metric=metric,
                n_components=n_components,
                init=init,
                max_iter=max_iter,
                verbose=verbose,
                eps=eps,
                random_state=random_state,
            )
            if best_stress is None or stress < best_stress:
                best_stress = stress
                best_pos = pos.copy()
                best_iter = n_iter_
    else:
        seeds = random_state.randint(np.iinfo(np.int32).max, size=n_init)
        results = Parallel(n_jobs=n_jobs, verbose=max(verbose - 1, 0))(
            delayed(_smacof_single_p)(
                similarities,
                n_uq,
                metric=metric,
                n_components=n_components,
                init=init,
                max_iter=max_iter,
                verbose=verbose,
                eps=eps,
                random_state=seed,
            )
            for seed in seeds
        )
        positions, stress, n_iters = zip(*results)
        best = np.argmin(stress)
        best_stress = stress[best]
        best_pos = positions[best]
        best_iter = n_iters[best]

    if return_n_iter:
        return best_pos, best_stress, best_iter
    else:
        return best_pos, best_stress


class MDSP(BaseEstimator):
    """Multidimensional Scaling Partial (MDSP)

    Parameters
    ----------
    metric : boolean, default=True
        compute metric or nonmetric SMACOF (Scaling by Majorizing a
        Complicated Function) algorithm.
    n_uq : int, default=1
        Number of unique values.
    n_components : int, default=2
        Number of dimension in which to immerse the similarities
        overridden if initial array is provided.
    n_init : int, default=4
        Number of time the smacof_p algorithm will be run with different
        initialisation. The final results will be the best output of the
        n_init consecutive runs in terms of stress.
    max_iter : int, default=300
        Maximum number of iterations of the SMACOF algorithm for a single run.
    verbose : int, default=0
        Level of verbosity.
    eps : float, default=1e-6
        Relative tolerance w.r.t stress to declare converge.
    n_jobs : int, default=1
        The number of jobs to use for the computation. This works by breaking
        down the pairwise matrix into n_jobs even slices and computing them in
        parallel.

        - If -1 all CPUs are used.
        - If 1 is given, no parallel computing code is used at all, which is
        useful for debugging.
        - For `n_jobs` below -1, `(n_cpus + 1 + n_jobs)` are used. Thus, for
        `n_jobs=-2, all CPUs but one are used.
    random_state : integer or numpy.RandomState, default=None
        The generator used to initialize the centers. If an integer is
        given, it fixes the seed. Defaults to the global numpy random
        number generator.
    dissimilarity : string
        Which dissimilarity measure to use. Supported are 'euclidean' and
        'precomputed'.

    Attributes
    ----------
    embedding_ : array-like of shape (n_components, n_samples)
        Stores the position of the dataset in the embedding space
    stress_ : float
        The final value of the stress (sum of squared distance of the
        disparities and the distances for all constrained points).

    References
    ----------
    .. [1] "Modern Multidimensional Scaling - Theory and Applications" Borg,
       I.; Groenen P. Springer Series in Statistics (1997)
    .. [2] "Nonmetric multidimensional scaling: a numerical method" Kruskal, J.
       Psychometrika, 29 (1964)
    .. [3] "Multidimensional scaling by optimizing goodness of fit to a
       nonmetric hypothesis" Kruskal, J. Psychometrika, 29, (1964)
    """

    def __init__(
        self,
        n_components=2,
        n_uq=1,
        metric=True,
        n_init=4,
        max_iter=300,
        verbose=0,
        eps=1e-3,
        n_jobs=1,
        random_state=None,
        dissimilarity="euclidean",
    ):
        self.n_components = n_components
        self.n_uq = n_uq
        self.dissimilarity = dissimilarity
        self.metric = metric
        self.n_init = n_init
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(self, X, y=None, init=None):
        """Compute the position of the points in the embedding space.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features), or \
                (n_samples, n_samples) if dissimilarity='precomputed'
            Input data.
        init : None or ndarray of shape (n_samples,), default=None
            - If `None`, randomly chooses the initial configuration.
            - If `np.ndarray`, initialize the SMACOF algorithm with this array.
        """
        self.fit_transform(X, init=init)
        return self

    def fit_transform(self, X, y=None, init=None):
        """Fit the data from X, and returns the embedded coordinates.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features), or \
                        (n_samples, n_samples) if dissimilarity='precomputed'
            Input data.
        init : None or ndarray of shape (n_samples,), default=None
            - If `None`, randomly chooses the initial configuration.
            - If `np.ndarray`, initialize the SMACOF algorithm with this array.
        """
        X = check_array(X)
        if X.shape[0] == X.shape[1] and self.dissimilarity != "precomputed":
            warnings.warn(
                "The MDS API has changed. `fit` now constructs an"
                " dissimilarity matrix from data. To use a custom "
                "dissimilarity matrix, set "
                "`dissimilarity=precomputed`."
            )

        if self.dissimilarity == "precomputed":
            self.dissimilarity_matrix_ = X
        elif self.dissimilarity == "euclidean":
            self.dissimilarity_matrix_ = euclidean_distances(X)
        else:
            raise ValueError(
                "Proximity must be 'precomputed' or 'euclidean'."
                " Got %s instead" % str(self.dissimilarity)
            )

        self.embedding_, self.stress_, self.n_iter_ = smacof_p(
            self.dissimilarity_matrix_,
            self.n_uq,
            metric=self.metric,
            n_components=self.n_components,
            init=init,
            n_init=self.n_init,
            n_jobs=self.n_jobs,
            max_iter=self.max_iter,
            verbose=self.verbose,
            eps=self.eps,
            random_state=self.random_state,
            return_n_iter=True,
        )

        return self.embedding_
