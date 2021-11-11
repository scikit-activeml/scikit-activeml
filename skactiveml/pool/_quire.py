import numpy as np

from ..base import SingleAnnotPoolBasedQueryStrategy


class QUIRE(SingleAnnotPoolBasedQueryStrategy):
    """QUIRE

    Querying Informative and Representative Examples (QUIRE)
    Query the most informative and representative samples where the metrics
    measuring and combining are done using min-max approach. Note that, QUIRE is
    not a batch mode active learning algorithm, it will select only one instance
    for querying at each iteration. Also, it does not need a model to evaluate the
    unlabeled data.
    The implementation refers to the project: https://github.com/ntucllab/libact

    Parameters
    ----------
    data_set: base.DataSet
        Data set containing samples, class labels, and optionally confidences of annotator(s).
    lambda: float, optional (default=1.0)
        A regularization parameter used in the regularization learning
        framework.
    S: array-like, shape (n_samples, n_samples)
        Similarity matrix defining the similarities between all paris of available samples, e.g., S[i,j] describes
        the similarity between the samples x_i and x_j.
        Default similarity matrix is the unit matrix.
    random_state: numeric | np.random.RandomState
        Random state for annotator selection.

    Attributes
    ----------
    data_set_: base.DataSet
        Data set containing samples, class labels, and optionally confidences of annotator(s).
    lambda_: float, optional (default=1.0)
        A regularization parameter used in the regularization learning
        framework.
    S_: array-like, shape (n_samples, n_samples)
        Similarity matrix defining the similarities between all paris of available samples, e.g., S[i,j] describes
        the similarity between the samples x_i and x_j.
        Default similarity matrix is the unit matrix.
    random_state_: numeric | np.random.RandomState
        Random state for annotator selection.
    L: array-like, shape (n_samples, n_samples)
        Inverse of the matrix (S + lambda * I).


    References
    ----------
    [1] Yang, Y.-Y.; Lee, S.-C.; Chung, Y.-A.; Wu, T.-E.; Chen, S.-A.; and Lin, H.-T. 2017.
        libact: Pool-based active learning in python. Technical report, National Taiwan University.
        available as arXiv preprint https://arxiv.org/abs/1710.00379.
    [2] Huang, S.; Jin, R.; and Zhou, Z. 2014. Active learning by querying informative and representative examples.
        IEEE Transactions on Pattern Analysis and Machine Intelligence
        36(10):1936-1949
    """

    def __init__(self, lmbda=1, random_state=None):
        super().__init__(random_state=random_state)
        self.lmbda_ = lmbda

    def query(self, X_cand, *args, batch_size=1, return_utilities=False,
              **kwargs):
        """Determines which for which candidate samples labels are to be
        queried.

        Parameters
        ----------
        X_cand : array-like, shape (n_samples, n_features)
            Candidate samples from which the strategy can select.
        batch_size : int, optional (default=1)
            The number of samples to be selected in one AL cycle.
        return_utilities : bool, optional (default=False)
            If true, also return the utilities based on the query strategy.

        Returns
        -------
        query_indices : numpy.ndarray, shape (batch_size)
            The query_indices indicate for which candidate sample a label is
            to queried, e.g., `query_indices[0]` indicates the first selected
            sample.
        utilities : numpy.ndarray, shape (batch_size, n_samples)
            The utilities of all candidate samples after each selected
            sample of the batch, e.g., `utilities[0]` indicates the utilities
            used for selecting the first sample (with index `query_indices[0]`)
            of the batch.
        """
        pass


def quire_utilities(K, lmbda, a, l, y_l):
    L = np.linalg.inv(K + lmbda * np.eye(len(K)))

    # efficient computation of inv(Laa)
    K_al = K[np.ix_(a, l)]
    K_aa = K[np.ix_(a, a)]
    K_ll = K[np.ix_(l, l)]
    I_a = np.eye(len(a))
    I_l = np.eye(len(l))
    L_aa_inv = (lmbda * I_a + K_aa)
    L_aa_inv -= K_al @ np.linalg.inv(lmbda * I_l + K_ll) @ K_al.T

    # determine determinant of L
    det_Laa = 1 / np.linalg.det(L_aa_inv)
    utilities = np.zeros_like(a)
    for s_idx, s in enumerate(a):
        u = a[a != s]
        L_ss = L_aa_inv[s, s]
        L_uu = L_aa_inv[np.ix_(u, u)]
        L_us = L_aa_inv[np.ix_(u, s)]
        L_uu_inv = L_uu - (1 / L_ss) * (L_us @ L_us.T)
        L_sl = L[np.ix_(s, l)]
        L_su = L[np.ix_(s, u)]
        L_ul = L[np.ix_(u, l)]
        utilities[s_idx] -= L_ss - (det_Laa / L_ss)
        utilities[s_idx] -= 2 * np.abs((L_sl - L_su @ L_uu_inv @ L_ul) @ y_l)

    return utilities
