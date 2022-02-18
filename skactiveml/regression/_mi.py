from skactiveml.base import SingleAnnotPoolBasedQueryStrategy


class MutualInformation(SingleAnnotPoolBasedQueryStrategy):
    """Regression based Mutual Information Maximization

    This class implements an Regression adaption of Query by Committee. It
    tries to estimate the model variance by a Committee of estimators.

    Parameters
    ----------
    random_state: numeric | np.random.RandomState, optional
        Random state for candidate selection.
    k_bootstraps: int, optional (default=1)
        The number of members in a committee.

    References
    ----------
    [1] Burbidge, Robert and Rowland, Jem J and King, Ross D. Active learning
        for regression based on query by committee. International conference on
        intelligent data engineering and automated learning, pages 209--218,
        2007.

    """

    def __init__(self, random_state=None):
        super().__init__(random_state=random_state)
        self.k_bootstraps = k_bootstraps

    def query(self, X_cand, batch_size=1, return_utilities=False, **kwargs):
        pass