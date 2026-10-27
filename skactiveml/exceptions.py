__all__ = ["MappingError"]


class MappingError(Exception):
    """Exception class to raise if a strategy needs a mapping between samples
    and candidates which is not available.
    """


class _ExhaustedCandidatePool(Exception):
    """Signals a query that has no candidate left to select from.

    The guard that :class:`skactiveml.base.PoolQueryStrategy` puts around
    every `query` answers this signal with the carried result. The result
    is shaped for the raising call, so a strategy must not call the
    `_validate_data` of another strategy.

    Parameters
    ----------
    result : numpy.ndarray or tuple of numpy.ndarray
        The empty acquisition result `query` is to return, already shaped
        according to the `return_utilities` of the aborted call.
    """

    def __init__(self, result):
        super().__init__(
            "The candidate pool is exhausted. This signal is internal to the "
            "pool query boundary and must not escape `query`."
        )
        self.result = result

    def __reduce__(self):
        # `args` carries the message rather than the result, so the default
        # reduction would rebuild this signal from its own message.
        return type(self), (self.result,)
