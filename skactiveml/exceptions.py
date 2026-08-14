__all__ = ["MappingError"]


class MappingError(Exception):
    """Exception class to raise if a strategy needs a mapping between samples
    and candidates which is not available.
    """


class _ExhaustedCandidatePool(Exception):
    """Signals a query that has no candidate left to select from.

    The shared pool validation raises this signal, and the guard that
    :class:`skactiveml.base.PoolQueryStrategy` puts around every `query`
    implementation answers it with the carried result. It therefore never
    leaves `query` and is not part of the public interface.

    The carried result is shaped for the call that raised the signal, so
    `_validate_data` must only ever be reached from the `query` of the same
    strategy. A strategy validating on behalf of another one would have its own
    guard answer with a result shaped for the other strategy's candidates.

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
