__all__ = ["MappingError"]


class MappingError(Exception):
    """
    Exception class to raise if a strategy needs a mapping between samples and
    candidates which is not available.
    """
