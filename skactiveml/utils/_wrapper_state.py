"""Fitted state of estimators wrapping another estimator."""

from sklearn.exceptions import NotFittedError

# Marks an in-progress lazy resolution in a wrapper's `__dict__`. Without a
# trailing underscore, `check_is_fitted` ignores it.
_RESOLVING_OWN_FITTED_STATE = "_resolving_own_fitted_state"


def _resolve_own_fitted_attribute(wrapper, item):
    """Resolve a wrapper-owned fitted attribute without asking the wrappee.

    Wrappers call this from `__getattr__` for the names listed in their
    `_own_fitted_attributes`, because a pre-fitted wrappee publishes
    attributes of the same name. A wrapper implementing
    `__sklearn_is_fitted__` resolves its fitted state here, at most once per
    lookup.

    Parameters
    ----------
    wrapper : object
        The wrapper whose `__getattr__` did not find `item`.
    item : str
        Name of the requested attribute, listed in the wrapper's
        `_own_fitted_attributes`.

    Returns
    -------
    value : object
        The value `wrapper` itself holds for `item`.

    Raises
    ------
    sklearn.exceptions.NotFittedError
        If `wrapper` holds no value for `item`. As an `AttributeError`, it
        makes `hasattr` report the attribute as absent.
    AttributeError
        If the lazy resolution of the wrapper's fitted state raises a
        `ValueError` or `TypeError`, e.g., because a declared class
        vocabulary contradicts the learned one.
    """
    resolve = getattr(type(wrapper), "__sklearn_is_fitted__", None)
    if resolve is not None and not wrapper.__dict__.get(
        _RESOLVING_OWN_FITTED_STATE, False
    ):
        wrapper.__dict__[_RESOLVING_OWN_FITTED_STATE] = True
        try:
            resolve(wrapper)
        except (ValueError, TypeError) as error:
            raise AttributeError(
                f"'{type(wrapper).__name__}' cannot answer '{item}' with the "
                f"wrapped estimator's value, and resolving its own fitted "
                f"state failed: {error}"
            ) from error
        finally:
            del wrapper.__dict__[_RESOLVING_OWN_FITTED_STATE]
        if item in wrapper.__dict__:
            return wrapper.__dict__[item]
    raise NotFittedError(
        f"This '{type(wrapper).__name__}' instance is not fitted yet. Call "
        f"'fit' with appropriate arguments before accessing '{item}'."
    )


def _restore_wrapper_attributes(wrapper, attributes):
    """Restore every attribute of a wrapper to a pre-call snapshot.

    Only the attribute mapping is restored. A wrapped estimator mutated by
    `partial_fit` and fitted state updated in place, e.g., a sliding window,
    are not rolled back.

    Parameters
    ----------
    wrapper : object
        The wrapper to roll back.
    attributes : dict
        Snapshot of `wrapper.__dict__` taken before the failing call.
    """
    wrapper.__dict__.clear()
    wrapper.__dict__.update(attributes)
