"""Fitted state of estimators wrapping another estimator."""

from sklearn.exceptions import NotFittedError

# Key marking an in-progress lazy resolution in a wrapper's `__dict__`. It ends
# without an underscore, so that `sklearn.utils.validation.check_is_fitted`
# never mistakes it for a fitted attribute while it is present.
_RESOLVING_OWN_FITTED_STATE = "_resolving_own_fitted_state"


def _resolve_own_fitted_attribute(wrapper, item):
    """Resolve a wrapper-owned fitted attribute without asking the wrappee.

    Every wrapper forwarding unknown attributes to the object it wraps calls
    this from its `__getattr__` for the names listed in its
    `_own_fitted_attributes`. Those names express one policy: a wrapper answers
    the fitted attributes it holds itself and never lets its wrappee answer
    them, because a pre-fitted wrappee publishes attributes of the same name
    long before the wrapper has resolved its own. Every other name keeps being
    forwarded, which is what the forwarding exists for. Each wrapper documents
    which names it owns, and why it excludes the near misses, next to its own
    `_own_fitted_attributes`.

    A wrapper deriving its fitted state lazily from a pre-fitted wrappee, i.e.,
    one implementing `__sklearn_is_fitted__`, resolves that state here instead
    of answering from the wrappee, so that the answer is always the wrapper's
    own. This resolution commits the wrapper's fitted state, which is why this
    is not a plain read. It runs at most once per lookup: a nested lookup of an
    owned name reaching this function while it is running reports the wrapper
    as unfitted rather than resolving again.

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
        If `wrapper` holds no value for `item`. Being an `AttributeError`, it
        also makes `hasattr` report the attribute as absent.
    AttributeError
        If the wrapper rejected a lazy resolution of its own fitted state,
        e.g., because a declared class vocabulary contradicts the learned one.
        The rejected state is never answered with the wrappee's value, and the
        rejection's reason is quoted in the message. Only the `ValueError` and
        `TypeError` a rejection raises are converted; any other exception
        propagates unchanged, so that a defect is not reported as a missing
        attribute.
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

    A failing fit must not leave a wrapper that reports fitted state it cannot
    serve. Because the attributes are restored as a whole rather than
    selectively deleted, an already fitted wrapper keeps exactly its pre-call
    values, and a previously unfitted one stays unfitted.

    Note that this restores the wrapper only. A `partial_fit` mutates the
    wrapped estimator in place, so a failing incremental update can leave that
    estimator itself in an implementation-defined state.

    A wrapper holding mutable fitted state that its fit updates in place, e.g.
    a sliding window, needs more than this: the snapshot holds the same object
    the fit mutated, so restoring the mapping alone does not undo the update.
    Such a wrapper has to copy that state when it snapshots and roll it back
    itself in addition to calling this function.

    Parameters
    ----------
    wrapper : object
        The wrapper to roll back.
    attributes : dict
        Snapshot of `wrapper.__dict__` taken before the failing call.
    """
    wrapper.__dict__.clear()
    wrapper.__dict__.update(attributes)
