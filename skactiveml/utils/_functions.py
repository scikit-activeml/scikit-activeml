import inspect
from functools import update_wrapper
from operator import attrgetter


def call_func(f_callable, only_mandatory=False, **kwargs):
    """Calls a function with the given parameters given in kwargs if they
    exist as parameters in f_callable.

    Parameters
    ----------
    f_callable : callable
        The function or object that is to be called
    only_mandatory : boolean
        If True only mandatory parameters are set.
    kwargs : kwargs
        All parameters that could be used for calling f_callable.

    Returns
    -------
    called object
    """
    params = inspect.signature(f_callable).parameters
    param_keys = params.keys()
    if only_mandatory:
        param_keys = list(
            filter(lambda k: params[k].default == inspect._empty, param_keys)
        )

    vars = dict(filter(lambda e: e[0] in param_keys, kwargs.items()))

    return f_callable(**vars)


def _available_if(method_name, has_available_if):
    if has_available_if:
        from sklearn.utils.metaestimators import available_if

        decorator = available_if(
            lambda self: _hasattr_array_like(self.estimator, method_name)
        )
    else:
        from sklearn.utils.metaestimators import if_delegate_has_method

        if not isinstance(method_name, (list, tuple)):
            decorator = if_delegate_has_method(delegate="estimator")
        else:
            decorator = _if_delegate_has_methods(
                delegate="estimator", method_names=method_name
            )

    return decorator


def _hasattr_array_like(obj, attribute_names):
    if not isinstance(attribute_names, (list, tuple)):
        attribute_names = [attribute_names]

    return any(hasattr(obj, attr) for attr in attribute_names)


class _IffHasAMethod:
    def __init__(self, fn, delegate_name, method_names):
        self.fn = fn
        self.delegate_name = delegate_name
        self.method_names = method_names

        # update the docstring of the descriptor
        update_wrapper(self, fn)

    def __get__(self, obj, owner=None):

        delegate = attrgetter(self.delegate_name)(obj)
        if not _hasattr_array_like(
            delegate, attribute_names=self.method_names
        ):
            raise AttributeError

        def out(*args, **kwargs):
            return self.fn(obj, *args, **kwargs)

        # update the docstring of the returned function
        update_wrapper(out, self.fn)
        return out


def _if_delegate_has_methods(delegate, method_names):
    return lambda fn: _IffHasAMethod(
        fn, delegate_name=delegate, method_names=method_names
    )
