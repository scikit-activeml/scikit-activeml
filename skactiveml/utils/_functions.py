import inspect
from functools import update_wrapper
from operator import attrgetter


def call_func(
    f_callable, only_mandatory=False, ignore_var_keyword=False, **kwargs
):
    """Calls a function with the given parameters given in kwargs if they
    exist as parameters in f_callable.

    Parameters
    ----------
    f_callable : callable
        The function or object that is to be called
    only_mandatory : boolean
        If True only mandatory parameters are set.
    ignore_var_keyword : boolean
        If False all kwargs are passed when f_callable uses a parameter that is
        of kind Parameter.VAR_KEYWORD, i.e., **kwargs. For further reference
        see inspect package.
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
            filter(lambda k: params[k].default == params[k].empty, param_keys)
        )

    has_var_keyword = any(
        filter(lambda p: p.kind == p.VAR_KEYWORD, params.values())
    )
    if has_var_keyword and not ignore_var_keyword and not only_mandatory:
        vars = kwargs
    else:
        vars = dict(filter(lambda e: e[0] in param_keys, kwargs.items()))

    return f_callable(**vars)
