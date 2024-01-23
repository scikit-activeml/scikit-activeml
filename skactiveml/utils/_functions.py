import inspect
from types import MethodType
from makefun import with_signature


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

class _MatchSignatureDescriptor:
    #TODO Docs
    def __init__(self, fn, reference_obj_lambda, func_name):
        self.fn = fn
        self.reference_obj_lambda = reference_obj_lambda
        self.func_name = func_name
        self.__name__ = func_name

    def __get__(self, obj, owner=None):
    #TODO Docs
        if obj is not None:
            reference_object = self.reference_obj_lambda(obj)
            if not hasattr(reference_object, self.func_name):
                attr_err = AttributeError(
                    f"This {repr(owner.__name__)} has no attribute {repr(self.attribute_name)}"
                )
                raise attr_err
        
            reference_function = getattr(reference_object, self.func_name)
            sig_str = f'{self.fn.__name__}(self, {str(inspect.signature(reference_function))[1:-1]})'
            fn = with_signature(sig_str)(self.fn)
            out = MethodType(fn, obj)
        else:
            out = self.fn

        return out

def match_signature(reference_obj_lambda, func_name):
    #TODO Docs
    return lambda fn: _MatchSignatureDescriptor(fn, reference_obj_lambda, func_name=func_name)