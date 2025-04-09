import inspect
from types import MethodType
from makefun import with_signature


def call_func(
    f_callable, only_mandatory=False, ignore_var_keyword=False, **kwargs
):
    """Calls a function with the given parameters given in `kwargs`, if they
    exist as parameters in `f_callable`.

    Parameters
    ----------
    f_callable : callable
        The function or object that is to be called.
    only_mandatory : boolean, default=False
        If `True`, only mandatory parameters are set.
    ignore_var_keyword : boolean, default=False
        If `False`, all kwargs are passed when `f_callable` uses a parameter
        that is of kind `Parameter.VAR_KEYWORD`, i.e., `**kwargs`. For further
        reference see the `inspect` package.
    kwargs : kwargs
        All parameters that could be used for calling f_callable.

    Returns
    -------
    f_callable_result : return type of `f_callable`
        The return value of f_callable.
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


def match_signature(wrapped_obj_name, func_name):
    """A decorator that matches the signature to a given method from a
    reference and hides it when the reference object does not have the wrapped
    function. This is especially helpful for wrapper classes whose functions
    should appear. This decorator is heavily inspired by the `available_if`
    decorator from `sklearn.utils.metaestimators`.

    Parameters
    ----------
    wrapped_obj_name : str
        The name of the object that will be wrapped.
    func_name : str
        The name of the function that will be wrapped.

    Returns
    -------
    wrapped_obj : callable
        The wrapped function.
    """

    class _MatchSignatureDescriptor:
        """_MatchSignatureDescriptor

        A descriptor that allows a wrapper to clone the signature of a
        method `func_name` from the wrapped object `wrapped_obj_name`.
        Furthermore, this extends upon the conditional property as implemented
        in `available_if` from `sklearn.utils.metaestimators`.

        Parameters
        ----------
        fn: MethodType
            The method that should be wrapped.
        wrapped_obj_name: str
            The name of the wrapped object within the wrapper class.
        func_name : str
            The method name of the function that should be wrapped.
        """

        def __init__(self, fn, wrapped_obj_name, func_name):
            self.fn = fn
            self.wrapped_obj_name = wrapped_obj_name
            self.func_name = func_name
            self.__name__ = func_name

        def __get__(self, obj, owner=None):
            """Wrap the method specified in `self.func_name` from the wrapped
            object `self.wrapped_obj_name` such that the signature will be the
            same.

            Parameters
            ----------
            obj : object
                The wrapper object. This parameter will be None, if the method
                is accessed via the class and not an instantiated object.
            owner : class, default=None
                The wrapper class.

            Returns
            -------
            The wrapped method.
            """
            if obj is not None:
                reference_object = getattr(obj, self.wrapped_obj_name)
                if not hasattr(reference_object, self.func_name):
                    raise AttributeError(
                        f"This {reference_object} has"
                        f" no method {self.func_name}."
                    )
                reference_function = getattr(reference_object, self.func_name)

                # Check if the refrenced function is a method of that object.
                # If it is, use `__func__` to copy the name of the `self`
                # parameter
                # If it is not, (i.e. it has been added as a lambda function),
                # add a provisory self argument in the first position
                if hasattr(reference_function, "__func__"):
                    new_sig = inspect.signature(reference_function.__func__)
                else:
                    reference_sig = inspect.signature(reference_function)
                    new_parameters = list(reference_sig.parameters.values())
                    new_parameters.insert(
                        0,
                        inspect.Parameter(
                            "self",
                            kind=inspect._ParameterKind.POSITIONAL_OR_KEYWORD,
                        ),
                    )
                    new_sig = inspect.Signature(
                        new_parameters,
                        return_annotation=reference_sig.return_annotation,
                    )

                # create a wrapper with the new signature and the correct
                # function name
                function_decorator = with_signature(
                    func_signature=new_sig,
                    func_name=self.fn.__name__,
                )
                fn = function_decorator(self.fn)
                out = MethodType(fn, obj)
            else:
                out = self.fn

            return out

    return lambda fn: _MatchSignatureDescriptor(
        fn, wrapped_obj_name, func_name=func_name
    )
