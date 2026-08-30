import inspect
from types import MethodType
from makefun import with_signature

successful_skorch_torch_import = False
try:
    import sys
    from collections.abc import Sequence
    from copy import deepcopy
    from torch import nn

    from ._validation import _check_forward_outputs

    successful_skorch_torch_import = True
except ImportError:  # pragma: no cover
    pass


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


if successful_skorch_torch_import:

    def make_criterion_tuple_aware(
        criterion,
        criterion_output_keys=None,
        forward_outputs=None,
    ):
        """
        Create a loss class (or wrap an existing instance) that selects part
        of a model's (possibly tuple-valued) output before passing it to the
        criterion.

        This utility generates (and caches) a dynamic subclass named
        `TupleAware<LossName>Idx<...>` of the given base loss class. The
        subclass overrides `forward` so that, if the `input` argument is a
        tuple (e.g., `(logits, embeddings, ...)`), only the element(s)
        selected by `criterion_output_keys` are passed to the base class's
        `forward`. If `input` is not a tuple, it is forwarded unchanged.

        If no selection is required (i.e., `criterion_output_keys` is
        `None` and `forward_outputs` is `None` so that the full input
        is passed unchanged), the original `criterion` is returned without
        wrapping.

        Parameters
        ----------
        criterion : torch.nn.Module.__class__ or torch.nn.Module
            - Either a loss class (subclass of `torch.nn.Module`),
              e.g. `nn.CrossEntropyLoss`, or
            - a loss instance, e.g. `nn.CrossEntropyLoss()`.
        criterion_output_keys : str or sequence of str or None, default=None
            Name or names of the forward outputs that are passed to the
            loss / criterion during training. Use this when `module.forward`
            returns multiple outputs (e.g. `(logits, embeddings, ...)`), but
            the criterion expects a single tensor input or a specific tuple of
            inputs. The names must refer to keys of `forward_outputs`. If
            `criterion_output_keys` is not `None` and `forward_outputs`
            is `None`, a `ValueError` is raised because the names
            cannot be resolved.

            - If a `str`, the corresponding named output of
              `module.forward` (i.e., the raw tensor selected via its index
              in `forward_outputs` before applying any transform) is passed
              to the criterion (e.g. `"logits"` to use only the class scores).
            - If a sequence of `str`, the selected named outputs are packed
              into a tuple and passed to the criterion in that order. Each raw
              forward output index may appear at most once: using multiple
              names that resolve to the same underlying index (e.g. `"proba"`
              and `"logits"` both pointing to index 0) is not allowed and
              results in a `ValueError`.
            - If `None`:

              - and `forward_outputs` is not `None`, the first output
                defined by `forward_outputs` is used as criterion input;
              - and `forward_outputs` is `None`, the full `input` is
                passed unchanged. In this case, the caller is responsible for
                ensuring that `module.forward` returns a single tensor if the
                criterion does not accept tuples.

        forward_outputs : dict[str, tuple[int, Callable | None]] or None,\
                default=None
            Dictionary from output names to `(idx, transform)` tuples, as used
            in the estimator's `forward_outputs` parameter. Only the keys and
            their associated indices `idx` are used here to resolve
            `criterion_output_keys` into raw output positions; the transform
            part is ignored by this helper. If `criterion_output_keys` is given
            as a string or sequence of strings, `forward_outputs` must be
            provided.

        Returns
        -------
        torch.nn.Module.__class__ or torch.nn.Module
            - If `criterion` is a class, returns the generated subclass
              `TupleAware<LossName>Idx<...>` (or the original class if no
              selection is required).
            - If `criterion` is an instance, returns a new instance
              (deep copy) whose class is that subclass. The original instance
              is not modified. If no selection is required, the original
              instance is returned unchanged.
        """
        # Validate criterion
        if isinstance(criterion, nn.Module):
            base_cls = criterion.__class__
        elif isinstance(criterion, type) and issubclass(criterion, nn.Module):
            base_cls = criterion
        else:
            raise TypeError(
                "criterion must be an `nn.Module` subclass or instance."
            )

        # Check invalid combination.
        if criterion_output_keys is not None and forward_outputs is None:
            raise ValueError(
                "`criterion_output_keys` is not `None`, but `forward_outputs` "
                "is `None`. Pass the same `forward_outputs` mapping that "
                "describes `module.forward`."
            )

        # No selection at all: pass full input unchanged.
        if criterion_output_keys is None and forward_outputs is None:
            return criterion

        # Validate forward_outputs if it is provided.
        _check_forward_outputs(forward_outputs=forward_outputs)

        # Resolve criterion_output_keys -> list of names.
        if criterion_output_keys is None:
            selected_names = [next(iter(forward_outputs))]
        elif isinstance(criterion_output_keys, str):
            selected_names = [criterion_output_keys]
        elif isinstance(criterion_output_keys, Sequence) and not isinstance(
            criterion_output_keys, bytes
        ):
            if len(criterion_output_keys) == 0:
                raise ValueError(
                    "`criterion_output_keys` must not be an empty sequence."
                )
            # Ensure all names are strings
            for name in criterion_output_keys:
                if not isinstance(name, str):
                    raise TypeError(
                        "All entries in `criterion_output_keys` must be "
                        "strings when a sequence is provided, got "
                        f"{type(name)} in {criterion_output_keys!r}."
                    )
            selected_names = list(criterion_output_keys)
        else:
            raise TypeError(
                "`criterion_output_keys` must be `None`, a string, or a "
                f"sequence of strings, got {type(criterion_output_keys)}."
            )

        # Map names -> raw indices using forward_outputs[name][0].
        indices = []
        for name in selected_names:
            spec = forward_outputs.get(name, None)
            if spec is None:
                raise ValueError(
                    f"Unknown forward output name {name!r} in "
                    "`criterion_output_keys`. Available names are: "
                    f"{list(forward_outputs.keys())}."
                )
            indices.append(spec[0])

        # Enforce: each raw index at most once.
        if len(set(indices)) != len(indices):
            raise ValueError(
                "`criterion_output_keys` must not contain multiple names "
                "that refer to the same raw forward output index. "
                f"Names {selected_names!r} resolve to indices {indices!r}."
            )

        # Build selector: int or tuple[int,...].
        if len(indices) == 1:
            selector = indices[0]
            max_idx = selector
        else:
            selector = tuple(indices)
            max_idx = max(selector)

        # Build a class-name key from the selector for caching / pickling.
        if isinstance(selector, int):
            idx_key = str(selector)
        else:
            idx_key = "_".join(str(i) for i in selector)

        cls_name = f"TupleAware{base_cls.__name__}Idx{idx_key}"

        # Build the wrapper `forward`.
        def forward(self, input, target, *args, **kwargs):
            if isinstance(input, tuple):
                if self._criterion_output_max_selector >= len(input):
                    raise ValueError(
                        f"`forward_outputs` references raw output index "
                        f"{self._criterion_output_max_selector}, but "
                        f"`module.forward` returned only {len(input)} "
                        f"object(s)."
                    )
                if isinstance(self._criterion_output_selector, int):
                    input = input[self._criterion_output_selector]
                else:
                    # selector is a tuple of indices
                    input = tuple(
                        input[i] for i in self._criterion_output_selector
                    )
            return base_cls.forward(self, input, target, *args, **kwargs)

        # Keep signature/tool-tips identical.
        forward.__signature__ = inspect.signature(base_cls.forward)
        forward.__doc__ = base_cls.forward.__doc__

        # Create/reuse subclass and expose it under this module.
        mod = sys.modules[__name__]
        TupleAwareCls = getattr(mod, cls_name, None)
        if TupleAwareCls is None:
            TupleAwareCls = type(
                cls_name,
                (base_cls,),
                {
                    "forward": forward,
                    "__module__": mod.__name__,
                    "_criterion_output_selector": selector,
                    "_criterion_output_max_selector": max_idx,
                },
            )
            setattr(mod, TupleAwareCls.__name__, TupleAwareCls)

        if isinstance(criterion, nn.Module):
            wrapped = deepcopy(criterion)
            object.__setattr__(wrapped, "__class__", TupleAwareCls)
            return wrapped
        else:
            return TupleAwareCls
