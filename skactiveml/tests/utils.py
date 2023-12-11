import inspect


def check_positional_args(func, func_name, param_dict, allow_kwargs=False):
    func_params = inspect.signature(func).parameters
    kwargs_var_keyword = []
    if not allow_kwargs:
        kwargs_var_keyword = list(
            filter(lambda p: p.kind == p.VAR_KEYWORD, func_params.values())
        )

    if param_dict is not None:
        for key, val in func_params.items():
            if (
                key != "self"
                and val not in kwargs_var_keyword
                and val.default == inspect._empty
                and key not in param_dict
            ):
                raise ValueError(
                    f"Missing positional argument `{key}` of `{func_name}` in "
                    f"`{func_name}_default_kwargs`."
                )


def check_test_param_test_availability(
    class_, func, func_name, not_test, logic_test=True
):
    # Get func parameters.
    func_params = inspect.signature(func).parameters
    kwargs_var_keyword = list(
        filter(lambda p: p.kind == p.VAR_KEYWORD, func_params.values())
    )

    # Check func parameters.
    for param, val in func_params.items():
        if param in not_test or val in kwargs_var_keyword:
            continue
        test_func_name = f"test_{func_name}_param_" + param
        with class_.subTest(msg=test_func_name):
            class_.assertTrue(
                hasattr(class_, test_func_name),
                msg=f"'{test_func_name}()' missing in {class_.__class__}",
            )
    if logic_test:
        # Check if func is being tested.
        with class_.subTest(msg=f"test_{func_name}"):
            class_.assertTrue(
                hasattr(class_, f"test_{func_name}"),
                msg=f"'test_{func_name}' missing in {class_.__class__}",
            )
