import inspect

from ..classifier import ParzenWindowClassifier


def check_positional_args(func, func_name, param_dict, kwargs_name=None):
    func_params = inspect.signature(func).parameters
    kwargs_var_keyword = []
    # Get kwargs variables
    kwargs_var_keyword = list(
        filter(lambda p: p.kind == p.VAR_KEYWORD, func_params.values())
    )

    # Test if each required key except for kwargs is included.
    if param_dict is not None:
        for key, val in func_params.items():
            if (
                key != "self"
                and val not in kwargs_var_keyword
                and val.default == inspect._empty
                and key not in param_dict
            ):
                if kwargs_name in None:
                    raise ValueError(
                        f"Missing positional argument `{key}` of `{func_name}`"
                        f" in `{func_name}_default_kwargs`."
                    )
                else:
                    raise ValueError(
                        f"Missing positional argument `{key}` of `{func_name}`"
                        f" in `{kwargs_name}`."
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


class ParzenWindowClassifierEmbedding(ParzenWindowClassifier):
    def predict(self, X, return_embeddings=False):
        y_pred = super().predict(X)
        if not return_embeddings:
            return y_pred
        return y_pred, X

    def predict_proba(self, X, return_embeddings=False):
        probas = super().predict_proba(X)
        if not return_embeddings:
            return probas
        return probas, X


class ParzenWindowClassifierTuple(ParzenWindowClassifier):
    def predict(self, X):
        y_pred = super().predict_proba(X).argmax(axis=-1)
        return y_pred, X

    def predict_proba(self, X):
        probas = super().predict_proba(X)
        return probas, X
