import inspect
import numpy as np

from ..classifier import ParzenWindowClassifier
from ..pool import uncertainty_scores


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


class ParzenWindowClassifierEmbeddingUncertainty(ParzenWindowClassifier):
    def predict(self, X, return_embeddings=False, return_uncertainties=False):
        out = self.predict_proba(
            X,
            return_embeddings=return_embeddings,
            return_uncertainties=return_uncertainties,
        )
        if isinstance(out, np.ndarray):
            return out.argmax(axis=-1)
        else:
            primary = out[0].argmax(axis=-1)
            return (primary,) + out[1:]

    def predict_proba(
        self, X, return_embeddings=False, return_uncertainties=False
    ):
        out = [super().predict_proba(X)]
        if return_embeddings:
            out.append(X)
        if return_uncertainties == "1d":
            out.append(uncertainty_scores(out[0], method="entropy"))
        elif return_uncertainties == "2d":
            out.append(
                uncertainty_scores(out[0], method="entropy").reshape(-1, 1)
            )
        elif return_uncertainties is False:
            pass
        else:
            raise ValueError(
                "`return_uncertainties` must be `1d` or `2d` or `False`."
            )
        if len(out) == 1:
            return out[0]
        else:
            return tuple(out)


class ParzenWindowClassifierTuple(ParzenWindowClassifier):
    def predict(self, X):
        y_pred = super().predict_proba(X).argmax(axis=-1)
        return y_pred, X

    def predict_proba(self, X):
        probas = super().predict_proba(X)
        return probas, X


class ParzenWindowClassifierTriplet(ParzenWindowClassifier):
    def predict(self, X):
        probas = super().predict_proba(X)
        y_pred = probas.argmax(axis=-1)
        unc = uncertainty_scores(probas, method="entropy")
        return y_pred, unc, X

    def predict_proba(self, X):
        probas = super().predict_proba(X)
        unc = uncertainty_scores(probas, method="entropy")
        return probas, unc, X


def _softmax_logits_from_probas(probas):
    return np.log(np.clip(probas, a_min=1e-12, a_max=1.0))


def _normalize_extra_outputs(extra_outputs):
    if extra_outputs is None:
        return []
    if isinstance(extra_outputs, str):
        return [extra_outputs]
    return list(extra_outputs)


class ParzenWindowClassifierLogitsEmbedding(ParzenWindowClassifier):
    fit_calls = 0

    @classmethod
    def reset_fit_calls(cls):
        cls.fit_calls = 0

    def fit(self, X, y, sample_weight=None):
        type(self).fit_calls += 1
        return super().fit(X, y, sample_weight=sample_weight)

    def predict_proba(
        self,
        X,
        return_embeddings=False,
        return_logits=False,
        extra_outputs=None,
    ):
        probas = super().predict_proba(X)
        logits = _softmax_logits_from_probas(probas)
        emb = np.asarray(X)
        extra_outputs = _normalize_extra_outputs(extra_outputs)
        if extra_outputs:
            out = [probas]
            for name in extra_outputs:
                if name == "logits":
                    out.append(logits)
                elif name in ["emb", "embedding", "embeddings"]:
                    out.append(emb)
                else:
                    raise ValueError(f"Unsupported extra output `{name}`.")
            return tuple(out)
        out = [probas]
        if return_logits:
            out.append(logits)
        if return_embeddings:
            out.append(emb)
        if len(out) == 1:
            return out[0]
        return tuple(out)


class ParzenWindowClassifierLogitsEmbeddingTuple(
    ParzenWindowClassifierLogitsEmbedding
):
    def predict_proba(self, X):
        probas = super().predict_proba(X)
        logits = _softmax_logits_from_probas(probas)
        return probas, logits, np.asarray(X)


class ParzenWindowClassifierWeirdTuple(ParzenWindowClassifier):
    def predict_proba(self, X, return_stuff=False):
        probas = super().predict_proba(X)
        if not return_stuff:
            return probas
        logits = _softmax_logits_from_probas(probas)
        return probas, np.asarray(X), logits


class ParzenWindowClassifierLogitsOnly(ParzenWindowClassifier):
    def predict_proba(self, X, return_logits=False):
        probas = super().predict_proba(X)
        if return_logits:
            return _softmax_logits_from_probas(probas)
        return probas
