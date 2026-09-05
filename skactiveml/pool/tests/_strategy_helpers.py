"""Private helpers shared by pool strategy contract tests."""

import inspect

from skactiveml.pool import RandomSampling


def _instantiate(strategy):
    """Instantiate a strategy by defaulting its required arguments."""
    init_params = {}
    for name, parameter in inspect.signature(
        strategy.__init__
    ).parameters.items():
        if name == "self":
            continue
        if name == "query_strategy":
            # Wrappers delegate their capabilities to the wrapped strategy,
            # so they must not be inspected with an absent one.
            init_params[name] = RandomSampling()
        elif parameter.default is inspect.Parameter.empty:
            init_params[name] = None
    return strategy(**init_params)
