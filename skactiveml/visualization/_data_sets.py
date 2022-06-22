import numpy as np
from scipy.stats import norm, uniform
from sklearn.utils import check_array

from skactiveml.utils import check_scalar, check_type, check_random_state
from skactiveml.visualization._misc import _check_interval_and_assign


def gaussian_noise_generator_1d(
    X, *intervals, interval_std=1, default_std=0.5, random_state=None
):
    """Generates gaussian distributed noise for each sample.

    Parameters
    ----------
    X : array-like of shape (n_samples, 1)
        Data set for which the noise is generated.
    intervals : tuple of shape (x_low, x_up) or (x_low, x_up, std)
        Adds further noise, with a standard deviation of `std` in the interval
        from `x_low` to `x_up`. If no standard deviation is specified, the
        `interval_std` is used
    interval_std : numeric, optional (default=0.5)
        The default standard deviation for an interval with no `std` specified.
    default_std : numeric or None, optional (default=0.0)
        The standard deviation of a gaussian noise added everywhere. This value
        might be zero.
    random_state : numeric | np.random.RandomState (default=None)
        The random state to use.

    Returns
    -------
    noise : np.ndarray of shape (n_samples)
        The noise for each sample.
    """

    X = check_array(X, allow_nd=False, ensure_2d=True)
    check_type(X.shape[1], "X.shape[1]", 1)
    check_scalar(
        interval_std,
        "interval_std",
        (int, float),
        min_val=0,
        min_inclusive=False,
    )
    check_scalar(default_std, "default_std", (int, float), min_val=0)
    random_state = check_random_state(random_state)

    intervals = _check_interval_and_assign(list(intervals), interval_std)

    x = X.flatten()
    noise = np.zeros_like(x, dtype=float)
    for a, b, std_itv in intervals:
        noise_itv = norm.rvs(
            scale=std_itv, size=x.shape, random_state=random_state
        )
        noise = noise + np.where((a <= x) & (x < b), noise_itv, 0)

    if default_std != 0:
        noise += norm.rvs(
            scale=default_std, size=x.shape, random_state=random_state
        )
    return noise


def sample_generator_1d(
    n_samples, *intervals, interval_density=1, random_state=None
):
    """Generate samples in a 1d space.

    Parameters
    ----------
    n_samples : int
        The number of samples to be generated.
    intervals : tuple of shape (x_low, x_up) or (x_low, x_up, density)
        The relative amount of samples in the region from `x_low` to `x_up`.
        If `density` is not given `interval_density` is used.
    interval_density : numeric, optional (default=1)
        The default density in an interval.
    random_state : numeric | np.random.RandomState (default=None)
        The random state to use.

    Returns
    -------
    X : np.ndarray of shape (n_samples, 1)
        The generated samples.
    """

    check_scalar(n_samples, "n_samples", int, min_val=1)
    check_scalar(
        interval_density,
        "interval_density",
        (int, float),
        min_val=0,
        min_inclusive=False,
    )
    random_state = check_random_state(random_state)
    intervals = _check_interval_and_assign(list(intervals), interval_density)

    total_weight = sum(
        ((b - a) * density_itv for a, b, density_itv in intervals)
    )
    interval_sizes = [
        int(n_samples * (b - a) * density_itv / total_weight)
        for a, b, density_itv in intervals
    ]
    currently_distributed = sum(interval_sizes)
    for i in range(n_samples - currently_distributed):
        interval_sizes[i] += 1
    X = np.zeros((0, 1))
    for size, (a, b, density_itv) in zip(interval_sizes, intervals):
        generated_samples = uniform.rvs(
            loc=a, scale=b - a, size=size, random_state=random_state
        )
        X = np.append(X, generated_samples)

    return np.sort(X).reshape(-1, 1)
