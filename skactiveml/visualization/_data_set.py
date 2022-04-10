import numpy as np
from scipy.stats import norm, uniform


def gaussian_noise_generator_1d(
    X, *intervals, std=1, default_std=0.5, random_state=None
):
    # interval : tuple of shape (a, b) or (a, b, std), a < b,
    # where std is the standard deviation of the
    # gaussian noise in the interval [a, b)

    intervals = list(intervals)

    for idx, interval in enumerate(intervals):
        if len(interval) != 3:
            intervals[idx] = (*interval, std)

    x = X.flatten()
    noise = np.zeros_like(x)
    for a, b, std_itv in intervals:
        noise_itv = norm.rvs(scale=std_itv, size=x.shape, random_state=random_state)
        noise = noise + np.where((a <= x) & (x < b), noise_itv, 0)
    noise += norm.rvs(scale=default_std, size=x.shape, random_state=random_state)
    return noise


def sample_generator_1d(n_samples, *intervals, density=1, random_state=None):
    # interval : tuple of shape (a, b) or (a, b, density), a < b

    intervals = list(intervals)

    for idx, interval in enumerate(intervals):
        if len(interval) != 3:
            intervals[idx] = (*interval, density)

    total_weight = sum(((b - a) * density_itv for a, b, density_itv in intervals))
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
