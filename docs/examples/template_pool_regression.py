# %%
# .. note::
# 
#    The generated animation can be found at the bottom of the page.
import numpy as np
from matplotlib import pyplot as plt, animation
from scipy.stats import uniform

from skactiveml.utils import MISSING_LABEL, labeled_indices, is_labeled

"$import_reg|from skactiveml.regressor import NICKernelRegressor"
"$import_misc"

random_state = np.random.RandomState(0)


def true_function(X_):
    return (X_**3 + 2 * X_**2 + X_ - 1).flatten()

n_samples = "$n_samples|100"
X = np.concatenate(
        [uniform.rvs(0, 1.5, 9 * n_samples // 10, random_state=random_state),
         uniform.rvs(1.5, 0.5, n_samples // 10, random_state=random_state)]
    ).reshape(-1, 1)

noise = np.vectorize(lambda x : random_state.rand() * 1.5 if x < 1
                                else random_state.rand() * 0.5)

# Build a dataset.
y_true = true_function(X) + noise(X).flatten()
y = np.full(shape=y_true.shape, fill_value=MISSING_LABEL)
X_test = np.linspace(0, 2, num="$res|100").reshape(-1, 1)

# Initialise the classifier.
reg = "$init_reg|NICKernelRegressor(random_state=random_state, metric_dict={'gamma': 15.0})"
# Initialise the query strategy.
qs = "$init_qs"
"$preproc"

# Preparation for plotting.
fig, (ax_1, ax_2) = plt.subplots(2, 1, sharex=True)
artists = []

# The active learning cycle:
n_cycles = "$n_cycles|20"
for c in range(n_cycles):
    # Fit the classifier.
    reg.fit(X, y)

    # Get labeled instances.
    X_labeled = X[labeled_indices(y)]

    # Query the next instance/s.
    query_idx = qs.query("$query_params")

    # Plot the labeled data.
    coll_old = list(ax_1.collections) + list(ax_2.collections)
    title = ax_1.text(
        0.5,
        1.05,
        f"Prediction after acquring {c} labels",
        size=plt.rcParams["axes.titlesize"],
        ha="center",
        transform=ax_1.transAxes,
    )

    _, utilities_test = qs.query(
        "$query_params", candidates=X_test, return_utilities=True
    )

    utilities_test = (utilities_test - utilities_test.min()).flatten()
    if np.any(utilities_test != utilities_test[0]):
        utilities_test /= utilities_test.max()

    is_lbld = is_labeled(y)

    (utility_line,) = ax_2.plot(X_test, utilities_test, c="green")
    utility_fill = plt.fill_between(
        X_test.flatten(), utilities_test, color="green", alpha=0.3
    )

    ax_1.scatter(X[~is_lbld], y_true[~is_lbld], c="lightblue")
    ax_1.scatter(X[is_lbld], y[is_lbld], c="orange")

    y_pred = reg.predict(X_test)
    (prediction_line,) = ax_1.plot(X_test, y_pred, c="black")

    coll_new = list(ax_1.collections) + list(ax_2.collections)
    coll_new.append(title)
    artists.append(
        [x for x in coll_new if (x not in coll_old)]
        + [utility_line, utility_fill, prediction_line]
    )

    # Label the queried instances.
    y[query_idx] = y_true[query_idx]

ani = animation.ArtistAnimation(fig, artists, interval=1000, blit=True)

# %%
# .. image:: ../../examples/pool_regression_legend.png
