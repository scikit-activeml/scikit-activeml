# %%
# .. note::
#    The generated animation can be found at the bottom of the page.
#
# | **Google Colab Note**: If the notebook fails to run after installing the
#   needed packages, try to restart the runtime (Ctrl + M) under
#   Runtime -> Restart session.
#
# .. image:: https://colab.research.google.com/assets/colab-badge.svg
#    :target: "$colab_link"
#
# | **Notebook Dependencies**
# | Uncomment the following cell to install all dependencies for this
#   tutorial.

"$install_dependencies|# !pip install scikit-activeml"

# %%
# .. raw:: html
#
#   <hr style="border-style: solid; border-top: 1px solid; border-right: 0; border-bottom: 0; border-left: 0;">
#

# %%
import numpy as np
from matplotlib import pyplot as plt, animation
from scipy.stats import uniform

from skactiveml.utils import MISSING_LABEL, labeled_indices, is_labeled

"$import_reg|from skactiveml.regressor import NICKernelRegressor"
"$import_misc"

# Set a fixed random state for reproducibility.
random_state = np.random.RandomState(0)


def true_function(X_):
    """Compute the true underlying function."""
    return (X_**3 + 2 * X_**2 + X_ - 1).flatten()


# Generate samples.
n_samples = "$n_samples|100"
X = np.concatenate(
    [
        uniform.rvs(0, 1.5, 9 * n_samples // 10, random_state=random_state),
        uniform.rvs(1.5, 0.5, n_samples // 10, random_state=random_state),
    ]
).reshape(-1, 1)

# Define noise: higher noise for X < 1 and lower otherwise.
noise = np.vectorize(
    lambda x: random_state.rand() * 1.5 if x < 1 else random_state.rand() * 0.5
)

# Build the dataset.
y_true = true_function(X) + noise(X).flatten()
y = np.full(shape=y_true.shape, fill_value=MISSING_LABEL)
X_test = np.linspace(0, 2, num="$res|100").reshape(-1, 1)

# Initialise the regressor.
reg = "$init_reg|NICKernelRegressor(random_state=random_state, metric_dict={'gamma': 15.0})"

# Initialise the query strategy.
qs = "$init_qs"
"$preproc"
# Prepare the plotting area.
fig, (ax_1, ax_2) = plt.subplots(2, 1, sharex=True)
artists = []

# Active learning cycle.
n_cycles = "$n_cycles|20"
for c in range(n_cycles):
    # Fit the regressor using the current labels.
    reg.fit(X, y)

    # Query the next sample(s).
    query_idx = qs.query("$query_params")

    # Record current plot elements.
    coll_old = list(ax_1.collections) + list(ax_2.collections)
    title = ax_1.text(
        0.5, 1.05,
        f"Prediction after acquiring {c} labels",
        size=plt.rcParams["axes.titlesize"],
        ha="center",
        transform=ax_1.transAxes,
    )

    # Compute utility values for the test candidates.
    _, utilities_test = qs.query("$query_params", candidates=X_test, return_utilities=True)
    utilities_test = (utilities_test - utilities_test.min()).flatten()
    if np.any(utilities_test != utilities_test[0]):
        utilities_test /= utilities_test.max()

    # Plot utility information on the second axis.
    (utility_line,) = ax_2.plot(X_test, utilities_test, c="green")
    utility_fill = plt.fill_between(X_test.flatten(), utilities_test, color="green", alpha=0.3)

    # Plot the samples and their labels.
    is_lbld = is_labeled(y)
    ax_1.scatter(X[~is_lbld], y_true[~is_lbld], c="lightblue")
    ax_1.scatter(X[is_lbld], y[is_lbld], c="orange")

    # Predict and plot the regressor's output.
    y_pred = reg.predict(X_test)
    (prediction_line,) = ax_1.plot(X_test, y_pred, c="black")

    # Capture new plot elements.
    coll_new = list(ax_1.collections) + list(ax_2.collections)
    coll_new.append(title)
    artists.append(
        [x for x in coll_new if (x not in coll_old)]
        + [utility_line, utility_fill, prediction_line]
    )

    # Update labels for the queried sample.
    y[query_idx] = y_true[query_idx]

# Create an animation from the collected frames.
ani = animation.ArtistAnimation(fig, artists, interval=1000, blit=True)

# %%
# .. image:: ../../examples/pool_regression_legend.png
