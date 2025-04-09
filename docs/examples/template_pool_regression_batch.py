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

# Set fixed random state for reproducibility.
random_state = np.random.RandomState(0)


def true_function(X_):
    """Compute the true function values for input X_."""
    return (X_**3 + 2 * X_**2 + X_ - 1).flatten()


# Generate dataset.
n_samples = "$n_samples|100"
X = np.concatenate(
    [
        uniform.rvs(0, 1.5, 9 * n_samples // 10, random_state=random_state),
        uniform.rvs(1.5, 0.5, n_samples // 10, random_state=random_state),
    ]
).reshape(-1, 1)

# Define noise: higher noise for x < 1 and lower otherwise.
noise = np.vectorize(
    lambda x: random_state.rand() * 1.5 if x < 1 else random_state.rand() * 0.5
)

# Build labels with added noise.
y_true = true_function(X) + noise(X).flatten()
y = np.full(shape=y_true.shape, fill_value=MISSING_LABEL)
X_test = np.linspace(0, 2, num="$res|1000").reshape(-1, 1)

# Initialise the classifier.
reg = "$init_reg|NICKernelRegressor(random_state=random_state, metric_dict={'gamma': 15.0})"

# Initialise the query strategy.
qs = "$init_qs"
"$preproc"
# Prepare the plotting area.
fig, (ax_1, ax_2) = plt.subplots(2, 1, sharex=True)
artists = []

# Active learning cycle.
n_cycles = "$n_cycles|20"
batch_size = "$batch_size|1"
for c in range(n_cycles):
    # Train the regressor with the current labels.
    reg.fit(X, y)

    # Query new sample(s) and obtain utilities.
    query_idx, utilities = qs.query("$query_params", batch_size=batch_size)

    # Capture the current plot state.
    coll_old = list(ax_1.collections) + list(ax_2.collections)
    title = ax_1.text(
        0.5, 1.05,
        f"Prediction after acquiring {c} labels",
        size=plt.rcParams["axes.titlesize"],
        ha="center",
        transform=ax_1.transAxes,
    )

    # Sort X values for smooth plotting of the utility curve.
    sort_mask = np.argsort(X.flatten())
    X_plot = X.flatten()[sort_mask]
    utilities_plot = utilities[0][sort_mask] / batch_size

    # Plot the utility curve on the second axis.
    (utility_line,) = ax_2.plot(X_plot, utilities_plot, c="green")
    utility_fill = plt.fill_between(X_plot, utilities_plot, color="green", alpha=0.3)

    # Plot the data: unlabeled points in light blue, labeled points in orange.
    is_lbld = is_labeled(y)
    ax_1.scatter(X[~is_lbld], y_true[~is_lbld], c="lightblue")
    ax_1.scatter(X[is_lbld], y[is_lbld], c="orange")

    # Predict and plot the regressor's output on the test set.
    y_pred = reg.predict(X_test)
    (prediction_line,) = ax_1.plot(X_test, y_pred, c="black")

    # Capture new plot elements for animation.
    coll_new = list(ax_1.collections) + list(ax_2.collections)
    coll_new.append(title)
    artists.append(
        [x for x in coll_new if (x not in coll_old)]
        + [utility_line, utility_fill, prediction_line]
    )

    # Update the labels of the queried samples.
    y[query_idx] = y_true[query_idx]

# Create the animation from collected frames.
ani = animation.ArtistAnimation(fig, artists, interval=1000, blit=True)

# %%
# .. image:: ../../examples/pool_regression_legend.png
