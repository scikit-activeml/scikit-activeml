# %%
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
import numpy as np
from matplotlib import pyplot as plt, animation
from matplotlib.ticker import MaxNLocator
from scipy.stats import uniform

from skactiveml.utils import MISSING_LABEL, is_labeled

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
y_test = true_function(X_test)

# Initialise the regressor.
reg = "$init_reg|NICKernelRegressor(random_state=random_state, metric_dict={'gamma': 15.0})"

# Initialise the query strategy.
qs = "$init_qs"
"$preproc"
# Preparation for plotting.
sample_color = "#2a78d6"
unlabeled_color = "#8c8c8c"
selection_color = "#eb6834"
line_color = "#222832"
utility_color = "#8c6bb1"
plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def scale_utilities(utilities):
    """Scale the utilities to [0, 1] and leave out excluded candidates."""
    utilities = np.where(np.isfinite(utilities), utilities, np.nan)
    low, high = np.nanmin(utilities), np.nanmax(utilities)
    if high == low:
        return np.where(np.isnan(utilities), np.nan, 1.0)
    return (utilities - low) / (high - low)


fig, (ax, ax_utility, ax_curve) = plt.subplots(
    3, 1, figsize=(6.4, 7.2), height_ratios=[2.4, 1, 1]
)
fig.subplots_adjust(left=0.12, right=0.95, bottom=0.07, top=0.94, hspace=0.45)
ax_utility.sharex(ax)
ax.tick_params(labelbottom=False)
ax.set_ylabel("Label")
ax.set_ylim(y_true.min() - 2, y_true.max() + 2)
ax.plot(X_test, y_test, c=unlabeled_color, ls="--", lw=1.5)
ax_utility.set_xlabel("Feature")
ax_utility.set_ylabel("Utility")
ax_utility.set_yticks([0, 1], ["low", "high"])
ax_utility.set_ylim(0, 1.1)
ax_curve.set_title("Learning curve of this run", loc="left")
ax_curve.set_xlabel("Acquired labels")
ax_curve.set_ylabel("Test R²")
ax_curve.xaxis.set_major_locator(MaxNLocator(integer=True))
ax_curve.grid(axis="y", color="#e3e5e8")
ax_curve.margins(x=0.03)
ax_curve.set_ylim(-1.5, 1.05)
artists = []
n_acquired, scores = [], []

# Active learning cycle.
n_cycles = "$n_cycles|20"
batch_size = "$batch_size|1"
for c in range(n_cycles):
    # Fit and evaluate the regressor.
    reg.fit(X, y)
    is_lbld = is_labeled(y, missing_label=MISSING_LABEL)
    n_acquired.append(np.sum(is_lbld))
    scores.append(reg.score(X_test, y_test))

    # Query the next sample.
    query_idx, utilities = qs.query(
        "$query_params", batch_size=batch_size, return_utilities=True
    )
    utilities = scale_utilities(utilities[0])

    # Capture the current plot state.
    coll_old = list(ax.collections) + list(ax_utility.collections)

    # Update plot with the prediction and samples.
    y_pred, y_std = reg.predict(X_test, return_std=True)
    ax.fill_between(
        X_test.ravel(),
        y_pred - y_std,
        y_pred + y_std,
        color=line_color,
        alpha=0.12,
    )
    (prediction,) = ax.plot(X_test, y_pred, c=line_color, lw=2)
    ax.scatter(
        X[~is_lbld],
        y_true[~is_lbld],
        s=16,
        facecolors="none",
        edgecolors=unlabeled_color,
        zorder=2,
    )
    ax.scatter(
        X[is_lbld],
        y_true[is_lbld],
        s=46,
        facecolors=sample_color,
        edgecolors=line_color,
        zorder=3,
    )
    ax.scatter(
        X[query_idx],
        y_true[query_idx],
        s=320,
        facecolors=(selection_color, 0.28),
        edgecolors=selection_color,
        linewidths=2.2,
        zorder=4,
    )
    title = ax.text(
        0,
        1.03,
        f"Cycle {c + 1}: {n_acquired[-1]} labels acquired",
        size=plt.rcParams["axes.titlesize"],
        transform=ax.transAxes,
    )

    # Update the utilities.
    ax_utility.vlines(X.ravel(), 0, utilities, colors=utility_color, lw=2)
    ax_utility.vlines(
        X[query_idx].ravel(), 0, 1.1, colors=selection_color, lw=2
    )

    # Update the learning curve.
    (curve,) = ax_curve.plot(n_acquired, scores, c=line_color, lw=2)
    point = ax_curve.scatter(n_acquired[-1], scores[-1], c=line_color)
    value = ax_curve.annotate(
        f"{scores[-1]:.2f}",
        (n_acquired[-1], scores[-1]),
        xytext=(6, 0),
        textcoords="offset points",
        va="center",
    )

    coll_new = list(ax.collections) + list(ax_utility.collections)
    artists.append(
        [x for x in coll_new if x not in coll_old]
        + [prediction, title, curve, point, value]
    )

    # Update labels based on query.
    y[query_idx] = y_true[query_idx]

ani = animation.ArtistAnimation(fig, artists, interval=1000)
