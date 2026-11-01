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
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_blobs

from skactiveml.utils import MISSING_LABEL
from skactiveml.visualization import plot_stream_decision_boundary

"$import_clf|from skactiveml.classifier import ParzenWindowClassifier"
"$import_misc"

# Set a fixed random state for reproducibility.
random_state = np.random.RandomState(0)

# Initial training set size.
init_size = "$init_size|0"  # e.g. 0 or any integer value

# Build a dataset.
n_test = 100
X, y_true = make_blobs(
    n_samples="$n_samples|200" + init_size + n_test,
    n_features=1,
    centers=[[0], [-3], [1], [2], [-0.5]],
    cluster_std=0.7,
    random_state=random_state,
)
y_true = y_true % 2  # Convert labels to binary (0, 1)

# Split the data into test, initial training, and streaming parts.
X_test, y_test = X[-n_test:], y_true[-n_test:]
X, y_true = X[:-n_test], y_true[:-n_test]
X_init = X[:init_size]
y_init = y_true[:init_size]
X_stream = X[init_size:]
y_stream = y_true[init_size:]

# Initialise the classifier.
clf = "$init_clf|ParzenWindowClassifier(classes=[0, 1], class_prior=1e-3, random_state=random_state)"

# Initialise the query strategy.
qs = "$init_qs"
plot_step = "$init_plot_step|5"
"$preproc"
# Initialize training data with initial examples.
X = list(X_init)
y = list(y_init)

# Preparation for plotting.
class_cmap = ListedColormap(["#2a78d6", "#1baf7a"])
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

fig = plt.figure(figsize=(6.4, 7.2))
ax = fig.add_axes([0.12, 0.53, 0.85, 0.41])
ax_utility = fig.add_axes([0.12, 0.35, 0.85, 0.14], sharex=ax)
ax_budget = fig.add_axes([0.12, 0.07, 0.33, 0.17])
ax_curve = fig.add_axes([0.6, 0.07, 0.3, 0.17])
axs = [ax, ax_utility, ax_budget, ax_curve]
ax.tick_params(labelbottom=False)
ax.set_xlim(-0.5, len(X_stream) + 0.5)
ax.set_ylim(X_stream.min() - 0.3, X_stream.max() + 0.3)
ax.set_ylabel("Feature")
ax_utility.set_xlabel("Observed samples")
ax_utility.set_ylabel("Utility")
ax_budget.set_title("Budget used", loc="left")
ax_budget.set_xlabel("Observed samples")
ax_budget.set_ylim(0, 1.02)
ax_budget.axhline(qs.budget, color="#8c8c8c", ls="--", lw=1.5)
ax_budget.grid(axis="y", color="#e3e5e8")
ax_curve.set_title("Learning curve of this run", loc="left")
ax_curve.set_xlabel("Observed samples")
ax_curve.set_ylabel("Test accuracy")
ax_curve.set_ylim(0.3, 1.02)
ax_curve.grid(axis="y", color="#e3e5e8")
for axis in (ax_budget, ax_curve):
    axis.set_xlim(-0.5, len(X_stream) + 0.5)
artists = []

queried_indices = [True] * len(y_init)
predictions_list = []
utilities_list, budgets, accuracies = [], [], []

# Process each streaming sample.
for t_x, (x_t, y_t) in enumerate(zip(X_stream, y_stream)):
    X_cand = x_t.reshape([1, -1])

    # Fit and evaluate the classifier.
    clf.fit(X, y)
    accuracies.append(clf.score(X_test, y_test))

    # Decide whether to query the current sample.
    sampled_indices, utilities = qs.query(
        "$query_params", return_utilities=True
    )
    qs.update("$update_params")

    X.append(x_t)
    y.append(y_t if len(sampled_indices) else MISSING_LABEL)
    queried_indices.append(len(sampled_indices) > 0)
    utilities_list.append(utilities[0])
    budgets.append(np.mean(queried_indices[len(y_init) :]))

    # Plot the current state at intervals defined by plot_step.
    if t_x % plot_step == 0:
        coll_old = [x for a in axs for x in a.collections]

        # Update the stream plot.
        ax, predictions_list = plot_stream_decision_boundary(
            ax,
            t_x,
            plot_step,
            clf,
            X_stream,
            predictions_list,
            color=line_color,
            res="$res|25",
        )
        observed = np.arange(1, t_x + 2)
        X_seen = X_stream[: t_x + 1, 0]
        y_seen = y_stream[: t_x + 1]
        queried = np.array(queried_indices[len(y_init) :])
        recent = queried & (observed > t_x + 1 - plot_step)
        ax.scatter(
            observed[~queried],
            X_seen[~queried],
            s=16,
            facecolors="none",
            edgecolors=class_cmap(y_seen[~queried]),
            zorder=2,
        )
        ax.scatter(
            observed[queried],
            X_seen[queried],
            s=46,
            facecolors=class_cmap(y_seen[queried]),
            edgecolors=line_color,
            zorder=3,
        )
        ax.scatter(
            observed[recent],
            X_seen[recent],
            s=320,
            facecolors=(selection_color, 0.28),
            edgecolors=selection_color,
            linewidths=2.2,
            zorder=4,
        )
        title = ax.text(
            0,
            1.03,
            f"{t_x + 1} of {len(X_stream)} samples observed",
            size=plt.rcParams["axes.titlesize"],
            transform=ax.transAxes,
        )

        # Update the utilities, budget, and learning curve.
        seen_utilities = np.array(utilities_list)
        ax_utility.vlines(
            observed[~queried],
            0,
            seen_utilities[~queried],
            colors=utility_color,
        )
        ax_utility.vlines(
            observed[queried],
            0,
            seen_utilities[queried],
            colors=selection_color,
        )
        (budget_line,) = ax_budget.plot(observed, budgets, c=line_color, lw=2)
        ax_budget.scatter(t_x + 1, budgets[-1], c=line_color)
        (curve,) = ax_curve.plot(
            np.arange(t_x + 1), accuracies, c=line_color, lw=2
        )
        ax_curve.scatter(t_x, accuracies[-1], c=line_color)
        value = ax_curve.annotate(
            f"{accuracies[-1]:.2f}",
            (t_x, accuracies[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
        )

        coll_new = [x for a in axs for x in a.collections]
        artists.append(
            [x for x in coll_new if x not in coll_old]
            + [title, budget_line, curve, value]
        )

# Show the utilities from low to high.
low, high = min(0, min(utilities_list)), max(utilities_list)
high = high if high > low else low + 1
ax_utility.set_ylim(low, high + 0.1 * (high - low))
ax_utility.set_yticks([low, high], ["low", "high"])

ani = animation.ArtistAnimation(fig, artists, interval=500)
