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
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.ticker import MaxNLocator
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

from skactiveml.utils import MISSING_LABEL, is_labeled
from skactiveml.visualization import plot_utilities, plot_decision_boundary

"$import_clf|from skactiveml.classifier import ParzenWindowClassifier"
"$import_misc"

random_state = np.random.RandomState(0)

# Build a dataset.
X_true, y_clusters = make_blobs(
    n_samples="$n_samples|400",
    n_features=2,
    centers=[[0, 1], [-3, 0.5], [-1, -1], [2, 1], [1, -0.5]],
    cluster_std=0.7,
    random_state=random_state,
)
y_true = y_clusters % 2

X_pool, X_test, y_pool, y_test = train_test_split(
    X_true, y_true, test_size=0.25, random_state=random_state
)

X = X_pool
y = np.full(shape=y_pool.shape, fill_value=MISSING_LABEL)

# Initialise the classifier.
clf = "$init_clf|ParzenWindowClassifier(classes=[0, 1], class_prior=1e-3, metric_dict={'gamma': 3}, random_state=random_state)"

# Initialise the query strategy.
qs = "$init_qs"
"$preproc"
# Preparation for plotting: colors of the two classes, the selected samples,
# the lines, and the utilities.
class_cmap = ListedColormap(["#2a78d6", "#1baf7a"])
selection_color = "#eb6834"
line_color = "#222832"
utility_cmap, utility_alpha = "Purples", 0.5
plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

# The feature space above, the learning curve below.
fig, (ax, ax_curve) = plt.subplots(
    2, 1, figsize=(6.4, 6.4), height_ratios=[3, 1]
)
fig.subplots_adjust(left=0.11, right=0.9, bottom=0.08, top=0.93, hspace=0.38)
margin = 0.05 * (X.max(axis=0) - X.min(axis=0))
feature_bound = [X.min(axis=0) - margin, X.max(axis=0) + margin]
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
colorbar = fig.colorbar(
    ScalarMappable(Normalize(0, 1), utility_cmap),
    ax=ax,
    ticks=[0, 1],
    fraction=0.04,
    pad=0.02,
)
colorbar.solids.set_alpha(utility_alpha)
colorbar.ax.set_yticklabels(["low", "high"])
colorbar.set_label("Utility", labelpad=-8)
colorbar.outline.set_visible(False)
ax_curve.set_title("Learning curve of this run", loc="left")
ax_curve.set_xlabel("Acquired labels")
ax_curve.set_ylabel("Test accuracy")
ax_curve.xaxis.set_major_locator(MaxNLocator(integer=True))
ax_curve.grid(axis="y", color="#e3e5e8")
ax_curve.margins(x=0.03, y=0.2)
artists = []
n_acquired, accuracies = [], []

# Active learning cycle:
n_cycles = "$n_cycles|20"
batch_size = "$batch_size|1"
for c in range(n_cycles):
    # Fit and evaluate the classifier with current labels.
    clf.fit(X, y)
    is_lbld = is_labeled(y, missing_label=MISSING_LABEL)
    n_acquired.append(np.sum(is_lbld))
    accuracies.append(clf.score(X_test, y_test))

    # Query the next sample(s).
    query_idx = qs.query("$query_params", batch_size=batch_size)

    # Capture the current plot state.
    coll_old = list(ax.collections)

    # Update plot with utility values, decision boundary, and samples.
    plot_utilities(
        qs,
        "$query_params",
        "$plot_utility_params|candidates=None",
        batch_size=batch_size,
        res="$res|25",
        feature_bound=feature_bound,
        ax=ax,
        contour_dict={"cmap": utility_cmap, "alpha": utility_alpha},
    )
    plot_decision_boundary(
        clf,
        feature_bound,
        ax=ax,
        cmap=class_cmap,
        boundary_dict={"colors": line_color},
    )
    ax.scatter(
        X[~is_lbld, 0],
        X[~is_lbld, 1],
        s=16,
        facecolors="none",
        edgecolors=class_cmap(y_pool[~is_lbld]),
        zorder=2,
    )
    ax.scatter(
        X[is_lbld, 0],
        X[is_lbld, 1],
        s=46,
        facecolors=class_cmap(y_pool[is_lbld]),
        edgecolors=line_color,
        zorder=3,
    )
    ax.scatter(
        X[query_idx, 0],
        X[query_idx, 1],
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

    # Extend the learning curve and label its newest point.
    (curve,) = ax_curve.plot(n_acquired, accuracies, c=line_color, lw=2)
    point = ax_curve.scatter(n_acquired[-1], accuracies[-1], c=line_color)
    value = ax_curve.annotate(
        f"{accuracies[-1]:.2f}",
        (n_acquired[-1], accuracies[-1]),
        xytext=(6, 0),
        textcoords="offset points",
        va="center",
    )

    coll_new = [x for x in ax.collections if x not in coll_old]
    artists.append(coll_new + [title, curve, point, value])

    # Update labels based on query.
    y[query_idx] = y_pool[query_idx]

ani = animation.ArtistAnimation(fig, artists, interval=1000, blit=True)
