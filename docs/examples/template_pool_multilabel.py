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
from matplotlib import animation, pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

from skactiveml.classifier import ParzenWindowClassifier
from skactiveml.utils import MISSING_LABEL, labeled_indices
from skactiveml.visualization import plot_decision_boundary, plot_utilities

"$import_misc"

random_state = np.random.RandomState(0)

# Build a dataset. Feature noise comes from each blob's spread, while the
# three binary labels are deterministic functions of the generating cluster.
X_true, y_clusters = make_blobs(
    n_samples="$n_samples|400",
    n_features=2,
    centers=[[0, 1], [-3, 0.5], [-1, -1], [2, 1], [1, -0.5]],
    cluster_std=0.7,
    random_state=random_state,
)
cluster_labels = np.array(
    [
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 1, 1],
    ]
)
y_true = cluster_labels[y_clusters]

X_pool, X_test, y_pool, y_test = train_test_split(
    X_true, y_true, test_size=0.25, random_state=random_state
)

X = X_pool
y = np.full(shape=y_pool.shape, fill_value=MISSING_LABEL)

# Initialise a native multi-label classifier.
clf = ParzenWindowClassifier(
    classes=[[0, 1]] * 3,
    class_prior=1e-3,
    metric_dict={"gamma": 3},
    target_type="multi-label",
    random_state=random_state,
)

# Initialise the query strategy.
qs = "$init_qs"
"$preproc"

# Preparation for plotting.
fig, axs = plt.subplots(2, 2, figsize=(1.5 * 6.4, 1.5 * 4.8))
fig.subplots_adjust(
    top=0.875, hspace=0.3, left=0.075, right=0.975, bottom=0.075
)
axes = axs.flatten()
label_axes = axes[:3]
utility_ax = axes[3]
feature_bound = [
    [min(X[:, 0]), min(X[:, 1])],
    [max(X[:, 0]), max(X[:, 1])],
]
artists = []

for label_idx, label_ax in enumerate(label_axes):
    label_ax.set_title(f"Label {label_idx + 1}")
    label_ax.set_xlabel("Feature 1")
    label_ax.set_ylabel("Feature 2")
utility_ax.set_title("Acquisition utility")
utility_ax.set_xlabel("Feature 1")
utility_ax.set_ylabel("Feature 2")

# Active learning cycle.
n_cycles = "$n_cycles|20"
for c in range(n_cycles):
    # Fit the classifier with the currently observed label vectors.
    clf.fit(X, y)

    # Query one complete label vector.
    query_idx = qs.query("$query_params")

    # Capture the current plot state.
    collections_before = [list(ax.collections) for ax in axes]
    title = fig.text(
        0.5,
        0.98,
        f"Decision boundaries and utility after acquiring {c} labels\n"
        f"Test Accuracy: {clf.score(X_test, y_test):.4f}",
        ha="center",
        va="top",
        fontsize=14,
    )

    # Plot one decision boundary and one binary target view per label output.
    plot_decision_boundary(
        clf,
        feature_bound,
        ax=label_axes,
        res="$res|25",
        confidence=None,
    )
    X_labeled = X[
        labeled_indices(
            y,
            missing_label=MISSING_LABEL,
            target_type="multi-label",
        )
    ]
    for label_idx, label_ax in enumerate(label_axes):
        label_ax.scatter(
            X[:, 0],
            X[:, 1],
            c=y_pool[:, label_idx],
            cmap="coolwarm",
            marker=".",
            zorder=2,
        )
        label_ax.scatter(
            X_labeled[:, 0],
            X_labeled[:, 1],
            c="grey",
            alpha=0.8,
            marker=".",
            s=300,
        )

    # Plot the single per-sample acquisition utility shared by all labels.
    plot_utilities(
        qs,
        "$query_params",
        "$plot_utility_params|candidates=None",
        res="$res|25",
        feature_bound=feature_bound,
        ax=utility_ax,
    )

    new_artists = [title]
    for ax, old_collections in zip(axes, collections_before):
        new_artists.extend(
            collection
            for collection in ax.collections
            if collection not in old_collections
        )
    artists.append(new_artists)

    # Observe the complete label vector selected in this cycle.
    y[query_idx] = y_pool[query_idx]

ani = animation.ArtistAnimation(fig, artists, interval=1000, blit=True)

# %%
# .. image:: ../../examples/pool_classification_legend.png
