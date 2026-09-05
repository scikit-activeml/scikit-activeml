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
from skactiveml.visualization import mesh, plot_decision_boundary

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
fig, label_axes = plt.subplots(1, 3, figsize=(1.5 * 6.4, 0.85 * 4.8))
fig.subplots_adjust(top=0.75, wspace=0.3, left=0.075, right=0.975, bottom=0.15)
feature_bound = [
    [min(X[:, 0]), min(X[:, 1])],
    [max(X[:, 0]), max(X[:, 1])],
]
res = "$res|25"
X_mesh, Y_mesh, mesh_samples = mesh(feature_bound, res)
artists = []

for label_idx, label_ax in enumerate(label_axes):
    label_ax.set_title(f"Label {label_idx + 1}")
    label_ax.set_xlabel("Feature 1")
    label_ax.set_ylabel("Feature 2")

# Active learning cycle.
n_cycles = "$n_cycles|20"
for c in range(n_cycles):
    # Fit the classifier with the currently observed label vectors.
    clf.fit(X, y)

    # Query one complete label vector.
    query_idx = qs.query("$query_params")

    # Capture the current plot state.
    collections_before = [list(ax.collections) for ax in label_axes]
    title = label_axes[1].text(
        0.5,
        1.18,
        f"Active learning cycle {c + 1}/{n_cycles} "
        f"after acquiring {c} label vectors\n"
        f"Test exact-match accuracy: {clf.score(X_test, y_test):.4f}",
        ha="center",
        va="bottom",
        fontsize=14,
        transform=label_axes[1].transAxes,
    )

    # Evaluate the single per-sample acquisition utility once and reuse the
    # resulting background for every label output.
    _, utilities = qs.query(
        "$query_params",
        candidates=mesh_samples,
        return_utilities=True,
    )
    utility_surface = utilities[0].reshape(X_mesh.shape)

    X_labeled = X[
        labeled_indices(
            y,
            missing_label=MISSING_LABEL,
            target_type="multi-label",
        )
    ]
    for label_idx, label_ax in enumerate(label_axes):
        label_ax.contourf(
            X_mesh,
            Y_mesh,
            utility_surface,
            cmap="Greens",
            alpha=0.75,
        )
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
            zorder=3,
        )

    # Plot one black decision boundary per label output.
    plot_decision_boundary(
        clf,
        feature_bound,
        ax=label_axes,
        res=res,
        boundary_dict={"colors": "black"},
        confidence=0.75,
    )

    new_artists = [title]
    for ax, old_collections in zip(label_axes, collections_before):
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
