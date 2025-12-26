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
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

from skactiveml.utils import MISSING_LABEL, labeled_indices
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
clf = "$init_clf|ParzenWindowClassifier(classes=[0, 1], random_state=random_state)"

# Initialise the query strategy.
qs = "$init_qs"
"$preproc"
# Preparation for plotting.
fig, ax = plt.subplots()
feature_bound = [
    [min(X[:, 0]), min(X[:, 1])],
    [max(X[:, 0]), max(X[:, 1])]
]
artists = []

# Active learning cycle:
n_cycles = "$n_cycles|20"
for c in range(n_cycles):
    # Fit the classifier with current labels.
    clf.fit(X, y)

    # Query the next sample(s).
    query_idx = qs.query("$query_params")

    # Capture the current plot state.
    coll_old = list(ax.collections)
    title = ax.text(
        0.5, 1.05,
        f"Decision boundary after acquiring {c} labels\n"
        f"Test Accuracy: {clf.score(X_test, y_test):.4f}",
        size=plt.rcParams["axes.titlesize"],
        ha="center", transform=ax.transAxes,
    )

    # Update plot with utility values, samples, and decision boundary.
    X_labeled = X[labeled_indices(y)]
    ax = plot_utilities(
        qs,
        "$query_params",
        "$plot_utility_params|candidates=None",
        res="$res|25",
        feature_bound=feature_bound,
        ax=ax,
    )
    ax.scatter(
        X[:, 0],
        X[:, 1],
        c=y_pool,
        cmap="coolwarm",
        marker=".",
        zorder=2
    )
    ax.scatter(
        X_labeled[:, 0],
        X_labeled[:, 1],
        c="grey",
        alpha=0.8,
        marker=".",
        s=300,
    )
    ax = plot_decision_boundary(clf, feature_bound, ax=ax)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')

    coll_new = list(ax.collections)
    coll_new.append(title)
    artists.append([x for x in coll_new if x not in coll_old])

    # Update labels based on query.
    y[query_idx] = y_pool[query_idx]

ani = animation.ArtistAnimation(fig, artists, interval=1000, blit=True)

# %%
# .. image:: ../../examples/pool_classification_legend.png
