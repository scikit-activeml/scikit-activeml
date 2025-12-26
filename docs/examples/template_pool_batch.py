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
from skactiveml.visualization import (
    plot_decision_boundary,
    plot_contour_for_samples,
)

"$import_clf|from skactiveml.classifier import ParzenWindowClassifier"
"$import_misc"

# Set a fixed random state for reproducibility.
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
# Preparation for plotting: create a 2x2 grid of subplots.
fig, axs = plt.subplots(2, 2, figsize=(1.5*6.4, 1.5*4.8))
fig.subplots_adjust(top=0.875, hspace=0.3, left=0.075, right=0.975, bottom=0.075)
feature_bound = [[min(X[:, 0]), min(X[:, 1])], [max(X[:, 0]), max(X[:, 1])]]
artists = [[] for j in range("$n_cycles|5")]

# Active learning cycle.
n_cycles = "$n_cycles|5"
for c in range(n_cycles):
    # Train the classifier with the current labels.
    clf.fit(X, y)

    # Query the next batch of samples; retrieve both indices and utility values.
    query_idx, utilities = qs.query("$query_params", batch_size=4, return_utilities=True)

    title = fig.text(
        0.5,
        0.98,
        f"Decision boundary and utilities after acquiring {c} labels\n"
        f"Test Accuracy: {clf.score(X_test, y_test):.4f}",
        ha='center',
        va='top',
        fontsize=14
    )
    artists[c].append(title)
    # Plot results on each subplot.
    for i, ax in enumerate(axs.flatten()):
        # Save current collections to identify new plot elements.
        coll_old = list(ax.collections)

        # Plot the utility contour for the current subplot.
        plot_contour_for_samples(
            X,
            utilities[i],
            res="$res|25",
            feature_bound=feature_bound,
            replace_nan=None,
            ax=ax,
        )
        # Scatter all samples with true labels.
        ax.scatter(X[:, 0], X[:, 1], c=y_pool, cmap="coolwarm", marker=".", zorder=2)
        # Highlight the labeled samples.
        X_labeled = X[labeled_indices(y)]
        ax.scatter(
            X_labeled[:, 0],
            X_labeled[:, 1],
            c="grey",
            alpha=0.8,
            marker=".",
            s=300,
        )
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        # Overlay the decision boundary.
        ax = plot_decision_boundary(clf, feature_bound, ax=ax)
        # Set the title indicating the current batch and subplot index.
        ax.set_title(f"Batch {c+1}, Utilities[{i}]")

        # Collect new artists (plot elements) added during this cycle.
        for x in ax.collections:
            if x not in coll_old:
                artists[c].append(x)

    # Update the labels for the queried samples.
    y[query_idx] = y_pool[query_idx]

# Create the animation using the collected artists.
ani = animation.ArtistAnimation(fig, artists, interval=1000)

# %%
# .. image:: ../../examples/pool_classification_legend.png
