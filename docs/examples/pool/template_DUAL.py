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
#.. raw:: html
#
#   <hr style="border-style: solid; border-top: 1px solid; border-right: 0; border-bottom: 0; border-left: 0;">
#

# %%
import numpy as np
from matplotlib import pyplot as plt, animation
from sklearn.datasets import make_blobs

from skactiveml.utils import MISSING_LABEL, is_labeled, simple_batch
from skactiveml.visualization import plot_decision_boundary, \
    plot_contour_for_samples

from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from skactiveml.classifier import SklearnClassifier
from skactiveml.pool import UncertaintySampling

random_state = np.random.RandomState(0)

# Build a dataset.
X, y_true = make_blobs(n_samples="$n_samples|200", n_features=2,
                       centers=[[0, 1], [-3, .5], [-1, -1], [2, 1], [1, -.5]],
                       cluster_std=.7, random_state=random_state)
y_true = y_true % 2
y = np.full(shape=y_true.shape, fill_value=MISSING_LABEL)

# Initialise the classifier.
clf = SklearnClassifier(LogisticRegression(), classes=np.unique(y_true))
# Initialise the query strategy.
qs = UncertaintySampling(method='least_confident', random_state=random_state)
gmm = GaussianMixture(init_params='kmeans', n_components=5)
gmm.fit(X)
density = np.exp(gmm.score_samples(X))
delta = 0.1
u_max = -np.inf
switching_point = False

# Preparation for plotting.
fig, ax = plt.subplots()
feature_bound = [[min(X[:, 0]), min(X[:, 1])], [max(X[:, 0]), max(X[:, 1])]]
artists = []

# The active learning cycle:
n_cycles = "$n_cycles|20"
for c in range(n_cycles):
    # Fit the classifier.
    clf.fit(X, y)

    # Get labeled samples.
    X_labeled = X[is_labeled(y)]

    # Query the next sample(s).
    if not switching_point:
        # DWUS
        query_idx, utils = qs.query(X=X, y=y, clf=clf, utility_weight=density, return_utilities=True)
        utilities = utils[0]
        switching_point = utilities[query_idx[0]] - u_max < delta
        u_max = utilities[query_idx[0]]
        strategy = "DWUS"
    else:
        # DWUS + US
        utils_US = qs.query(X=X, y=y, clf=clf, return_utilities=True)[1][0]
        err = np.nanmean(utils_US)
        utilities = (1-err)*utils_US + err*density
        query_idx = simple_batch(utilities, random_state)
        strategy = "DWUS + US"

    # Plot the labeled data.
    coll_old = list(ax.collections)
    title = ax.text(
        0.5, 1.05, f"Decision boundary after acquring {c} labels with {strategy}",
        size=plt.rcParams["axes.titlesize"], ha="center",
        transform=ax.transAxes
    )
    ax = plot_contour_for_samples(X, utilities, feature_bound=feature_bound,
                                  res=31, ax=ax, replace_nan=None)
    ax.scatter(X[:, 0], X[:, 1], c=y_true, cmap="coolwarm", marker=".",
               zorder=2)
    ax.scatter(X_labeled[:, 0], X_labeled[:, 1], c="grey", alpha=.8,
               marker=".", s=300)
    ax = plot_decision_boundary(clf, feature_bound, ax=ax)

    coll_new = list(ax.collections)
    coll_new.append(title)
    artists.append([x for x in coll_new if (x not in coll_old)])

    # Label the queried samples.
    y[query_idx] = y_true[query_idx]

ani = animation.ArtistAnimation(fig, artists, interval=1000, blit=True)
