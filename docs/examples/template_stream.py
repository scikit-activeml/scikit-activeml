import numpy as np
from matplotlib import pyplot as plt, animation
from sklearn.datasets import make_blobs

from skactiveml.utils import MISSING_LABEL, labeled_indices
from skactiveml.visualization import (
    plot_stream_utilities,
    plot_decision_boundary
)

"$import_clf|from skactiveml.classifier import ParzenWindowClassifier"
"$import_misc"

random_state = np.random.RandomState(0)

# Build a dataset.
X, y_true = make_blobs(
    n_samples="$n_samples|200",
    n_features=2,
    centers=[[0, 1], [-3, 0.5], [-1, -1], [2, 1], [1, -0.5]],
    cluster_std=0.7,
    random_state=random_state,
)
y_true = y_true % 2
init_size = "$init_size|2"
X_init = X[:init_size, :]
y_init = y_true[:init_size]
X_stream = X[init_size:, :]
y_stream = y_true[init_size:]


# Initialise the classifier.
clf = "$init_clf|ParzenWindowClassifier(classes=[0, 1], random_state=random_state)"
# Initialise the query strategy.
qs = "$init_qs"
"$preproc"

# Preparation for plotting.
fig, ax = plt.subplots()
feature_bound = [[min(X[:, 0]), min(X[:, 1])], [max(X[:, 0]), max(X[:, 1])]]
artists = []

feature_bound = [[min(X[:, 0]), min(X[:, 1])], [max(X[:, 0]), max(X[:, 1])]]
artists = []

X_train = []
X_train.extend(X_init)
y_train = []
y_train.extend(y_init)
queried_count = 0
budget_list = []

for t, (x_t, y_t) in enumerate(zip(X_stream, y_stream)):
    X_cand = x_t.reshape([1, -1])
    y_cand = y_t
    clf.fit(X_train, y_train)
    # Get labeled instances.
    X_labeled = np.array(X_train)[labeled_indices(y_train)]
    # check whether to sample the instance or not
    sampled_indices, utilities = qs.query(
        "$query_params", return_utilities=True
    )

    # Plot the labeled data.
    coll_old = list(ax.collections)
    ax = plot_stream_utilities(
        qs,
        "$query_params",
        "$plot_possible_queries|plot_annotators=None",
        res="$res|25",
        feature_bound=feature_bound,
        ax=ax,
    )
    ax.scatter(
        np.array(X_train)[:, 0],
        np.array(X_train)[:, 1],
        c=y_true[: t + init_size],
        cmap="coolwarm",
        marker=".",
        zorder=2,
    )
    ax.scatter(
        X_labeled[:, 0],
        X_labeled[:, 1],
        c="grey",
        alpha=0.8,
        marker=".",
        s=300,
    )
    ax.scatter(
        X_stream[t, 0], X_stream[t, 1], c="grey", alpha=0.8, marker="*", s=300,
    )
    ax = plot_decision_boundary(clf, feature_bound, ax=ax)

    queried_count += len(sampled_indices)
    budget_list.append(queried_count / (t + 1))
    title_string = (f"Decision boundary after {t} new instances \n" +
                    f"with utility: {utilities[0]: .4f} " +
                    f"budget is {budget_list[-1]:.4f}")
    title = ax.text(
        x=0.5,
        y=1.05,
        s=title_string,
        size=plt.rcParams["axes.titlesize"],
        ha="center",
        transform=ax.transAxes,
    )
    coll_new = list(ax.collections)
    coll_new.append(title)
    artists.append([x for x in coll_new if (x not in coll_old)])

    # update the query strategy and budget_manager to calculate the right budget
    qs.update("$update_params")
    # Label the queried instances.
    X_train.append(x_t)
    if len(sampled_indices):
        y_train.append(y_t)
    else:
        y_train.append(MISSING_LABEL)

ani = animation.ArtistAnimation(fig, artists, interval=500, blit=True)
