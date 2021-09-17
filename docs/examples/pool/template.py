import numpy as np
from matplotlib import pyplot as plt, animation
from sklearn.datasets import make_classification
from sklearn.exceptions import NotFittedError
from skactiveml.utils import MISSING_LABEL, is_unlabeled
from skactiveml.visualization import plot_utility, plot_decision_boundary
#_ import

random_state = np.random.RandomState(0)

# Build a dataset.
X, y_true = make_classification(n_features=2, n_redundant=0,
                                random_state=random_state)
y = np.full(shape=y_true.shape, fill_value=MISSING_LABEL)
# Initialise the classifier.
clf = "#_init_clf"
# Initialise the query strategy.
qs = "#_init_qs"

# Preparation for plotting.
fig, ax = plt.subplots()
x1_min = min(X[:, 0])
x1_max = max(X[:, 0])
x2_min = min(X[:, 1])
x2_max = max(X[:, 1])
bound = [[x1_min, x2_min], [x1_max, x2_max]]
artists = []

# The active learning cycle:
n_cycles = 20
for c in range(n_cycles):
    # Set X_cand to the unlabeled instances.
    unlbld_idx = np.where(is_unlabeled(y))[0]
    X_cand = X[unlbld_idx]

    # Query the next instance/s.
    query_idx = unlbld_idx[qs.query(X_cand)]  #_43 query_params

    # Plot the labeled data.
    coll_old = list(ax.collections)
    title = ax.text(
        0.5, 1.05, f"Decision boundry after acquring {c} labels",
        size=plt.rcParams["axes.titlesize"], ha="center",
        transform=ax.transAxes
    )
    ax = plot_utility(qs, bound=bound, ax=ax)  #_25 {query_params}
    ax.scatter(X_cand[:, 0], X_cand[:, 1], c="k", marker=".")
    ax.scatter(X[:, 0], X[:, 1], c=-y, cmap="coolwarm_r", alpha=.9, marker=".")
    try:
        ax = plot_decision_boundary(clf, bound, ax=ax)
    except NotFittedError:
        pass
    coll_new = list(ax.collections)
    coll_new.append(title)
    artists.append([x for x in coll_new if (x not in coll_old)])

    # Label the queried instances.
    y[query_idx] = y_true[query_idx]

    # Fit the classifier.
    clf.fit(X, y)

ani = animation.ArtistAnimation(fig, artists, blit=True)
