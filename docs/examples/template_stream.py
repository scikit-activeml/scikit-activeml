# %%
# .. note::
# 
#    The generated animation can be found at the bottom of the page.
import numpy as np
from matplotlib import pyplot as plt, animation
from sklearn.datasets import make_blobs

from skactiveml.utils import MISSING_LABEL
from skactiveml.visualization import (
    plot_stream_training_data,
    plot_stream_decision_boundary,
)

"$import_clf|from skactiveml.classifier import ParzenWindowClassifier"
"$import_misc"

random_state = np.random.RandomState(0)
init_size = "$init_size|0"
# Build a dataset.
X, y_true = make_blobs(
    n_samples="$n_samples|200" + init_size,
    n_features=1,
    centers=[[0], [-3], [1], [2], [-0.5]],
    cluster_std=0.7,
    random_state=random_state,
)
y_true = y_true % 2
X_init = X[:init_size]
y_init = y_true[:init_size]
X_stream = X[init_size:]
y_stream = y_true[init_size:]


# Initialise the classifier.
clf = "$init_clf|ParzenWindowClassifier(classes=[0, 1], random_state=random_state)"
# Initialise the query strategy.
qs = "$init_qs"
plot_step = "$init_plot_step|5"
"$preproc"

X_train = []
X_train.extend(X_init)
y_train = []
y_train.extend(y_init)
classes = np.unique(y_true)


# Preparation for plotting.
fig, ax = plt.subplots()

feature_bound = [[0, len(X)], [min(X), max(X)]]
ax.set_xlim(0, len(X))
ax.set_ylim(bottom=min(X), top=max(X))
artists = []

X_train = []
X_train.extend(X_init)
y_train = []
y_train.extend(y_init)

queried_indices = [True] * len(y_init)

predictions_list = []

for t_x, (x_t, y_t) in enumerate(zip(X_stream, y_stream)):

    X_cand = x_t.reshape([1, -1])
    y_cand = y_t
    clf.fit(X_train, y_train)
    # Check whether to sample the instance or not
    sampled_indices, utilities = qs.query(
        "$query_params", return_utilities=True
    )
    budget_manager_param_dict = {"utilities": utilities}
    # Update the query strategy and budget_manager to calculate the right budget
    qs.update("$update_params")
    # Label the queried instances.
    X_train.append(x_t)
    if len(sampled_indices):
        y_train.append(y_t)
    else:
        y_train.append(MISSING_LABEL)

    queried_indices.append(len(sampled_indices) > 0)

    # Plot the labeled data.
    if t_x % plot_step == 0:

        coll_old = list(ax.collections)
        ax, predictions_list = plot_stream_decision_boundary(
            ax, t_x, plot_step, clf, X, predictions_list, res="$res|25"
        )
        data_lines = plot_stream_training_data(
            ax, X_train, y_train, queried_indices, classes, feature_bound
        )

        title_string = (
            f"Decision boundary after {t_x} new instances \n"
            + f"with utility: {utilities[0]: .4f} "
            + f"budget is {sum(queried_indices) / (t_x + 1):.4f}"
        )
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

        artists.append(
            [x for x in coll_new if (x not in coll_old)] + data_lines
        )

ani = animation.ArtistAnimation(fig, artists, interval=500, blit=True)

# %%
# .. image:: ../../examples/stream_classification_legend.png
