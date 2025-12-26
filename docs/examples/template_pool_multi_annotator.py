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
# ---

# %%
import numpy as np
from matplotlib import pyplot as plt, animation
from matplotlib.ticker import MaxNLocator
from sklearn.datasets import make_blobs

from skactiveml.utils import (
    MISSING_LABEL,
    majority_vote,
    is_labeled,
)
from skactiveml.visualization import (
    plot_utilities,
    plot_decision_boundary,
    mesh,
)

"$import_clf|from skactiveml.classifier import ParzenWindowClassifier"
"$import_misc"

random_state = np.random.RandomState(0)
rng = np.random.default_rng(seed=0)
# Build a dataset.
X_all, y_true_all = make_blobs(
    n_samples="$n_samples|400",
    n_features=2,
    centers=[[0, 1], [-3, 0.5], [-1, -1], [2, 1], [1, -0.5]],
    cluster_std=0.7,
    random_state=random_state,
)
X, X_test = X_all[: len(X_all) // 2], X_all[len(X_all) // 2 :]
y_true_all = y_true_all % 2
y_true, y_true_test = (
    y_true_all[: len(X_all) // 2],
    y_true_all[len(X_all) // 2 :],
)
n_annotators = 5
y_annot = np.zeros(shape=(len(X), n_annotators), dtype=int)
annotator_error_prob = np.linspace(0.0, 0.3, num=n_annotators)
for i, p in enumerate(annotator_error_prob):
    y_noise = rng.binomial(1, p, len(X))
    y_annot[:, i] = y_noise ^ y_true
y = np.full(shape=y_annot.shape, fill_value=MISSING_LABEL)
y_mv = majority_vote(y, missing_label=MISSING_LABEL, random_state=random_state)
# Initialise the classifier.
clf = "$init_clf|ParzenWindowClassifier(classes=[0, 1], random_state=random_state)"

# Initialise the query strategy.
qs = "$init_qs"
"$preproc"
# Preparation for plotting.
fig = plt.figure(figsize=(7, 5))
ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
ax2 = plt.subplot2grid((3, 1), (2, 0), rowspan=1)
feature_bound = [[min(X[:, 0]), min(X[:, 1])], [max(X[:, 0]), max(X[:, 1])]]
artists = []

# Active learning cycle:
n_cycles = "$n_cycles|20"
for c in range(n_cycles):
    # Fit the classifier with current labels.
    clf.fit(X, y_mv)

    # Fit the annotation performance model
    if np.all(np.any(is_labeled(y), axis=0)):
        A_perf_clf = np.sum(
            np.where(is_labeled(y), y_annot == clf.predict(X)[:, None], 0),
            axis=0,
        ) / np.sum(is_labeled(y), axis=0)
    else:
        A_perf_clf = None

    A_perf_clf_individual = np.full(n_annotators, np.nan)
    has_labels = np.any(is_labeled(y), axis=0)
    A_perf_clf_individual[has_labels] = np.sum(
        np.where(
            is_labeled(y)[:, has_labels],
            y_annot[:, has_labels] == clf.predict(X)[:, None],
            0,
        ),
        axis=0,
    ) / np.sum(is_labeled(y)[:, has_labels], axis=0)

    # Query the next sample(s).
    query_idx = qs.query("$query_params")

    # Capture the current plot state.
    coll_old = list(ax1.collections) + list(ax2.collections)
    title = ax1.text(
        0.5,
        1.05,
        f"Decision boundary after acquiring {c} labels\n"
        f"Test Accuracy: {clf.score(X_test, y_true_test):.4f}",
        size=plt.rcParams["axes.titlesize"],
        ha="center",
        transform=ax1.transAxes,
    )

    y_mv = majority_vote(y, random_state=0)
    is_labeled_sample = np.any(is_labeled(y), axis=1)
    is_correctly_labeled_sample = is_labeled_sample & (y_mv == y_true)
    is_wrongly_labeled_sample = is_labeled_sample & (y_mv != y_true)

    axes = [ax1, ax2]
    # axes = plot_annotator_utilities(ma_qs, X=X, y=y, clf=clf, axes=axes, feature_bound=bound)
    X_mesh, Y_mesh, mesh_samples = mesh(feature_bound, 25)
    _, utilities = qs.query(
        "$query_params", return_utilities=True, candidates=mesh_samples
    )
    ax1.contourf(
        X_mesh,
        Y_mesh,
        np.mean(utilities[0], axis=1).reshape(X_mesh.shape),
        **{"cmap": "Greens", "alpha": 0.75},
    )
    # for a in range(n_annotators):
    plot_decision_boundary(clf, ax=ax1, feature_bound=feature_bound)
    ax1.scatter(
        X[~is_labeled_sample, 0],
        X[~is_labeled_sample, 1],
        c=y_true[~is_labeled_sample],
        cmap="coolwarm",
        marker=".",
        zorder=2,
        s=10,
    )
    ax1.scatter(
        X[is_correctly_labeled_sample, 0],
        X[is_correctly_labeled_sample, 1],
        c=y_mv[is_correctly_labeled_sample],
        cmap="coolwarm",
        marker="o",
        s=20,
        zorder=100,
        vmin=0,
        vmax=1,
    )
    ax1.scatter(
        X[is_wrongly_labeled_sample, 0],
        X[is_wrongly_labeled_sample, 1],
        c=y_mv[is_wrongly_labeled_sample],
        cmap="coolwarm",
        marker="x",
        s=20,
        zorder=100,
        vmin=0,
        vmax=1,
    )
    ax1.scatter(
        X[is_labeled_sample, 0],
        X[is_labeled_sample, 1],
        c="grey",
        alpha=0.8,
        marker=".",
        edgecolors="black",
        s=300,
    )
    ax1.set_xlabel("Feature 1")
    ax1.set_ylabel("Feature 2")

    requests_per_annotator = np.sum(is_labeled(y), axis=0)
    bar_labels = ax2.bar(
        np.arange(n_annotators),
        requests_per_annotator,
        width=0.4,
        color="grey",
    )

    ax2.set_xlabel("Annotators")
    ax2.set_xticks(
        np.arange(n_annotators),
        [f"(AP={1-ep})" for ep in annotator_error_prob],
    )
    ax2.set_ylabel("Requested Labels")
    text_elements = []
    for i in range(n_annotators):
        if not np.isnan(A_perf_clf_individual[i]):
            text = ax2.text(
                i,
                requests_per_annotator[i] + 0.1,
                r"($\widehat{\text{AP}}$=" + f"{A_perf_clf_individual[i]:.2})",
                horizontalalignment="center",
                color="black",
                fontsize=10,
            )
            text_elements.append(text)
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))

    coll_new = list(ax1.collections) + list(ax2.collections)
    coll_new.append(title)
    artists.append(
        [x for x in coll_new if x not in coll_old]
        + bar_labels.get_children()
        + text_elements
    )

    # Update labels based on query.
    y[query_idx[:, 0], query_idx[:, 1]] = y_annot[
        query_idx[:, 0], query_idx[:, 1]
    ]
lower_y_limit, upper_y_limit = ax2.get_ylim()
ax2.set_ylim((lower_y_limit, upper_y_limit * 1.2))
ani = animation.ArtistAnimation(fig, artists, interval=1000, blit=True)
# %%
# .. image:: ../../examples/pool_multi_annotator_legend.png
