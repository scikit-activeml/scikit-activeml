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
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.ticker import MaxNLocator
from sklearn.datasets import make_blobs

from skactiveml.utils import MISSING_LABEL, majority_vote, is_labeled
from skactiveml.visualization import plot_decision_boundary, mesh

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
# Initialise the classifier.
clf = "$init_clf|ParzenWindowClassifier(classes=[0, 1], class_prior=1e-3, metric_dict={'gamma': 3}, random_state=random_state)"

# Initialise the query strategy.
qs = "$init_qs"
"$preproc"
# Preparation for plotting.
class_cmap = ListedColormap(["#2a78d6", "#1baf7a"])
selection_color = "#eb6834"
line_color = "#222832"
utility_cmap, utility_alpha = "Purples", 0.35
plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

fig = plt.figure(figsize=(6.4, 8.4))
ax = fig.add_axes([0.11, 0.55, 0.75, 0.39])
ax_requests = fig.add_axes([0.11, 0.3, 0.34, 0.16])
ax_accuracy = fig.add_axes([0.52, 0.3, 0.34, 0.16], sharey=ax_requests)
ax_curve = fig.add_axes([0.11, 0.06, 0.75, 0.14])
cax = fig.add_axes([0.88, 0.55, 0.02, 0.39])
margin = 0.05 * (X.max(axis=0) - X.min(axis=0))
feature_bound = [X.min(axis=0) - margin, X.max(axis=0) + margin]
X_mesh, Y_mesh, mesh_samples = mesh(feature_bound, "$res|25")
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
colorbar = fig.colorbar(
    ScalarMappable(Normalize(0, 1), utility_cmap), cax=cax, ticks=[0, 1]
)
colorbar.solids.set_alpha(utility_alpha)
colorbar.ax.set_yticklabels(["low", "high"])
colorbar.set_label("Mean utility", labelpad=-8)
colorbar.outline.set_visible(False)
annotators = np.arange(n_annotators)
ax_requests.set_title("Requested labels", loc="left")
ax_requests.set_yticks(annotators, [f"{a + 1}" for a in annotators])
ax_requests.set_ylabel("Annotator")
ax_requests.invert_yaxis()
ax_requests.xaxis.set_major_locator(MaxNLocator(5, integer=True))
ax_accuracy.set_title("Annotator accuracy", loc="left")
ax_accuracy.tick_params(labelleft=False)
ax_accuracy.set_xlim(-0.03, 1.02)
ax_accuracy.grid(axis="x", color="#e3e5e8")
ax_accuracy.scatter(
    1 - annotator_error_prob, annotators, marker="|", s=250, c=line_color
)
ax_curve.set_title("Learning curve of this run", loc="left")
ax_curve.set_xlabel("Acquired labels")
ax_curve.set_ylabel("Test accuracy")
ax_curve.xaxis.set_major_locator(MaxNLocator(integer=True))
ax_curve.grid(axis="y", color="#e3e5e8")
ax_curve.margins(x=0.03)
ax_curve.set_ylim(0.3, 1.02)
artists = []
n_acquired, accuracies = [], []

# Active learning cycle:
n_cycles = "$n_cycles|20"
batch_size = "$batch_size|3"
for c in range(n_cycles):
    # Fit and evaluate the classifier.
    y_mv = majority_vote(y, missing_label=MISSING_LABEL, random_state=0)
    clf.fit(X, y_mv)
    is_lbld = is_labeled(y, missing_label=MISSING_LABEL)
    n_acquired.append(np.sum(is_lbld))
    accuracies.append(clf.score(X_test, y_true_test))

    # Estimate the annotator accuracies.
    agrees = is_lbld & (y_annot == clf.predict(X)[:, None])
    with np.errstate(invalid="ignore"):
        A_perf_estimate = agrees.sum(axis=0) / is_lbld.sum(axis=0)
    A_perf_clf = None if np.isnan(A_perf_estimate).any() else A_perf_estimate

    # Query the next sample-annotator pairs.
    query_idx = qs.query("$query_params", batch_size=batch_size)

    # Capture the current plot state.
    coll_old = list(ax.collections) + list(ax_accuracy.collections)
    patches_old = list(ax_requests.patches)

    # Update plot with the mean utilities, decision boundary, and samples.
    _, utilities = qs.query(
        "$query_params",
        batch_size=batch_size,
        return_utilities=True,
        candidates=mesh_samples,
    )
    ax.contourf(
        X_mesh,
        Y_mesh,
        np.mean(utilities[0], axis=1).reshape(X_mesh.shape),
        cmap=utility_cmap,
        alpha=utility_alpha,
    )
    plot_decision_boundary(
        clf,
        feature_bound,
        ax=ax,
        cmap=class_cmap,
        boundary_dict={"colors": line_color},
    )
    has_labels = is_lbld.any(axis=1)
    is_correct = has_labels & (y_mv == y_true)
    is_wrong = has_labels & (y_mv != y_true)
    ax.scatter(
        X[~has_labels, 0],
        X[~has_labels, 1],
        s=16,
        facecolors="none",
        edgecolors=class_cmap(y_true[~has_labels]),
        zorder=2,
    )
    for is_vote, marker, size in [(is_correct, "o", 46), (is_wrong, "X", 80)]:
        ax.scatter(
            X[is_vote, 0],
            X[is_vote, 1],
            s=size,
            marker=marker,
            facecolors=class_cmap(y_mv[is_vote].astype(int)),
            edgecolors=line_color,
            zorder=3,
        )
    selected = np.unique(query_idx[:, 0])
    ax.scatter(
        X[selected, 0],
        X[selected, 1],
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

    # Update the annotator plots.
    requests = is_lbld.sum(axis=0)
    new_requests = np.bincount(query_idx[:, 1], minlength=n_annotators)
    ax_requests.barh(annotators, requests, color="#b9bec6")
    ax_requests.barh(
        annotators, new_requests, left=requests, color=selection_color
    )
    ax_accuracy.scatter(A_perf_estimate, annotators, s=36, c=line_color)

    # Update the learning curve.
    (curve,) = ax_curve.plot(n_acquired, accuracies, c=line_color, lw=2)
    point = ax_curve.scatter(n_acquired[-1], accuracies[-1], c=line_color)
    value = ax_curve.annotate(
        f"{accuracies[-1]:.2f}",
        (n_acquired[-1], accuracies[-1]),
        xytext=(6, 0),
        textcoords="offset points",
        va="center",
    )

    coll_new = list(ax.collections) + list(ax_accuracy.collections)
    artists.append(
        [x for x in coll_new if x not in coll_old]
        + [x for x in ax_requests.patches if x not in patches_old]
        + [title, curve, point, value]
    )

    # Update labels based on query.
    y[query_idx[:, 0], query_idx[:, 1]] = y_annot[
        query_idx[:, 0], query_idx[:, 1]
    ]

ani = animation.ArtistAnimation(fig, artists, interval=1000)
