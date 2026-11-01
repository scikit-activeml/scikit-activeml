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
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.ticker import MaxNLocator
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

from skactiveml.classifier import ParzenWindowClassifier
from skactiveml.utils import MISSING_LABEL, is_labeled
from skactiveml.visualization import mesh, plot_decision_boundary

"$import_misc"

random_state = np.random.RandomState(0)

# Build a dataset with three binary labels per sample.
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
class_cmap = ListedColormap(["#2a78d6", "#1baf7a"])
selection_color = "#eb6834"
line_color = "#222832"
utility_cmap, utility_alpha = "Purples", 0.5
plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

fig, axs = plt.subplots(2, 2, figsize=(6.4, 6.4))
fig.subplots_adjust(
    left=0.11, right=0.86, top=0.9, bottom=0.09, hspace=0.4, wspace=0.42
)
label_axes, ax_curve = [axs[0, 0], axs[0, 1], axs[1, 0]], axs[1, 1]
for label_ax in label_axes[1:]:
    label_ax.sharex(label_axes[0])
    label_ax.sharey(label_axes[0])
label_axes[0].tick_params(labelbottom=False)
label_axes[1].tick_params(labelleft=False)
label_axes[0].set_ylabel("Feature 2")
label_axes[2].set_ylabel("Feature 2")
label_axes[1].set_xlabel("Feature 1")
label_axes[2].set_xlabel("Feature 1")
colorbar = fig.colorbar(
    ScalarMappable(Normalize(0, 1), utility_cmap),
    cax=fig.add_axes([0.88, 0.55, 0.02, 0.35]),
    ticks=[0, 1],
)
colorbar.solids.set_alpha(utility_alpha)
colorbar.ax.set_yticklabels(["low", "high"])
colorbar.set_label("Utility", labelpad=-8)
colorbar.outline.set_visible(False)
margin = 0.05 * (X.max(axis=0) - X.min(axis=0))
feature_bound = [X.min(axis=0) - margin, X.max(axis=0) + margin]
res = "$res|25"
X_mesh, Y_mesh, mesh_samples = mesh(feature_bound, res)
for label_idx, label_ax in enumerate(label_axes):
    label_ax.set_title(f"Label {label_idx + 1}", loc="left")
ax_curve.set_title("Learning curve of this run", loc="left")
ax_curve.set_xlabel("Acquired label vectors")
ax_curve.set_ylabel("Exact-match accuracy")
ax_curve.xaxis.set_major_locator(MaxNLocator(integer=True))
ax_curve.grid(axis="y", color="#e3e5e8")
ax_curve.margins(x=0.03)
ax_curve.set_ylim(-0.05, 1.05)
artists = []
n_acquired, accuracies = [], []

# Active learning cycle.
n_cycles = "$n_cycles|20"
batch_size = "$batch_size|1"
for c in range(n_cycles):
    # Fit and evaluate the classifier.
    clf.fit(X, y)
    is_lbld = is_labeled(y, missing_label=MISSING_LABEL).all(axis=1)
    n_acquired.append(np.sum(is_lbld))
    accuracies.append(clf.score(X_test, y_test))

    # Query one complete label vector.
    query_idx = qs.query("$query_params", batch_size=batch_size)

    # Capture the current plot state.
    coll_old = [x for ax in label_axes for x in ax.collections]

    # Compute the utilities shared by all labels.
    _, utilities = qs.query(
        "$query_params",
        batch_size=batch_size,
        candidates=mesh_samples,
        return_utilities=True,
    )
    utility_surface = utilities[0].reshape(X_mesh.shape)

    # Update the panel of each label.
    for label_idx, label_ax in enumerate(label_axes):
        label_ax.contourf(
            X_mesh,
            Y_mesh,
            utility_surface,
            cmap=utility_cmap,
            alpha=utility_alpha,
        )
        label_ax.scatter(
            X[~is_lbld, 0],
            X[~is_lbld, 1],
            s=10,
            facecolors="none",
            edgecolors=class_cmap(y_pool[~is_lbld, label_idx]),
            zorder=2,
        )
        label_ax.scatter(
            X[is_lbld, 0],
            X[is_lbld, 1],
            s=30,
            facecolors=class_cmap(y_pool[is_lbld, label_idx]),
            edgecolors=line_color,
            zorder=3,
        )
        label_ax.scatter(
            X[query_idx, 0],
            X[query_idx, 1],
            s=180,
            facecolors=(selection_color, 0.28),
            edgecolors=selection_color,
            linewidths=2,
            zorder=4,
        )

    # Plot one decision boundary per label.
    plot_decision_boundary(
        clf,
        feature_bound,
        ax=label_axes,
        res=res,
        cmap=class_cmap,
        boundary_dict={"colors": line_color},
    )
    title = fig.text(
        0.11,
        0.985,
        f"Cycle {c + 1}: {n_acquired[-1]} label vectors acquired",
        size=plt.rcParams["axes.titlesize"],
        va="top",
    )

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

    coll_new = [
        x for ax in label_axes for x in ax.collections if x not in coll_old
    ]
    artists.append(coll_new + [title, curve, point, value])

    # Observe the complete label vector selected in this cycle.
    y[query_idx] = y_pool[query_idx]

ani = animation.ArtistAnimation(fig, artists, interval=1000)
