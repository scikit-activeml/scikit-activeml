import numpy as np
import torch

from datasets import load_dataset
from matplotlib import pyplot as plt, animation
from matplotlib.lines import Line2D
from PIL import Image
from sentence_transformers import SentenceTransformer
from skactiveml.classifier import SkorchClassifier
from skactiveml.pool import Badge
from skactiveml.utils import is_labeled
from sklearn.base import clone
from sklearn.manifold import TSNE
from skorch.callbacks import LRScheduler
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR

# Define the device depending on its availability.
device = "cuda" if torch.cuda.is_available() else "cpu"

# Seed the initialization and training of the network.
torch.manual_seed(0)

# Load data from Huggingface and encode it via `sentence_transformers`.
ds_train = load_dataset("yangwang825/reuters-21578", split="train")
ds_test = load_dataset("yangwang825/reuters-21578", split="test")
mdl = SentenceTransformer("all-MiniLM-L6-v2", device=device)
X_pool = mdl.encode(ds_train["text"])
y_pool = np.asarray(ds_train["label"], dtype=np.int64)
X_test = mdl.encode(ds_test["text"])
y_test = np.asarray(ds_test["label"], dtype=np.int64)
n_features, classes = X_pool.shape[1], np.unique(y_pool)
missing_label = -1


# Build your `torch` module for classification, which outputs:
# - classification logits,
# - learned sample embeddings.
class ClassificationModule(nn.Module):
    def __init__(self, n_features, n_classes, n_hidden_units):
        super().__init__()
        self.linear_1 = nn.Linear(n_features, n_hidden_units)
        self.linear_2 = nn.Linear(n_hidden_units, n_classes)
        self.activation = nn.ReLU()

    def forward(self, x):
        x_embed = self.linear_1(x)
        logits = self.linear_2(self.activation(x_embed))
        return logits, x_embed


# Wrap your torch module via a `skactiveml` wrapper, which requires the
# definition of training parameters.
clf = SkorchClassifier(
    module=ClassificationModule,
    criterion=nn.CrossEntropyLoss,
    forward_outputs={"proba": (0, nn.Softmax(dim=-1)), "emb": (1, None)},
    neural_net_param_dict={
        # Module-related parameters.
        "module__n_features": n_features,
        "module__n_hidden_units": 128,
        "module__n_classes": len(classes),
        # Optimizer-related parameters.
        "max_epochs": 100,
        "batch_size": 16,
        "lr": 0.01,
        "optimizer": torch.optim.RAdam,
        "callbacks": [
            ("lr_scheduler", LRScheduler(policy=CosineAnnealingLR, T_max=100))
        ],
        # General parameters.
        "verbose": 0,
        "device": device,
        "train_split": False,
        "iterator_train__shuffle": True,
    },
    classes=classes,
    missing_label=missing_label,
).initialize()

# Start the active learning cycle with zero initial labels.
y_train = np.full_like(y_pool, missing_label)

# Create a deep active learning query strategy.
qs = Badge(
    missing_label=missing_label,
    clf_embedding_flag_name={"extra_outputs": "emb"},
    random_state=0,
)

# Define the active learning parameters.
n_cycles = 15
batch_size = 4

# Plotting stuff.
X_tsne = TSNE(n_components=2, random_state=0).fit_transform(X_pool)
tab10 = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#bcbd22",  # olive
    "#17becf",  # cyan
    "#000000",  # black
]
unlabeled_color = "#c9c8c1"
selection_color = "#eb6834"
line_color = "#222832"
muted_color = "#6b6a65"
neutral_color = "#8c8c8c"
grid_color = "#e3e5e8"
plt.rcParams.update(
    {
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
fig = plt.figure(figsize=(6.4, 7.2))
fig.text(
    0.04,
    0.975,
    "Pool-based Active Learning with BADGE",
    size=14,
    weight="bold",
    va="top",
)
fig.text(
    0.04,
    0.938,
    "Reuters news, t-SNE of sentence-transformer embeddings",
    color=muted_color,
    va="top",
)
ax = fig.add_axes([0.04, 0.39, 0.92, 0.47])
ax.set_axis_off()
ax.scatter(X_tsne[:, 0], X_tsne[:, 1], s=5, c=unlabeled_color, lw=0)
handles = [
    Line2D([], [], ls="", marker="o", ms=5, mfc=unlabeled_color, mec="none"),
    Line2D([], [], ls="", marker="o", ms=8, mfc=neutral_color, mec=line_color),
    Line2D(
        [],
        [],
        ls="",
        marker="o",
        ms=15,
        mfc=(selection_color, 0.28),
        mec=selection_color,
        mew=2,
    ),
]
fig.legend(
    handles,
    ["Unlabeled", "Labeled (color: class)", "New label"],
    loc="upper left",
    bbox_to_anchor=(0.02, 0.385),
    ncol=3,
    frameon=False,
    handletextpad=0.3,
    columnspacing=1.0,
    fontsize=11,
)
ax_curve = fig.add_axes([0.13, 0.07, 0.78, 0.2])
ax_curve.set_title("Learning curve", loc="left")
ax_curve.set_xlabel("Acquired labels")
ax_curve.set_ylabel("Test accuracy")
ax_curve.set_xticks(np.arange(0, n_cycles * batch_size + 1, 12))
ax_curve.set_xlim(0, 1.02 * n_cycles * batch_size)
ax_curve.set_ylim(0, 1.02)
ax_curve.grid(axis="y", color=grid_color)
artists = []

# Storage for the test accuracies throughout the cycles.
accuracies = []

# Execute active learning cycles.
for c in range(-1, n_cycles):
    if c > -1:
        query_idx = qs.query(
            X=X_pool,
            y=y_train,
            batch_size=batch_size,
            clf=clf,
            fit_clf=False,
        )
        y_train[query_idx] = y_pool[query_idx]
        clf.fit(X_pool, y_train)

    # Plotting stuff. The README snippet does not score here, so scoring
    # must not advance torch's random state.
    with torch.random.fork_rng(devices=[]):
        accuracies.append(clf.score(X_test, y_test))
    is_lbld = is_labeled(y_train, missing_label=missing_label)
    n_labels = np.arange(c + 2) * batch_size
    n_covered = len(np.unique(y_train[is_lbld]))
    frame = [
        ax.scatter(
            X_tsne[is_lbld, 0],
            X_tsne[is_lbld, 1],
            s=46,
            facecolors=[tab10[k] for k in y_train[is_lbld]],
            edgecolors=line_color,
            zorder=3,
        ),
        ax.text(
            0,
            1.02,
            f"Cycle {c + 1}: {n_labels[-1]} labels from {n_covered} of "
            f"{len(classes)} classes",
            size=13,
            transform=ax.transAxes,
        ),
    ]
    if c > -1:
        frame.append(
            ax.scatter(
                X_tsne[query_idx, 0],
                X_tsne[query_idx, 1],
                s=320,
                facecolors=(selection_color, 0.28),
                edgecolors=selection_color,
                linewidths=2.2,
                zorder=2,
            )
        )
    (curve,) = ax_curve.plot(n_labels, accuracies, c=line_color, lw=2)
    point = ax_curve.scatter(
        n_labels[-1], accuracies[-1], c=line_color, clip_on=False
    )
    low = accuracies[-1] < 0.5
    value = ax_curve.annotate(
        f"{accuracies[-1]:.2f}",
        (n_labels[-1], accuracies[-1]),
        xytext=(6, 2 if low else -2),
        textcoords="offset points",
        va="bottom" if low else "top",
    )
    artists.append(frame + [curve, point, value])


class SharedPaletteWriter(animation.PillowWriter):
    """Map all frames to one palette that keeps `colors` exact."""

    def __init__(self, colors, **kwargs):
        super().__init__(**kwargs)
        self.colors = colors

    def finish(self):
        frames = np.stack([np.asarray(f.convert("RGB")) for f in self._frames])
        codes = frames.astype(np.int32) @ np.array([65536, 256, 1])
        values, inverse, counts = np.unique(
            codes, return_inverse=True, return_counts=True
        )
        exact = [int(color[1:], 16) for color in self.colors]
        frequent = values[np.argsort(-counts)].tolist()
        # One index stays free, so that the GIF can store unchanged
        # pixels as transparent.
        palette = list(dict.fromkeys(exact + frequent))[:255]
        palette = np.array(
            [[p >> 16, (p >> 8) & 255, p & 255] for p in palette]
        )
        values = np.stack([values >> 16, (values >> 8) & 255, values & 255], 1)
        distances = ((values[:, None] - palette[None]) ** 2).sum(-1)
        indices = distances.argmin(1)[inverse].reshape(codes.shape)
        self._frames = []
        for frame in indices.astype(np.uint8):
            image = Image.fromarray(frame, "P")
            image.putpalette(palette.ravel().tolist())
            self._frames.append(image)
        super().finish()


# Plotting stuff.
full_accuracy = clone(clf).fit(X_pool, y_pool).score(X_test, y_test)
ax_curve.axhline(full_accuracy, color=neutral_color, ls="--", lw=1.5)
ax_curve.text(
    1,
    full_accuracy - 0.04,
    f"All {len(y_pool):,} labels: {full_accuracy:.2f}",
    size=10,
    color=muted_color,
    va="top",
)
artists += [artists[-1]] * 3
ani = animation.ArtistAnimation(fig, artists, interval=1000)
colors = tab10 + [
    unlabeled_color,
    selection_color,
    line_color,
    muted_color,
    neutral_color,
    grid_color,
    "#ffffff",
]
writer = SharedPaletteWriter(colors, fps=1)
ani.save(filename="logos/readme_pool.gif", writer=writer)
