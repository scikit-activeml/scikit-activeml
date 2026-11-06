import numpy as np
import torch

from datasets import load_dataset
from matplotlib import pyplot as plt, animation
from matplotlib.lines import Line2D
from skactiveml.classifier import SkorchClassifier
from skactiveml.stream import Split
from skactiveml.utils import is_labeled
from sklearn.base import clone
from sklearn.manifold import TSNE
from skorch.callbacks import LRScheduler
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoImageProcessor, Dinov2Model

from _scripts.readme_utils import SharedPaletteWriter

# Define the device depending on its availability.
device = "cuda" if torch.cuda.is_available() else "cpu"

# Seed the initialization and training of the network.
torch.manual_seed(0)

# Load data and embed the stream and test images.
ds = load_dataset("cifar10")
processor = AutoImageProcessor.from_pretrained(
    "facebook/dinov2-small", use_fast=True
)
model = Dinov2Model.from_pretrained("facebook/dinov2-small").to(device).eval()


def embed(batch):
    inputs = processor(images=batch["img"], return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs).last_hidden_state[:, 0]
    batch["emb"] = out.cpu().numpy()
    return batch


ds_stream = ds["train"].select(range(300))
ds_stream = ds_stream.map(embed, batched=True, batch_size=128)
ds_test = ds["test"].map(embed, batched=True, batch_size=128)
X_stream = np.stack(ds_stream["emb"], dtype=np.float32)
y_stream = np.array(ds_stream["label"], dtype=np.int64)
X_test = np.stack(ds_test["emb"], dtype=np.float32)
y_test = np.array(ds_test["label"], dtype=np.int64)
n_features, classes = X_stream.shape[1], np.unique(y_stream)
missing_label = -1


# Build `torch` module for classification, outputting classification logits.
class ClassificationModule(nn.Module):
    def __init__(self, n_features, n_classes, n_hidden_units):
        super().__init__()
        self.linear_1 = nn.Linear(n_features, n_hidden_units)
        self.linear_2 = nn.Linear(n_hidden_units, n_classes)
        self.activation = nn.ReLU()

    def forward(self, x):
        x_embed = self.linear_1(x)
        logits = self.linear_2(self.activation(x_embed))
        return logits


# Wrap your torch module via a `skactiveml` wrapper, which requires the
# definition of training parameters.
clf = SkorchClassifier(
    module=ClassificationModule,
    criterion=nn.CrossEntropyLoss,
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

# Start the stream with zero initial labels.
y_train = np.full_like(y_stream, missing_label)

# Create the query strategy.
qs = Split(random_state=0, budget=0.1)
n_cycles = len(X_stream)

# Plotting stuff.
X_tsne = TSNE(n_components=2, random_state=0).fit_transform(X_stream)
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
    "Stream-based Active Learning with Split",
    size=14,
    weight="bold",
    va="top",
)
fig.text(
    0.04,
    0.938,
    "CIFAR-10 images, t-SNE of vision-transformer embeddings",
    color=muted_color,
    va="top",
)
ax = fig.add_axes([0.04, 0.39, 0.92, 0.47])
ax.set_axis_off()
pad = 0.04 * np.ptp(X_tsne, axis=0)
ax.set_xlim(X_tsne[:, 0].min() - pad[0], X_tsne[:, 0].max() + pad[0])
ax.set_ylim(X_tsne[:, 1].min() - pad[1], X_tsne[:, 1].max() + pad[1])
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
ax_budget = fig.add_axes([0.13, 0.07, 0.31, 0.2])
ax_budget.set_ylabel("Share labeled")
ax_budget.axhline(qs.budget, color=neutral_color, ls="--", lw=1.5)
ax_budget.text(
    n_cycles,
    qs.budget + 0.03,
    f"Budget: {qs.budget:.0%}",
    size=10,
    color=muted_color,
    ha="right",
    va="bottom",
)
ax_curve = fig.add_axes([0.63, 0.07, 0.28, 0.2])
ax_curve.set_title("Learning curve", loc="left")
ax_curve.set_ylabel("Test accuracy")
for axis in (ax_budget, ax_curve):
    axis.set_xlabel("Observed samples")
    axis.set_xticks(np.arange(0, n_cycles + 1, 100))
    axis.set_xlim(0, 1.02 * n_cycles)
    axis.set_ylim(0, 1.02)
    axis.grid(axis="y", color=grid_color)
plot_step = 5
artists = []

# Execute active learning cycle.
accuracies = []
queried, shares = [], []
query_idx = []
for t in range(-1, n_cycles):
    if t > -1:
        query_idx = qs.query(candidates=X_stream[[t]], clf=clf, fit_clf=False)
        qs.update(candidates=X_stream[[t]], queried_indices=query_idx)
        if len(query_idx) > 0:
            y_train[t] = y_stream[t]
            clf.fit(X_stream, y_train)

        # Plotting stuff.
        window = qs.budget_manager_.w
        queried.append(len(query_idx) > 0)
        shares.append(np.mean(queried[-window:]))
    # The README snippet does not score here, so scoring must not advance
    # torch's random state.
    if len(query_idx) > 0 or len(accuracies) == 0:
        with torch.random.fork_rng(devices=[]):
            accuracies.append(clf.score(X_test, y_test))
    else:
        accuracies.append(accuracies[-1])

    # Plotting stuff.
    n_observed = t + 1
    if n_observed % plot_step > 0:
        continue
    observed = np.arange(n_observed)
    is_lbld = is_labeled(y_train[:n_observed], missing_label=missing_label)
    recent = observed[is_lbld & (observed >= n_observed - 3 * plot_step)]
    frame = [
        ax.scatter(
            X_tsne[observed[~is_lbld], 0],
            X_tsne[observed[~is_lbld], 1],
            s=16,
            c=unlabeled_color,
            lw=0,
        ),
        ax.scatter(
            X_tsne[observed[is_lbld], 0],
            X_tsne[observed[is_lbld], 1],
            s=46,
            facecolors=[tab10[k] for k in y_train[observed[is_lbld]]],
            edgecolors=line_color,
            zorder=3,
        ),
        ax.scatter(
            X_tsne[recent, 0],
            X_tsne[recent, 1],
            s=320,
            facecolors=(selection_color, 0.28),
            edgecolors=selection_color,
            linewidths=2.2,
            zorder=2,
        ),
        ax.text(
            0,
            1.02,
            f"{n_observed} of {n_cycles} samples observed, "
            f"{is_lbld.sum()} labels",
            size=13,
            transform=ax.transAxes,
        ),
    ]
    (share_line,) = ax_budget.plot(observed + 1, shares, c=line_color, lw=2)
    (curve,) = ax_curve.plot(
        np.arange(n_observed + 1), accuracies, c=line_color, lw=2
    )
    point = ax_curve.scatter(
        n_observed, accuracies[-1], c=line_color, clip_on=False
    )
    low = accuracies[-1] < 0.5
    value = ax_curve.annotate(
        f"{accuracies[-1]:.2f}",
        (n_observed, accuracies[-1]),
        xytext=(6, 2 if low else -2),
        textcoords="offset points",
        va="bottom" if low else "top",
    )
    artists.append(frame + [share_line, curve, point, value])


# Plotting stuff.
ax_budget.set_title(f"Labels in the last {window} samples", loc="left")
full_accuracy = clone(clf).fit(X_stream, y_stream).score(X_test, y_test)
ax_curve.axhline(full_accuracy, color=neutral_color, ls="--", lw=1.5)
ax_curve.text(
    0.02 * n_cycles,
    full_accuracy - 0.04,
    f"All {n_cycles} labels: {full_accuracy:.2f}",
    size=10,
    color=muted_color,
    va="top",
)
artists += [artists[-1]] * 20
ani = animation.ArtistAnimation(fig, artists, interval=150)
colors = tab10 + [
    unlabeled_color,
    selection_color,
    line_color,
    muted_color,
    neutral_color,
    grid_color,
    "#ffffff",
]
writer = SharedPaletteWriter(colors, fps=1000 / 150)
ani.save(filename="logos/readme_stream.gif", writer=writer)
