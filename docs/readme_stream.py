import numpy as np
import torch

from datasets import load_dataset
from matplotlib import pyplot as plt, animation
from skactiveml.classifier import SkorchClassifier
from skactiveml.stream import Split
from sklearn.manifold import TSNE
from skorch.callbacks import LRScheduler
from skactiveml.utils import is_labeled
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoImageProcessor, Dinov2Model


# Define the device depending on its availability.
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load data.
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


ds = ds.map(embed, batched=True, batch_size=128)
X_stream = np.stack(ds["train"]["emb"], dtype=np.float32)[:1000]
y_stream = np.array(ds["train"]["label"], dtype=np.int64)[:1000]
X_test = np.stack(ds["test"]["emb"], dtype=np.float32)
y_test = np.array(ds["test"]["label"], dtype=np.int64)
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

# Plotting stuff.
X_tsne = TSNE(n_components=2, random_state=0).fit_transform(X_stream)
fig, (ax_1, ax_2) = plt.subplots(
    2, 1, figsize=(6, 6), gridspec_kw={"height_ratios": [3, 1]}
)
ax_3 = ax_2.twinx()
fig.suptitle(
    "Stream-based AL with Split for Image Classification (CIFAR10)",
    fontweight="bold",
    fontsize=12,
)
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
artists = []

# Initialize training data as empty lists.
y_train = np.full_like(y_stream, missing_label)

# Execute active learning cycle.
qs = Split(random_state=0, budget=0.1)
n_cycles = 300
accuracies = []
budgets = [0]
query_idx = []
for t in range(-1, n_cycles):
    if t > -1:
        query_idx = qs.query(
            candidates=X_stream[[t]], y=y_stream[t], clf=clf, fit_clf=False
        )
        qs.update(candidates=X_stream[[t]], queried_indices=query_idx)
        if len(query_idx) > 0:
            y_train[t] = y_stream[t]
            clf.fit(X_stream, y_train)
        budgets.append(
            is_labeled(y_train, missing_label=missing_label).sum() / (t + 1)
        )
    if len(query_idx) > 0 or len(accuracies) == 0:
        accuracies.append(clf.score(X_test, y_test))
    else:
        accuracies.append(accuracies[-1])

    # Plotting stuff.
    X_labeled = X_tsne[is_labeled(y_train, missing_label=missing_label)]
    coll_old = (
        list(ax_1.collections)
        + list(ax_2.collections)
        + list(ax_3.collections)
    )
    ax_1.scatter(
        X_tsne[: t + 1, 0],
        X_tsne[: t + 1, 1],
        c="grey",
        marker=".",
        alpha=0.3,
        label="unlabeled",
        s=150,
    )
    for color in classes:
        ax_1.scatter(
            X_tsne[y_train == color][:, 0],
            X_tsne[y_train == color][:, 1],
            c=tab10[color],
            marker=".",
            alpha=1,
            s=300,
            edgecolors="k",
        )
    ax_1.set_xlabel("$t$-SNE Feature 1", fontsize=12)
    ax_1.set_ylabel("$t$-SNE Feature 2", fontsize=12)
    cycles = np.arange(n_cycles + 1)
    (accuracy_line,) = ax_2.plot(cycles[: t + 2], accuracies, c="C7")
    ax_2.set_xticks(np.arange(0, n_cycles + 1, 30))
    ax_2.set_yticks(np.arange(0.0, 1.2, 0.2))
    ax_2.set_xlabel("Time", fontsize=12)
    ax_2.set_ylabel("Test Accuracy", fontsize=12, color="C7")
    ax_2.set_ylim(ymin=-0.1, ymax=1.1)
    (budget_line,) = ax_3.plot(cycles[: t + 2], budgets, c="C0")
    ax_3.set_yticks(np.arange(0.0, 1.2, 0.2))
    ax_3.set_ylim(ymin=-0.1, ymax=1.1)
    ax_3.set_ylabel("Budget", fontsize=12, color="C0")
    coll_new = (
        list(ax_1.collections)
        + list(ax_2.collections)
        + list(ax_3.collections)
    )
    artists.append(
        [x for x in coll_new if (x not in coll_old)]
        + [accuracy_line, budget_line]
    )

# Plotting stuff.
fig.tight_layout()
ani = animation.ArtistAnimation(fig, artists, interval=100, blit=True)
ani.save(filename="logos/readme_stream.gif", writer="pillow")
