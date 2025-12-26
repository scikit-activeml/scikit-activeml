import numpy as np
import torch

from datasets import load_dataset
from matplotlib import pyplot as plt, animation
from sentence_transformers import SentenceTransformer
from skactiveml.classifier import SkorchClassifier
from skactiveml.pool import Badge
from sklearn.manifold import TSNE
from skorch.callbacks import LRScheduler
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR

# Define the device depending on its availability.
device = "cuda" if torch.cuda.is_available() else "cpu"

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

# Plotting stuff.
X_tsne = TSNE(n_components=2).fit_transform(X_pool)
fig, (ax_1, ax_2) = plt.subplots(
    2, 1, figsize=(6, 6), gridspec_kw={"height_ratios": [3, 1]}
)
fig.suptitle(
    "Pool-based AL with BADGE for Text Classification (Reuters)",
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

# Start the active learning cycle with zero initial labels.
y_train = np.full_like(y_pool, missing_label)

# Create a deep active learning query strategy.
qs = Badge(
    missing_label=missing_label,
    clf_embedding_flag_name={"extra_outputs": "emb"},
)

# Define the active learning parameters.
n_cycles = 15
batch_size = 4

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
    accuracies.append(clf.score(X_test, y_test))

    # Plotting stuff.
    coll_old = list(ax_1.collections) + list(ax_2.collections)
    ax_1.scatter(
        X_tsne[:, 0],
        X_tsne[:, 1],
        c="gray",
        marker=".",
        alpha=0.3,
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
    cycles = np.arange(n_cycles + 1) * batch_size
    (accuracy_line,) = ax_2.plot(cycles[: c + 2], accuracies, c="C7")
    ax_2.set_xticks(cycles)
    ax_2.set_yticks(np.arange(0.0, 1.2, 0.2))
    ax_2.set_ylim(ymin=-0.1, ymax=1.1)
    ax_2.set_xlabel("Number of Labeled Samples", fontsize=12)
    ax_2.set_ylabel("Test Accuracy", fontsize=12, color="C7")
    coll_new = list(ax_1.collections) + list(ax_2.collections)
    artists.append(
        [x for x in coll_new if (x not in coll_old)] + [accuracy_line]
    )

# Plotting stuff.
fig.tight_layout()
ani = animation.ArtistAnimation(fig, artists, interval=1000, blit=True)
ani.save(filename="logos/readme_pool.gif", writer="pillow")
