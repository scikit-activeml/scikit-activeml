import numpy as np
from sklearn.datasets import make_blobs
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.model_selection import train_test_split

from skactiveml.classifier import SklearnClassifier
from skactiveml.pool import UncertaintySampling, TypiClust

# Generate data set.
X, y_true = make_blobs(
    n_samples=800, centers=8, random_state=0, cluster_std=1.0
)
X, X_test, y_true, y_test = train_test_split(X, y_true, test_size=0.2)
y = np.full(shape=y_true.shape, fill_value=-1)
y[:10] = y_true[:10]

# Create classifier and query strategies
clf = SklearnClassifier(GaussianProcessClassifier(), missing_label=-1, classes=np.unique(y_true))
clf.fit(X, y)
qs_1 = TypiClust(missing_label=-1)
qs_2 = UncertaintySampling(missing_label=-1)

# Execute active learning cycle.
n_cycles = 20
scores = []
for c in range(n_cycles):
    if c < 5:
        query_idx = qs_1.query(X=X, y=y)
    else:
        query_idx = qs_2.query(X=X, y=y, clf=clf)
    y[query_idx] = y_true[query_idx]
    clf.fit(X, y)
    scores.append(clf.score(X_test, y_test))
