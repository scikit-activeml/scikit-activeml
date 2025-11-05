import numpy as np
from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.datasets import make_blobs
from skactiveml.pool import UncertaintySampling
from skactiveml.utils import MISSING_LABEL
from skactiveml.classifier import SklearnClassifier

# Generate data set.
X, y_true = make_blobs(n_samples=200, centers=8, random_state=0, cluster_std=0.7)
y_true = y_true % 4
y = np.full(shape=y_true.shape, fill_value=MISSING_LABEL)
plt.scatter(X[:, 0], X[:, 1], c=y_true)
plt.show()
# Use the first 10 samples as initial training data.
y[:10] = y_true[:10]



# Create classifier and query strategy.
clf = SklearnClassifier(
    GaussianProcessClassifier(random_state=0),
    classes=np.unique(y_true),
    random_state=0
)
qs = UncertaintySampling(method='entropy')

# Execute active learning cycle.
n_cycles = 20
for c in range(n_cycles):
    query_idx = qs.query(X=X, y=y, clf=clf)
    y[query_idx] = y_true[query_idx]

# Fit final classifier.
clf.fit(X, y)