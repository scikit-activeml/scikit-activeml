import numpy as np
from sklearn.datasets import make_blobs
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.model_selection import train_test_split
from sklearn.semi_supervised import LabelSpreading

from skactiveml.classifier import SklearnClassifier
from skactiveml.pool import TypiClust

# Generate data set.
X, y_true = make_blobs(n_samples=800, centers=8)
X, X_test, y_true, y_test = train_test_split(X, y_true, test_size=0.2)
y = np.full(shape=y_true.shape, fill_value=-1)
y[:10] = y_true[:10]

# Create label propagation method, classifier and query strategy
prop = LabelSpreading(gamma=1.0)
clf = SklearnClassifier(
    GaussianProcessClassifier(),
    missing_label=-1,
    classes=np.unique(y_true)
)
qs = TypiClust(missing_label=-1)

# Execute active learning cycle.
n_cycles = 20
scores = []
for c in range(n_cycles):
    query_idx = qs.query(X=X, y=y)
    y[query_idx] = y_true[query_idx]
    prop.fit(X, y)
    clf.fit(X, prop.predict(X))
    scores.append(clf.score(X_test, y_test))
