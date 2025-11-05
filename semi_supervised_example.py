import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.semi_supervised import LabelSpreading

from skactiveml.classifier import SklearnClassifier
from skactiveml.pool import UncertaintySampling

MISSING_LABEL = -1
# Generate data set.
X_all, y_true = make_blobs(
    n_samples=800, centers=8, random_state=0, cluster_std=1.0
)
y_true = y_true % 4

plt.scatter(X_all[:, 0], X_all[:, 1], c=y_true)
plt.show()
X, X_test = X_all[:len(X_all)//2], X_all[len(X_all)//2:]
y_true, y_true_test = y_true[:len(X_all)//2], y_true[len(X_all)//2:]


class SSLClassifier(SklearnClassifier):
    def __init__(
        self,
        ssl_model,
        estimator,
        classes=None,
        missing_label=MISSING_LABEL,
        cost_matrix=None,
        random_state=None,
    ):
        super().__init__(
            estimator=estimator,
            missing_label=missing_label,
            cost_matrix=cost_matrix,
            random_state=random_state,
            classes=classes,
        )
        self.ssl_model = ssl_model

    def fit(self, X, y, **fit_kwargs):
        self.ssl_model.fit(X, y)
        y_ssl = self.ssl_model.predict(X)
        return super().fit(X, y_ssl, **fit_kwargs)

# Create classifiers and query strategy.
clf = SklearnClassifier(
    estimator=GaussianProcessClassifier(random_state=0),
    classes=np.unique(y_true),
    random_state=0,
    missing_label=MISSING_LABEL,
)
clf_ssl = SSLClassifier(
    LabelSpreading(kernel="rbf", gamma=1.0),
    estimator=GaussianProcessClassifier(random_state=0),
    classes=np.unique(y_true),
    random_state=0,
    missing_label=MISSING_LABEL,
)
clfs = [
    clf,
    clf_ssl,
]
for clf in clfs:
    y = np.full(shape=y_true.shape, fill_value=MISSING_LABEL)
    y[:10] = y_true[:10]

    qs = UncertaintySampling(method="margin_sampling", missing_label=MISSING_LABEL)
    print(clf.__class__.__name__)
    # Execute active learning cycle.
    n_cycles = 20
    for c in range(n_cycles):
        clf.fit(X, y)
        query_idx = qs.query(X=X, y=y, clf=clf, fit_clf=False)
        y[query_idx] = y_true[query_idx]
        print(np.mean(clf.predict(X_test) == y_true_test))

    # Fit final classifier.
    clf.fit(X, y)
