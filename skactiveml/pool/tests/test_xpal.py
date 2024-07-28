import numpy as np
from sklearn.datasets import make_classification
from sklearn.utils import check_random_state
from skactiveml.classifier import ParzenWindowClassifier
from skactiveml.pool._xpal import XPAL
from ..utils import MISSING_LABEL

def create_sample_data(n_samples=100, n_features=2, n_classes=2, random_state=None):
    from sklearn.datasets import make_classification
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features,
        n_redundant=0,
        n_clusters_per_class=1,  
        n_classes=n_classes,
        random_state=random_state
    )
    return X, y


def test_xpal_init():
    xpal = XPAL(prior=0.1, m_max=2, kernel='linear', kernel_params={'C': 1.0})
    assert xpal.prior == 0.1
    assert xpal.m_max == 2
    assert xpal.kernel == 'linear'
    assert xpal.kernel_params == {'C': 1.0}

    xpal_default = XPAL()
    assert xpal_default.prior == 0.001
    assert xpal_default.m_max == 1
    assert xpal_default.kernel == 'rbf'
    assert xpal_default.kernel_params is None


def test_xpal_query():
    random_state = check_random_state(42)
    X, y = create_sample_data(random_state=random_state)
    clf = ParzenWindowClassifier(random_state=random_state)
    xpal = XPAL(random_state=random_state)
    labeled_indices = random_state.choice(len(X), size=5, replace=False)
    y_partial = np.full(len(y), xpal.missing_label)
    y_partial[labeled_indices] = y[labeled_indices]
    clf.fit(X, y_partial)
    queried_indices = xpal.query(X=X, y=y_partial, clf=clf, batch_size=5)
    assert len(queried_indices) == 5
    assert all(0 <= idx < len(X) for idx in queried_indices)
    assert 0 not in queried_indices

def test_xpal_imbalanced():
    random_state = check_random_state(42)
    X, y = make_classification(n_samples=1000, n_features=5, n_classes=2,
                               n_informative=2, n_redundant=1, weights=[0.9, 0.1],
                               random_state=random_state)
    clf = ParzenWindowClassifier(random_state=random_state)
    xpal = XPAL(random_state=random_state)
    labeled_indices = random_state.choice(len(X), size=50, replace=False)
    y_partial = np.full(len(y), xpal.missing_label)
    y_partial[labeled_indices] = y[labeled_indices]
    clf.fit(X, y_partial)
    queried_indices = xpal.query(X=X, y=y_partial, clf=clf, batch_size=20)
    assert len(queried_indices) == 20
    assert all(0 <= idx < len(X) for idx in queried_indices)
    assert all(idx not in labeled_indices for idx in queried_indices)
    minority_class = np.argmin(np.bincount(y))
    assert np.any(y[queried_indices] == minority_class)


def test_xpal_query_with_candidates():
    random_state = check_random_state(1337)
    X, y = create_sample_data(random_state=random_state)

    clf = ParzenWindowClassifier(random_state=random_state)
    xpal = XPAL(random_state=random_state)

    labeled_indices = random_state.choice(len(X), size=5, replace=False)
    y_partial = np.full(len(y), xpal.missing_label)
    y_partial[labeled_indices] = y[labeled_indices]

    clf.fit(X, y_partial)

    candidates = random_state.choice(len(X), size=50, replace=False)
    queried_indices = xpal.query(X=X, y=y_partial, clf=clf, candidates=candidates, batch_size=5)

    assert len(queried_indices) == 5
    assert all(idx in candidates for idx in queried_indices)
    assert all(0 <= idx < len(X) for idx in queried_indices)


def test_xpal_query_with_utilities():
    random_state = check_random_state(1337)
    X, y = create_sample_data(random_state=random_state)
    clf = ParzenWindowClassifier(random_state=random_state)
    xpal = XPAL(random_state=random_state)
    labeled_indices = random_state.choice(len(X), size=5, replace=False)
    y_partial = np.full(len(y), xpal.missing_label)
    y_partial[labeled_indices] = y[labeled_indices]
    clf.fit(X, y_partial)
    queried_indices, utilities = xpal.query(X=X, y=y_partial, clf=clf, batch_size=5, return_utilities=True)

    assert len(queried_indices) == 5
    assert len(utilities) == len(X)
    assert np.all(np.isfinite(utilities))
    assert np.any(utilities != utilities.min())


def test_xpal_m_max():
    random_state = check_random_state(1337)
    X, y = create_sample_data(random_state=random_state)
    clf = ParzenWindowClassifier(random_state=random_state)
    xpal = XPAL(m_max=5, random_state=random_state)
    labeled_indices = random_state.choice(len(X), size=5, replace=False)
    y_partial = np.full(len(y), xpal.missing_label)
    y_partial[labeled_indices] = y[labeled_indices]
    clf.fit(X, y_partial)
    queried_indices = xpal.query(X=X, y=y_partial, clf=clf, batch_size=5)

    assert len(queried_indices) == 5
    assert all(0 <= idx < len(X) for idx in queried_indices)
    assert all(idx not in labeled_indices for idx in queried_indices)


def test_xpal_prior():
    random_state = check_random_state(1337)
    X, y = create_sample_data(random_state=random_state)
    clf = ParzenWindowClassifier(random_state=random_state)
    xpal = XPAL(prior=0.5, random_state=random_state)
    labeled_indices = random_state.choice(len(X), size=5, replace=False)
    y_partial = np.full(len(y), xpal.missing_label)
    y_partial[labeled_indices] = y[labeled_indices]
    clf.fit(X, y_partial)
    queried_indices = xpal.query(X=X, y=y_partial, clf=clf, batch_size=5)
    
    assert len(queried_indices) == 5
    assert all(0 <= idx < len(X) for idx in queried_indices)
    assert all(idx not in labeled_indices for idx in queried_indices)


def test_xpal_multiclass():
    random_state = check_random_state(1337)
    X, y = create_sample_data(n_samples=100, n_features=2, n_classes=3, random_state=random_state)
    clf = ParzenWindowClassifier(random_state=random_state)
    xpal = XPAL(random_state=random_state)

    labeled_indices = random_state.choice(len(X), size=10, replace=False)
    y_partial = np.full(len(y), xpal.missing_label)
    y_partial[labeled_indices] = y[labeled_indices]

    clf.fit(X, y_partial)
    queried_indices = xpal.query(X=X, y=y_partial, clf=clf, batch_size=5)

    assert len(queried_indices) == 5


def test_xpal_edge_cases():
    random_state = check_random_state(1337)
    X, y = create_sample_data(random_state=random_state)

    clf = ParzenWindowClassifier(random_state=random_state)
    xpal = XPAL(random_state=random_state)

    y_all_labeled = y.copy()
    clf.fit(X, y_all_labeled)
    queried_indices = xpal.query(X=X, y=y_all_labeled, clf=clf, batch_size=5)
    assert len(queried_indices) == 0

    y_none_labeled = np.full(len(y), xpal.missing_label)
    clf.classes = [0, 1]
    clf.fit(X, y_none_labeled)
    queried_indices = xpal.query(X=X, y=y_none_labeled, clf=clf, batch_size=5)
    assert len(queried_indices) == 5

    X_small, y_small = X[:10], y[:10]
    y_small_partial = np.full(len(y_small), xpal.missing_label)
    y_small_partial[0] = y_small[0]
    clf.fit(X_small, y_small_partial)
    queried_indices = xpal.query(X=X_small, y=y_small_partial, clf=clf, batch_size=5)
    assert len(queried_indices) == 5
    assert all(0 <= idx < len(X_small) for idx in queried_indices)
    assert 0 not in queried_indices
    

def test_xpal_importance_sampling():
    random_state = check_random_state(1337)
    X, y = create_sample_data(random_state=random_state)
    clf = ParzenWindowClassifier(random_state=random_state)
    xpal = XPAL(random_state=random_state, m_max=3)
    labeled_indices = random_state.choice(len(X), size=5, replace=False)
    y_partial = np.full(len(y), xpal.missing_label)
    y_partial[labeled_indices] = y[labeled_indices]
    clf.fit(X, y_partial)
    queried_indices = xpal.query(X=X, y=y_partial, clf=clf, batch_size=5)
    assert len(queried_indices) == 5


def test_xpal_kernel_similarity():
    random_state = check_random_state(1337)
    X, y = create_sample_data(random_state=random_state)
    clf = ParzenWindowClassifier(random_state=random_state)
    xpal = XPAL(kernel='rbf', kernel_params={'gamma': 0.1}, random_state=random_state)
    labeled_indices = random_state.choice(len(X), size=5, replace=False)
    y_partial = np.full(len(y), xpal.missing_label)
    y_partial[labeled_indices] = y[labeled_indices]
    clf.fit(X, y_partial)
    queried_indices = xpal.query(X=X, y=y_partial, clf=clf, batch_size=5)
    assert len(queried_indices) == 5
    

def test_xpal_kernel_options():
    random_state = check_random_state(1337)
    X, y = create_sample_data(random_state=random_state)
    labeled_indices = random_state.choice(len(X), size=5, replace=False)
    y_partial = np.full(len(y), MISSING_LABEL)
    y_partial[labeled_indices] = y[labeled_indices]
    
    kernels = ['linear', 'poly', 'sigmoid']
    kernel_params = {
        'poly': {'degree': 3, 'coef0': 1, 'gamma': 1},
        'sigmoid': {'coef0': 0, 'gamma': 1}
    }

    for kernel in kernels:
        clf = ParzenWindowClassifier(random_state=random_state)
        params = kernel_params.get(kernel, {})
        xpal = XPAL(kernel=kernel, kernel_params=params, random_state=random_state)
        clf.fit(X, y_partial)
        queried_indices = xpal.query(X=X, y=y_partial, clf=clf, batch_size=5)
        assert len(queried_indices) == 5
        assert all(0 <= idx < len(X) for idx in queried_indices)

