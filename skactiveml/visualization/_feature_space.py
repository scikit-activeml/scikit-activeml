import numpy as np
import pylab as plt

from skactiveml.utils import call_func, is_unlabeled, is_labeled
from sklearn.utils.multiclass import type_of_target

from skactiveml.utils import call_func, is_unlabeled, is_labeled, ExtLabelEncoder
from sklearn.utils import check_array, check_consistent_length, column_or_1d
from skactiveml.classifier import SklearnClassifier

def plot_decision_boundary(self, X, y, y_oracle, clf, selector, res=21):
    # Validatet input parameters
    X = check_array(X)
    y = np.array(y)
    check_consistent_length(X, y)
    check_consistent_length(X, y_oracle)
    is_lbdl = is_labeled(y, self.missing_label)
    if len(y[is_lbdl]) > 0:
        y_type = type_of_target(y[is_lbdl])
        if y_type not in ['binary', 'multiclass', 'multiclass-multioutput',
                          'multilabel-indicator', 'multilabel-sequences',
                          'unknown']:
            raise ValueError("Unknown label type: %r" % y_type)
    self._le = ExtLabelEncoder(classes=self.classes,
                               missing_label=self.missing_label)
    y = self._le.fit_transform(y)
    if len(self._le.classes_) == 0:
        raise ValueError("No class label is known because 'y' contains no "
                         "actual class labels and 'classes' is not "
                         "defined. Change at least on of both to overcome "
                         "this error.")

    # Restrictions for clf
    if clf not in SklearnClassifier:
        raise TypeError("It only supports SklearnClassifiers")



    # create mesh for plotting
    x_1_vec = np.linspace(min(X[:, 0]), max(X[:, 0]), res)
    x_2_vec = np.linspace(min(X[:, 1]), max(X[:, 1]), res)
    X_1_mesh, X_2_mesh = np.meshgrid(x_1_vec, x_2_vec)
    X_mesh = np.array([X_1_mesh.reshape(-1), X_2_mesh.reshape(-1)]).T

    # compute gains
    clf.fit(X, y)
    posteriors = clf.predict_proba(X_mesh)[:,0].reshape(X_1_mesh.shape)

    # compute gains
    _, scores = call_func(selector.query, X_cand=X_mesh, X=X, y=y, X_eval=X,
                          return_utilities=True)
    scores = scores.reshape(X_1_mesh.shape)

    # get indizes for plotting
    labeled_indices = np.where(is_labeled(y))[0]
    unlabeled_indices = np.where(is_unlabeled(y))[0]

    # setup figure
    fig = plt.figure(figsize=(6, 4))
    plt.xlim(min(X[:, 0]), max(X[:, 0]))
    plt.ylim(min(X[:, 1]), max(X[:, 1]))
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.title('Decision boundry after acquring {} labels'.format(len(labeled_indices)))
    cmap = plt.get_cmap('coolwarm')

    plt.scatter(X[labeled_indices, 0], X[labeled_indices, 1], c=[[.2, .2, .2]], s=90, marker='o', zorder=3.8)
    plt.scatter(X[labeled_indices, 0], X[labeled_indices, 1], c=[[.8, .8, .8]], s=60, marker='o', zorder=4)
    for cl, marker in zip([0,1],['D','s']):
        cl_labeled_idx = labeled_indices[y[labeled_indices] == cl]
        cl_unlabeled_idx = unlabeled_indices[y_oracle[unlabeled_indices]==cl]
        plt.scatter(X[cl_labeled_idx, 0], X[cl_labeled_idx, 1], c=np.ones(len(cl_labeled_idx))*cl, marker=marker, vmin=-0.2, vmax=1.2, cmap='coolwarm', s=20, zorder=5)
        plt.scatter(X[cl_unlabeled_idx, 0], X[cl_unlabeled_idx, 1], c=np.ones(len(cl_unlabeled_idx)) * cl, marker=marker, vmin=-0.2, vmax=1.2, cmap='coolwarm', s=20, zorder=3)
        plt.scatter(X[cl_unlabeled_idx, 0], X[cl_unlabeled_idx, 1], c='k', marker=marker, vmin=-0.1, vmax=1.1, cmap='coolwarm', s=30, zorder=2.8)

    CS = plt.contourf(X_1_mesh, X_2_mesh, scores, cmap='Greens', alpha=.75)
    CS = plt.contour(X_1_mesh, X_2_mesh, posteriors, [.5], colors='k', linewidths=[2], zorder=1)
    CS = plt.contour(X_1_mesh, X_2_mesh, posteriors, [.25,.75], cmap='coolwarm_r', linewidths=[2,2],
                     zorder=1, linestyles='--', alpha=.9, vmin=.2, vmax=.8)

    fig.tight_layout()
    plt.show()

    return fig