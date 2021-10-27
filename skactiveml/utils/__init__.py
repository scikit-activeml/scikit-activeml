"""
The :mod:`skactiveml.utils` module includes various utilities.
"""
from ._aggregation import compute_vote_vectors, majority_vote
from ._functions import call_func, simple_batch, fit_if_not_fitted
from ._label import is_unlabeled, is_labeled, labeled_indices, \
    unlabeled_indices, \
    ExtLabelEncoder
from ._multi_annot import ext_confusion_matrix
from ._selection import rand_argmax, rand_argmin
from ._validation import check_classes, check_missing_label, check_scalar, \
    check_cost_matrix, check_classifier_params, check_X_y, MISSING_LABEL, \
    check_random_state, check_class_prior, check_type


__all__ = ['rand_argmax', 'rand_argmin', 'compute_vote_vectors',
           'majority_vote', 'is_unlabeled', 'is_labeled', 'ExtLabelEncoder',
           'check_classes', 'check_missing_label', 'check_cost_matrix',
           'check_scalar', 'check_classifier_params', 'check_X_y',
           'check_random_state', 'MISSING_LABEL', 'call_func', 'simple_batch',
           'check_class_prior', 'ext_confusion_matrix', 'fit_if_not_fitted',
           'labeled_indices', 'unlabeled_indices', 'check_type']
