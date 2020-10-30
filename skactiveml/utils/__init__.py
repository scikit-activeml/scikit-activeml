"""
The :mod:`skactiveml.utils` module includes various utilities.
"""
from ._aggregation import compute_vote_vectors
from ._functions import initialize_class_with_kwargs
from ._label import is_unlabeled, is_labeled, ExtLabelEncoder, MISSING_LABEL
from ._selection import rand_argmax, rand_argmin
from ._validation import check_classes, check_missing_label, check_cost_matrix, \
    check_classifier_params, check_X_y
from ._multi_annot import ext_confusion_matrix

__all__ = ['rand_argmax', 'rand_argmin', 'compute_vote_vectors',
           'is_unlabeled', 'is_labeled', 'ExtLabelEncoder', 'check_classes',
           'check_missing_label', 'check_cost_matrix',
           'check_classifier_params', 'check_X_y', 'MISSING_LABEL',
           'initialize_class_with_kwargs', 'ext_confusion_matrix']
