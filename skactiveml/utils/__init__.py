"""
The :mod:`skactiveml.utils` module includes various utilities.
"""
from ._selection import rand_argmax, rand_argmin
from ._aggregation import compute_vote_vectors
from ._label import is_unlabeled, is_labeled, ExtLabelEncoder, MISSING_LABEL
from ._validation import check_classes, check_missing_label, check_cost_matrix

__all__ = ['rand_argmax', 'rand_argmin', 'compute_vote_vectors', 'is_unlabeled', 'is_labeled', 'ExtLabelEncoder',
           'check_classes', 'check_missing_label', 'check_cost_matrix', 'MISSING_LABEL']