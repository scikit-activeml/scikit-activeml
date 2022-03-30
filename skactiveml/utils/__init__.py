"""
The :mod:`skactiveml.utils` module includes various utilities.
"""
from ._aggregation import compute_vote_vectors, majority_vote
from ._functions import call_func
from ._label import (
    is_unlabeled,
    is_labeled,
    labeled_indices,
    unlabeled_indices,
    MISSING_LABEL,
    check_missing_label,
    check_equal_missing_label,
)
from ._label_encoder import ExtLabelEncoder
from ._multi_annot import ext_confusion_matrix
from ._selection import rand_argmax, rand_argmin, simple_batch
from ._validation import (
    check_classes,
    check_scalar,
    check_cost_matrix,
    check_classifier_params,
    check_X_y,
    check_random_state,
    check_class_prior,
    check_type,
    check_bound,
    check_budget_manager,
    check_indices,
)

__all__ = [
    "rand_argmax",
    "rand_argmin",
    "compute_vote_vectors",
    "majority_vote",
    "is_unlabeled",
    "is_labeled",
    "ExtLabelEncoder",
    "check_classes",
    "check_missing_label",
    "check_cost_matrix",
    "check_scalar",
    "check_classifier_params",
    "check_X_y",
    "check_random_state",
    "MISSING_LABEL",
    "call_func",
    "check_class_prior",
    "ext_confusion_matrix",
    "labeled_indices",
    "unlabeled_indices",
    "check_type",
    "check_bound",
    "check_equal_missing_label",
    "check_budget_manager",
    "check_indices",
    "simple_batch",
]
