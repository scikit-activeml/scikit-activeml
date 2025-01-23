import numpy as np
from sklearn import clone
from sklearn.cluster import KMeans

from ..base import SingleAnnotatorPoolQueryStrategy, SkactivemlClassifier
from ..utils import (
    MISSING_LABEL,
    check_type,
    check_equal_missing_label,
    unlabeled_indices,
    check_scalar,
    is_unlabeled,
    simple_batch,
)


class LabelCardinalityinconsistency(SingleAnnotatorPoolQueryStrategy):

    def __init__(
            self,
            clf_embedding_flag_name=None,
            missing_label=MISSING_LABEL,
            random_state=None,
    ):
        self.clf_embedding_flag_name = clf_embedding_flag_name
        super().__init__(
            missing_label=missing_label, random_state=random_state
        )

    def query(
        self,
        X,
        y,
        clf,
        fit_clf=True,
        sample_weight=None,
        candidates=None,
        batch_size=1,
        return_utilities=False,
    ):

        is_multilabel = np.array(y).ndim == 2

        # Validate input parameters
        X, y, candidates, batch_size, return_utilities = self._validate_data(
            X, y, candidates, batch_size, return_utilities, reset=True, is_multilabel=is_multilabel
        )

        X_cand, mapping = self._transform_candidates(candidates, X, y, is_multilabel=is_multilabel)

        # Validate classifier type
        check_type(clf, "clf", SkactivemlClassifier)
        check_equal_missing_label(clf.missing_label, self.missing_label_)
        check_scalar(fit_clf, "fit_clf", bool)
        if self.clf_embedding_flag_name is not None:
            check_scalar(
                self.clf_embedding_flag_name, "clf_embedding_flag_name", str
            )

        # Fit the classifier
        if fit_clf:
            if sample_weight is None:
                clf = clone(clf).fit(X, y)
            else:
                clf = clone(clf).fit(X, y, sample_weight)

        # find the unlabeled dataset
        if candidates is None:
            X_unlbld = X_cand
        elif mapping is not None:
            if not is_multilabel:
                unlbld_mapping = unlabeled_indices(
                    y[mapping], missing_label=self.missing_label
                )
            else:
                unlbld_mapping = np.unique(np.argwhere(is_unlabeled(y[mapping], MISSING_LABEL)[:, 0]))
            X_unlbld = X_cand[unlbld_mapping]

        else:
            X_unlbld = X_cand

        n_lbld = X.shape[0] - X_unlbld.shape[0]

        y_label_cardinality = 0 # TODO what if no labeled instances
        if n_lbld != 0:
            y_label_cardinality = np.nansum(y) / n_lbld

        Y_pred = clf.predict(X_unlbld)
        pred_mean_cardinality = Y_pred.sum(axis=-1)

        utilities_cand = np.abs(pred_mean_cardinality - y_label_cardinality)

        if mapping is None:
            utilities = utilities_cand
        else:
            utilities = np.full(len(X), np.nan)
            utilities[mapping] = utilities_cand

        return simple_batch(
            utilities,
            self.random_state_,
            batch_size=batch_size,
            return_utilities=return_utilities,
        )



