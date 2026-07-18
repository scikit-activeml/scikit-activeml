.. _target-semantics:

================================
Target Semantics in Version 1.1
================================

Target semantics describe the task, target type, and annotation type that give
target data its meaning.  Version 1.1 resolves those semantics once, represents
them with :class:`~skactiveml.utils.TargetSpec`, and then checks whether a
consumer supports the exact resolved target capability.  Resolution and
capability checking are deliberately separate: a target can be meaningful
even when a particular estimator or query strategy cannot execute it.

Public API
==========

:func:`~skactiveml.utils.resolve_target_spec` combines ``y`` with the declared
``task``, ``target_type``, ``annotation_type``, optional class vocabularies,
and ``missing_label``.  It returns a frozen target specification with four
fields:

``task``
    ``"classification"`` or ``"regression"``.
``target_type``
    The concrete value ``"single-output"``, ``"multi-label"``, or
    ``"multi-output"``.  A resolved specification never stores ``"auto"``.
``annotation_type``
    ``"single-annotator"`` or ``"multi-annotator"``.
``classes``
    The canonical immutable class vocabulary for classification, or ``None``
    for regression.

Public estimators and pool query strategies expose a ``target_type``
constructor parameter.  Its default, ``"auto"``, preserves unambiguous
single-output calls.  After a successful fit, classifiers and regressors expose
the concrete ``target_spec_`` attribute.
Query strategies resolve semantics for each call and do not retain a
last-query specification.  When a fitted classifier is passed to a strategy,
the classifier's ``target_spec_`` is authoritative.

Explicit multi-label classification
===================================

Declare a multi-label target rather than relying on its two-dimensional shape.
With ``classes=None``, every output column must expose both binary classes.
The following example fits a classifier and queries one complete label vector;
it is executed as part of the documentation tests.

.. doctest::

   >>> import numpy as np
   >>> from sklearn.linear_model import LogisticRegression
   >>> from sklearn.multioutput import MultiOutputClassifier
   >>> from skactiveml.classifier import SklearnClassifier
   >>> from skactiveml.pool import UncertaintySampling
   >>> X = np.array([
   ...     [-2.0, -1.0], [-1.0, 1.0], [1.0, -1.0],
   ...     [2.0, 1.0], [0.0, -0.2], [0.0, 0.2],
   ... ])
   >>> y = np.array([
   ...     [0.0, 0.0], [0.0, 1.0], [1.0, 0.0],
   ...     [1.0, 1.0], [np.nan, np.nan], [np.nan, np.nan],
   ... ])
   >>> clf = SklearnClassifier(
   ...     MultiOutputClassifier(LogisticRegression(random_state=0)),
   ...     classes=None,
   ...     target_type="multi-label",
   ...     random_state=0,
   ... )
   >>> _ = clf.fit(X, y)
   >>> assert clf.target_spec_.target_type == "multi-label"
   >>> assert clf.target_spec_.classes == ((0.0, 1.0), (0.0, 1.0))
   >>> strategy = UncertaintySampling(
   ...     method="entropy", target_type="multi-label", random_state=0
   ... )
   >>> query_indices = strategy.query(
   ...     X=X, y=y, clf=clf, fit_clf=False
   ... )
   >>> assert query_indices.shape == (1,)
   >>> assert query_indices[0] in (4, 5)

Class vocabularies and complete rows
------------------------------------

Multi-label classification uses one binary class vocabulary per label output.
Explicit vocabularies let fitting start before both classes have been observed
and allow non-numeric labels.  Their input order is not the probability-column
order: each vocabulary is normalized to the same canonical ordering used by
fitted ``classes_``.

.. doctest::

   >>> from skactiveml.utils import resolve_target_spec
   >>> string_y = np.array([
   ...     ["present", "yes"],
   ...     ["absent", "no"],
   ... ])
   >>> spec = resolve_target_spec(
   ...     string_y,
   ...     task="classification",
   ...     target_type="multi-label",
   ...     classes=(("present", "absent"), ("yes", "no")),
   ...     missing_label=None,
   ... )
   >>> assert spec.classes == (("absent", "present"), ("no", "yes"))

Without explicit ``classes``, resolution never invents a ``(0, 1)``
vocabulary.  A column with fewer than two observed classes raises an error.

.. doctest::

   >>> under_observed = np.array([
   ...     [0.0, 0.0], [1.0, 0.0], [np.nan, np.nan]
   ... ])
   >>> _ = resolve_target_spec(  # doctest: +IGNORE_EXCEPTION_DETAIL
   ...     under_observed,
   ...     task="classification",
   ...     target_type="multi-label",
   ... )
   Traceback (most recent call last):
   ...
   ValueError:

Version 1.1 treats a multi-label sample as one complete vector.  Each row must
therefore be wholly observed or wholly missing; mixed rows would imply
partial-label training or acquisition, which is not supported.

.. doctest::

   >>> mixed_row_y = np.array([[0.0, 1.0], [np.nan, 0.0]])
   >>> _ = resolve_target_spec(  # doctest: +IGNORE_EXCEPTION_DETAIL
   ...     mixed_row_y,
   ...     task="classification",
   ...     target_type="multi-label",
   ...     classes=((0, 1), (0, 1)),
   ... )
   Traceback (most recent call last):
   ...
   ValueError:

Ambiguous two-dimensional classification
========================================

For single-annotator classification, a bare two-dimensional ``y`` with
``target_type="auto"`` and ``classes=None`` is ambiguous: columns could be
binary label outputs or distinct future outputs.  Values that happen to look
binary do not resolve that ambiguity.  Disambiguate by declaring
``target_type``, supplying a flat or nested class vocabulary, or passing a
fitted estimator whose ``target_spec_`` already provides resolved evidence.

.. doctest::

   >>> _ = resolve_target_spec(  # doctest: +IGNORE_EXCEPTION_DETAIL
   ...     np.array([[0, 1], [1, 0]]), task="classification"
   ... )
   Traceback (most recent call last):
   ...
   ValueError:

A flat vocabulary under ``"auto"`` means single-output classification.  A
nested set of binary vocabularies means multi-label classification.  A nested
vocabulary containing a non-binary output resolves to future multi-output
classification, which version 1.1 components then reject as unsupported.

Target-aware masks and indices
==============================

:func:`~skactiveml.utils.is_labeled`,
:func:`~skactiveml.utils.is_unlabeled`,
:func:`~skactiveml.utils.labeled_indices`, and
:func:`~skactiveml.utils.unlabeled_indices` accept the keyword-only
``target_type`` argument.  The default ``"single-output"`` behavior remains
elementwise, including for multi-annotator matrices.  With
``target_type="multi-label"``, the helpers validate complete-or-missing rows
and return sample-level masks or indices.  They do not accept ``"auto"``;
pass the concrete value from a resolved specification.

.. doctest::

   >>> from skactiveml.utils import is_unlabeled, unlabeled_indices
   >>> is_unlabeled(y, target_type=clf.target_spec_.target_type).tolist()
   [False, False, False, False, True, True]
   >>> unlabeled_indices(y, target_type="multi-label").tolist()
   [4, 5]

Regression
==========

Regressors accept ``target_type="auto"`` and ``"single-output"`` for
version 1.1 execution.  One-dimensional numerical targets resolve to
single-output regression, and column vectors remain compatible.  Multiple
target columns resolve to recognized ``"multi-output"`` regression semantics,
but regressors reject that valid specification because multi-output execution
is not yet a declared capability.  Regression target specifications always
have ``classes=None``.

Multiple annotators
===================

Target type and annotation type are independent axes.  In a multi-annotator
matrix, columns identify annotators supplying observations for the same
single-output label; they are not label-output columns.  Mixed observed and
missing entries within one sample remain valid.  Existing multi-annotator
estimators and strategies keep that annotation type internally and preserve
sample-annotator acquisition: query results still identify ``(sample,
annotator)`` pairs.

Errors and component capabilities
=================================

Invalid semantics fail during resolution.  Examples include an unknown
``target_type`` or ``target_type="multi-label"`` with regression.  A valid
target specification that is absent from a component's exact capabilities
fails later with a capability error naming both the requested and supported
combinations.  This distinction tells users whether to correct the meaning of
their input or choose a component that implements it.

After fitting, inspect ``estimator.target_spec_`` instead of repeating target
resolution from array shape.  In particular, inspect its ``target_type`` and
``annotation_type`` before selecting downstream behavior, and use its
``classes`` as the canonical classification vocabulary.

Recognized future semantics
===========================

Version 1.1 deliberately recognizes without executing multi-output
classification and multi-output regression.  It also defers partial-label
querying and multi-label multi-annotator querying.  These are capability and
acquisition-scope limits, not architectural conflations: target type remains
separate from annotation type, and acquisition granularity is not stored in
``TargetSpec``.  Future support can therefore add exact component capabilities
and an explicit acquisition model without changing the meanings published
here.
