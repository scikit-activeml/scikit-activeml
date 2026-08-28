:orphan:

.. _target-semantics:

===============================
Target and Annotation Semantics
===============================

Target semantics describe the task, target type, and annotation type that give
target data its meaning.  The target contract resolves those semantics once,
represents them with :class:`~skactiveml.utils.TargetSpec`, and then checks
whether a consumer supports the exact resolved target capability.  Resolution
and capability checking are deliberately separate: a target can be meaningful
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

.. _multilabel-strategy-inventory:

Multi-label pool strategy capabilities
--------------------------------------

The following pool strategies accept complete multi-label targets.  This
categorized inventory is checked against the exact capability inventory in
``skactiveml/pool/tests/test_multilabel_contracts.py``.  That test groups
strategies by probability consumption, while this user-facing inventory groups
them by how their acquisition method relates to multi-label data.  Adding a
multi-label-capable strategy therefore requires classifying it in both places.
The :doc:`Strategy Overview <generated/strategy_overview>` provides a
``Multi-Label`` filter and links to the available examples.

**Native multi-label methods**
    :class:`~skactiveml.pool.MaxLossReductionMaxConfidence` and
    :class:`~skactiveml.pool.LabelCardinalityInconsistency` cite methods
    designed for multi-label acquisition.  Maximum Loss Reduction with
    Maximal Confidence is commonly shortened to **MMC** in the literature and
    tutorials; the importable class is
    :class:`~skactiveml.pool.MaxLossReductionMaxConfidence`, and no separate
    ``MMC`` alias is provided.

**Extensions of single-output methods**
    :class:`~skactiveml.pool.Badge`, :class:`~skactiveml.pool.Clue`,
    :class:`~skactiveml.pool.DropQuery`, :class:`~skactiveml.pool.Falcun`,
    :class:`~skactiveml.pool.ProbCover`,
    :class:`~skactiveml.pool.UHerding`, and
    :class:`~skactiveml.pool.UncertaintySampling` document how the library
    extends their cited single-output method.  Most compute per-label scores
    and reduce them to one sample utility; ``ProbCover`` instead reads the
    observed label rows when choosing its default radius.  Follow each class
    link for the precise extension and reduction contract.

**Representation- and mask-only methods**
    :class:`~skactiveml.pool.CoreSet`,
    :class:`~skactiveml.pool.DiscriminativeAL`,
    :class:`~skactiveml.pool.GreedySamplingX`,
    :class:`~skactiveml.pool.MaxHerding`,
    :class:`~skactiveml.pool.RandomSampling`, and
    :class:`~skactiveml.pool.TypiClust` operate on sample representations and
    the labeled/unlabeled mask; label values do not enter their acquisition.

:class:`~skactiveml.pool.ParallelUtilityEstimationWrapper` and
:class:`~skactiveml.pool.SubSamplingWrapper` inherit multi-label behavior from
their wrapped strategy.

Estimator capability for multi-label wrapping
---------------------------------------------

``SklearnClassifier`` admits an estimator for multi-label classification only
when it is a ``scikit-learn`` classifier, implements ``predict_proba``, and
positively declares either ``target_tags.multi_output`` or
``classifier_tags.multi_label``.  Capability is never discovered by fitting on
generated data.  An estimator such as plain ``LogisticRegression`` exposes
``predict_proba`` but declares neither tag, so it is rejected before any
fitted state is committed instead of silently degrading to prior-only
predictions.

.. doctest::

   >>> rejected = SklearnClassifier(
   ...     LogisticRegression(),
   ...     classes=[[0, 1], [0, 1]],
   ...     missing_label=-1,
   ... )
   >>> try:
   ...     _ = rejected.fit(np.zeros((2, 2)), np.array([[0, 1], [1, 0]]))
   ... except ValueError as error:
   ...     print("target_tags.multi_output" in str(error))
   True
   >>> assert not hasattr(rejected, "target_spec_")

Pre-fitted estimators
---------------------

A pre-fitted ``estimator`` already published the target semantics of its own
predictions, so ``SklearnClassifier`` reconciles the declared semantics with its
learned classes by class identity before publishing any fitted attribute.
Declared ``classes`` may extend the learned vocabulary, and the additional
classes then receive zero-filled probability columns in the declared order.
They can neither reinterpret learned classes nor change the number of predicted
outputs, so equally wide but disjoint vocabularies are rejected rather than
silently relabeled.

.. doctest::

   >>> X_prefit = np.array([[-2.0], [-1.0], [1.0], [2.0]])
   >>> estimator = LogisticRegression().fit(X_prefit, [0, 0, 1, 1])
   >>> extended = SklearnClassifier(
   ...     estimator, classes=[0, 1, 2], missing_label=-1
   ... )
   >>> assert np.all(extended.predict_proba(X_prefit)[:, 2] == 0.0)
   >>> assert extended.target_spec_.classes == (0, 1, 2)
   >>> relabeled = SklearnClassifier(
   ...     estimator, classes=[2, 3], missing_label=-1
   ... )
   >>> try:
   ...     _ = relabeled.predict(X_prefit)
   ... except ValueError as error:
   ...     print("learned the class labels" in str(error))
   True
   >>> assert not hasattr(relabeled, "target_spec_")

A fitted multi-label estimator is accepted when it publishes one binary class
vocabulary per label output, as ``MultiOutputClassifier`` and a multi-output
``RandomForestClassifier`` do, or when it publishes explicit multi-label
metadata whose flat ``classes_`` identifies its label outputs, as
``OneVsRestClassifier`` does.  In the latter case, the flat classes are output
identifiers rather than one binary vocabulary, and each output carries a binary
indicator, so ``[[0, 1], ...]`` has to be declared.  A pre-fitted estimator
publishing neither representation cannot be declared multi-label at all,
because a flat learned vocabulary is indistinguishable from single-output
classification; fit such an estimator through the wrapper instead.

.. doctest::

   >>> from sklearn.multiclass import OneVsRestClassifier
   >>> y_prefit = np.array([[0, 1], [0, 1], [1, 0], [1, 0]])
   >>> one_vs_rest = OneVsRestClassifier(LogisticRegression()).fit(
   ...     X_prefit, y_prefit
   ... )
   >>> declared = SklearnClassifier(
   ...     one_vs_rest, classes=[[0, 1], [0, 1]], missing_label=-1
   ... )
   >>> assert declared.predict(X_prefit).shape == (4, 2)
   >>> assert declared.target_spec_.target_type == "multi-label"

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

All label outputs must declare classes of one dtype kind, because one array
holds every output of a sample.  Outputs may declare different vocabularies
and different widths of the same kind, e.g. ``("no", "yes")`` beside
``("off", "always")``.  Mixing kinds is rejected during resolution: strings
with numbers, integers with floats, and booleans with integers alike.  Were a
mixture accepted, the array would coerce the outputs to a common dtype, and
the labels describing a sample would no longer be the labels that were
declared, e.g. the integer ``0`` of one output would come back as the string
``'0'``.

.. doctest::

   >>> mixed_y = np.empty((2, 2), dtype=object)
   >>> mixed_y[:] = [["no", 0], ["yes", 1]]
   >>> _ = resolve_target_spec(  # doctest: +IGNORE_EXCEPTION_DETAIL
   ...     mixed_y,
   ...     task="classification",
   ...     target_type="multi-label",
   ...     classes=(("no", "yes"), (0, 1)),
   ...     missing_label=None,
   ... )
   Traceback (most recent call last):
   ...
   ValueError:

Predictions are described by the declared class labels, not by the wider
dtype that also has to represent ``missing_label``.  ``predict`` therefore
returns the dtype of ``classes_`` for a single-output target, and the dtype
its per-output vocabularies have in common for a multi-label target.  Integer
classes combined with the default ``missing_label=np.nan`` are the common
case: targets are held as ``float64`` so that missing labels fit beside them,
while predictions come back as ``int64`` and stay usable where class labels
are expected, e.g. as indices.

.. doctest::

   >>> from skactiveml.classifier import ParzenWindowClassifier
   >>> dtype_X = np.zeros((3, 1))
   >>> dtype_y = np.array([0, np.nan, 1])
   >>> dtype_clf = ParzenWindowClassifier(classes=[0, 1])
   >>> _ = dtype_clf.fit(dtype_X, dtype_y)
   >>> dtype_clf.predict(dtype_X).dtype == dtype_clf.classes_.dtype
   True

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

The current contract treats a multi-label sample as one complete vector.  Each
row must therefore be wholly observed or wholly missing; mixed rows would imply
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
classification, which current components reject as unsupported.

Single-output column vectors
============================

A target of shape ``(n_samples, 1)`` is accepted once its target semantics
resolve to single-output.  For classification, an explicit
``target_type="single-output"`` or a flat class vocabulary provides the
necessary evidence; classifiers and pool query strategies then convert the
column to the canonical one-dimensional representation and emit a
``DataConversionWarning``.  A bare classification column under
``target_type="auto"`` and ``classes=None`` remains an ambiguous
two-dimensional target.

For regression, both ``target_type="auto"`` and an explicit
``target_type="single-output"`` accept a column vector, preserving the
existing regression contract.  This holds wherever the task is known, i.e.
at a regressor and at a strategy that resolves its targets by one.  A
task-agnostic strategy knows neither the task nor a class vocabulary, so it
treats every bare two-dimensional target as ambiguous, whether its values
are continuous or discrete.  A target with more than one column is not a
single-output target for either task.

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
currently supported execution.  One-dimensional numerical targets resolve to
single-output regression, and column vectors remain compatible.  Multiple
target columns resolve to recognized ``"multi-output"`` regression semantics,
but regressors reject that valid specification because multi-output execution
is not yet a declared capability.  Regression target specifications always
have ``classes=None``.

A single-output regression target is described by one value per sample, so
``predict`` returns an array of shape ``(n_samples,)``.  A wrapped estimator
may describe one sample by a column instead; its predictions are narrowed to
the declared target type.  Predictions describing several target columns are
rejected rather than flattened, because flattening would silently turn them
into ``n_samples * n_outputs`` values that no longer describe a sample each.

.. doctest::

   >>> from skactiveml.regressor import SklearnRegressor
   >>> from sklearn.linear_model import LinearRegression
   >>> shape_X = np.zeros((3, 1))
   >>> shape_y = np.array([0.0, np.nan, 1.0])
   >>> shape_reg = SklearnRegressor(LinearRegression())
   >>> _ = shape_reg.fit(shape_X, shape_y)
   >>> shape_reg.predict(shape_X).shape
   (3,)

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

The contract deliberately recognizes without executing multi-output
classification and multi-output regression.  It also defers partial-label
querying and multi-label multi-annotator querying.  These are capability and
acquisition-scope limits, not architectural conflations: target type remains
separate from annotation type, and acquisition granularity is not stored in
``TargetSpec``.  Future support can therefore add exact component capabilities
and an explicit acquisition model without changing the meanings published
here.
