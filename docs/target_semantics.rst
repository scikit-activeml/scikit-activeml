.. _target-semantics:

===============================
Target and Annotation Semantics
===============================

Target semantics specify three properties of target data: the task, the target
type, and the annotation type. The target contract first resolves these
properties into a :class:`~skactiveml.utils.TargetSpec`. Each estimator or query
strategy can then check whether it supports that exact specification.

Resolution and capability checking are separate. A target specification can be
valid even if a particular estimator or query strategy does not support it.

Public API
==========

:func:`~skactiveml.utils.resolve_target_spec` determines the target semantics
from ``y`` and the declared ``task``, ``target_type``, ``annotation_type``,
optional class vocabularies, and ``missing_label``. It returns a frozen target
specification with four fields:

``task``
    ``"classification"`` or ``"regression"``.

``target_type``
    The resolved value ``"single-output"``, ``"multi-label"``, or
    ``"multi-output"``. A resolved specification never contains ``"auto"``.

``annotation_type``
    ``"single-annotator"`` or ``"multi-annotator"``.

``classes``
    The canonical immutable class vocabulary for classification, or ``None``
    for regression.

Public estimators and pool query strategies expose a ``target_type``
constructor parameter. The default ``"auto"`` keeps existing unambiguous
single-output calls working. After a successful fit, classifiers and regressors
expose the resolved ``target_spec_`` attribute.

Query strategies resolve the target semantics separately for each call and do
not store the specification from the previous query. If a fitted classifier is
passed to a strategy, its ``target_spec_`` is authoritative, i.e., it
determines what ``y`` means for that query.

Explicit multi-label classification
===================================

For multi-label classification, declare ``target_type="multi-label"`` instead
of relying on ``y`` being two-dimensional. If ``classes=None``, each output
column must contain both binary classes so that its vocabulary can be inferred.

The following example fits a classifier and queries one complete label vector.
It is executed as part of the documentation tests.

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

The following pool strategies support complete multi-label targets. This list
is checked against the exact capability inventory in
``skactiveml/pool/tests/test_multilabel_contracts.py``. The test groups
strategies by how they consume probabilities, whereas this documentation groups
them by how their acquisition method uses multi-label data. A new
multi-label-capable strategy therefore has to be added to the appropriate group
in both places.

The :doc:`Strategy Overview <generated/strategy_overview>` provides a
``Multi-Label`` filter and links to the available examples.

**Native multi-label methods**

    :class:`~skactiveml.pool.MaxLossReductionMaxConfidence` and
    :class:`~skactiveml.pool.LabelCardinalityInconsistency` implement methods
    designed for multi-label acquisition. Maximum Loss Reduction with Maximal
    Confidence is commonly abbreviated as **MMC** in the literature and
    tutorials. The importable class is
    :class:`~skactiveml.pool.MaxLossReductionMaxConfidence`; there is no separate
    ``MMC`` alias.

**Extensions of single-output methods**

    :class:`~skactiveml.pool.Badge`, :class:`~skactiveml.pool.Clue`,
    :class:`~skactiveml.pool.DropQuery`, :class:`~skactiveml.pool.Falcun`,
    :class:`~skactiveml.pool.ProbCover`,
    :class:`~skactiveml.pool.UHerding`, and
    :class:`~skactiveml.pool.UncertaintySampling` document how the corresponding
    single-output method is extended to multi-label data. Most of these methods
    compute a score for each label and reduce the scores to one utility per
    sample. ``ProbCover`` differs in that it uses the observed label rows when
    choosing its default radius. See the documentation of each class for the
    exact extension and reduction rule.

**Representation- and mask-only methods**

    :class:`~skactiveml.pool.CoreSet`,
    :class:`~skactiveml.pool.DiscriminativeAL`,
    :class:`~skactiveml.pool.GreedySamplingX`,
    :class:`~skactiveml.pool.MaxHerding`,
    :class:`~skactiveml.pool.RandomSampling`, and
    :class:`~skactiveml.pool.TypiClust` use sample representations and the
    labeled/unlabeled mask. The label values themselves do not affect the
    acquisition.

:class:`~skactiveml.pool.ParallelUtilityEstimationWrapper` and
:class:`~skactiveml.pool.SubSamplingWrapper` inherit multi-label support from
the strategy they wrap.

Estimator capability for multi-label wrapping
---------------------------------------------

``SklearnClassifier`` accepts an estimator for multi-label classification only
if the estimator is a ``scikit-learn`` classifier, implements ``predict_proba``,
and declares either ``target_tags.multi_output`` or
``classifier_tags.multi_label`` as supported. This capability is determined
from the estimator metadata and is never inferred by fitting generated data.

For example, a plain ``LogisticRegression`` implements ``predict_proba`` but
declares neither tag. It is therefore rejected before any fitted state is stored
instead of silently falling back to prior-only predictions.

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

A pre-fitted ``estimator`` already has learned target semantics. Before
``SklearnClassifier`` exposes any fitted attributes, it checks that the declared
semantics are consistent with the estimator's learned classes by class identity.

Declared ``classes`` may extend the learned class vocabulary. Any additional
classes then receive zero-filled probability columns in the declared order.
However, the declaration may neither reinterpret learned classes nor change the
number of predicted outputs. Consequently, equally wide but disjoint class
vocabularies are rejected instead of being silently relabeled.

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

A fitted multi-label estimator is accepted in either of two cases. First, it may
provide one binary class vocabulary per label output, as
``MultiOutputClassifier`` and a multi-output ``RandomForestClassifier`` do.
Second, it may provide explicit multi-label metadata together with a flat
``classes_`` that identifies the label outputs, as ``OneVsRestClassifier`` does.

For ``OneVsRestClassifier``, the flat classes identify outputs rather than the
binary class vocabulary of each output. Because each output is a binary
indicator, ``[[0, 1], ...]`` must therefore be declared explicitly. A pre-fitted
estimator that provides neither representation cannot be declared as multi-label:
a flat learned class vocabulary alone cannot be distinguished from
single-output classification. Such an estimator must instead be fitted through
the wrapper.

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

Multi-label classification uses one binary class vocabulary for each label
output. Explicit vocabularies allow fitting to start before both classes have
been observed and also support non-numeric labels. The order in which a
vocabulary is provided does not define the probability-column order. Each
vocabulary is normalized to the same canonical order used by fitted
``classes_``.

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

All label outputs must use classes of the same dtype kind because one array
stores all outputs of a sample. Different outputs may still use different
binary vocabularies within the same kind, for example ``("no", "yes")`` next
to ``("off", "always")``.

Mixing dtype kinds is rejected during resolution, including strings with
numbers, integers with floats, and booleans with integers. Otherwise, the array
would coerce the outputs to a common dtype and could change the declared class
labels. For example, the integer ``0`` of one output could be returned as the
string ``'0'``.

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

Predictions use the dtype of the declared class labels, not the potentially
wider dtype required by ``missing_label``. For a single-output target,
``predict`` therefore returns the dtype of ``classes_``. For a multi-label
target, it returns the common dtype of the per-output vocabularies.

A common case is integer classes together with the default
``missing_label=np.nan``. The target array then uses ``float64`` so that it can
contain ``np.nan``, while predictions use ``int64`` and remain valid wherever
class labels are expected, for example as indices.

.. doctest::

   >>> from skactiveml.classifier import ParzenWindowClassifier
   >>> dtype_X = np.zeros((3, 1))
   >>> dtype_y = np.array([0, np.nan, 1])
   >>> dtype_clf = ParzenWindowClassifier(classes=[0, 1])
   >>> _ = dtype_clf.fit(dtype_X, dtype_y)
   >>> dtype_clf.predict(dtype_X).dtype == dtype_clf.classes_.dtype
   True

If ``classes`` is not specified, target resolution never assumes a ``(0, 1)``
vocabulary. A label column with fewer than two observed classes therefore raises
an error.

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

The current contract treats the multi-label target of a sample as one complete
label vector. A row must therefore be either fully observed or fully missing.
A partially observed row would require partial-label training or acquisition,
which is not supported.

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

For single-annotator classification, a two-dimensional ``y`` is ambiguous when
``target_type="auto"`` and ``classes=None``. Its columns could represent binary
label outputs or distinct outputs of a future multi-output classification task.
Binary-looking values do not resolve this ambiguity.

Specify ``target_type``, provide a flat or nested class vocabulary, or pass a
fitted estimator whose ``target_spec_`` already resolves the target semantics.

.. doctest::

   >>> _ = resolve_target_spec(  # doctest: +IGNORE_EXCEPTION_DETAIL
   ...     np.array([[0, 1], [1, 0]]), task="classification"
   ... )
   Traceback (most recent call last):
   ...
   ValueError:

With ``target_type="auto"``, a flat class vocabulary resolves to single-output
classification. A nested set of binary vocabularies resolves to multi-label
classification. A nested vocabulary containing a non-binary output resolves to
multi-output classification. This target type is recognized but is not yet
supported by current components.

Single-output column vectors
============================

A target with shape ``(n_samples, 1)`` is accepted once its semantics resolve to
single-output. For classification, either an explicit
``target_type="single-output"`` or a flat class vocabulary provides enough
information. Classifiers and pool query strategies then convert the column to
the canonical one-dimensional representation and emit a
``DataConversionWarning``.

A classification column with ``target_type="auto"`` and ``classes=None`` remains
an ambiguous two-dimensional target.

For regression, both ``target_type="auto"`` and an explicit
``target_type="single-output"`` accept a column vector, preserving the existing
regression behavior. This applies whenever the task is known, for example in a
regressor or in a strategy that resolves its targets through a regressor.

A task-agnostic strategy has neither a known task nor a class vocabulary.
Therefore, it treats every bare two-dimensional target as ambiguous, regardless
of whether the values are continuous or discrete. A target with more than one
column is not single-output for either classification or regression.

Target-aware masks and indices
==============================

:func:`~skactiveml.utils.is_labeled`,
:func:`~skactiveml.utils.is_unlabeled`,
:func:`~skactiveml.utils.labeled_indices`, and
:func:`~skactiveml.utils.unlabeled_indices` accept a keyword-only ``target_type``
argument.

With the default ``target_type="single-output"``, their behavior remains
elementwise, including for multi-annotator matrices. With
``target_type="multi-label"``, they require each row to be fully observed or
fully missing and return sample-level masks or indices.

These helpers do not accept ``"auto"``. Pass the concrete ``target_type`` from a
resolved target specification.

.. doctest::

   >>> from skactiveml.utils import is_unlabeled, unlabeled_indices
   >>> is_unlabeled(y, target_type=clf.target_spec_.target_type).tolist()
   [False, False, False, False, True, True]
   >>> unlabeled_indices(y, target_type="multi-label").tolist()
   [4, 5]

Regression
==========

For currently supported regression, regressors accept ``target_type="auto"``
and ``target_type="single-output"``. One-dimensional numeric targets resolve to
single-output regression, and column vectors remain supported.

Targets with multiple columns resolve to the recognized
``target_type="multi-output"`` semantics. The specification itself is valid,
but regressors reject it because multi-output regression is not yet a supported
capability. Regression target specifications always have ``classes=None``.

A single-output regression target contains one value per sample, so ``predict``
returns an array of shape ``(n_samples,)``. A wrapped estimator may instead
return one prediction per sample as a column; this column is reduced to the
shape required by the declared single-output target type.

Predictions with several target columns are rejected rather than flattened.
Flattening them would produce ``n_samples * n_outputs`` values and would no
longer preserve one target value per sample.

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

Target type and annotation type are independent. In a target observation matrix
with multi-annotator annotation type, the columns represent annotators that
provide observations for the same single-output target. They do not represent
separate label outputs.

A sample may therefore contain both observed and missing annotator labels.
Existing multi-annotator estimators and strategies retain the multi-annotator
annotation type and continue to query sample-annotator pairs, so query results
still identify ``(sample, annotator)`` pairs.

Errors and component capabilities
=================================

Invalid target semantics raise an error during resolution. Examples include an
unknown ``target_type`` or ``target_type="multi-label"`` for regression.

A different case is a valid target specification that a particular component
does not support. This passes resolution and fails during capability checking.
The resulting error reports both the requested specification and the supported
combinations. The distinction indicates whether the target declaration itself
must be corrected or a different component is required.

After fitting, use ``estimator.target_spec_`` rather than inferring the semantics
again from the shape of the target array. In particular, use its ``target_type``
and ``annotation_type`` to choose downstream behavior and its ``classes`` as the
canonical class vocabulary for classification.

Recognized future semantics
===========================

The target contract already recognizes multi-output classification and
multi-output regression, although current components do not execute them.
Partial-label querying and multi-label multi-annotator querying are also not yet
supported.

These are limits of current component capabilities and acquisition scope. They
do not change the distinction between target type and annotation type, and
acquisition granularity is not part of ``TargetSpec``. Future support can
therefore add the required component capabilities and an explicit acquisition
model without changing the target semantics defined here.
