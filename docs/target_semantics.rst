.. _target-semantics:

===============================
Target and Annotation Semantics
===============================

The meaning of ``y`` depends on the task, target type, and annotation type.
Target resolution combines these properties in a
:class:`~skactiveml.utils.TargetSpec`. An estimator or query strategy then
checks whether it supports that specification. A specification can be valid
even when a component does not support it.

These semantics apply to pool and stream workflows alike. The
:doc:`Core Workflow <core_workflow>` defines each property, and the
:doc:`Strategy Overview <generated/strategy_overview>` filters strategies by
them.

Public API
==========

:func:`~skactiveml.utils.resolve_target_spec` uses ``y``, ``task``,
``target_type``, ``annotation_type``, ``missing_label``, and any declared class
vocabularies to return an immutable specification with four fields:

``task``
    ``"classification"`` or ``"regression"``.

``target_type``
    The resolved value ``"single-output"`` or ``"multi-label"``. A resolved
    specification never contains ``"auto"``.

``annotation_type``
    ``"single-annotator"`` or ``"multi-annotator"``.

``classes``
    The normalized class vocabulary for classification, stored as a tuple or
    a tuple of tuples. For regression, this is ``None``.

Components that interpret the labels themselves accept a ``target_type``
constructor parameter: classifiers that support multi-label targets, e.g.,
:class:`~skactiveml.classifier.ParzenWindowClassifier`, and pool strategies
that use no estimator for ``y``, e.g., :class:`~skactiveml.pool.RandomSampling`.
Other strategies take the target type from the estimator they receive, and the
remaining components support single-output labels only. A class's docstring
states whether it accepts ``target_type``. The default, ``"auto"``, infers the
type for unambiguous single-output inputs. After a successful fit, classifiers
and regressors store the resolved specification in ``target_spec_``.

Query strategies resolve the specification on each call. If you pass a fitted
estimator, its ``target_spec_`` determines how the strategy interprets ``y``.
For single-output classification, single-annotator pool strategies expect one
label per sample even when a fitted classifier was trained on several
annotators' labels; see :ref:`multiple-annotators`. Multi-label queries retain
one label per output for each sample. Strategies do not retain a specification
between queries.

An unfitted estimator declares the target semantics through its
``missing_label`` and, for classifiers, ``classes`` and, where available,
``target_type``, which are resolved together with ``y``. A
:class:`~skactiveml.classifier.SlidingWindowClassifier` without ``classes``
declares the classes of its ``estimator``. A strategy that fits a classifier,
e.g., with ``fit_clf=True``, first checks that the classifier supports
single-annotator labels, then fits a clone and uses its ``target_spec_``. A
target specification that the strategy does not support may therefore be
rejected only after training.

.. _label-contract:

Label and Missing-value Contract
================================

The following rules define valid class vocabularies, observed labels, and
missing labels. They apply to classifiers, regressors, the label encoder,
label aggregation, label helpers, and query strategies. The rules are
deliberately simple: labels are strings or real numbers, and rare combinations
of inputs are not handled separately.

``classes``
    The allowed categories for classification. Single-output classification
    has one vocabulary; multi-label classification has one binary vocabulary
    per output. Missing labels are excluded from class vocabularies.
    Regression has no class vocabulary.

``y``
    Observed labels together with missing entries. Classification labels are
    categories, and regression labels are numerical. Missing entries are
    excluded before the observed labels are checked against a vocabulary or
    used to infer one.

``missing_label``
    The value used for an unknown label. Numeric ``np.nan`` and the string
    ``"nan"`` are different values. Only the configured value marks a missing
    entry. For example, with ``missing_label=None``, an observed ``np.nan``
    is invalid.

Supported Values and Missing Labels
-----------------------------------

The contract consists of seven rules:

1. **Conversion.** Lists and tuples follow NumPy's conversion, so ``[0, 1.5]``
   becomes a floating-point array. The exception is a list that NumPy would
   turn into strings, such as ``["cat", np.nan]`` or ``["a", 1]``. It becomes
   an object array instead, so its values are checked as they are, and a NaN
   marker cannot become the class ``"nan"``. Arrays are taken as they are.
2. **Values.** Labels and classes are strings or real numbers. Boolean values,
   integers, and floating-point values are all numbers and may be mixed.
   Mixing strings and numbers raises a ``TypeError``, as do complex numbers,
   bytes, datetimes, and timedeltas.
3. **Vocabularies.** A class vocabulary is one-dimensional, unique, and
   finite, and it contains neither ``None`` nor the missing label. Classes are
   unique by value, so ``[0, 0.0]`` contains a duplicate.
4. **Missing labels.** String labels accept ``None`` or a string as the
   missing label. Numeric labels accept ``None``, ``np.nan``, or a finite
   number. Numeric ``np.nan`` never matches the string ``"nan"``.
5. **Multi-label.** Each row is fully labeled or fully missing, every output
   is binary, and the outputs are either all strings or all numbers.
6. **Regression.** Labels are finite real numbers other than Boolean values.
7. **Dtypes.** Predictions use the dtype of ``classes_``. Labels with missing
   entries are stored in NumPy's common dtype of the classes and the missing
   label, or in ``object`` for ``missing_label=None``.

.. doctest::

   >>> import numpy as np
   >>> from skactiveml.classifier import ParzenWindowClassifier
   >>> label_X = np.zeros((3, 1))
   >>> label_y = np.array([0, np.nan, 1.5])
   >>> label_clf = ParzenWindowClassifier(classes=[0, 1.5])
   >>> label_clf.fit(label_X, label_y).classes_.tolist()
   [0.0, 1.5]
   >>> ParzenWindowClassifier(classes=["cat", 1]).fit(label_X, label_y)
   Traceback (most recent call last):
       ...
   TypeError: `classes` mixes strings and numbers. ...

For ``y``, validation uses the array dtype, so inferring the classes from
``np.array([0.0, 1.0])`` gives floating-point classes even though the values
are integral. Declare ``classes`` explicitly when you need to control the
prediction dtype.

For classification, the missing label must be absent from every class
vocabulary. For example, ``classes=[0, 1]`` with ``missing_label=1`` is invalid
because ``1`` would also be a known class.

An empty or entirely missing ``y`` has no observed labels from which to infer
a vocabulary. Components that need one, such as the target resolver and label
encoder, require ``classes`` in this case.

Single-output Classification Examples
-------------------------------------

Every nonmissing label must match a declared class by value. Numeric labels
may use a wider storage dtype to accommodate missing values: ``0.0`` matches
the declared integer class ``0``, but the string ``"0"`` does not. Boolean
labels are numbers as well, so ``True`` matches the declared class ``1``.
Predictions use the declared class dtype.

``np.array([True, np.nan, False])`` produces floating-point values. The example
below uses object storage and ``missing_label=None`` to preserve Boolean
labels.

.. doctest::

   >>> bool_y = np.array([True, None, False], dtype=object)
   >>> bool_clf = ParzenWindowClassifier(classes=[0, 1], missing_label=None)
   >>> bool_clf.fit(label_X, bool_y).predict(label_X).dtype
   dtype('int64')

In the following tables, ``NaN`` denotes numeric ``np.nan``.

.. list-table:: Single-output classification outcomes
   :header-rows: 1
   :widths: 22 15 28 35

   * - ``classes``
     - ``missing_label``
     - ``y``
     - Outcome
   * - ``[0, 1]``
     - ``NaN``
     - ``[0, NaN, 1]``
     - Accept: missing entries are not labels.
   * - ``[0, 1]``
     - ``-2.5``
     - ``[0, -2.5, 1]``
     - Accept: the missing label may be fractional.
   * - ``[False, True]``
     - ``None``
     - ``[False, None, True]``
     - Accept: Boolean categories with a missing entry.
   * - ``[0, 1]``
     - ``None``
     - ``[True, None, False]``
     - Accept: Boolean labels match numeric classes by value.
   * - ``[0, 1.5]``
     - ``NaN``
     - ``[0, 1.5]``
     - Accept: integers and floating-point values are both numbers.
   * - ``["cat", "dog"]``
     - ``"?"``
     - ``["cat", "?", "dog"]``
     - Accept: a string missing label outside the vocabulary.
   * - ``[0, 1]``
     - ``NaN``
     - ``[0, 2, NaN]``
     - Reject: ``2`` is not a declared class.
   * - ``[0, 1]``
     - ``1``
     - ``[0, 1]``
     - Reject: the missing label is also a class.
   * - ``["cat", 1]``
     - ``None``
     - ``["cat", 1]``
     - Reject: strings and numbers are mixed.
   * - ``None``
     - ``NaN``
     - ``[0, 1.5]``
     - Accept: the list becomes a floating-point array; infer ``(0.0, 1.5)``.
   * - ``None``
     - ``"?"``
     - ``["cat", "?", "dog"]``
     - Accept: infer the vocabulary from ``"cat"`` and ``"dog"``.
   * - ``None``
     - ``None``
     - ``[0.0, NaN, 1.0]``
     - Reject: NaN is not the configured missing label and is not a category.
   * - ``None``
     - ``NaN``
     - ``[NaN, NaN]``
     - Reject: no vocabulary can be inferred; declare ``classes``.

For example, ``missing_label=None`` makes NaN an invalid observed label:

.. doctest::

   >>> nan_y = np.array([0.0, np.nan, 1.0])
   >>> ParzenWindowClassifier(missing_label=None).fit(label_X, nan_y)
   Traceback (most recent call last):
       ...
   ValueError: `y` contains NaN, which is only valid as the configured `missing_label`.

Multi-label Classification Examples
-----------------------------------

For ``target_type="multi-label"``, each output has exactly two classes and
each sample has one value per output. One array stores all outputs, so the
vocabularies of all outputs must be either all strings or all numbers.
Otherwise, NumPy would convert the numbers to strings, such as an integer class
``0`` into ``"0"``. Numeric vocabularies may differ in dtype, and different
binary vocabularies are allowed.

All outputs share one missing label. Each row must be fully observed or fully
missing. Partially observed rows are rejected because partial-label training
and acquisition are unsupported.

.. list-table:: Multi-label classification outcomes
   :header-rows: 1
   :widths: 29 15 26 30

   * - ``classes``
     - ``missing_label``
     - Example row in ``y``
     - Outcome
   * - ``[[0, 1], [2, 3]]``
     - ``NaN``
     - ``[0, 2]`` or ``[NaN, NaN]``
     - Accept: different integer vocabularies and complete rows.
   * - ``[[0, 1], [2, 3]]``
     - ``NaN``
     - ``[0, NaN]``
     - Reject: partially missing vector.
   * - ``[["no", "yes"], ["off", "on"]]``
     - ``None``
     - ``["no", "off"]`` or ``[None, None]``
     - Accept: string vocabularies with a compatible missing label.
   * - ``[[0, 1], [0.0, 1.0]]``
     - ``NaN``
     - ``[0, 1.0]``
     - Accept: integer and floating-point vocabularies are both numeric.
   * - ``[["no", "yes"], [0, 1]]``
     - ``None``
     - ``["no", 1]``
     - Reject: string and numeric vocabularies.
   * - ``[[0, 1, 2], [0, 1]]``
     - ``NaN``
     - ``[0, 1]``
     - Reject: one vocabulary is not binary, so this is not multi-label.

With ``classes=None``, each column must contain exactly two observed categories
to infer its binary vocabulary. The resolver does not assume ``(0, 1)``. For
example, ``[[0, 0], [1, 0]]`` contains only one class for the second output,
so you must declare ``classes``.

The complete-row rule applies to label outputs. In a multi-annotator matrix,
each column represents an annotator, and ``[0, NaN]`` is valid. See
:ref:`multiple-annotators`.

Regression Examples
-------------------

Regression requires ``classes=None``. Observed labels must be finite real
numbers other than Boolean values.

.. list-table:: Regression outcomes, with no class vocabulary
   :header-rows: 1
   :widths: 18 35 47

   * - ``missing_label``
     - ``y``
     - Outcome
   * - ``NaN``
     - ``[0, 1.5, NaN]``
     - Accept: integer and floating-point labels.
   * - ``None``
     - ``[0, 1.5, None]``
     - Accept: numerical labels with a missing entry.
   * - ``-999``
     - ``[0, 1.5, -999]``
     - Accept: ``-999`` denotes missing, not a label.
   * - ``"?"``
     - ``[0, 1.5, "?"]``
     - Reject: a string missing label beside numerical labels.
   * - ``None``
     - ``[0, NaN, None]``
     - Reject: NaN is not the configured missing label.
   * - ``NaN``
     - ``[True, False]``
     - Reject: Boolean values are not numerical labels.

The label helpers :func:`~skactiveml.utils.is_labeled` and
:func:`~skactiveml.utils.is_unlabeled` do not know the prediction task. They
check the rules that classification and regression share. Components that know
the task also apply the regression rule.

.. doctest::

   >>> from skactiveml.utils import is_unlabeled
   >>> is_unlabeled(["cold", "?", "warm"], missing_label="?")
   array([False,  True, False])
   >>> is_unlabeled(
   ...     np.array([0, 1.5, None], dtype=object), missing_label=None
   ... )
   array([False, False,  True])
   >>> is_unlabeled(["cold", 2])
   Traceback (most recent call last):
       ...
   TypeError: `y` mixes strings and numbers. ...

With ``missing_label=-1``, every ``-1`` is treated as missing, including an
intended regression value. Use ``np.nan`` or ``None`` when a numeric
placeholder could also be an observed label.

The string ``"nan"`` is incompatible with numeric labels. Use ``np.nan`` to
mark numeric NaN values as missing.

.. doctest::

   >>> is_unlabeled(np.array([0.0, np.nan, 1.0]), missing_label="nan")
   Traceback (most recent call last):
       ...
   TypeError: `missing_label='nan'` is a string and is not compatible with
   the numeric labels in `y`. ...
   >>> is_unlabeled(np.array([0.0, np.nan, 1.0]), missing_label=np.nan)
   array([False,  True, False])

.. _label-storage:

Storage and Prediction Dtypes
-----------------------------

Predictions use the dtype of the declared classes. Storing ``y`` may require a
wider dtype to accommodate the missing label. For declared ``int64`` classes
``[0, 1]`` and ``missing_label=np.nan``, ``[0, np.nan, 1]`` is stored as
``float64``, while ``predict`` returns ``int64``. Multi-label predictions use
the common dtype of the output vocabularies.

.. doctest::

   >>> import numpy as np
   >>> from skactiveml.classifier import ParzenWindowClassifier
   >>> dtype_X = np.zeros((3, 1))
   >>> dtype_y = np.array([0, np.nan, 1])
   >>> dtype_clf = ParzenWindowClassifier(classes=[0, 1])
   >>> _ = dtype_clf.fit(dtype_X, dtype_y)
   >>> dtype_clf.predict(dtype_X).dtype == dtype_clf.classes_.dtype
   True

Integer labels above ``2**53`` are unsupported, because ``float64``
represents integers exactly only up to ``2**53``.

By default, the label encoder's ``inverse_transform`` uses a dtype that can
store the configured missing label. Set ``prefer_class_dtype=True`` to decode
fully observed labels into the class dtype. For example, integer classes with
``missing_label=None`` use object storage for missing entries but keep their
integer dtype for fully observed labels.

.. doctest::

   >>> from skactiveml.utils import ExtLabelEncoder
   >>> encoder = ExtLabelEncoder(classes=[0, 1], missing_label=None)
   >>> encoder = encoder.fit([0, None])
   >>> encoder.inverse_transform([0, -1]).tolist()
   [0, None]
   >>> encoder.inverse_transform([0, 1], prefer_class_dtype=True).dtype
   dtype('int64')

Explicit Multi-label Classification
===================================

Set ``target_type="multi-label"`` on the classifier to declare multi-label
classification. A two-dimensional ``y`` alone does not identify the target
type. A strategy that receives the classifier takes the target type from it.
The example fits a classifier and queries one complete label vector.

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
   >>> strategy = UncertaintySampling(method="entropy", random_state=0)
   >>> query_indices = strategy.query(
   ...     X=X, y=y, clf=clf, fit_clf=False
   ... )
   >>> assert query_indices.shape == (1,)
   >>> assert query_indices[0] in (4, 5)

.. _multilabel-strategy-inventory:

Multi-label Pool Strategy Capabilities
--------------------------------------

The following pool strategies query complete multi-label vectors. Use the
``Multi-label`` filter in the
:doc:`Strategy Overview <generated/strategy_overview>` to find compatible
methods. Their API class pages link to the associated demonstrations.

Native multi-label methods
    :class:`~skactiveml.pool.MaxLossReductionMaxConfidence` and
    :class:`~skactiveml.pool.LabelCardinalityInconsistency` implement methods
    designed for multi-label data. Papers and tutorials abbreviate Maximum Loss
    Reduction with Maximal Confidence as MMC. Import
    :class:`~skactiveml.pool.MaxLossReductionMaxConfidence`; the package has no
    ``MMC`` alias.

Extensions of single-output methods
    :class:`~skactiveml.pool.Badge`, :class:`~skactiveml.pool.Clue`,
    :class:`~skactiveml.pool.DropQuery`, :class:`~skactiveml.pool.Falcun`,
    :class:`~skactiveml.pool.ProbCover`,
    :class:`~skactiveml.pool.UHerding`, and
    :class:`~skactiveml.pool.UncertaintySampling` extend single-output methods
    to multi-label data. Most compute scores for each label and combine them
    into one utility per sample. ``ProbCover`` uses observed label rows to
    choose its default radius. Each class documents its extension and score
    reduction rule.

Methods based on features and label availability
    :class:`~skactiveml.pool.CoreSet`,
    :class:`~skactiveml.pool.DiscriminativeAL`,
    :class:`~skactiveml.pool.GreedySamplingX`,
    :class:`~skactiveml.pool.MaxHerding`,
    :class:`~skactiveml.pool.RandomSampling`, and
    :class:`~skactiveml.pool.TypiClust` use sample representations and the
    labeled/unlabeled mask. Label values do not affect which samples they
    select.

:class:`~skactiveml.pool.ParallelUtilityEstimationWrapper` and
:class:`~skactiveml.pool.SubSamplingWrapper` inherit multi-label support from
the strategy they wrap.

Estimator Capability for Multi-label Wrapping
---------------------------------------------

To use an estimator for multi-label classification, ``SklearnClassifier``
requires a scikit-learn classifier with ``predict_proba`` and at least one of
the tags ``target_tags.multi_output`` or ``classifier_tags.multi_label`` set
to true. The wrapper checks the estimator's metadata without fitting test data.

A plain ``LogisticRegression`` has ``predict_proba`` but declares neither tag.
The wrapper rejects it before storing fitted state.

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

Pre-fitted Estimators
---------------------

When wrapping a pre-fitted estimator, ``SklearnClassifier`` checks the declared
target specification against the learned classes before exposing fitted
attributes.

The declared ``classes`` may add classes to the learned vocabulary. Extra
classes receive zero-filled probability columns in the wrapper's class order.
The declaration must retain the learned class labels and the number of outputs.
Replacing learned classes with different values is invalid, even if the
vocabularies have the same length.

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

A fitted multi-label estimator must provide either:

- One binary class vocabulary per output, as ``MultiOutputClassifier`` and a
  multi-output ``RandomForestClassifier`` do.
- Explicit multi-label metadata and a flat ``classes_`` identifying the label
  outputs, as ``OneVsRestClassifier`` does.

For ``OneVsRestClassifier``, each output is a binary indicator. Declare its
vocabulary explicitly with ``classes=[[0, 1], ...]``. The estimator's flat
``classes_`` identifies the outputs and does not supply these vocabularies.

If an estimator provides neither representation, fit it through the wrapper.
A flat learned class vocabulary alone is insufficient to identify a fitted
multi-label estimator.

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

Multi-label Vocabulary Resolution
---------------------------------

Declare vocabularies to start fitting before both classes of each output have
been observed. Vocabularies can also contain non-numeric labels. Probability
columns follow the normalized order in fitted ``classes_``, regardless of the
order in which you declared the classes.

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

The next examples show three invalid inputs under the :ref:`label-contract`:
strings in one output beside numbers in another, too few observed classes to
infer a binary vocabulary, and a partially observed row.

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
   TypeError:

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

Ambiguous Two-dimensional Classification
========================================

For single-annotator classification, a two-dimensional ``y`` is ambiguous with
``target_type="auto"`` and ``classes=None``. Its columns could be binary
multi-label outputs, or the annotators of a label matrix passed to a
single-annotator component by mistake. The shape and values alone do not
determine their meaning.

Declare multi-label targets where the labels are interpreted: on the
classifier, with ``target_type="multi-label"`` or a nested class vocabulary, or
with ``target_type="multi-label"`` on a strategy that uses no estimator for
``y``. Pass the labels of several annotators to a multi-annotator component. A
fitted estimator with a resolved ``target_spec_`` also suffices.

.. doctest::

   >>> _ = resolve_target_spec(  # doctest: +IGNORE_EXCEPTION_DETAIL
   ...     np.array([[0, 1], [1, 0]]), task="classification"
   ... )
   Traceback (most recent call last):
   ...
   ValueError:

With ``target_type="auto"``, a flat vocabulary selects single-output
classification, and a nested vocabulary selects multi-label classification.
Every nested vocabulary must be binary; see :ref:`current-limitations`.

Single-output Column Vectors
============================

Labels with shape ``(n_samples, 1)`` are accepted when the target type resolves
to single-output. For classification, set ``target_type="single-output"``,
declare a flat class vocabulary, or use a component without multi-label
support. In pool and stream queries, the estimator decides: the column stays
ambiguous if it supports multi-label targets. Classifiers and pool query
strategies convert a single-output column to a one-dimensional array and emit
``DataConversionWarning``. With ``target_type="auto"`` and ``classes=None``,
the column remains ambiguous for components that support multi-label targets.

For regression, a column vector is accepted whenever the task is known, for
example in a regressor or in a strategy that resolves the target specification
through a regressor.

A strategy without a known task or class vocabulary treats a two-dimensional
``y`` alone as ambiguous, including when it has continuous values. For both
classification and regression, single-output labels can have at most one
column.

Target-aware Masks and Indices
==============================

:func:`~skactiveml.utils.is_labeled`,
:func:`~skactiveml.utils.is_unlabeled`,
:func:`~skactiveml.utils.labeled_indices`, and
:func:`~skactiveml.utils.unlabeled_indices` accept a keyword-only ``target_type``
argument.

With the default, ``target_type="single-output"``, these helpers check each
entry, including entries in multi-annotator matrices. With
``target_type="multi-label"``, they check that each row is fully observed or
fully missing and return masks or indices for samples.

These helpers do not accept ``"auto"``. Pass ``target_type`` from a
resolved target specification.

.. doctest::

   >>> from skactiveml.utils import is_unlabeled, unlabeled_indices
   >>> is_unlabeled(y, target_type=clf.target_spec_.target_type).tolist()
   [False, False, False, False, True, True]
   >>> unlabeled_indices(y, target_type="multi-label").tolist()
   [4, 5]

Regression
==========

Regressors have no ``target_type`` parameter. One-dimensional numeric labels
resolve to single-output regression, and column vectors are supported.

By default, :class:`~skactiveml.regressor.SklearnRegressor` passes observed
labels to the wrapped estimator as ``float64``. This also applies to integer
labels and object arrays used with ``missing_label=None``.

With ``include_unlabeled_samples=True``, the wrapper passes ``y`` without dtype
conversion so that it retains the missing label. The wrapped estimator must
handle missing entries. The skorch regressor casts labels to ``float32`` in
both cases.

Labels with multiple columns are rejected because multi-output regression is
unsupported; see :ref:`current-limitations`.

For single-output regression, ``predict`` returns one value per sample in an
array of shape ``(n_samples,)``. If a wrapped estimator returns a single
column, the wrapper reduces it to this shape. Predictions with several columns
are rejected, as are samples drawn with ``sample_y`` that contain several
target columns: flattening them would produce ``n_samples * n_outputs`` values.
Non-finite predictions and samples are rejected as well.

.. doctest::

   >>> from skactiveml.regressor import SklearnRegressor
   >>> from sklearn.linear_model import LinearRegression
   >>> shape_X = np.zeros((3, 1))
   >>> shape_y = np.array([0.0, np.nan, 1.0])
   >>> shape_reg = SklearnRegressor(LinearRegression())
   >>> _ = shape_reg.fit(shape_X, shape_y)
   >>> shape_reg.predict(shape_X).shape
   (3,)

.. _multiple-annotators:

Multiple Annotators
===================

The target type describes the label to predict. The annotation type describes
who provides it. In a multi-annotator matrix, each column holds one annotator's
labels for the same task. A row such as ``[0, NaN]`` is valid: one annotator
provided a label, and the other has not.

Multi-annotator estimators retain this annotation type in their fitted
specification. Multi-annotator strategies return ``(sample, annotator)`` pairs
so that each requested label identifies both the sample and its annotator.
For pool queries, ``batch_size`` counts these pairs. Repeated requests for the
same sample count separately. In a single-annotator workflow, it instead counts
selected samples, each supplying one label or a complete multi-label vector.

Multi-annotator specifications currently require the single-output target type.
With ``target_type="auto"``, the resolver assigns ``"single-output"`` and
rejects nested class vocabularies. Combining
``annotation_type="multi-annotator"`` with ``target_type="multi-label"`` fails
during resolution. The two-dimensional label matrix has no separate dimensions
for annotators and label outputs. Supporting both would require another
representation; see :ref:`current-limitations`.

Compatible single-annotator pool strategies, such as
:class:`~skactiveml.pool.UncertaintySampling`, :class:`~skactiveml.pool.Badge`,
and :class:`~skactiveml.pool.Falcun`, can use a fitted multi-annotator
classifier with ``fit_clf=False`` and aggregated query labels, e.g., majority
votes. The classifier predicts one class distribution per sample. Query
validation therefore uses a single-annotator prediction specification while
preserving the classifier's original ``target_spec_`` and its exact class
vocabulary. This does not enable fitting a multi-annotator classifier on
aggregated labels with ``fit_clf=True``. Algorithm-specific classifier
requirements still apply, including any internal refitting a strategy performs.

:class:`~skactiveml.pool.MonteCarloEER` and
:class:`~skactiveml.pool.ValueOfInformationEER` require fitting hypothetical
single-annotator labels and therefore reject multi-annotator classifiers even
with ``fit_clf=False``. :class:`~skactiveml.pool.UHerding` accepts these
classifiers with a fixed ``temperatures`` value, such as ``temperatures=1.0``
or ``temperatures=[1.0]``. Automatic calibration, including the default
``temperatures=None``, requires internal refitting and is rejected for these
classifiers.

With ``candidates=None``, aggregated labels determine which samples are
unlabeled. Pass explicit candidates to include samples that have an aggregate
label but still lack labels from some annotators. Alternatively,
:class:`~skactiveml.pool.multiannotator.SingleAnnotatorWrapper` aggregates the
label matrix and selects sample-annotator pairs; it also accepts the fitted
classifier through ``fit_clf=False``.

An :class:`~skactiveml.classifier.multiannotator.AnnotatorEnsembleClassifier`
used with ``fit_ensemble=False`` as a committee in
:class:`~skactiveml.pool.QueryByCommittee`,
:class:`~skactiveml.pool.BatchBALD`, or :class:`~skactiveml.pool.GreedyBALD`
requires fitted member classifiers. Fitting this ensemble on an empty dataset
leaves its members unfitted, so committee queries raise a ``NotFittedError``.
To initialize a committee without observed labels, fit it with declared
``classes`` on a nonempty pool with an all-missing label matrix.

Stream query strategies apply the same rules: they accept a fitted
multi-annotator classifier with ``fit_clf=False`` and check labels ``y``, if
given, against its single-annotator prediction specification. They reject an
unfitted multi-annotator classifier, so fit it with declared ``classes`` on an
all-missing label matrix to start without labels. Before refitting, the
strategy checks that the classifier supports single-annotator labels,
including when ``y`` is a column vector. Their algorithm-specific requirements
apply as well.

Errors and Component Capabilities
=================================

Errors occur at two stages:

- Resolution rejects an invalid declaration, such as an unknown
  ``target_type`` or ``target_type="multi-label"`` for regression. Correct the
  declaration to match the task and labels.
- Capability checking rejects a valid specification that the component cannot
  use. The error reports the requested specification and supported
  combinations. Choose a component that supports the specification.

After fitting, read ``estimator.target_spec_`` to determine how to handle
labels. Use its ``target_type`` and ``annotation_type`` to interpret training
observations and its ``classes`` for the normalized classification vocabulary.
Single-annotator query strategies check their capability against the
single-annotator prediction specification of a fitted multi-annotator
classifier; see :ref:`multiple-annotators`.

Stream Query Strategies
-----------------------

Model-based stream query strategies support single-output, single-annotator
classification. They have no ``target_type`` parameter and use the classifier
passed to ``query`` to determine the target specification. If ``y`` is given,
it must match the single-annotator prediction specification derived from a
fitted classifier's ``target_spec_``, as for pool strategies. For an unfitted
classifier, resolution uses its ``classes``, ``missing_label``, and
``target_type`` together with ``y``. Without ``y``, only these declarations
count.

With ``fit_clf=True``, the strategy fits a clone and then uses its
``target_spec_``. The classifier's own capabilities are checked against its
training specification, or the resolved specification if it is unfitted. The
strategy's capability is checked against the single-annotator prediction
specification. A multi-label classifier fails the capability check, whereas a
fitted multi-annotator classifier passes both checks; see
:ref:`multiple-annotators`.

Passed labels are validated even with ``fit_clf=False``. This rejects some
calls accepted by earlier versions, including label matrices, invalid missing
labels, and labels outside the fitted class vocabulary. For example, a
classifier fitted on a shorter window without declared ``classes`` may not
know every class in the supplied history. Declare the full ``classes`` before
fitting, refit the classifier on the supplied labels, or use ``fit_clf=True``
with a compatible classifier and class declaration. If the strategy does not
use training labels when ``fit_clf=False``, omit the optional ``y`` argument.

:class:`~skactiveml.stream.StreamRandomSampling` and
:class:`~skactiveml.stream.PeriodicSampling` use only candidates and a budget.
They support single-output classification, multi-label classification, and
regression without inspecting labels. Budget managers use utilities to decide
which queries fit the budget.
:class:`~skactiveml.stream.budgetmanager.FixedUncertaintyBudgetManager` also
requires a flat class vocabulary to compute its threshold from the number of
classes.

.. doctest::

   >>> from sklearn.linear_model import SGDClassifier
   >>> from skactiveml.stream import VariableUncertainty
   >>> stream_X = np.array([[0.0], [1.0], [2.0], [3.0]])
   >>> stream_y = np.array([[0, 1], [1, 0], [-1, -1], [-1, -1]])
   >>> stream_clf = SklearnClassifier(
   ...     MultiOutputClassifier(
   ...         SGDClassifier(loss="log_loss", random_state=0)
   ...     ),
   ...     classes=[[0, 1], [0, 1]],
   ...     missing_label=-1,
   ...     target_type="multi-label",
   ... ).fit(stream_X, stream_y)
   >>> stream_qs = VariableUncertainty(random_state=0)
   >>> stream_qs.query(  # doctest: +IGNORE_EXCEPTION_DETAIL
   ...     candidates=stream_X, clf=stream_clf, X=stream_X, y=stream_y
   ... )
   Traceback (most recent call last):
       ...
   ValueError: VariableUncertainty does not support target capability ...

The same check rejects an unfitted classifier with a declared multi-label
vocabulary, including when ``y`` is absent:

.. doctest::

   >>> unfitted_clf = SklearnClassifier(
   ...     MultiOutputClassifier(
   ...         SGDClassifier(loss="log_loss", random_state=0)
   ...     ),
   ...     classes=[[0, 1], [0, 1]],
   ...     missing_label=-1,
   ...     target_type="multi-label",
   ... )
   >>> stream_qs.query(  # doctest: +IGNORE_EXCEPTION_DETAIL
   ...     candidates=stream_X, clf=unfitted_clf
   ... )
   Traceback (most recent call last):
       ...
   ValueError: VariableUncertainty does not support target capability ...

.. _current-limitations:

Unsupported Targets and Queries
===============================

The current limits are:

- Multi-output classification, where each column is a separate classification
  task with any number of classes, and multi-output regression are
  unsupported. A nested class vocabulary must therefore be binary, and a
  regression label must be one-dimensional or a column vector.
- Partial-label querying is unsupported. Multi-label queries request complete
  label vectors.
- Multi-label targets with multiple annotators fail during resolution. The
  label representation cannot express both annotator and output dimensions;
  see :ref:`multiple-annotators`.

``TargetSpec`` describes target and annotation types. It does not specify
whether a query acquires a whole label vector or part of one. Supporting
partial-label queries would also require an explicit model for that choice.
