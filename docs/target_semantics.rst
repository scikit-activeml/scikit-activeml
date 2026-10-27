.. _target-semantics:

===============================
Target and Annotation Semantics
===============================

Target semantics specify three properties of the labels ``y``: the task, the
target type, and the annotation type. Target resolution first resolves these
properties into a :class:`~skactiveml.utils.TargetSpec`. Each estimator or
query strategy can then check whether it supports that exact specification.

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
    The resolved value ``"single-output"`` or ``"multi-label"``. A resolved
    specification never contains ``"auto"``.

``annotation_type``
    ``"single-annotator"`` or ``"multi-annotator"``.

``classes``
    The canonical immutable class vocabulary for classification, or ``None``
    for regression.

Only components that interpret ``y`` themselves and support multi-label
targets have a ``target_type`` constructor parameter: classifiers such as
:class:`~skactiveml.classifier.ParzenWindowClassifier` and pool strategies
that use no estimator for ``y``, such as
:class:`~skactiveml.pool.RandomSampling`. Strategies that receive an estimator
take the target type from it, pool wrappers leave it to the strategy they
wrap, and stream strategies that use no labels, such as
:class:`~skactiveml.stream.StreamRandomSampling`, need none. All other
components support single-output labels only. A class's docstring states
whether it accepts ``target_type``. The default ``"auto"`` keeps unambiguous single-output calls
working. After a successful fit, classifiers and regressors expose the
resolved ``target_spec_`` attribute.

Query strategies resolve the target semantics separately for each call and do
not store the specification from the previous query. If a fitted estimator is
passed to a strategy, its ``target_spec_`` determines the task, target type,
and class vocabulary of ``y`` for that query. For single-output classification,
single-annotator pool strategies expect one label per sample even if a fitted
classifier was trained on multiple annotators' labels, see
:ref:`multiple-annotators`. Multi-label queries retain one label per output
for each sample.

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

Label and Missing-Value Contract
================================

This contract defines which class vocabularies, labels, and missing labels
are valid. It applies consistently to classifiers, regressors, the label
encoder, label aggregation, the public label helpers, and query strategies.
Its rules are deliberately simple: labels are strings or real numbers, and
rare combinations of inputs are not handled separately.

Three distinct concepts determine whether an input is valid:

``classes``
    The allowed categories for classification: one vocabulary for single-output
    classification, or one binary vocabulary per multi-label output. Missing
    labels are never classes. Regression has no class vocabulary.

``y``
    Observed labels together with missing entries. Classification labels are
    categories, and regression labels are numerical. Missing entries are
    excluded before the observed labels are checked against a vocabulary or
    used to infer one.

``missing_label``
    The single configured value denoting an unknown label. Only the
    configured value denotes missingness: with ``missing_label=None``, an
    observed ``np.nan`` is rejected rather than treated as another missing
    label.

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
   bytes, datetimes, and timedeltas. Integer labels above ``2**53`` are not
   supported, see :ref:`label-storage`.
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
   label, or in ``object`` for ``missing_label=None``, see
   :ref:`label-storage`.

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

Observed labels ``y`` are judged by their array representation, so inferring
the classes from ``np.array([0.0, 1.0])`` yields floating-point classes even
though the values are integral. If the dtype of predictions matters, declare
``classes`` explicitly.

An empty or entirely missing ``y`` does not show whether the labels are strings
or numbers; a declared vocabulary does. A classification consumer that must
infer a vocabulary, such as the target resolver or the label encoder,
therefore rejects an entirely missing ``y`` when ``classes=None``.

Single-output Classification Examples
-------------------------------------

Every nonmissing label must match a declared class by value. Numeric labels
may use a wider storage dtype to accommodate missing values: ``0.0`` matches
the declared integer class ``0``, but the string ``"0"`` does not. Boolean
labels are numbers as well, so ``True`` matches the declared class ``1``.
Predictions use the declared class dtype. Note that a floating-point missing
label would convert ``np.array([True, np.nan, False])`` to a floating-point
array. The following example therefore uses object storage with
``missing_label=None`` to keep the Boolean values.

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

The row with ``missing_label=None`` and an observed NaN raises:

.. doctest::

   >>> nan_y = np.array([0.0, np.nan, 1.0])
   >>> ParzenWindowClassifier(missing_label=None).fit(label_X, nan_y)
   Traceback (most recent call last):
       ...
   ValueError: `y` contains NaN, which is only valid as the configured `missing_label`.

Multi-label Classification Examples
-----------------------------------

For ``target_type="multi-label"``, each output has exactly two classes, and
each sample has one value per output. One array stores all outputs of a
sample, so the vocabularies of all outputs must be either all strings or all
numbers. Otherwise, NumPy would turn the numbers into strings, e.g., the
integer ``0`` of one output into the string ``"0"``. Numeric vocabularies may
differ in dtype, and different binary vocabularies are allowed.

One missing label is shared across all outputs. The complete-row rule
applies: the label vector of a sample must be fully observed or fully
missing. A partially
observed row would require partial-label training or acquisition, which is not
supported.

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
to infer its binary vocabulary; resolution never assumes a ``(0, 1)``
vocabulary. For example, the labels ``[[0, 0], [1, 0]]`` do not supply both
classes for the second output; explicit classes are needed.

The complete-row rule applies to label outputs only. The columns of a
multi-annotator matrix are annotators, not outputs, so a row ``[0, NaN]`` is
valid there. See :ref:`multiple-annotators`.

Regression Examples
-------------------

Regression has no class vocabulary: the target specification has
``classes=None``, and no class vocabulary may be supplied. Nonmissing labels
must be finite real numbers other than Boolean values.

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

The public label helpers :func:`~skactiveml.utils.is_labeled` and
:func:`~skactiveml.utils.is_unlabeled` are task-agnostic. They check the rules
that classification and regression share. Consumers that know the task apply
the stricter regression rule separately.

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

Regression cannot detect a collision between a numeric missing label and an
intended label: with ``missing_label=-1``, every ``-1`` denotes missing.
Prefer ``np.nan`` or ``None`` if a numeric placeholder could be a real
label. The string missing label ``"nan"`` is incompatible with numerical
labels (rule 4):

.. doctest::

   >>> is_unlabeled(np.array([0.0, np.nan, 1.0]), missing_label="nan")
   Traceback (most recent call last):
       ...
   TypeError: `missing_label='nan'` is a string and is not compatible with the numeric labels in `y`. ...
   >>> is_unlabeled(np.array([0.0, np.nan, 1.0]), missing_label=np.nan)
   array([False,  True, False])

.. _label-storage:

Storage and Prediction Dtypes
-----------------------------

Predictions use the dtype of the declared classes, not the potentially wider
dtype that ``missing_label`` requires for storing ``y``. For declared ``int64``
classes ``[0, 1]`` and ``missing_label=np.nan``, the labels ``[0, np.nan, 1]``
are stored as ``float64``, while ``predict`` returns ``int64``. For a
multi-label target type, predictions use the common dtype of the per-output
vocabularies.

.. doctest::

   >>> import numpy as np
   >>> from skactiveml.classifier import ParzenWindowClassifier
   >>> dtype_X = np.zeros((3, 1))
   >>> dtype_y = np.array([0, np.nan, 1])
   >>> dtype_clf = ParzenWindowClassifier(classes=[0, 1])
   >>> _ = dtype_clf.fit(dtype_X, dtype_y)
   >>> dtype_clf.predict(dtype_X).dtype == dtype_clf.classes_.dtype
   True

``float64`` represents integers exactly only up to ``2**53``, so integer
labels above ``2**53`` are not supported.

By default, ``inverse_transform`` of the label encoder decodes into the dtype
that also holds the configured missing label. With
``prefer_class_dtype=True``, fully observed labels are decoded into the class
dtype instead. Integer classes with ``missing_label=None``, for example, use
object storage for missing entries but keep their integer dtype for fully
observed labels.

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

For multi-label classification, declare ``target_type="multi-label"`` on the
classifier instead of relying on ``y`` being two-dimensional. A strategy that
receives the classifier takes the target type from it. The following example
fits a classifier and queries one complete label vector.

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

The following pool strategies support complete multi-label vectors. This list
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

Estimator Capability for Multi-label Wrapping
---------------------------------------------

``SklearnClassifier`` accepts an estimator for multi-label classification only
if the estimator is a ``scikit-learn`` classifier, implements ``predict_proba``,
and declares either ``target_tags.multi_output`` or
``classifier_tags.multi_label`` as supported. This capability is determined
from the estimator metadata and is never inferred by fitting generated data.

For example, a plain ``LogisticRegression`` implements ``predict_proba`` but
declares neither tag. It is therefore rejected instead of silently falling back
to prior-only predictions.

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

Pre-fitted Estimators
---------------------

A pre-fitted ``estimator`` already has learned target semantics.
``SklearnClassifier`` checks that the declared semantics are consistent with
the estimator's learned classes by value.

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

Multi-label Vocabulary Resolution
---------------------------------

Explicit vocabularies allow fitting to start before both classes of an output
have been observed and also support non-numeric labels. The order in which a
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

The following checks show the multi-label rules of the
:ref:`label-contract` during resolution: strings in one output beside numbers
in another, an output with fewer than two observed classes and no declared
vocabulary, and a partially observed row are rejected.

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

For single-annotator classification, a two-dimensional ``y`` is ambiguous when
``target_type="auto"`` and ``classes=None``. Its columns could represent binary
label outputs, or the annotators of a label matrix passed to a single-annotator
component by mistake. Binary-looking values do not resolve this ambiguity.

Declare multi-label targets where the labels are interpreted: on the
classifier, with ``target_type="multi-label"`` or a nested class vocabulary, or
with ``target_type="multi-label"`` on a strategy that uses no estimator for
``y``. Pass the labels of several annotators to a multi-annotator component. A
fitted estimator whose ``target_spec_`` resolves the target semantics also
suffices.

.. doctest::

   >>> _ = resolve_target_spec(  # doctest: +IGNORE_EXCEPTION_DETAIL
   ...     np.array([[0, 1], [1, 0]]), task="classification"
   ... )
   Traceback (most recent call last):
   ...
   ValueError:

With ``target_type="auto"``, a flat class vocabulary resolves to single-output
classification, and a nested vocabulary resolves to multi-label
classification. Every nested vocabulary must be binary, see
:ref:`current-limitations`.

Single-output Column Vectors
============================

Labels with shape ``(n_samples, 1)`` are accepted once their semantics resolve
to single-output. For classification, an explicit
``target_type="single-output"``, a flat class vocabulary, or a component
without multi-label support provides enough information. In pool and stream
queries, the estimator's declarations or its fitted ``target_spec_`` provide
this information. Classifiers and pool query strategies convert a
single-output column to the canonical one-dimensional representation and emit
a ``DataConversionWarning``. A classification column with ``target_type="auto"``
and ``classes=None`` remains ambiguous for components that support
multi-label targets.

For regression, a column vector is accepted whenever the task is known, for
example in a regressor or in a strategy that resolves its labels through a
regressor.

A task-agnostic strategy has neither a known task nor a class vocabulary.
Therefore, it treats every bare two-dimensional ``y`` as ambiguous, regardless
of whether the values are continuous or discrete. Labels with more than one
column are not single-output for either classification or regression.

Target-aware Masks and Indices
==============================

:func:`~skactiveml.utils.is_labeled`,
:func:`~skactiveml.utils.is_unlabeled`,
:func:`~skactiveml.utils.labeled_indices`, and
:func:`~skactiveml.utils.unlabeled_indices` accept a keyword-only ``target_type``
argument.

With the default ``target_type="single-output"``, their behavior is
elementwise, including for multi-annotator matrices. With
``target_type="multi-label"``, they enforce the complete-row rule and return
sample-level masks or indices.

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

Regressors have no ``target_type`` parameter. One-dimensional numeric labels
resolve to single-output regression, and column vectors are supported.

The estimator wrapped by :class:`~skactiveml.regressor.SklearnRegressor`
receives the observed labels as ``float64``, so that integer labels and the
object storage that ``missing_label=None`` requires reach it as ordinary
floating-point values, whereas ``include_unlabeled_samples=True`` passes the
raw ``y`` on unchanged, because only its own representation holds the missing
label. The skorch regressor casts its labels to ``float32`` in either case.

Labels with multiple columns are rejected, because multi-output regression is
not supported, see :ref:`current-limitations`.

A single-output regression label is one value per sample, so ``predict``
returns an array of shape ``(n_samples,)``. A wrapped estimator may instead
return one prediction per sample as a column. This column is reduced to shape
``(n_samples,)``.

Predictions with several columns are rejected rather than flattened, and so
are samples drawn with ``sample_y`` that contain several columns.
Flattening them would produce ``n_samples * n_outputs`` values and would no
longer preserve one label per sample. Non-finite predictions and samples are
rejected as well.

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

Target type and annotation type are separate properties of a target
specification: the target type describes what one label is, and the
annotation type describes who provides it. In a label matrix with
multi-annotator annotation type, the columns represent annotators that provide
labels for the same sample. They do not represent separate label outputs, so a
row ``[0, NaN]`` means that one annotator supplied a label and the other did
not.

A sample may therefore contain both observed and missing annotator labels.
Multi-annotator estimators retain the multi-annotator annotation type in their
fitted specification. Multi-annotator strategies query sample-annotator pairs,
so query results identify ``(sample, annotator)`` pairs.

Currently, a multi-annotator specification is always single-output. With
``target_type="auto"``, resolution assigns ``"single-output"`` and rejects
nested class vocabularies. Declaring ``target_type="multi-label"`` together
with ``annotation_type="multi-annotator"`` is rejected during resolution, not
during capability checking, because the two-dimensional label matrix can hold
either annotators or label outputs but not both. Lifting this restriction would
require a representation with both dimensions, see :ref:`current-limitations`.

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

Invalid target semantics raise an error during resolution. Examples include an
unknown ``target_type`` or ``target_type="multi-label"`` for regression.

A different case is a valid target specification that a particular component
does not support. This passes resolution and fails during capability checking.
The resulting error reports both the requested specification and the supported
combinations. The distinction indicates whether the target declaration itself
must be corrected or a different component is required.

After fitting, use ``estimator.target_spec_`` rather than inferring the semantics
again from the shape of ``y``. In particular, use its ``target_type`` and
``annotation_type`` to interpret its training observations and its ``classes``
as the canonical class vocabulary for classification. Single-annotator query
strategies check their capability against the single-annotator prediction
specification of a fitted multi-annotator classifier, see
:ref:`multiple-annotators`.

Stream Query Strategies
-----------------------

Stream query strategies that use a classifier support only single-output
single-annotator classification and have no ``target_type`` parameter. They
resolve the classifier and ``y`` as pool query strategies do: given labels
``y`` must match the single-annotator prediction specification of the
classifier, an unfitted classifier contributes its declarations, and with
``fit_clf=True`` the fitted clone's ``target_spec_`` applies. A multi-label
classifier is therefore rejected with the capability error described above,
whereas a fitted multi-annotator classifier passes, see
:ref:`multiple-annotators`.

Passed labels are validated even with ``fit_clf=False``. This rejects some
calls accepted by earlier versions, including label matrices, invalid missing
labels, and labels outside the fitted class vocabulary. For example, a
classifier fitted on a shorter window without declared ``classes`` may not
know every class in the supplied history. Declare the full ``classes`` before
fitting, refit the classifier on the supplied labels, or use ``fit_clf=True``
with a compatible classifier and class declaration. If the strategy does not
use training labels when ``fit_clf=False``, omit the optional ``y`` argument.

The label-free baselines :class:`~skactiveml.stream.StreamRandomSampling` and
:class:`~skactiveml.stream.PeriodicSampling` act on candidates and the budget
only. Like the representation- and mask-only pool methods above, they are
task-agnostic and declare single-output classification, multi-label
classification, and regression. The budget managers consume utilities only.
:class:`~skactiveml.stream.budgetmanager.FixedUncertaintyBudgetManager` merely
requires a flat class vocabulary, because it computes its threshold from the
number of classes.

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

An unfitted classifier declares the label semantics, so a declared
multi-label vocabulary is rejected the same way, even without labels ``y``.

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

Current Limitations
===================

Multi-output classification, where each column is a separate classification
task with any number of classes, and multi-output regression are not
supported. A nested class vocabulary must therefore be binary, and a
regression label must be one-dimensional or a column vector. Partial-label
querying is not supported either. Multi-label targets with multiple annotators
are rejected, see :ref:`multiple-annotators`.

Supporting these cases later would add component capabilities, a label
representation with an annotator dimension, or an explicit acquisition model.
It would not change the distinction between target type and annotation type
defined here.
