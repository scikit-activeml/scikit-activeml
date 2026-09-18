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
not store the specification from the previous query. If a fitted estimator is
passed to a strategy, its ``target_spec_`` determines what ``y`` means for
that query. With the default ``target_type="auto"``, the strategy adopts the
target type of the fitted estimator. An explicit strategy ``target_type`` must
agree with the fitted estimator; a conflicting declaration raises a
``ValueError`` rather than silently overriding either side.

.. _label-contract:

Label and Missing-Value Contract
================================

This contract defines which class vocabularies, labels, and missing labels
are valid. It applies consistently to classifiers, regressors, the label
encoder, label aggregation, the public label helpers, and query strategies.
Its guiding principle is predictability over flexibility: a rule is preferred
if users can anticipate its outcome without knowing implementation details
and if developers can enforce it in one place. Flexibility that only holds for
some estimators is not part of the contract; explicit object arrays retain
their value-by-value type information as described below.

Three distinct concepts determine whether an input is valid:

``classes``
    The allowed categories for classification: one vocabulary for single-output
    classification, one binary vocabulary per multi-label output, or one
    vocabulary per output for the recognized multi-output classification, see
    :ref:`recognized-future-semantics`. Missing labels are never classes.
    Regression has no class vocabulary.

``y``
    Observed labels together with missing entries. Classification labels are
    categories, and regression labels are numerical. Missing entries are
    excluded before the observed labels are checked against a vocabulary or
    used to infer one.

``missing_label``
    The single configured value denoting an unknown label. Numeric ``np.nan``
    and the string ``"nan"`` are different values. Only the configured value
    denotes missingness: with ``missing_label=None``, an observed ``np.nan``
    is rejected rather than treated as another missing label.

Supported Values and Missing Labels
-----------------------------------

Each class vocabulary contains one family: Boolean values, integers, finite
floating-point values, or Unicode strings. Integer values must fit a common
signed or unsigned integer dtype of at most 64 bits. Floating-point values may
use ``float16``, ``float32``, or ``float64``, including ordinary Python
``float`` values. Python and NumPy scalar equivalents are accepted.

Complex numbers, integers requiring more than 64 bits, extended-precision
floats, bytes, arbitrary numeric objects, and nested objects are outside this
contract. Neither ``None`` nor nonfinite numbers are categories. Object arrays
remain valid storage if their entries satisfy these rules; object storage alone
does not grant support for additional scalar types.

A declared ``classes`` argument is checked value by value. It is short, and its
exact types define the dtype of predictions. For example, ``classes=[0, 1.5]``
is rejected because it mixes integers and floating-point values, while
``classes=[0.0, 1.5]`` is accepted.

.. doctest::

   >>> import numpy as np
   >>> from skactiveml.classifier import ParzenWindowClassifier
   >>> family_X = np.zeros((3, 1))
   >>> family_y = np.array([0, np.nan, 1.5])
   >>> ParzenWindowClassifier(classes=[0, 1.5]).fit(family_X, family_y)
   Traceback (most recent call last):
       ...
   TypeError: `classes` must contain one label family, but mixes integer and floating-point values. ...
   >>> family_clf = ParzenWindowClassifier(classes=[0.0, 1.5])
   >>> family_clf.fit(family_X, family_y).classes_.tolist()
   [0.0, 1.5]

Observed labels ``y`` are judged by their array representation. A Python list
or tuple generally follows the NumPy conversion rules, so ``y=[0, 1.5]``
becomes a floating-point array, and with ``classes=None`` the inferred
vocabulary is ``(0.0, 1.5)``. The one exception is a lossy conversion: if
NumPy would round an integer, as for ``[2**53, 2**53 + 1, np.nan]``, or turn a
non-string value into a string, the conversion uses object storage instead,
so that every supported scalar keeps its value and its family. The family
rules then apply to the preserved values. Likewise, inferring from
``np.array([0.0, 1.0])`` yields
floating-point classes even though the values are integral. Only object arrays
are inspected value by value, because their dtype carries no information about
the values they hold. Thus an explicit object array containing integer and
floating-point entries mixes classification families and is rejected, even
where ordinary NumPy conversion of the same list produces one floating-point
array. The same holds for the Boolean and integer families: the list
``[False, 2]`` becomes an integer array and is accepted, whereas
``np.array([False, 2], dtype=object)`` mixes families and is rejected.
Ordinary label arrays are not scanned entry by entry. The large-integer case
of the lossless fallback is described in :ref:`label-storage`. If the dtype of
predictions matters, declare ``classes`` explicitly.

Numeric missing labels use the supported real scalar types above. NaN is
permitted specifically as a missing label, while infinities are rejected
both as missing labels and as labels.

.. list-table:: Missing-label compatibility
   :header-rows: 1
   :widths: 36 12 14 20 18

   * - Observed labels
     - ``None``
     - ``np.nan``
     - Finite numeric missing label
     - String missing label
   * - Numeric or Boolean categories
     - Yes
     - Yes
     - Yes
     - No
   * - String categories
     - Yes
     - No
     - No
     - Yes
   * - Numerical labels
     - Yes
     - Yes
     - Yes
     - No

For classification, the missing label must also be absent from every class
vocabulary. For example, ``classes=[0, 1]`` with ``missing_label=1`` is
invalid: ``1`` cannot mean both a known category and an unknown label. An
empty or entirely missing ``y`` provides no evidence about the label family;
a declared vocabulary supplies it. A classification consumer that must infer
a vocabulary, such as the target resolver or the label encoder, therefore
rejects an entirely missing ``y`` when ``classes=None``.

Single-output Classification Examples
-------------------------------------

Every nonmissing label must match a declared class by value. Numeric labels
may use a wider storage dtype to accommodate missing values: ``0.0`` matches
the declared integer class ``0``, but the string ``"0"`` does not. This does
not permit a mixed-family declaration such as ``classes=[0, 1.0]``.

Boolean labels match integer classes by value, and integer labels match
Boolean classes the same way, because both families describe the same
categories: ``True`` is the declared integer class ``1``. A Boolean/integer
mixture within one ``y`` or one ``classes``, such as
``np.array([False, 2], dtype=object)`` or ``classes=[False, 1]``, is
rejected as any other family mixture is. Predictions use the declared class
dtype, so the class vocabulary decides which of the two families is
returned. Note that a floating-point missing label would convert
``np.array([True, np.nan, False])`` to a floating-point array, which is then
judged by the floating-point rule above. The following example therefore uses
object storage with ``missing_label=None`` to keep the Boolean values.

.. doctest::

   >>> bool_y = np.array([True, None, False], dtype=object)
   >>> bool_clf = ParzenWindowClassifier(classes=[0, 1], missing_label=None)
   >>> bool_clf.fit(family_X, bool_y).predict(family_X).dtype
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
     - Accept: missing entries do not mix class families.
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
     - Accept: Boolean labels match integer classes by value.
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
   * - ``[0, 1.5]``
     - ``NaN``
     - ``[0, 1.5]``
     - Reject: mixed integer/float declaration. Declare ``[0.0, 1.5]`` instead.
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

Only the configured ``missing_label`` denotes a missing label, so an
observed NaN beside ``missing_label=None`` is an error rather than a second
one.

.. doctest::

   >>> nan_y = np.array([0.0, np.nan, 1.0])
   >>> ParzenWindowClassifier(missing_label=None).fit(family_X, nan_y)
   Traceback (most recent call last):
       ...
   ValueError: `y` contains NaN, which is only valid as the configured `missing_label`. ...

Multi-label Classification Examples
-----------------------------------

For ``target_type="multi-label"``, each output has exactly two classes, and
each sample has one value per output. All output vocabularies must share one
dtype kind: Boolean, signed integer, unsigned integer, floating-point, or
Unicode string. Different widths within a kind and different binary
vocabularies are allowed. One array stores all outputs of a sample, so mixed
kinds would be coerced to a common dtype, e.g., the integer ``0`` of one
output could be returned as the string ``"0"``. Following the principle of
predictability over flexibility, the contract forbids mixed kinds across
outputs outright, even where a lossless common dtype exists. A signed integer
vocabulary beside an unsigned integer vocabulary is therefore rejected, even
if their individual values fit.

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
     - Reject: integer and floating-point output vocabularies.
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
``classes=None``, and no class vocabulary may be supplied. Labels are
numerical. Unlike classification labels and class vocabularies, regression
labels may mix integer and floating-point families. Nonmissing labels must be
finite real numbers of the supported numeric types.

.. list-table:: Regression outcomes, with no class vocabulary
   :header-rows: 1
   :widths: 18 35 47

   * - ``missing_label``
     - ``y``
     - Outcome
   * - ``NaN``
     - ``[0, 1.5, NaN]``
     - Accept: mixed integer/float labels.
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

The public label helpers :func:`~skactiveml.utils.is_labeled` and
:func:`~skactiveml.utils.is_unlabeled` are task-agnostic. For single-output
targets they accept values valid for either classification or regression.
Consequently, an object array mixing integers and floating-point values is
accepted as possible regression data, while a Boolean/integer mixture is
rejected because it is valid for neither task. Consumers that know the task
apply the stricter classification or regression rule separately.

.. doctest::

   >>> from skactiveml.utils import is_unlabeled
   >>> is_unlabeled(["cold", "?", "warm"], missing_label="?")
   array([False,  True, False])
   >>> is_unlabeled(
   ...     np.array([0, 1.5, None], dtype=object), missing_label=None
   ... )
   array([False, False,  True])
   >>> is_unlabeled(np.array([False, 2], dtype=object))
   Traceback (most recent call last):
       ...
   TypeError: `y` must contain one label family, ...

Regression cannot detect a collision between a numeric missing label and an
intended label: with ``missing_label=-1``, every ``-1`` denotes missing.
Prefer ``np.nan`` or ``None`` if a numeric placeholder could be a real
label. Numeric NaN never matches the string missing label
``"nan"``; the two are different values, so the string missing label is
incompatible with numerical labels instead of denoting the NaN among them.

.. doctest::

   >>> is_unlabeled(np.array([0.0, np.nan, 1.0]), missing_label="nan")
   Traceback (most recent call last):
       ...
   TypeError: `missing_label='nan'` is a string and is not compatible with the floating-point labels in `y`. ...
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

Floating-point storage is lossless only for integers that ``float64``
represents exactly. Above ``2**53``, adjacent integer identifiers may become
the same floating-point value. When constructing an array containing large
integer identifiers and ``np.nan``, use ``dtype=object`` to preserve both the
identifiers and the missing label. Label validation and the label encoder
preserve exact integers in Python lists by choosing object storage when
needed; they cannot recover precision already lost in an array constructed by
the caller.

.. doctest::

   >>> from skactiveml.utils import ExtLabelEncoder
   >>> large_classes = [2**53, 2**53 + 1]
   >>> large_y = np.array([*large_classes, np.nan], dtype=object)
   >>> large_encoder = ExtLabelEncoder(classes=large_classes).fit(large_y)
   >>> large_encoder.transform(large_y).tolist()
   [0, 1, -1]
   >>> large_encoder.transform([*large_classes, np.nan]).tolist()
   [0, 1, -1]
   >>> large_encoder.inverse_transform([0, 1], prefer_class_dtype=True).tolist()
   [9007199254740992, 9007199254740993]

Mixed NumPy signed and unsigned integer scalars are normalized to a lossless
common integer dtype before sorting, so their identity is preserved as well.

.. doctest::

   >>> scalar_classes = [np.int64(2**53), np.uint64(2**53 + 1)]
   >>> scalar_encoder = ExtLabelEncoder(classes=scalar_classes).fit([])
   >>> scalar_encoder.classes_.tolist()
   [9007199254740992, 9007199254740993]

By default, ``inverse_transform`` decodes into a lossless dtype that also
accommodates the configured missing label. With ``prefer_class_dtype=True``,
fully observed labels are decoded into the lossless common class dtype instead.
Integer classes with ``None``, or large integer classes with ``np.nan``, can
therefore use object storage for missing entries while fully observed decoding
uses the class dtype.

Explicit Multi-label Classification
===================================

For multi-label classification, declare ``target_type="multi-label"`` instead
of relying on ``y`` being two-dimensional. The following example fits a
classifier and queries one complete label vector.

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

Pre-fitted Estimators
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
:ref:`label-contract` during resolution: mixed dtype kinds across outputs, an
output with fewer than two observed classes and no declared vocabulary, and a
partially observed row are rejected.

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
label outputs or distinct outputs of a multi-output classification task.
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
multi-output classification, see :ref:`recognized-future-semantics`.

Single-output Column Vectors
============================

Labels with shape ``(n_samples, 1)`` are accepted once their semantics resolve
to single-output. For classification, either an explicit
``target_type="single-output"`` or a flat class vocabulary provides enough
information. Classifiers and pool query strategies then convert the column to
the canonical one-dimensional representation and emit a
``DataConversionWarning``. A classification column with ``target_type="auto"``
and ``classes=None`` remains ambiguous.

For regression, both ``target_type="auto"`` and an explicit
``target_type="single-output"`` accept a column vector. This applies whenever
the task is known, for example in a regressor or in a strategy that resolves
its labels through a regressor.

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

Regressors accept ``target_type="auto"`` and ``target_type="single-output"``.
One-dimensional numeric labels resolve to single-output regression, and column
vectors are supported.

The estimator wrapped by :class:`~skactiveml.regressor.SklearnRegressor`
receives the observed labels as ``float64``, so that integer labels and the
object storage that ``missing_label=None`` requires reach it as ordinary
floating-point values, whereas ``include_unlabeled_samples=True`` passes the
raw ``y`` on unchanged, because only its own representation holds the missing
label. The skorch regressor casts its labels to ``float32`` in either case.

Labels with multiple columns resolve to the recognized
``target_type="multi-output"`` semantics. The specification itself is valid,
but regressors reject it because multi-output regression is not yet a supported
capability.

A single-output regression label is one value per sample, so ``predict``
returns an array of shape ``(n_samples,)``. A wrapped estimator may instead
return one prediction per sample as a column; this column is reduced to the
shape required by the declared single-output target type.

Predictions with several columns are rejected rather than flattened.
Flattening them would produce ``n_samples * n_outputs`` values and would no
longer preserve one label per sample.

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
labels for the same target. They do not represent separate label outputs, so a
row ``[0, NaN]`` means that one annotator supplied a label and the other did
not.

A sample may therefore contain both observed and missing annotator labels.
Multi-annotator estimators and strategies retain the multi-annotator annotation
type and query sample-annotator pairs, so query results identify
``(sample, annotator)`` pairs.

Currently, a multi-annotator specification is always single-output. With
``target_type="auto"``, resolution assigns ``"single-output"`` and rejects
nested class vocabularies. Declaring ``target_type="multi-label"`` or
``target_type="multi-output"`` together with
``annotation_type="multi-annotator"`` is rejected during resolution, not
during capability checking, because the two-dimensional label matrix can hold
either annotators or outputs but not both. Lifting this restriction would
require a representation with both dimensions, see
:ref:`recognized-future-semantics`.

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
``annotation_type`` to choose downstream behavior and its ``classes`` as the
canonical class vocabulary for classification.

Stream Query Strategies
-----------------------

Stream query strategies declare the single-output single-annotator
classification capability and have no ``target_type`` parameter, because they
do not resolve the labels ``y`` themselves. The classifier passed to ``query``
is the target authority. A fitted classifier carries its ``target_spec_``. An
unfitted classifier declares the meaning of ``y`` through its ``classes``,
``missing_label``, and ``target_type``, from which the specification is
resolved together with ``y`` exactly as for pool query strategies; without
``y``, only these declarations count. The specification is checked against
that capability before committing any query state such as
``budget_manager_``. A multi-label or multi-annotator classifier is therefore
rejected with the capability error described above.

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
   >>> hasattr(stream_qs, "budget_manager_")
   False

An unfitted classifier is the target authority through its declarations, so a
declared multi-label vocabulary is rejected the same way, even without labels
``y``.

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
   >>> hasattr(stream_qs, "budget_manager_")
   False

.. _recognized-future-semantics:

Recognized Future Semantics
===========================

Target resolution already recognizes multi-output classification and
multi-output regression, although current components do not execute them.
Partial-label querying is not supported either. Multi-label or multi-output
targets with multiple annotators are not recognized yet: resolution rejects
the combination, see :ref:`multiple-annotators`.

The first two are limits of current component capabilities and acquisition
scope; the last is a limit of the current label representation. None of them
changes the distinction between target type and annotation type, and
acquisition granularity is not part of ``TargetSpec``. Future support can
therefore add the required component capabilities, a label representation
with an annotator dimension, and an explicit acquisition model without
changing the target semantics defined here.
