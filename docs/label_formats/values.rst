.. _label-contract:

===============================
Label Values and Missing Labels
===============================

The following rules define valid class vocabularies, observed labels, and
missing labels. They apply to classifiers, regressors, the label encoder,
label aggregation, label helpers, and query strategies. The rules are
deliberately simple: labels are strings or real numbers, and rare combinations
of inputs are not handled separately.

``classes``
    The allowed categories for classification. Single-output classification
    has one vocabulary. Multi-label classification has one binary
    vocabulary per output. Missing labels are excluded from class vocabularies.
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

Rules
=====

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

For ``y``, validation uses the array dtype, so inferring the classes from
``np.array([0.0, 1.0])`` gives floating-point classes even though the values
are integral. Declare ``classes`` explicitly when you need to control the
prediction dtype.

An empty or entirely missing ``y`` has no observed labels from which to infer
a vocabulary. Components that need one, such as
:func:`~skactiveml.utils.resolve_target_spec` and the label encoder, require
``classes`` in this case.

Choosing the Missing Label
==========================

Which missing labels are compatible depends on whether the labels are strings
or numbers. Boolean class labels count as numbers. The missing label must not
be a class.

.. list-table:: Compatible missing labels
   :header-rows: 1
   :stub-columns: 1
   :widths: 16 14 14 28 28

   * - Labels
     - ``None``
     - ``np.nan``
     - Finite number, e.g., ``-1``
     - String, e.g., ``"?"``
   * - Strings
     - Yes
     - No
     - No
     - Yes
   * - Numbers
     - Yes
     - Yes
     - Yes
     - No

With ``missing_label=-1``, every ``-1`` is treated as missing, including an
intended regression value. Use ``np.nan`` or ``None`` when a numeric
placeholder could also be an observed label.

The string ``"nan"`` is incompatible with numeric labels. Use ``np.nan`` to
mark numeric NaN values as missing.

.. doctest::

   >>> from skactiveml.utils import is_unlabeled
   >>> is_unlabeled(np.array([0.0, np.nan, 1.0]), missing_label="nan")
   Traceback (most recent call last):
       ...
   TypeError: `missing_label='nan'` is a string and is not compatible with
   the numeric labels in `y`. ...
   >>> is_unlabeled(np.array([0.0, np.nan, 1.0]), missing_label=np.nan)
   array([False,  True, False])

Examples by Setting
===================

The tabs list accepted and rejected inputs for each setting. In their tables,
``NaN`` denotes numeric ``np.nan``.

.. tab-set::

   .. tab-item:: Single-output Classification

      Every nonmissing label must match a declared class by value. Numeric
      labels may use a wider storage dtype to accommodate missing values:
      ``0.0`` matches the declared integer class ``0``, but the string ``"0"``
      does not. Boolean labels are numbers as well, so ``True`` matches the
      declared class ``1``. Predictions use the declared class dtype.

      ``np.array([True, np.nan, False])`` produces floating-point values. The
      example below uses object storage and ``missing_label=None`` to
      preserve Boolean labels.

      .. doctest::

         >>> bool_y = np.array([True, None, False], dtype=object)
         >>> bool_clf = ParzenWindowClassifier(
         ...     classes=[0, 1], missing_label=None
         ... )
         >>> bool_clf.fit(label_X, bool_y).predict(label_X).dtype
         dtype('int64')

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
           - Accept: the list becomes a floating-point array with the
             vocabulary ``(0.0, 1.5)``.
         * - ``None``
           - ``"?"``
           - ``["cat", "?", "dog"]``
           - Accept: infer the vocabulary from ``"cat"`` and ``"dog"``.
         * - ``None``
           - ``None``
           - ``[0.0, NaN, 1.0]``
           - Reject: NaN is not the configured missing label and is not a
             category.
         * - ``None``
           - ``NaN``
           - ``[NaN, NaN]``
           - Reject: no vocabulary can be inferred, so declare
             ``classes``.

      For example, ``missing_label=None`` makes NaN an invalid observed
      label:

      .. doctest::

         >>> nan_y = np.array([0.0, np.nan, 1.0])
         >>> ParzenWindowClassifier(missing_label=None).fit(label_X, nan_y)
         Traceback (most recent call last):
             ...
         ValueError: `y` contains NaN, which is only valid as the configured `missing_label`.

   .. tab-item:: Multi-label Classification

      For ``target_type="multi-label"``, each output has exactly two classes
      and each sample has one value per output. One array stores all outputs,
      so the vocabularies of all outputs must be either all strings or all
      numbers. Otherwise, NumPy would convert the numbers to strings, such as
      an integer class ``0`` into ``"0"``. Numeric vocabularies may differ in
      dtype, and different binary vocabularies are allowed.

      All outputs share one missing label. Each row must be fully observed or
      fully missing. Partially observed rows are rejected because
      partial-label training and acquisition are unsupported.

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
           - Accept: integer and floating-point vocabularies are both
             numeric.
         * - ``[["no", "yes"], [0, 1]]``
           - ``None``
           - ``["no", 1]``
           - Reject: string and numeric vocabularies.
         * - ``[[0, 1, 2], [0, 1]]``
           - ``NaN``
           - ``[0, 1]``
           - Reject: one vocabulary is not binary, so this is not
             multi-label.

      With ``classes=None``, each column must contain exactly two observed
      categories to infer its binary vocabulary. The resolver does not assume
      ``(0, 1)``. For example, ``[[0, 0], [1, 0]]`` contains only one class
      for the second output, so you must declare ``classes``.

      The complete-row rule applies to label outputs. In a multi-annotator
      matrix, each column represents an annotator, and ``[0, NaN]`` is valid.
      See :ref:`multiple-annotators`.

   .. tab-item:: Regression

      Regression requires ``classes=None``. Observed labels must be finite
      real numbers other than Boolean values. Regressors have no
      ``target_type`` parameter, and one-dimensional numeric labels resolve to
      single-output regression. Labels with several columns are rejected,
      because multi-output regression is unsupported. Predictions have the
      shape ``(n_samples,)``. Wrapper regressors, such as
      :class:`~skactiveml.regressor.SklearnRegressor`, reduce a single
      prediction column of the wrapped estimator to this shape and reject
      predictions or samples with several columns or non-finite values.

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
      :func:`~skactiveml.utils.is_unlabeled` do not know the prediction task.
      They check the rules that classification and regression share.
      Components that know the task also apply the regression rule.

      .. doctest::

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

.. _label-storage:

Storage and Prediction Dtypes
=============================

Predictions use the dtype of the declared classes. Storing ``y`` may require a
wider dtype to accommodate the missing label. For declared ``int64`` classes
``[0, 1]`` and ``missing_label=np.nan``, ``[0, np.nan, 1]`` is stored as
``float64``, while ``predict`` returns ``int64``. Multi-label predictions use
the common dtype of the output vocabularies.

.. doctest::

   >>> dtype_X = np.zeros((3, 1))
   >>> dtype_y = np.array([0, np.nan, 1])
   >>> dtype_clf = ParzenWindowClassifier(classes=[0, 1])
   >>> _ = dtype_clf.fit(dtype_X, dtype_y)
   >>> dtype_clf.predict(dtype_X).dtype == dtype_clf.classes_.dtype
   True

Integer labels above ``2**53`` are unsupported, because ``float64``
represents integers exactly only up to ``2**53``.

The label encoder decodes labels into a dtype that can store the missing
label. To decode fully observed labels into the class dtype, pass
``prefer_class_dtype=True`` to
:meth:`ExtLabelEncoder.inverse_transform
<skactiveml.utils.ExtLabelEncoder.inverse_transform>`.
