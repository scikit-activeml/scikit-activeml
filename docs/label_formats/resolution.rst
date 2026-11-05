.. _label-resolution:

=====================
Resolution and Errors
=====================

Estimators and query strategies interpret labels in two steps. They resolve a
target specification from the labels and declarations, and then check whether
they support it. A specification can be valid even when a component does not
support it.

.. The figure is embedded inline so that its text follows the light and dark
   theme. Its title and description provide the alternative text.

.. container:: inline-figure

   .. raw:: html
      :file: ../logos/label-resolution.svg

   The two steps raise the errors in the red boxes. The component determines
   the task and annotation type, and a fitted estimator supplies the
   declarations through its ``target_spec_``.

Target Specification
====================

:func:`~skactiveml.utils.resolve_target_spec` uses ``y``, ``task``,
``target_type``, ``annotation_type``, ``missing_label``, and any declared class
vocabularies to return an immutable :class:`~skactiveml.utils.TargetSpec`
with four fields:

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

After a successful fit, classifiers and regressors store the resolved
specification in ``target_spec_``. Use its ``target_type`` and
``annotation_type`` to interpret training labels and its ``classes`` for the
normalized classification vocabulary.

Declarations
============

Only components that interpret ``y`` themselves and support multi-label
classification have a ``target_type`` constructor parameter: classifiers such
as :class:`~skactiveml.classifier.ParzenWindowClassifier` and pool strategies
that use no estimator for ``y``, such as
:class:`~skactiveml.pool.RandomSampling`. Strategies that receive an estimator
take the target type from it, and pool wrappers leave it to the strategy they
wrap. Stream strategies that use no labels, such as
:class:`~skactiveml.stream.StreamRandomSampling`, need none. All other
components support single-output labels only. A class's docstring states
whether it accepts ``target_type``. The default, ``"auto"``, resolves
unambiguous single-output inputs.

Query strategies resolve the specification on each call. If you pass a fitted
estimator, its ``target_spec_`` determines how the strategy interprets ``y``,
and the strategy checks ``y`` against it. For single-output classification,
single-annotator pool strategies expect one label per sample even when a
fitted classifier was trained on several annotators' labels (see
:ref:`multiple-annotators`). Multi-label queries retain one label per output
for each sample. Strategies do not retain a specification between queries.

An unfitted estimator declares how to interpret ``y`` through its
``missing_label`` and, for classifiers, ``classes`` and, where available,
``target_type``, which are resolved together with ``y``. A
:class:`~skactiveml.classifier.SlidingWindowClassifier` without ``classes``
declares the classes of its ``estimator``. A strategy that fits a classifier,
e.g., with ``fit_clf=True``, first checks that the classifier supports
single-annotator labels, then fits a clone and uses its ``target_spec_``. A
target specification that the strategy does not support may therefore be
rejected only after training.

Errors
======

Errors occur at two stages:

- Resolution rejects invalid labels or declarations, such as labels outside
  the declared ``classes``, an unknown ``target_type``, or
  ``target_type="multi-label"`` for regression. The
  :doc:`values` page lists the rules for labels. Correct the labels or the
  declaration.
- Capability checking rejects a valid specification that the component cannot
  use. The error reports the requested specification and supported
  combinations. Choose a component that supports the specification.

Single-annotator query strategies check their capability against the
single-annotator prediction specification of a fitted multi-annotator
classifier (see :ref:`multiple-annotators`).

.. _label-resolution-stream:

Stream Query Strategies
=======================

Stream query strategies that use a classifier support only single-output
single-annotator classification and have no ``target_type`` parameter. They
resolve the classifier and ``y`` as pool query strategies do: given labels
``y`` must match the single-annotator prediction specification of the
classifier, an unfitted classifier contributes its declarations, and with
``fit_clf=True`` the fitted clone's ``target_spec_`` applies. A multi-label
classifier is therefore rejected with a capability error, whereas a fitted
multi-annotator classifier passes, see :ref:`multiple-annotators`.

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

.. dropdown:: Examples of rejected multi-label classifiers

   A fitted multi-label classifier is rejected:

   .. doctest::

      >>> import numpy as np
      >>> from sklearn.linear_model import SGDClassifier
      >>> from sklearn.multioutput import MultiOutputClassifier
      >>> from skactiveml.classifier import SklearnClassifier
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
