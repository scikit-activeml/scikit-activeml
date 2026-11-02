==========================
Multi-label Classification
==========================

In multi-label classification, each sample has a vector of binary labels, one
per label output, and a query requests the complete vector. The
:doc:`multi-label tutorial
</generated/tutorials/30_pool_multilabel_getting_started>` shows a complete
active learning cycle.

Declaring Multi-label Classification
====================================

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

For single-annotator classification, a two-dimensional ``y`` with more than
one column is ambiguous when ``target_type="auto"`` and ``classes=None``. Its
columns could represent binary label outputs, or the annotators of a label
matrix passed to a single-annotator component by mistake. Binary-looking
values do not resolve this ambiguity, so the resolution fails:

.. doctest::

   >>> from skactiveml.utils import resolve_target_spec
   >>> _ = resolve_target_spec(  # doctest: +IGNORE_EXCEPTION_DETAIL
   ...     np.array([[0, 1], [1, 0]]), task="classification"
   ... )
   Traceback (most recent call last):
   ...
   ValueError:

Declare multi-label classification where the labels are interpreted:

* On the classifier, with ``target_type="multi-label"`` or a nested class
  vocabulary. With ``target_type="auto"``, a flat vocabulary selects
  single-output classification, and a nested vocabulary selects multi-label
  classification. Every nested vocabulary must be binary.
* On a strategy that uses no estimator for ``y``, such as
  :class:`~skactiveml.pool.RandomSampling`, with
  ``target_type="multi-label"``.

A fitted estimator with a resolved ``target_spec_`` also suffices. If the
columns hold the labels of several annotators, pass the matrix to a
multi-annotator component, see :doc:`multiannotator`. If multi-label
classification is declared, a column vector of shape ``(n_samples, 1)`` is one
binary label output.

Class Vocabularies
==================

Declare vocabularies to start fitting before both classes of each output have
been observed. Vocabularies can also contain non-numeric labels. Probability
columns follow the normalized order in fitted ``classes_``, regardless of the
order in which you declared the classes.

.. doctest::

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

The :ref:`rules <label-contract>` reject strings in one output beside numbers
in another, too few observed classes to infer a binary vocabulary, and
partially observed rows.

.. dropdown:: Examples of rejected labels

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

Labeled and Unlabeled Samples
=============================

:func:`~skactiveml.utils.is_labeled`,
:func:`~skactiveml.utils.is_unlabeled`,
:func:`~skactiveml.utils.labeled_indices`, and
:func:`~skactiveml.utils.unlabeled_indices` accept a keyword-only
``target_type`` argument.

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

.. _multilabel-strategy-inventory:

Query Strategies
================

The following pool strategies query complete multi-label vectors. Use the
``Multi-label`` filter in the
:doc:`Strategy Overview </generated/strategy_overview>` to find compatible
methods. Their API class pages link to the associated demonstrations.

Native multi-label methods
    :class:`~skactiveml.pool.MaxLossReductionMaxConfidence` and
    :class:`~skactiveml.pool.LabelCardinalityInconsistency` implement methods
    designed for multi-label data. Papers and tutorials abbreviate Maximum Loss
    Reduction with Maximal Confidence as MMC. The package has no ``MMC``
    alias, so import
    :class:`~skactiveml.pool.MaxLossReductionMaxConfidence`.

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

Of the stream strategies, only :class:`~skactiveml.stream.StreamRandomSampling`
and :class:`~skactiveml.stream.PeriodicSampling` support multi-label
classification. They use no labels, and the other stream strategies reject
multi-label classifiers, see :ref:`Resolution and Errors
<label-resolution-stream>`.

.. _multilabel-classifiers:

Classifiers
===========

The classifiers that support multi-label classification take a
``target_type`` parameter:
:class:`~skactiveml.classifier.ParzenWindowClassifier`,
:class:`~skactiveml.classifier.MixtureModelClassifier`,
:class:`~skactiveml.classifier.SklearnClassifier`, and
:class:`~skactiveml.classifier.SkorchClassifier`.

To wrap a scikit-learn classifier for multi-label classification,
``SklearnClassifier`` requires ``predict_proba`` and the estimator tag
``target_tags.multi_output`` or ``classifier_tags.multi_label``, as
``MultiOutputClassifier`` declares. A plain ``LogisticRegression`` declares
neither tag and is rejected. The class documentation also describes how the
wrapper checks the declared ``classes`` of a pre-fitted estimator, such as a
``OneVsRestClassifier``.
