.. _multiple-annotators:

===================
Multiple Annotators
===================

The target type describes the label to predict. The annotation type describes
who provides it. In a multi-annotator label matrix, each row represents a
sample and each column an annotator. A row such as ``[0, NaN]`` is valid: one
annotator provided a label for the sample, and the other has not. The
:doc:`multiple annotators tutorial
</generated/tutorials/10_multiple_annotators_getting_started>` shows a
complete active learning cycle.

Queries
=======

Multi-annotator estimators retain the annotation type in their fitted
specification. Multi-annotator strategies return ``(sample, annotator)`` pairs
so that each requested label identifies both the sample and its annotator.
For pool queries, ``batch_size`` counts these pairs. Repeated requests for the
same sample count separately. In a single-annotator workflow, it instead
counts selected samples, each supplying one label or a complete multi-label
vector.

Multi-label Classification
==========================

Multi-annotator specifications currently require the single-output target
type. With ``target_type="auto"``, the resolver assigns ``"single-output"``
and rejects nested class vocabularies. Combining
``annotation_type="multi-annotator"`` with ``target_type="multi-label"`` fails
during resolution. The two-dimensional label matrix has no separate
dimensions for annotators and label outputs. Supporting both would require
another representation.

Single-annotator Query Strategies
=================================

Compatible single-annotator pool strategies, such as
:class:`~skactiveml.pool.UncertaintySampling`, :class:`~skactiveml.pool.Badge`,
and :class:`~skactiveml.pool.Falcun`, can use a fitted multi-annotator
classifier with ``fit_clf=False`` and aggregated query labels, e.g., majority
votes. The classifier predicts one class distribution per sample. Query
validation therefore uses a single-annotator prediction specification while
preserving the classifier's original ``target_spec_`` and its exact class
vocabulary. This does not enable fitting a multi-annotator classifier on
aggregated labels with ``fit_clf=True``. Algorithm-specific classifier
requirements still apply, including any internal refitting a strategy
performs.

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
label matrix and selects sample–annotator pairs. It also accepts the fitted
classifier through ``fit_clf=False``.

An :class:`~skactiveml.classifier.multiannotator.AnnotatorEnsembleClassifier`
used with ``fit_ensemble=False`` as a committee in
:class:`~skactiveml.pool.QueryByCommittee`,
:class:`~skactiveml.pool.BatchBALD`, or :class:`~skactiveml.pool.GreedyBALD`
requires fitted member classifiers. Fitting this ensemble on an empty dataset
leaves its members unfitted, so committee queries raise a ``NotFittedError``.
To initialize a committee without observed labels, fit it with declared
``classes`` on a nonempty pool with an all-missing label matrix.

Stream Query Strategies
=======================

Stream query strategies apply the same rules: they accept a fitted
multi-annotator classifier with ``fit_clf=False`` and check labels ``y``, if
given, against its single-annotator prediction specification. They reject an
unfitted multi-annotator classifier, so fit it with declared ``classes`` on an
all-missing label matrix to start without labels. Before refitting, the
strategy checks that the classifier supports single-annotator labels,
including when ``y`` is a column vector. Their algorithm-specific requirements
apply as well.
