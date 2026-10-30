.. _core-workflow:
.. _concepts:

========
Concepts
========

In active learning, a query strategy selects which samples to label for
training an estimator. Annotators provide the labels of the selected samples.
scikit-activeml supports pool-based and stream-based active learning.


.. _the-label-vector-y:
.. _choosing-the-missing-label:
.. _common-pitfalls:
.. _the-active-learning-cycle:
.. _pool-and-stream-acquisition:
.. _pool-based-classification:
.. _candidates-batches-and-utilities:
.. _pool-based-acquisition:
.. _stream-based-active-learning:
.. _stream-based-acquisition:

Components and Responsibilities
===============================

.. The panels are embedded inline so that their text follows the light and
   dark theme, and they stack on narrow screens. Their titles and
   descriptions provide the alternative text.

.. container:: concepts-figure

   .. container:: concepts-panels

      .. raw:: html
         :file: logos/concepts-pool.svg

      .. raw:: html
         :file: logos/concepts-stream.svg

   Blue boxes show what scikit-activeml provides. Your application manages
   the data, queries the annotator, and controls training and stopping. The
   stream panel highlights the current ``candidates``.

The feature matrix ``X`` describes the samples. The label array ``y`` stores
known labels and marks unknown labels with ``missing_label``.

Pool-based and stream-based query strategies both use ``query()`` to select
samples:

* **Pool-based:** the query strategy selects one sample or a batch of samples
  from the pool. The application can stop when its labeling budget is
  exhausted.
* **Stream-based:** the query strategy decides for each arriving sample
  whether to query its label and can skip all current candidates. It
  controls the fraction of arriving samples whose labels are queried, and
  its ``update()`` method advances its state after each decision, including
  when no label is queried.

The :doc:`pool-based <generated/tutorials/00_pool_getting_started>` and
:doc:`stream-based <generated/tutorials/20_stream_getting_started>` tutorials
show runnable examples.


.. _workflow-variations:
.. _regression:
.. _multi-label-classification:

Learning Settings
=================

.. The former Multiple Annotators heading supplied a local HTML anchor.
   Keep it local because target_semantics.rst owns the global Sphinx label.

.. raw:: html

   <span id="multiple-annotators"></span>

A learning setting combines three properties of the labels:

* **Task:** classification predicts class labels; regression predicts
  continuous labels.
* **Target type:** single-output means one value per sample; multi-label
  means a vector of binary labels per sample, and a query asks for the
  complete vector.
* **Annotation type:** a single annotator provides one label or label vector
  per sample; with multiple annotators, labels are stored per annotator, and
  a query selects sample–annotator pairs.

The supported combinations depend on the estimator and query strategy. The
:doc:`Strategy Overview <generated/strategy_overview>` filters query
strategies by these properties, and the :doc:`Tutorials <tutorials>` page
starts with a getting-started tutorial for each of these settings.

.. note::

   scikit-activeml does not support general multi-output classification and
   regression, or multi-label classification with multiple annotators. The
   :doc:`Target and Annotation Semantics <target_semantics>` page defines
   label representations and these limits.


Where to Go Next
================

* The :doc:`Tutorials <tutorials>` page continues with evaluation, deep
  learning, and collecting human labels.
* The :doc:`API Reference <generated/api/skactiveml>` documents the parameters
  and behavior of estimators and query strategies.
