.. _label-formats:

=============
Label Formats
=============

The format of the label array ``y`` depends on the learning setting, which the
:doc:`Concepts <concepts>` page defines by task, target type, and annotation
type. This page shows the format for each setting and which settings
scikit-activeml supports. The formats apply to pool-based and stream-based
active learning alike, and the pages listed at the end give the rules and
examples.

.. The panels are embedded inline so that their text follows the light and
   dark theme, and they stack on narrow screens. Their titles and
   descriptions provide the alternative text.

.. container:: inline-figure

   .. container:: inline-figure-panels

      .. raw:: html
         :file: logos/label-format-single-output.svg

      .. raw:: html
         :file: logos/label-format-regression.svg

      .. raw:: html
         :file: logos/label-format-multilabel.svg

      .. raw:: html
         :file: logos/label-format-multiannotator.svg

   Each row of ``y`` belongs to one sample. Blue cells hold observed labels,
   dashed cells mark missing labels, and orange cells show what a query
   requests. A multi-label row must be complete or entirely missing, so the
   partially labeled row (✗) is rejected. A row of annotator labels may be
   partially labeled.

Two-dimensional Labels
======================

Multi-label classification and multiple annotators both store ``y`` as a
matrix with one row per sample. The shape alone does not tell them apart, so
declare the setting:

* For multi-label classification, set ``target_type="multi-label"`` or a
  nested binary class vocabulary on the classifier. Query strategies take the
  declaration from it. A pool query strategy that uses no estimator for
  ``y``, such as :class:`~skactiveml.pool.RandomSampling`, takes
  ``target_type="multi-label"`` itself. See
  :doc:`label_formats/multilabel`.
* For multiple annotators, pass the label matrix to a multi-annotator
  classifier or query strategy. See :doc:`label_formats/multiannotator`.

As in scikit-learn, labels with shape ``(n_samples, 1)`` are single-output
unless multi-label classification is declared. This holds for classification,
regression, and strategies that use no estimator for ``y``. Single-annotator
classifiers and pool query strategies convert such a column to shape
``(n_samples,)`` and emit a ``DataConversionWarning``.

.. _current-limitations:

Supported Settings
==================

The table lists the settings that the query strategies support. Its links open
the :doc:`Strategy Overview <generated/strategy_overview>` with the matching
filters.

.. list-table::
   :header-rows: 1
   :widths: 34 33 33

   * - Setting
     - Single annotator
     - Multiple annotators
   * - Single-output classification
     - `Pool and stream
       <generated/strategy_overview.html?task=classification&target=single-output&annotators=single-annotator>`__
     - `Pool
       <generated/strategy_overview.html?annotators=multi-annotator>`__
   * - Multi-label classification
     - `Pool, stream (random and periodic sampling only)
       <generated/strategy_overview.html?target=multi-label>`__
     - Unsupported
   * - Regression
     - `Pool, stream (random and periodic sampling only)
       <generated/strategy_overview.html?task=regression>`__
     - Unsupported
   * - Multi-output classification or regression
     - Unsupported
     - Unsupported

In multi-output classification, each column of ``y`` is a separate
classification task with any number of classes. Multi-label classification
covers the binary case, so a nested class vocabulary must be binary. Two more
limits apply:

* Multi-label queries request complete label vectors. Querying a part of a
  label vector is unsupported.
* A multi-annotator matrix has no separate dimension for label outputs, so
  multi-label classification with multiple annotators fails when the labels
  are resolved.

Rules and Examples
==================

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Label Values and Missing Labels
      :link: label_formats/values
      :link-type: doc

      Which values labels, classes, and missing labels may take, with
      accepted and rejected examples for each setting.

   .. grid-item-card:: Multi-label Classification
      :link: label_formats/multilabel
      :link-type: doc

      Declare label vectors and their class vocabularies, and find the
      strategies that query complete label vectors.

   .. grid-item-card:: Multiple Annotators
      :link: label_formats/multiannotator
      :link-type: doc

      Query sample–annotator pairs, and use a multi-annotator classifier with
      single-annotator query strategies.

   .. grid-item-card:: Resolution and Errors
      :link: label_formats/resolution
      :link-type: doc

      How estimators and query strategies interpret labels and declarations,
      and why they reject them.

.. toctree::
   :hidden:

   label_formats/values
   label_formats/multilabel
   label_formats/multiannotator
   label_formats/resolution
