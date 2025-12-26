.. _tutorials:

==================
In-depth Tutorials
==================

The following sections summarize a selection of our in-depth tutorials.
Each entry lists the data modality and models used in the tutorial,
while the active learning scenario and prediction task are reflected by the
subsections.


🏊 Pool-based Active Learning
-----------------------------

In pool-based active learning, a model has access to a large pool of unlabeled
samples. In each iteration it selects one or several informative samples
from this pool, queries their labels, and retrains on the enlarged labeled set.
This setting is common when data can be stored and queried flexibly, while
labeling is the main bottleneck.


Classification
^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :class: no-tag-filter

   * - Tutorial
     - Data
     - Models
   * - :doc:`Pool-based Active Learning: Getting Started </generated/tutorials/00_pool_getting_started>`
     - Synthetic
     - - Logistic Regression
   * - :doc:`Pool-based Active Learning: Simple Evaluation Study </generated/tutorials/04_pool_simple_evaluation_study>`
     - Tabular
     - - Gaussian Process Classifier
       - Parzen Window Classifier
   * - :doc:`Deep Active Learning for Fine-tuning Vision Transformers </generated/tutorials/01_deep_pool_al_with_skorch>`
     - Image
     - - Vision Transformer with Full Fine-tuning
   * - :doc:`Deep Active Learning with Frozen Vision Transformers </generated/tutorials/05_pool_al_with_self_supervised_learning>`
     - Image
     - - Vision Transformer with Linear Probing
   * - :doc:`Semi-supervised Active Learning </generated/tutorials/08_pool_ssl>`
     - Image
     - - Vision Transformer with Linear Probing
       - Self-training
   * - :doc:`Bayesian Active Learning </generated/tutorials/09_bayesian_al>`
     - Audio
     - - Wav2Vec with Multi-layer Perceptron Probing
   * - :doc:`Image Annotation Tool </generated/tutorials/03_pool_oracle_annotations>`
     - Image
     - - Multi-layer Perceptron
   * - :doc:`Paper Annotation Tool </generated/tutorials/06_pool_al_text_annotation_tool>`
     - Text
     - - Text Transformer with Linear Probing


Regression
^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :class: no-tag-filter

   * - Tutorial
     - Data
     - Models
   * - :doc:`Pool-based Active Learning for Regression: Getting Started </generated/tutorials/02_pool_regression_getting_started>`
     - Synthetic
     - - Kernel Regressor
   * - :doc:`Advanced Active Learning for Regression Tasks </generated/tutorials/07_pool_advanced_regression>`
     - Tabular
     - - Extreme Gradient Boosted Tree
       - Multi-layer Perceptron
       - Random Forest


Multi-annotator Learning
^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :class: no-tag-filter


   * - Tutorial
     - Data
     - Models
   * - :doc:`Multi-annotator Active Learning: Getting Started </generated/tutorials/10_multiple_annotators_getting_started>`
     - Synthetic
     - - Logistic Regression
   * - :doc:`Advanced Multi-annotator Active Learning </generated/tutorials/11_multiple_annotators_advanced>`
     - Image
     - - Convolutional Neural Network


🌊 Stream-based Active Learning
-------------------------------

In stream-based active learning, samples arrive sequentially as a data stream.
For each incoming sample, the learner must immediately decide whether to query
its label or discard it, typically under a strict labeling budget. This setting
is relevant when data cannot be stored indefinitely or when decisions need to
be made online.


Classification
^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :class: no-tag-filter

   * - Tutorial
     - Data
     - Models
   * - :doc:`Stream-based Active Learning: Getting Started </generated/tutorials/20_stream_getting_started>`
     - Text
     - - Sentence Transformer with Parzen Window Classifier
   * - :doc:`Stream-based Active Learning in Batches </generated/tutorials/21_stream_batch_with_pool_al>`
     - Synthetic
     - - Parzen Window Classifier
