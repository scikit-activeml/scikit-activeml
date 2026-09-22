.. _tutorials:

==================
In-depth Tutorials
==================

These notebooks explain complete active learning workflows, from representing
missing labels to evaluating a model or collecting annotations.

Start with :doc:`pool-based classification
</generated/tutorials/00_pool_getting_started>` to learn the basic cycle. The
:doc:`evaluation study </generated/tutorials/04_pool_simple_evaluation_study>`
then shows how to compare methods using the same folds and initial labels.
The sections below lead to regression, deep learning, multiple annotators,
multi-label classification, and data streams.

Check each notebook's dependencies and device settings before running it.
Run annotation notebooks in Jupyter to use their widgets. The documentation
shows a static view.


Pool-based Active Learning
--------------------------

In pool-based active learning, a query strategy selects samples from a fixed
pool. An oracle supplies the requested labels, and the model is fitted on
the labels acquired so far.


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
     - - Logistic regression
   * - :doc:`Pool-based Active Learning: A Simple Evaluation Study </generated/tutorials/04_pool_simple_evaluation_study>`
     - Tabular
     - - Gaussian process classifier
       - Parzen window classifier
   * - :doc:`Deep Active Learning by Fine-tuning DINOv2 </generated/tutorials/01_deep_pool_al_with_skorch>`
     - Image
     - - DINOv2 with a trainable backbone and classification head
   * - :doc:`Deep Active Learning with a Frozen Vision Transformer </generated/tutorials/05_pool_al_with_self_supervised_learning>`
     - Image
     - - Frozen DINOv2 embeddings and logistic regression
   * - :doc:`Deep Active Learning with Self-training </generated/tutorials/08_pool_ssl>`
     - Image
     - - Frozen DINOv2 embeddings and logistic regression
       - Self-training
   * - :doc:`Bayesian Active Learning with Audio Embeddings </generated/tutorials/09_bayesian_al>`
     - Audio
     - - Wav2Vec 2.0 embeddings and an MLP with Monte Carlo dropout
   * - :doc:`Annotate Images with an Active Learning Widget </generated/tutorials/03_pool_oracle_annotations>`
     - Image
     - - Multilayer perceptron (MLP)
   * - :doc:`Annotate Research Papers with Text Embeddings </generated/tutorials/06_pool_al_text_annotation_tool>`
     - Text
     - - Precomputed text embeddings and logistic regression


.. _multilabel-classification:

Multi-label Classification
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :class: no-tag-filter

   * - Tutorial
     - Data
     - Models
   * - :doc:`Pool-based Multilabel Active Learning: Getting Started </generated/tutorials/30_pool_multilabel_getting_started>`
     - Audio
     - - Perch v2 embeddings and an MLP


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
     - - NIC kernel regressor
   * - :doc:`Compare Active Learning Strategies for Regression </generated/tutorials/07_pool_advanced_regression>`
     - Tabular
     - - Gradient-boosted trees (XGBoost)
       - MLP
       - Random forest


Multi-annotator Learning
^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :class: no-tag-filter


   * - Tutorial
     - Data
     - Models
   * - :doc:`Multi-annotator Pool-based Active Learning: Getting Started </generated/tutorials/10_multiple_annotators_getting_started>`
     - Synthetic
     - - Parzen window classifier with majority voting
   * - :doc:`Advanced Active Learning with Multiple Annotators </generated/tutorials/11_multiple_annotators_advanced>`
     - Image
     - - Convolutional network with majority voting or AnnotMix


Stream-based Active Learning
----------------------------

In stream-based active learning, samples arrive over time. The query strategy
decides whether to request each incoming sample's label. The batch tutorial
shows how to use pool-based strategies when several arrivals can be collected
before making that decision.


Classification
^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :class: no-tag-filter

   * - Tutorial
     - Data
     - Models
   * - :doc:`Stream-based Active Learning with scikit-learn, River, and CapyMOA </generated/tutorials/20_stream_getting_started>`
     - Text
     - - SentenceTransformer embeddings with scikit-learn, River, and CapyMOA classifiers
   * - :doc:`Batch-stream Active Learning with Pool-based Query Strategies </generated/tutorials/21_stream_batch_with_pool_al>`
     - Synthetic
     - - Parzen window classifier
