.. _tutorials:

==================
In-depth Tutorials
==================

Each notebook is a complete experiment or annotation workflow built with
scikit-activeml. The notebooks assume familiarity with Python, NumPy,
supervised learning, and the scikit-learn estimator interface. They explain
active learning concepts where the code uses them. A tutorial uses the
single-output target type and a single annotator unless its title names
multi-label or multiple annotators. The :doc:`Core Workflow </core_workflow>`
defines these types, and the :doc:`Strategy Overview
</generated/strategy_overview>` filters strategies by them.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Goal
     - Tutorials
   * - Learn the basic cycle
     - :doc:`Pool-based: Getting Started </generated/tutorials/00_pool_getting_started>`,
       then :doc:`A Simple Evaluation Study </generated/tutorials/04_pool_simple_evaluation_study>`
   * - Evaluate a method
     - :doc:`A Simple Evaluation Study </generated/tutorials/04_pool_simple_evaluation_study>`,
       before interpreting differences between learning curves
   * - Work with continuous labels
     - :doc:`Regression: Getting Started </generated/tutorials/02_pool_regression_getting_started>`,
       then :doc:`Advanced Regression </generated/tutorials/07_pool_advanced_regression>`
   * - Learn from images or audio
     - :doc:`Frozen Vision Transformer </generated/tutorials/05_pool_al_with_self_supervised_learning>`,
       then :doc:`Self-training </generated/tutorials/08_pool_ssl>`,
       :doc:`Fine-tuning DINOv2 </generated/tutorials/01_deep_pool_al_with_skorch>`\*,
       or :doc:`Bayesian Active Learning </generated/tutorials/09_bayesian_al>`\*
   * - Collect human labels
     - :doc:`Pool-based: Getting Started </generated/tutorials/00_pool_getting_started>`,
       then :doc:`Annotate Images </generated/tutorials/03_pool_oracle_annotations>`
       or :doc:`Annotate Research Papers </generated/tutorials/06_pool_al_text_annotation_tool>`
   * - Query labels from several annotators
     - :doc:`Multi-annotator: Getting Started </generated/tutorials/10_multiple_annotators_getting_started>`,
       then :doc:`Advanced Multiple Annotators </generated/tutorials/11_multiple_annotators_advanced>`\*
   * - Use several binary labels per sample
     - :doc:`Multi-label: Getting Started </generated/tutorials/30_pool_multilabel_getting_started>`
   * - Process a data stream
     - :doc:`Stream-based Active Learning </generated/tutorials/20_stream_getting_started>`,
       then :doc:`Batch-stream Active Learning </generated/tutorials/21_stream_batch_with_pool_al>`

Tutorials marked with \* assume basic PyTorch knowledge.


Run a Tutorial
--------------

* Install the dependencies listed at the start of the notebook, then run its
  cells in order from a fresh kernel. Each notebook contains its own helper
  code. In Colab, restart the session if imports fail after installation.
* Notebooks with a ``QUICK_RUN`` flag were run with ``False`` for the
  published comparison. Set the flag to ``True`` for a smaller run, or comment
  out methods in the strategy lists.
* Dataset sizes, training settings, and annotation budgets are editable.
  Changing them produces different results.
* First runs may download datasets and pretrained weights, which later runs
  can reuse.
* Annotation buttons require a live Jupyter session. The saved outputs show an
  untouched image session and a completed automatic research-paper session
  with reference labels. Set ``AUTOMATIC=False`` and rerun the notebook to
  label the papers yourself.


Pool-based Active Learning
--------------------------

In pool-based active learning, a query strategy selects samples from a fixed
pool. An oracle supplies the requested labels, and the model is fitted on
the labels acquired so far.


Classification
^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :class: tutorial-table

   * - Tutorial
     - Data
     - Models
     - Learning objective
   * - :doc:`Pool-based Active Learning: Getting Started </generated/tutorials/00_pool_getting_started>`
     - Synthetic
     - - Logistic regression
     - Represent missing labels and complete a fit–query–annotate cycle.
   * - :doc:`Pool-based Active Learning: A Simple Evaluation Study </generated/tutorials/04_pool_simple_evaluation_study>`
     - Tabular
     - - Gaussian process classifier
       - Parzen window classifier
     - Compare classifiers and query strategies with paired folds, seeds, and annotation budgets.
   * - :doc:`Deep Active Learning by Fine-tuning DINOv2 </generated/tutorials/01_deep_pool_al_with_skorch>`
     - Image
     - - DINOv2 with a trainable backbone and classification head
     - Continue fine-tuning while exposing probabilities and embeddings to queries.
   * - :doc:`Deep Active Learning with a Frozen Vision Transformer </generated/tutorials/05_pool_al_with_self_supervised_learning>`
     - Image
     - - Frozen DINOv2 embeddings and logistic regression
     - Separate feature extraction from repeated classifier fitting and querying.
   * - :doc:`Deep Active Learning with Self-training </generated/tutorials/08_pool_ssl>`
     - Image
     - - Frozen DINOv2 embeddings and logistic regression
       - Self-training
     - Track acquired labels and pseudo-labels under the same annotation budget.
   * - :doc:`Bayesian Active Learning with Audio Embeddings </generated/tutorials/09_bayesian_al>`
     - Audio
     - - Wav2Vec 2.0 embeddings and an MLP with Monte Carlo dropout
     - Use consistent model members for individual and joint batch selection.
   * - :doc:`Annotate Images with an Active Learning Widget </generated/tutorials/03_pool_oracle_annotations>`
     - Image
     - - Multilayer perceptron (MLP)
     - Collect image labels and manage batches, skips, undo, and export.
   * - :doc:`Annotate Research Papers with Text Embeddings </generated/tutorials/06_pool_al_text_annotation_tool>`
     - Text
     - - Precomputed text embeddings and logistic regression
     - Connect text queries to human labels and annotation sessions.


   * - .. _multilabel-classification:

       :doc:`Pool-based Multi-label Active Learning: Getting Started </generated/tutorials/30_pool_multilabel_getting_started>`
     - Audio
     - - Perch v2 embeddings and an MLP
     - Query complete label vectors and interpret multi-label probabilities and metrics.
   * - .. _multi-annotator-learning:

       :doc:`Multi-annotator Pool-based Active Learning: Getting Started </generated/tutorials/10_multiple_annotators_getting_started>`
     - Synthetic
     - - Parzen window classifier with majority voting
     - Select sample–annotator pairs and aggregate several labels per sample.
   * - :doc:`Advanced Active Learning with Multiple Annotators </generated/tutorials/11_multiple_annotators_advanced>`
     - Image
     - - Convolutional network with majority voting or AnnotMix
     - Separate classifier learning, annotator modeling, and annotator selection.


Regression
^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :class: tutorial-table

   * - Tutorial
     - Data
     - Models
     - Learning objective
   * - :doc:`Pool-based Active Learning for Regression: Getting Started </generated/tutorials/02_pool_regression_getting_started>`
     - Synthetic
     - - NIC kernel regressor
     - Build a numerical annotation loop and relate predictive distributions, query utilities, and error.
   * - :doc:`Advanced Pool-based Active Learning for Regression </generated/tutorials/07_pool_advanced_regression>`
     - Tabular
     - - Gradient-boosted trees (XGBoost)
       - MLP
       - Random forest
     - Distinguish model comparisons from query-strategy comparisons, and keep evaluation data separate from acquired labels.


Stream-based Active Learning
----------------------------

In stream-based active learning, samples arrive over time. The query strategy
decides whether to request each incoming sample's label. The batch-stream
tutorial uses pool-based strategies when several arrivals can be collected
before that decision.


.. _id1:

Classification
^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :class: tutorial-table

   * - Tutorial
     - Data
     - Models
     - Learning objective
   * - :doc:`Stream-based Active Learning with scikit-learn, River, and CapyMOA </generated/tutorials/20_stream_getting_started>`
     - Text
     - - SentenceTransformer embeddings with scikit-learn, River, and CapyMOA classifiers
     - Predict each arrival before learning from it, update the budget, and compare incremental models and query strategies.
   * - :doc:`Batch-stream Active Learning with Pool-based Query Strategies </generated/tutorials/21_stream_batch_with_pool_al>`
     - Synthetic
     - - Parzen window classifier
     - Map incoming and retained samples to pool queries, including partial batches.

.. toctree::
   :hidden:
   :maxdepth: 1

   generated/tutorials/00_pool_getting_started
   generated/tutorials/04_pool_simple_evaluation_study
   generated/tutorials/01_deep_pool_al_with_skorch
   generated/tutorials/05_pool_al_with_self_supervised_learning
   generated/tutorials/08_pool_ssl
   generated/tutorials/09_bayesian_al
   generated/tutorials/03_pool_oracle_annotations
   generated/tutorials/06_pool_al_text_annotation_tool
   generated/tutorials/30_pool_multilabel_getting_started
   generated/tutorials/10_multiple_annotators_getting_started
   generated/tutorials/11_multiple_annotators_advanced
   generated/tutorials/02_pool_regression_getting_started
   generated/tutorials/07_pool_advanced_regression
   generated/tutorials/20_stream_getting_started
   generated/tutorials/21_stream_batch_with_pool_al
