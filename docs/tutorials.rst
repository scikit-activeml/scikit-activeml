.. _tutorials:

==================
In-depth Tutorials
==================

These notebooks demonstrate how to design experiments and annotation
workflows with scikit-activeml. Each focuses on a scenario and shows the
framework's relevant capabilities through a complete example. They assume
familiarity with Python, NumPy, supervised learning, and the scikit-learn
estimator interface. Active learning concepts are explained where they help
readers understand and adapt the workflow.

Start with :doc:`pool-based classification
</generated/tutorials/00_pool_getting_started>` to learn the basic cycle. The
:doc:`evaluation study </generated/tutorials/04_pool_simple_evaluation_study>`
shows how to compare methods using the same folds and initial labels.
The sections below lead to regression, deep learning, multi-label
classification, multiple annotators, and data streams.

Pool and Stream describe how samples become available. Classification and
Regression are prediction tasks. Within each task, the target type
(``single-output`` or ``multi-label``) and annotation setting
(``single-annotator`` or ``multi-annotator``) describe the workflow shown.
The italic labels beneath each tutorial title identify its target and
annotation setting. These dimensions also organize the
:doc:`Visualizations </generated/sphinx_gallery_examples/index>` and the
filters in the :doc:`Strategy Overview </generated/strategy_overview>`.

Choose a path matching your goal. Readers familiar with the framework can
enter the relevant scenario directly:

* **Evaluate a method:** use the evaluation study before interpreting
  differences between learning curves.
* **Predict numerical targets:** start with the regression introduction, then
  compare regression models and query strategies in the advanced notebook.
* **Learn from images or audio:** start with frozen vision embeddings, then
  explore self-training, backbone fine-tuning, or Bayesian audio learning.
  The fine-tuning and Bayesian notebooks assume basic familiarity with PyTorch.
* **Collect human labels:** use the image or research-paper annotation notebook
  after learning how pool queries and acquired labels are represented.
* **Use richer labels:** start with the multi-annotator introduction before
  its neural extension, or use the multi-label tutorial for several binary
  labels per sample.
* **Handle arriving data:** read the streaming introduction before adapting
  pool strategies to batches from a stream.


Run a Tutorial
--------------

Install the dependencies listed at the start of the notebook, then run its
cells in order from a fresh kernel. Each notebook includes its own support
code. The Colab setup explains when to restart after installation.

Notebooks with ``QUICK_RUN`` use ``False`` for the published comparison.
Set it to ``True`` for a smaller example, or comment out methods in the
visible strategy lists. Dataset sizes, training settings, and annotation
budgets are editable; changing them produces different results. First runs
may download datasets and pretrained weights, which later runs can reuse.

Annotation buttons require a live Jupyter session. The image example starts
with an untouched session and supports labeling, skip, and undo. The
research-paper example shows an automatic reference-label demonstration;
set ``AUTOMATIC=False`` to collect your own judgments. Saved widget views
show the recorded session state.


.. _pool:

Pool-based Active Learning
--------------------------

In pool-based active learning, a query strategy selects samples from a fixed
pool. An oracle supplies the requested labels, and the model is fitted on
the labels acquired so far.


Classification
^^^^^^^^^^^^^^

Begin with the single-output introduction and evaluation study. The multi-label
example acquires a complete binary label vector for each selected sample.
The multi-annotator introduction then extends the pool cycle to a
sample–annotator matrix; read it before the advanced neural example, which
also requires basic PyTorch knowledge. Its budgets count annotations,
including repeated labels for a sample.

.. list-table::
   :header-rows: 1
   :class: no-tag-filter

   * - Tutorial
     - Data
     - Models
     - Learning objective
   * - :doc:`Pool-based Active Learning: Getting Started </generated/tutorials/00_pool_getting_started>`

       *Single-output; single annotator.*
     - Synthetic
     - - Logistic regression
     - Represent missing labels and complete a fit–query–annotate cycle.
   * - :doc:`Pool-based Active Learning: A Simple Evaluation Study </generated/tutorials/04_pool_simple_evaluation_study>`

       *Single-output; single annotator.*
     - Tabular
     - - Gaussian process classifier
       - Parzen window classifier
     - Compare classifiers and query strategies with paired folds, seeds, and label budgets.
   * - :doc:`Deep Active Learning by Fine-tuning DINOv2 </generated/tutorials/01_deep_pool_al_with_skorch>`

       *Single-output; single annotator.*
     - Image
     - - DINOv2 with a trainable backbone and classification head
     - Continue fine-tuning while exposing probabilities and embeddings to queries.
   * - :doc:`Deep Active Learning with a Frozen Vision Transformer </generated/tutorials/05_pool_al_with_self_supervised_learning>`

       *Single-output; single annotator.*
     - Image
     - - Frozen DINOv2 embeddings and logistic regression
     - Separate feature extraction from repeated classifier fitting and querying.
   * - :doc:`Deep Active Learning with Self-training </generated/tutorials/08_pool_ssl>`

       *Single-output; single annotator.*
     - Image
     - - Frozen DINOv2 embeddings and logistic regression
       - Self-training
     - Track acquired labels and pseudo-labels under the same annotation budget.
   * - :doc:`Bayesian Active Learning with Audio Embeddings </generated/tutorials/09_bayesian_al>`

       *Single-output; single annotator.*
     - Audio
     - - Wav2Vec 2.0 embeddings and an MLP with Monte Carlo dropout
     - Use consistent model members for individual and joint batch selection.
   * - :doc:`Annotate Images with an Active Learning Widget </generated/tutorials/03_pool_oracle_annotations>`

       *Single-output; single annotator.*
     - Image
     - - Multilayer perceptron (MLP)
     - Collect image labels and manage batches, skips, undo, and export.
   * - :doc:`Annotate Research Papers with Text Embeddings </generated/tutorials/06_pool_al_text_annotation_tool>`

       *Single-output; single annotator.*
     - Text
     - - Precomputed text embeddings and logistic regression
     - Connect text queries to human judgments and annotation sessions.


   * - .. _multilabel-classification:

       :doc:`Pool-based Multi-label Active Learning: Getting Started </generated/tutorials/30_pool_multilabel_getting_started>`

       *Multi-label; single annotator.*
     - Audio
     - - Perch v2 embeddings and an MLP
     - Query complete label rows and interpret multi-label probabilities and metrics.
   * - .. _multi-annotator-learning:

       :doc:`Multi-annotator Pool-based Active Learning: Getting Started </generated/tutorials/10_multiple_annotators_getting_started>`

       *Single-output; multi-annotator.*
     - Synthetic
     - - Parzen window classifier with majority voting
     - Select sample–annotator pairs and aggregate multiple answers.
   * - :doc:`Advanced Active Learning with Multiple Annotators </generated/tutorials/11_multiple_annotators_advanced>`

       *Single-output; multi-annotator.*
     - Image
     - - Convolutional network with majority voting or AnnotMix
     - Separate classifier learning, annotator modeling, and annotator selection.


Regression
^^^^^^^^^^

Begin with the synthetic example to understand predictive uncertainty and a
numerical oracle. It connects a query–annotate–refit loop to selection utilities
and prediction error. The advanced comparison then adds real data and several
model families while keeping evaluation data separate from acquired labels.

.. list-table::
   :header-rows: 1
   :class: no-tag-filter

   * - Tutorial
     - Data
     - Models
     - Learning objective
   * - :doc:`Pool-based Active Learning for Regression: Getting Started </generated/tutorials/02_pool_regression_getting_started>`

       *Single-output; single annotator.*
     - Synthetic
     - - NIC kernel regressor
     - Build a numerical annotation loop and relate predictive distributions, query utilities, and error.
   * - :doc:`Advanced Pool-based Active Learning for Regression </generated/tutorials/07_pool_advanced_regression>`

       *Single-output; single annotator.*
     - Tabular
     - - Gradient-boosted trees (XGBoost)
       - MLP
       - Random forest
     - Distinguish model comparisons from query-strategy comparisons.


.. _stream:

Stream-based Active Learning
----------------------------

In stream-based active learning, samples arrive over time. The query strategy
decides whether to request each incoming sample's label. The batch tutorial
shows how to use pool-based strategies when several arrivals can be collected
before making that decision.
The streaming introduction explains evaluation before training, budget
updates, and learning only from acquired labels. Its classifier wrappers and
missing-label representation also appear in the pool-based introduction.
Continue with the batch example for retained windows and index mappings.
An arrival batch groups incoming samples; a query batch counts the samples
whose labels are acquired. Collecting ten arrivals and requesting one label
uses an arrival batch of ten and a query batch of one.


.. _id1:

Classification
^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :class: no-tag-filter

   * - Tutorial
     - Data
     - Models
     - Learning objective
   * - :doc:`Stream-based Active Learning with scikit-learn, River, and CapyMOA </generated/tutorials/20_stream_getting_started>`

       *Single-output; single annotator.*
     - Text
     - - SentenceTransformer embeddings with scikit-learn, River, and CapyMOA classifiers
     - Build a per-arrival annotation loop and compare incremental models and query strategies.
   * - :doc:`Batch-stream Active Learning with Pool-based Query Strategies </generated/tutorials/21_stream_batch_with_pool_al>`

       *Single-output; single annotator.*
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
