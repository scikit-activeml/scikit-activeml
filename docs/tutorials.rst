.. _tutorials:

==================
In-depth Tutorials
==================

Each tutorial is a runnable notebook built with scikit-activeml. The
tutorials assume familiarity with Python, NumPy, supervised learning, and the
scikit-learn estimator interface. They explain active learning concepts where
the code uses them. Tutorials tagged PyTorch assume basic PyTorch knowledge.

.. Earlier versions of this page had sections and table rows with these ids.
   Keep them so old links open the getting-started tutorials.

.. raw:: html

   <span id="pool-based-active-learning"></span>
   <span id="classification"></span>
   <span id="regression"></span>
   <span id="stream-based-active-learning"></span>

.. _multilabel-classification:
.. _multi-annotator-learning:
.. _id1:

Getting Started
---------------

Start with the tutorial that matches how your samples arrive and what your
labels look like. The :doc:`Concepts </concepts>` page explains both, and the
:doc:`Strategy Overview </generated/strategy_overview>` lists the matching
query strategies.

.. grid:: 1 2 3 3
   :gutter: 3

   .. grid-item-card:: Pool-based Classification
      :link: generated/tutorials/00_pool_getting_started
      :link-type: doc
      :img-top: logos/tutorial-pool.svg
      :img-alt:
      :class-card: tutorial-start
      :class-img-top: dark-light

      Represent missing labels and follow the fit–query–annotate cycle.
      +++
      :bdg-primary-line:`Synthetic`

   .. grid-item-card:: Regression
      :link: generated/tutorials/02_pool_regression_getting_started
      :link-type: doc
      :img-top: logos/tutorial-regression.svg
      :img-alt:
      :class-card: tutorial-start
      :class-img-top: dark-light

      Query continuous labels and relate predictive distributions to query
      utilities.
      +++
      :bdg-primary-line:`Synthetic`

   .. grid-item-card:: Multi-label Classification
      :link: generated/tutorials/30_pool_multilabel_getting_started
      :link-type: doc
      :img-top: logos/tutorial-multilabel.svg
      :img-alt:
      :class-card: tutorial-start
      :class-img-top: dark-light

      Query complete label vectors and keep missing labels distinct from
      negatives.
      +++
      :bdg-primary-line:`Audio` :bdg-secondary-line:`PyTorch`

   .. grid-item-card:: Multiple Annotators
      :link: generated/tutorials/10_multiple_annotators_getting_started
      :link-type: doc
      :img-top: logos/tutorial-multiannotator.svg
      :img-alt:
      :class-card: tutorial-start
      :class-img-top: dark-light

      Select sample–annotator pairs and fit a classifier on majority votes.
      +++
      :bdg-primary-line:`Synthetic`

   .. grid-item-card:: Stream-based Classification
      :link: generated/tutorials/20_stream_getting_started
      :link-type: doc
      :img-top: logos/tutorial-stream.svg
      :img-alt:
      :class-card: tutorial-start
      :class-img-top: dark-light

      Decide which arriving documents to label, with scikit-learn, River, and
      CapyMOA classifiers.
      +++
      :bdg-primary-line:`Text`


Go Further
----------

Each tutorial names the getting-started tutorial to read first.

.. grid:: 1 2 3 3
   :gutter: 3

   .. grid-item-card:: A Simple Evaluation Study
      :link: generated/tutorials/04_pool_simple_evaluation_study
      :link-type: doc
      :class-card: tutorial-topic topic-evaluation
      :class-header: kicker

      Evaluation
      ^^^
      Compare query strategies with paired folds, seeds, and annotation
      budgets.
      +++
      :bdg-primary-line:`Tabular`

   .. grid-item-card:: Advanced Regression
      :link: generated/tutorials/07_pool_advanced_regression
      :link-type: doc
      :class-card: tutorial-topic topic-setting
      :class-header: kicker

      Regression
      ^^^
      Separate model choice from query-strategy choice with three regressors.
      +++
      :bdg-primary-line:`Tabular`
      :bdg-secondary-line:`PyTorch`

   .. grid-item-card:: Frozen Vision Transformer
      :link: generated/tutorials/05_pool_al_with_self_supervised_learning
      :link-type: doc
      :class-card: tutorial-topic topic-deep-learning
      :class-header: kicker

      Deep learning
      ^^^
      Embed images once with DINOv2, then compare query strategies.
      +++
      :bdg-primary-line:`Image`
      :bdg-secondary-line:`PyTorch`

   .. grid-item-card:: Self-training
      :link: generated/tutorials/08_pool_ssl
      :link-type: doc
      :class-card: tutorial-topic topic-deep-learning
      :class-header: kicker

      Deep learning
      ^^^
      Add pseudo-labels to acquired labels under the same annotation budget.
      +++
      :bdg-primary-line:`Image`
      :bdg-secondary-line:`PyTorch`

   .. grid-item-card:: Fine-tuning DINOv2
      :link: generated/tutorials/01_deep_pool_al_with_skorch
      :link-type: doc
      :class-card: tutorial-topic topic-deep-learning
      :class-header: kicker

      Deep learning
      ^^^
      Fine-tune a vision transformer while it informs each query.
      +++
      :bdg-primary-line:`Image`
      :bdg-secondary-line:`PyTorch`

   .. grid-item-card:: Bayesian Active Learning
      :link: generated/tutorials/09_bayesian_al
      :link-type: doc
      :class-card: tutorial-topic topic-deep-learning
      :class-header: kicker

      Deep learning
      ^^^
      Use Monte Carlo dropout for individual and joint batch selection.
      +++
      :bdg-primary-line:`Audio`
      :bdg-secondary-line:`PyTorch`

   .. grid-item-card:: Annotate Images
      :link: generated/tutorials/03_pool_oracle_annotations
      :link-type: doc
      :class-card: tutorial-topic topic-human-labels
      :class-header: kicker

      Human labels
      ^^^
      Label images in a live notebook with batches, skips, undo, and export.
      +++
      :bdg-primary-line:`Image`

   .. grid-item-card:: Annotate Research Papers
      :link: generated/tutorials/06_pool_al_text_annotation_tool
      :link-type: doc
      :class-card: tutorial-topic topic-human-labels
      :class-header: kicker

      Human labels
      ^^^
      Label queried arXiv papers from their titles and abstracts.
      +++
      :bdg-primary-line:`Text`

   .. grid-item-card:: Advanced Multiple Annotators
      :link: generated/tutorials/11_multiple_annotators_advanced
      :link-type: doc
      :class-card: tutorial-topic topic-setting
      :class-header: kicker

      Multiple annotators
      ^^^
      Learn from noisy labels while choosing which annotator to ask.
      +++
      :bdg-primary-line:`Image`
      :bdg-secondary-line:`PyTorch`

   .. grid-item-card:: Batch-stream Active Learning
      :link: generated/tutorials/21_stream_batch_with_pool_al
      :link-type: doc
      :class-card: tutorial-topic topic-setting
      :class-header: kicker

      Stream
      ^^^
      Apply pool-based query strategies to batches of arriving samples.
      +++
      :bdg-primary-line:`Synthetic`


Run a Tutorial
--------------

* Install the dependencies listed at the start of the notebook, then run its
  cells in order from a fresh kernel. Each notebook contains its own helper
  code.
* Notebooks with a ``QUICK_RUN`` flag were run with ``False`` for the
  published comparison. Set the flag to ``True`` for a smaller run. First runs
  may download datasets and pretrained weights.

.. toctree::
   :hidden:
   :maxdepth: 1

   generated/tutorials/00_pool_getting_started
   generated/tutorials/02_pool_regression_getting_started
   generated/tutorials/30_pool_multilabel_getting_started
   generated/tutorials/10_multiple_annotators_getting_started
   generated/tutorials/20_stream_getting_started
   generated/tutorials/04_pool_simple_evaluation_study
   generated/tutorials/07_pool_advanced_regression
   generated/tutorials/05_pool_al_with_self_supervised_learning
   generated/tutorials/08_pool_ssl
   generated/tutorials/01_deep_pool_al_with_skorch
   generated/tutorials/09_bayesian_al
   generated/tutorials/03_pool_oracle_annotations
   generated/tutorials/06_pool_al_text_annotation_tool
   generated/tutorials/11_multiple_annotators_advanced
   generated/tutorials/21_stream_batch_with_pool_al
