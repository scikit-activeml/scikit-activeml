.. _core-workflow:

=============
Core Workflow
=============

Labels can cost expert time, money, or a laboratory experiment. Active learning
selects which labels to acquire within that budget. A query strategy chooses
samples whose labels are expected to help train the model. You then request
those labels from an *oracle*, such as a domain expert.

``scikit-activeml`` builds on
`scikit-learn <https://scikit-learn.org/stable/>`_. You pass the data and known
labels to an estimator, use a query strategy to select samples, and add the
requested labels to the data. The examples below show this cycle for
classification, regression, multiple annotators, and data streams.


The Active Learning Cycle
=========================

The following terms are used throughout the documentation.

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Term
     - Meaning
   * - Sample
     - One row of the feature matrix ``X``.
   * - Label
     - The value to be predicted for a sample, stored in the label array
       ``y``.
   * - Oracle
     - A source of labels, for example a domain expert, a
       crowdsourcing platform, or a costly experiment.
   * - Estimator
     - The classifier (``clf``) or regressor (``reg``) trained on the labels
       acquired so far.
   * - Query strategy
     - The object (``qs``) that decides which samples are labeled next.
   * - Candidates
     - The samples the query strategy is allowed to select from.
   * - Batch
     - The samples selected in one cycle. ``batch_size`` sets their number.
   * - Budget
     - The number of labels that can be afforded, which usually determines
       when the cycle stops.

One classification cycle has three steps:

.. code-block:: text

    ┌──► clf.fit(X, y)
    │       Train the estimator on the labels known so far.
    │
    │    query_idx = qs.query(X=X, y=y, clf=clf)
    │       Select samples to label.
    │
    │    y[query_idx] = ...
    │       Ask the oracle for the labels and store them in y.
    │       │
    └───────┘

    Repeat until the labeling budget is exhausted.

In the third step, request labels from a labeling service, a domain expert, or
a measurement. The examples simulate this request by reading the selected
labels from a ground-truth array, ``y_true``. They also use ``y_true`` to set up
initial labels and report scores. Pass only the acquired labels in ``y`` to
``fit()`` and ``query()``; passing ``y_true`` would expose labels that are
unavailable during a real query.


The Label Vector ``y``
======================

For single-output tasks, ``X`` contains the labeled and unlabeled samples, and
``y`` has one entry per sample. Fill unknown entries with
:doc:`MISSING_LABEL <skactiveml.utils.MISSING_LABEL>`.

.. doctest::

   >>> import numpy as np
   >>> from skactiveml.utils import (
   ...     MISSING_LABEL, labeled_indices, unlabeled_indices
   ... )
   >>> y = np.full(shape=5, fill_value=MISSING_LABEL)
   >>> y[[1, 3]] = [0, 1]
   >>> y
   array([nan,  0., nan,  1., nan])
   >>> labeled_indices(y)
   array([1, 3])
   >>> unlabeled_indices(y)
   array([0, 2, 4])

``X[i]`` and ``y[i]`` refer to the same sample. When querying from ``X``, use
the returned indices to store labels with ``y[query_idx] = ...``. Keeping the
rows in place preserves the indices from earlier queries.

Pass the full ``X`` and the partially labeled ``y`` to the estimator.
Estimators that use unlabeled samples, such as self-training or density-based
models, can use those rows during fitting. Supervised estimators train on the
observed labels. The same ``(X, y)`` pair gives query strategies access to both
the labeled samples and the samples they may query next.

For regression, ``y`` stores continuous labels. For multi-label classification,
each row is a label vector. With multiple annotators, ``y[i, j]`` stores the
label that annotator ``j`` provided for sample ``i``.


Choosing the Missing Label
--------------------------

``MISSING_LABEL`` is ``np.nan``. To store it with numeric labels, use a
floating-point or object array. Use the ``missing_label`` parameter to choose
another placeholder, such as ``-1`` for integer labels or ``None`` for string
labels.

.. doctest::

   >>> y_int = np.full(shape=5, fill_value=-1)
   >>> y_int[[1, 3]] = [0, 1]
   >>> labeled_indices(y_int, missing_label=-1)
   array([1, 3])

Use the same ``missing_label`` throughout the workflow so that estimators,
query strategies, and label helpers agree on which entries are missing.


Pool-based Classification
=========================

In pool-based active learning, ``X`` contains a fixed pool of samples and ``y``
records which of them have been labeled so far. The example below uses two
clusters generated with :func:`~sklearn.datasets.make_blobs`, a logistic
regression classifier, and uncertainty sampling, which selects the samples
whose predicted class is least certain.

:class:`~skactiveml.classifier.SklearnClassifier` lets a scikit-learn classifier
fit a partially labeled ``y``. Set ``classes`` to declare the class vocabulary
before all classes have been observed. This example uses ``y_true`` to select
one initial sample per class. In an application, obtain initial labels from
the oracle, for example by labeling random samples until each class has been
observed.

.. doctest::

   >>> import numpy as np
   >>> from sklearn.datasets import make_blobs
   >>> from sklearn.linear_model import LogisticRegression
   >>> from skactiveml.classifier import SklearnClassifier
   >>> from skactiveml.pool import UncertaintySampling
   >>> from skactiveml.utils import MISSING_LABEL, labeled_indices
   >>> X, y_true = make_blobs(
   ...     n_samples=100,
   ...     centers=[[-3.0, 0.0], [3.0, 0.0]],
   ...     cluster_std=1.5,
   ...     random_state=0,
   ... )
   >>> y = np.full(shape=len(X), fill_value=MISSING_LABEL)
   >>> initial_idx = [np.flatnonzero(y_true == c)[0] for c in [0, 1]]
   >>> y[initial_idx] = y_true[initial_idx]
   >>> clf = SklearnClassifier(
   ...     LogisticRegression(), classes=[0, 1], random_state=0
   ... )
   >>> qs = UncertaintySampling(method="entropy", random_state=0)
   >>> _ = clf.fit(X, y)
   >>> print(f"{len(labeled_indices(y))} labels: {clf.score(X, y_true):.2f}")
   2 labels: 0.87
   >>> for cycle in range(10):
   ...     query_idx = qs.query(
   ...         X=X, y=y, clf=clf, fit_clf=False, batch_size=1
   ...     )
   ...     # Replace this lookup by a request to your labeling process.
   ...     y[query_idx] = y_true[query_idx]
   ...     _ = clf.fit(X, y)
   >>> print(f"{len(labeled_indices(y))} labels: {clf.score(X, y_true):.2f}")
   12 labels: 1.00

In this example, the classifier predicts the pool labels correctly after two
initial labels and ten queried labels. The score measures accuracy on this
pool. Each query fills entries in ``y``; the rows of ``X`` stay in place.

Model-based query strategies fit the estimator by default. The example fits
``clf`` before querying, so it passes ``fit_clf=False`` to reuse the fitted
estimator. Regression strategies use ``fit_reg`` for the same purpose. Set
these flags to ``False`` when you have already fitted the estimator to avoid
fitting it twice for the same query.


Candidates, Batches, and Utilities
----------------------------------

For a single-annotator pool strategy, ``candidates`` controls which samples may
be selected and how the returned indices are interpreted.

.. list-table::
   :header-rows: 1
   :widths: 25 40 35

   * - ``candidates``
     - Candidate set
     - Meaning of returned indices
   * - ``None``
     - Unlabeled samples in ``(X, y)``
     - Positions in ``X``
   * - One-dimensional integer array
     - The indicated subset of ``X``
     - Positions in ``X``
   * - Feature array
     - Samples supplied separately from ``X``
     - Positions in ``candidates``

Support for separate feature arrays depends on the strategy. ``batch_size``
sets how many samples are selected in one cycle. A batch lets you request
several labels at once.

With ``return_utilities=True``, the strategy also returns the scores used to
select samples. The utilities have one row per selection in the batch, with
``np.nan`` at positions outside the candidate set.

.. doctest::

   >>> query_idx, utilities = qs.query(
   ...     X=X, y=y, clf=clf, fit_clf=False, batch_size=3,
   ...     return_utilities=True,
   ... )
   >>> query_idx.shape
   (3,)
   >>> utilities.shape
   (3, 100)
   >>> bool(np.all(np.isnan(utilities[0, labeled_indices(y)])))
   True

See each strategy's :meth:`query() <skactiveml.base.QueryStrategy.query>`
documentation for the accepted inputs and utility shapes.


Workflow Variations
===================

The following examples change the estimator, the label representation, or the
query interface for each task and annotation setting.


Regression
----------

Regression uses continuous labels and a regression query strategy.
:class:`~skactiveml.pool.GreedySamplingTarget` selects the samples whose
predicted labels differ most from the labels acquired so far.

.. doctest::

   >>> from sklearn.datasets import make_regression
   >>> from skactiveml.pool import GreedySamplingTarget
   >>> from skactiveml.regressor import NICKernelRegressor
   >>> X, y_true = make_regression(
   ...     n_samples=100, n_features=1, noise=5.0, random_state=0
   ... )
   >>> y = np.full(shape=len(X), fill_value=MISSING_LABEL)
   >>> y[[0, 1]] = y_true[[0, 1]]
   >>> reg = NICKernelRegressor(random_state=0)
   >>> qs = GreedySamplingTarget(random_state=0)
   >>> for cycle in range(10):
   ...     query_idx = qs.query(X=X, y=y, reg=reg, batch_size=1)
   ...     y[query_idx] = y_true[query_idx]
   >>> _ = reg.fit(X, y)
   >>> print(f"{len(labeled_indices(y))} labels: {reg.score(X, y_true):.2f}")
   12 labels: 0.96

The query strategy handles fitting during each query. The final
``reg.fit(X, y)`` fits the regressor used for scoring.


Multi-label Classification
--------------------------

In multi-label classification, ``y`` has one column per binary label output.
Declare ``target_type="multi-label"`` explicitly on both the classifier and
the query strategy. Each query requests a complete label vector. Every row of
``y`` must be fully labeled or fully missing.

.. doctest::

   >>> from sklearn.datasets import make_multilabel_classification
   >>> from sklearn.multioutput import MultiOutputClassifier
   >>> from skactiveml.utils import is_labeled
   >>> X, y_true = make_multilabel_classification(
   ...     n_samples=100, n_features=10, n_classes=3, n_labels=2,
   ...     random_state=0,
   ... )
   >>> y = np.full(shape=y_true.shape, fill_value=MISSING_LABEL)
   >>> y[:5] = y_true[:5]
   >>> clf = SklearnClassifier(
   ...     MultiOutputClassifier(LogisticRegression()),
   ...     classes=[[0, 1], [0, 1], [0, 1]],
   ...     target_type="multi-label",
   ...     random_state=0,
   ... )
   >>> qs = UncertaintySampling(
   ...     method="entropy", target_type="multi-label", random_state=0
   ... )
   >>> _ = clf.fit(X, y)
   >>> for cycle in range(5):
   ...     query_idx = qs.query(X=X, y=y, clf=clf, fit_clf=False)
   ...     y[query_idx] = y_true[query_idx]
   ...     _ = clf.fit(X, y)
   >>> is_labeled(y, target_type="multi-label").shape
   (100,)
   >>> int(np.sum(is_labeled(y, target_type="multi-label")))
   10

With ``target_type="multi-label"``, ``is_labeled`` returns one entry per
sample. See :doc:`Target and Annotation Semantics <target_semantics>` for class
vocabularies, label shapes, and supported multi-label components.


Multiple Annotators
-------------------

A multi-annotator strategy selects both a sample and an annotator, returning
``(sample, annotator)`` pairs. Store each acquired label in ``y[i, j]``, where
``i`` is the sample index and ``j`` is the annotator index.

The example simulates two annotators for the clusters used above. The first
returns the true label; the second flips it with probability 0.3.
:func:`~skactiveml.utils.majority_vote` combines their labels into one vector
for fitting a classifier.

.. doctest::

   >>> from skactiveml.classifier import ParzenWindowClassifier
   >>> from skactiveml.pool import RandomSampling
   >>> from skactiveml.pool.multiannotator import SingleAnnotatorWrapper
   >>> from skactiveml.utils import majority_vote
   >>> X, y_true = make_blobs(
   ...     n_samples=100,
   ...     centers=[[-3.0, 0.0], [3.0, 0.0]],
   ...     cluster_std=1.5,
   ...     random_state=0,
   ... )
   >>> rng = np.random.default_rng(0)
   >>> is_confused = rng.random(len(X)) < 0.3
   >>> y_true_annotators = np.column_stack(
   ...     [y_true, np.where(is_confused, 1 - y_true, y_true)]
   ... )
   >>> y = np.full(shape=y_true_annotators.shape, fill_value=MISSING_LABEL)
   >>> qs = SingleAnnotatorWrapper(
   ...     RandomSampling(random_state=0), random_state=0
   ... )
   >>> query_pairs = qs.query(
   ...     X=X, y=y, batch_size=20, n_annotators_per_sample=2
   ... )
   >>> query_pairs.shape
   (20, 2)
   >>> sample_idx, annotator_idx = query_pairs[:, 0], query_pairs[:, 1]
   >>> y[sample_idx, annotator_idx] = y_true_annotators[
   ...     sample_idx, annotator_idx
   ... ]
   >>> y_aggregated = majority_vote(y, random_state=0)
   >>> clf = ParzenWindowClassifier(classes=[0, 1], random_state=0)
   >>> _ = clf.fit(X, y_aggregated)
   >>> int(np.sum(is_labeled(y_aggregated)))
   10

The 20 acquired labels cover 10 samples, because each selected sample was
labeled by both annotators.


Stream-based Active Learning
----------------------------

In stream-based active learning, samples arrive in sequence. For each sample,
decide whether to acquire its label before seeing future samples. Pass the
current sample as ``candidates``. ``query()`` proposes a selection; call
``update()`` to record the decision and advance the strategy's budget and
state. Call it after every decision, including when nothing was queried.

The example uses :class:`~skactiveml.stream.PeriodicSampling` to query labels
at a fixed rate.

.. doctest::

   >>> from skactiveml.stream import PeriodicSampling
   >>> X, y_true = make_blobs(
   ...     n_samples=20,
   ...     centers=[[-3.0, 0.0], [3.0, 0.0]],
   ...     cluster_std=1.5,
   ...     random_state=0,
   ... )
   >>> y = np.full(shape=len(X), fill_value=MISSING_LABEL)
   >>> qs = PeriodicSampling(budget=0.2, random_state=0)
   >>> for t in range(len(X)):
   ...     candidates = X[[t]]
   ...     query_idx = qs.query(candidates=candidates)
   ...     # update() is required even if query_idx is empty.
   ...     _ = qs.update(candidates=candidates, queried_indices=query_idx)
   ...     if len(query_idx) > 0:
   ...         y[t] = y_true[t]
   >>> qs.observed_samples_
   20
   >>> int(np.sum(is_labeled(y)))
   4

The strategy queried 4 of the 20 samples, matching the 20% budget. Model-based
stream strategies also receive the estimator and the labels observed so far.


Common Pitfalls
===============

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Pitfall
     - What happens, and what to do instead
   * - Different ``missing_label`` values
     - Components disagree on which entries are missing. Pass the
       same ``missing_label`` to every estimator, query strategy, and utility
       function.
   * - Fitting the estimator twice
     - Model-based strategies fit the estimator inside ``query()`` by default,
       so an additional explicit ``fit()`` trains the same model twice per
       cycle. Either drop the explicit call or pass ``fit_clf=False``
       (``fit_reg=False`` for regression).
   * - Writing ``MISSING_LABEL`` into an integer ``y``
     - ``np.nan`` cannot be stored in an integer array, so the assignment
       raises ``ValueError: cannot convert float NaN to integer``. Use a
       floating-point ``y`` or an integer placeholder such as
       ``missing_label=-1``.
   * - Omitting ``classes``
     - The classifier infers classes from observed labels, so ``predict_proba``
       may have fewer columns during early cycles. Declare the full vocabulary
       with ``classes``.
   * - Shrinking ``X`` after a query
     - Removing labeled samples from the pool invalidates the indices of
       earlier queries and hides those samples from strategies that reason
       about the whole data set. Keep ``X`` fixed and only write into ``y``.
   * - Skipping ``update()`` in a stream
     - ``query()`` does not advance the internal state of a stream strategy, so
       its budget is never tracked and
       :class:`~skactiveml.stream.PeriodicSampling`, for example, stops
       querying altogether. Call ``update()`` after every decision, including
       when nothing was queried.
   * - Passing ``y_true`` to ``fit()`` or ``query()``
     - Fitting or querying with ground-truth labels produces results that a
       real application cannot reproduce. Pass the partially labeled ``y``
       containing only the labels acquired so far.


Where to Go Next
================

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Goal
     - Documentation
   * - Build a complete pool-based classification experiment
     - :doc:`Pool-based Active Learning: Getting Started <generated/tutorials/00_pool_getting_started>`
   * - Use deep models and learned representations
     - :doc:`Deep Active Learning by Fine-tuning DINOv2 <generated/tutorials/01_deep_pool_al_with_skorch>`
   * - Work with continuous labels
     - :doc:`Pool-based Active Learning for Regression: Getting Started <generated/tutorials/02_pool_regression_getting_started>`
   * - Query labels from several annotators
     - :doc:`Multi-annotator Pool-based Active Learning: Getting Started <generated/tutorials/10_multiple_annotators_getting_started>`
   * - Process a data stream
     - :doc:`Stream-based Active Learning with scikit-learn, River, and CapyMOA <generated/tutorials/20_stream_getting_started>`
   * - Query complete label vectors
     - :doc:`Pool-based Multilabel Active Learning: Getting Started <generated/tutorials/30_pool_multilabel_getting_started>`
   * - Select a query strategy
     - :doc:`Strategy Overview <generated/strategy_overview>`
   * - Inspect classes, methods, and parameters
     - :doc:`API Reference <generated/api/skactiveml>`
