.. _core-workflow:

=============
Core Workflow
=============

Labeling data is usually the expensive part of building a machine learning
model. Unlabeled samples are often cheap to collect, but every label costs
expert time, money, or a laboratory experiment. Active learning spends this
budget deliberately: instead of labeling a large random subset, it repeatedly
asks an *oracle* for the labels of those samples from which the model is
expected to learn the most.

``scikit-activeml`` implements this idea on top of
`scikit-learn <https://scikit-learn.org/stable/>`_. Regardless of the
prediction task or the active learning scenario, a workflow is built from the
same three parts: the **data**, an **estimator** that is trained on the
labels acquired so far, and a **query strategy** that decides which samples are
labeled next. This page explains how these parts fit together and shows a
complete, runnable cycle for each supported scenario.


🔁 The Active Learning Cycle
============================

The following terms are used throughout the documentation.

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Term
     - Meaning
   * - Sample
     - One row of the feature matrix ``X``.
   * - Label
     - The value to be predicted for a sample, stored in the label vector
       ``y``.
   * - Oracle
     - Whatever provides labels on request, for example a domain expert, a
       crowdsourcing platform, or a costly experiment.
   * - Estimator
     - The classifier (``clf``) or regressor (``reg``) trained on the labels
       acquired so far.
   * - Query strategy
     - The object (``qs``) that decides which samples are labeled next.
   * - Candidates
     - The samples the query strategy is allowed to select from.
   * - Batch
     - The number of samples selected in one cycle (``batch_size``).
   * - Budget
     - The number of labels that can be afforded, which usually determines
       when the cycle stops.

Written as code, one cycle consists of three steps:

.. code-block:: text

    ┌──► clf.fit(X, y)
    │       Train the estimator on the labels known so far.
    │
    │    query_idx = qs.query(X=X, y=y, clf=clf)
    │       Let the query strategy select the most useful samples.
    │
    │    y[query_idx] = ...
    │       Ask the oracle for the labels and store them in y.
    │       │
    └───────┘

    Repeat until the labeling budget is exhausted.

The third step is the only one that leaves the program. In a real application,
it is replaced by a call to a labeling service, a form shown to a domain
expert, or a measurement. Because this documentation has to run without a human
in the loop, the examples below simulate the oracle with a ground-truth array
``y_true`` and read from it only in that third step. Never use ``y_true`` for
fitting an estimator or for querying, as this would leak labels that a real
application does not have.


🏷️ The Label Vector ``y``
==========================

``scikit-activeml`` does not split the data into a labeled and an unlabeled
set. Instead, ``X`` always contains *all* samples and ``y`` always has one
entry per sample. Entries whose label is not known yet are filled with the
missing label :doc:`MISSING_LABEL <skactiveml.utils.MISSING_LABEL>`.

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

Keeping known and unknown labels in one array has three practical consequences.

**Indexing stays simple.** ``X[i]`` and ``y[i]`` always refer to the same
sample, so the indices returned by ``query()`` can be applied directly to both
arrays. Acquiring labels is a single assignment, ``y[query_idx] = ...``, and no
other part of the loop has to be updated. With separate labeled and unlabeled
arrays, every cycle would move rows from one array into the other and thereby
invalidate indices that were already stored.

**Semi-supervised learning is native.** An estimator is always fitted on the
complete ``X`` together with the labels known so far. Estimators that exploit
unlabeled samples, such as self-training or density-based models, therefore
need no additional interface, whereas purely supervised estimators simply
ignore the samples with a missing label.

**Query strategies see both parts.** A query strategy typically needs the
labeled samples to judge what the model already knows and the unlabeled samples
to decide what to ask for next. Passing a single ``(X, y)`` pair provides both.

The same convention is used in every scenario. For regression, ``y`` stores
continuous labels; for multi-label classification, each row of ``y`` is a label
vector; and for multiple annotators, ``y[i, j]`` stores the label that
annotator ``j`` provided for sample ``i``.


Choosing the Missing Label
--------------------------

``MISSING_LABEL`` is ``np.nan``, which requires a floating-point ``y`` and
therefore does not fit integer or string labels. Every estimator, query
strategy, and utility function accepts a ``missing_label`` parameter, so the
placeholder can be adapted to the labels at hand, for example ``-1`` for
integer labels or ``None`` for string labels.

.. doctest::

   >>> y_int = np.full(shape=5, fill_value=-1)
   >>> y_int[[1, 3]] = [0, 1]
   >>> labeled_indices(y_int, missing_label=-1)
   array([1, 3])

All objects taking part in one workflow must agree on the same
``missing_label``, since otherwise the same entry would be considered labeled
by one object and unlabeled by another.


🏊 Pool-based Classification
============================

In pool-based active learning, ``X`` contains a fixed pool of samples and ``y``
records which of them have been labeled so far. The example below uses two
clusters generated with :func:`~sklearn.datasets.make_blobs`, a logistic
regression classifier, and uncertainty sampling, which selects the samples
whose predicted class is least certain.

:class:`~skactiveml.classifier.SklearnClassifier` wraps a scikit-learn
classifier so that it can be fitted on a partially labeled ``y``. Passing
``classes`` explicitly makes the classifier independent of which classes happen
to be labeled already. The cycle starts from one labeled sample per class,
which in practice corresponds to labeling a small random subset until every
class has been observed.

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

Ten queried labels are enough to classify the whole pool correctly. Note that
the loop never removes a sample from ``X``: querying only turns a missing entry
of ``y`` into a known label.

By default, a model-based query strategy fits the estimator it receives, which
is convenient when the caller does not maintain a fitted estimator. The example
above fits ``clf`` itself, so it passes ``fit_clf=False`` to reuse that fitted
state. Regression strategies use ``fit_reg`` for the same purpose. Choose one
owner for the fit that precedes a query, because fitting the estimator
explicitly and additionally leaving the flag at its default performs the same
work twice.


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

Support for separately supplied feature arrays depends on the strategy.
``batch_size`` sets how many samples are selected in one cycle. Requesting a
batch is useful whenever labels can be acquired in parallel, for instance by
several annotators at once. With ``return_utilities=True``, the strategy also
returns the scores behind its decision, with one row per selection in the batch
and ``np.nan`` for every sample that was not a candidate.

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

Consult the :meth:`query() <skactiveml.base.QueryStrategy.query>`
documentation of a strategy for its exact inputs and the shape of its
utilities.


🔀 Workflow Variations
======================

The cycle stays the same across prediction tasks, annotator setups, and
active learning scenarios. The differences are the estimator, the shape and
meaning of ``y``, and the arguments and return value of ``query()``.


📈 Regression
-------------

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

This loop leaves the fit to the query strategy, which is why no ``fit_reg``
argument is passed and ``reg`` is fitted only once at the end for scoring.


📑 Multi-label Classification
-----------------------------

In multi-label classification, every sample carries several binary labels, so
``y`` becomes a matrix with one column per label. Declare
``target_type="multi-label"`` explicitly on both the classifier and the query
strategy. A row of ``y`` is queried as a whole: it is either completely labeled
or completely missing, but never partially labeled.

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

Because complete rows are queried, ``is_labeled`` returns one entry per sample
rather than one per label. See
:doc:`Target and Annotation Semantics <target_semantics>` for class
vocabularies, label shapes, and the complete multi-label capability contract.


👥 Multiple Annotators
----------------------

If several annotators are available and they do not agree, the sample alone no
longer determines what a query costs or what it is worth. ``y`` therefore
becomes a matrix in which ``y[i, j]`` stores the label that annotator ``j``
provided for sample ``i``, and a multi-annotator strategy returns
``(sample, annotator)`` pairs instead of sample indices.

In the example, two annotators label the clusters from above: the first one is
reliable, whereas the second one confuses the classes in about 30% of the
cases. :func:`~skactiveml.utils.majority_vote` aggregates the collected labels
into a single label vector that an ordinary classifier can be fitted on.

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


🌊 Stream-based Active Learning
-------------------------------

In stream-based active learning, samples arrive one after another and the
decision whether to acquire a label has to be made immediately, without access
to future samples. The current sample is passed as ``candidates``, and
``query()`` only proposes a decision: it does not advance the budget or any
other internal state of the strategy. Call ``update()`` after every decision,
including when nothing was queried.

The example uses :class:`~skactiveml.stream.PeriodicSampling`, which spends its
budget at a constant rate, so that the decisions are easy to follow.

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

The stream strategy labeled 4 of the 20 observed samples, matching the budget
of 20%. Model-based stream strategies additionally receive the estimator and
the labels observed so far.


⚠️ Common Pitfalls
===================

The following mistakes are easy to make and typically show up as a workflow
that runs without an error but does not behave as intended.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Pitfall
     - What happens, and what to do instead
   * - Different ``missing_label`` values
     - The estimator considers an entry labeled while the query strategy
       considers it unlabeled, so samples are queried twice or never. Pass the
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
     - A classifier fitted on a partially labeled ``y`` only knows the classes
       it has already seen, so ``predict_proba`` returns fewer columns during
       the early cycles. Pass ``classes`` explicitly.
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
   * - Using ``y_true`` outside the labeling step
     - Fitting or querying with ground-truth labels produces results that a
       real application cannot reproduce. Read from the oracle only where the
       acquired labels are written into ``y``.


🧭 Where to Go Next
===================

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Goal
     - Documentation
   * - Build a complete pool-based classification experiment
     - :doc:`Pool-based Active Learning: Getting Started <generated/tutorials/00_pool_getting_started>`
   * - Use deep models and learned representations
     - :doc:`Deep Active Learning for Fine-tuning Vision Transformers <generated/tutorials/01_deep_pool_al_with_skorch>`
   * - Work with continuous labels
     - :doc:`Pool-based Active Learning for Regression: Getting Started <generated/tutorials/02_pool_regression_getting_started>`
   * - Query labels from several annotators
     - :doc:`Multi-annotator Active Learning: Getting Started <generated/tutorials/10_multiple_annotators_getting_started>`
   * - Process a data stream
     - :doc:`Stream-based Active Learning: Getting Started <generated/tutorials/20_stream_getting_started>`
   * - Query complete label vectors
     - :doc:`Pool-based Multilabel Active Learning: Getting Started <generated/tutorials/30_pool_multilabel_getting_started>`
   * - Select a query strategy
     - :doc:`Strategy Overview <generated/strategy_overview>`
   * - Inspect classes, methods, and parameters
     - :doc:`API reference <generated/api/skactiveml>`
