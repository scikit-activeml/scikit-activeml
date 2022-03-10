.. _api_reference:

=============
API Reference
=============

This is an overview of the API.

.. module:: skactiveml

:mod:`skactiveml.base`: Base classes
====================================

.. automodule:: skactiveml.base
    :no-members:
    :no-inherited-members:

.. currentmodule:: skactiveml.base

.. autosummary::
   :nosignatures:
   :toctree: generated/api
   :template: class.rst

   QueryStrategy
   SingleAnnotatorPoolQueryStrategy
   MultiAnnotatorPoolQueryStrategy
   SkactivemlClassifier
   ClassFrequencyEstimator
   AnnotatorModelMixin


:mod:`skactiveml.pool`: Pool-based strategies
=============================================

.. automodule:: skactiveml.pool
    :no-members:
    :no-inherited-members:

.. currentmodule:: skactiveml.pool

Classes
-------

.. autosummary::
   :nosignatures:
   :toctree: generated/api
   :template: class.rst

   RandomSampling
   ProbabilisticAL
   UncertaintySampling
   EpistemicUncertaintySampling
   ExpectedErrorReduction
   QueryByCommittee
   FourDs
   CostEmbeddingAL

Functions
---------

.. autosummary::
   :nosignatures:
   :toctree: generated/api
   :template: function.rst

   cost_reduction
   uncertainty_scores
   expected_average_precision
   expected_error_reduction
   average_kl_divergence
   vote_entropy

:mod:`skactiveml.pool.multi`: Multi-annotator pool-based strategies
===================================================================

.. automodule:: skactiveml.pool.multi
    :no-members:
    :no-inherited-members:

.. currentmodule:: skactiveml.pool.multi

Classes
-------

.. autosummary::
   :nosignatures:
   :toctree: generated/api
   :template: class.rst

   IntervalEstimationThreshold
   IntervalEstimationAnnotModel
   SingleAnnotatorWrapper

:mod:`skactiveml.classifier`: Classifier
========================================

.. automodule:: skactiveml.classifier
    :no-members:
    :no-inherited-members:

.. currentmodule:: skactiveml.classifier

Classes
-------

.. autosummary::
   :nosignatures:
   :toctree: generated/api
   :template: class.rst

   ParzenWindowClassifier
   MixtureModelClassifier
   SklearnClassifier

:mod:`skactiveml.classifier.multi`: Multi-annotator classifier
==============================================================

.. automodule:: skactiveml.classifier.multi
    :no-members:
    :no-inherited-members:

.. currentmodule:: skactiveml.classifier.multi

Classes
-------

.. autosummary::
   :nosignatures:
   :toctree: generated/api
   :template: class.rst

   AnnotatorEnsembleClassifier
   AnnotatorLogisticRegression

:mod:`skactiveml.visualization`: Visualization functions
========================================================

.. automodule:: skactiveml.visualization
    :no-members:
    :no-inherited-members:

.. currentmodule:: skactiveml.visualization

.. autosummary::
   :nosignatures:
   :toctree: generated/api
   :template: function.rst

   plot_utilities
   plot_decision_boundary

:mod:`skactiveml.utils`: Utility classes, functions and constants
=================================================================

.. automodule:: skactiveml.utils
    :no-members:
    :no-inherited-members:

.. currentmodule:: skactiveml.utils

Classes
-------

.. autosummary::
   :nosignatures:
   :toctree: generated/api
   :template: class.rst

   ExtLabelEncoder

Functions
---------

.. autosummary::
   :nosignatures:
   :toctree: generated/api
   :template: function.rst

   rand_argmax
   rand_argmin
   compute_vote_vectors
   is_unlabeled
   is_labeled
   check_classes
   check_missing_label
   check_cost_matrix
   check_scalar
   check_classifier_params
   check_X_y
   check_random_state
   call_func
   simple_batch
   check_class_prior
   ext_confusion_matrix
   fit_if_not_fitted
   labeled_indices
   unlabeled_indices
   check_type

Constants
---------
.. currentmodule:: skactiveml

.. autosummary::
   :nosignatures:
   :toctree: generated/api
   :template: module.rst

   utils.MISSING_LABEL

.. currentmodule:: skactiveml

.. autodata:: skactiveml.utils.MISSING_LABEL

.. list-table::

   * - :doc:`skactiveml.utils.MISSING_LABEL <../skactiveml.utils.MISSING_LABEL>`
     - Define constant for missing label used throughout the package.
