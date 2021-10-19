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

.. currentmodule:: skactiveml

.. autosummary::
   :nosignatures:
   :toctree: generated/api
   :template: class.rst

   skactiveml.QueryStrategy
   skactiveml.SingleAnnotPoolBasedQueryStrategy
   skactiveml.MultiAnnotPoolBasedQueryStrategy
   skactiveml.SkactivemlClassifier
   skactiveml.ClassFrequencyEstimator
   skactiveml.AnnotModelMixin


:mod:`skactiveml.pool`: Pool-based strategies
=============================================

.. automodule:: skactiveml.pool
    :no-members:
    :no-inherited-members:

Classes
-------
.. currentmodule:: skactiveml

.. autosummary::
   :nosignatures:
   :toctree: generated/api
   :template: class.rst

   skactiveml.pool.RandomSampler
   skactiveml.pool.McPAL
   skactiveml.pool.UncertaintySampling
   skactiveml.pool.EpistemicUncertainty
   skactiveml.pool.ExpectedErrorReduction
   skactiveml.pool.QBC
   skactiveml.pool.FourDS
   skactiveml.pool.ALCE

Functions
---------
.. currentmodule:: skactiveml

.. autosummary::
   :nosignatures:
   :toctree: generated/api
   :template: function.rst

   skactiveml.pool.cost_reduction
   skactiveml.pool.uncertainty_scores
   skactiveml.pool.expected_average_precision
   skactiveml.pool.expected_error_reduction
   skactiveml.pool.average_kl_divergence
   skactiveml.pool.vote_entropy

:mod:`skactiveml.classifier`: Classifier
========================================

.. automodule:: skactiveml.classifier
    :no-members:
    :no-inherited-members:

.. currentmodule:: skactiveml

.. autosummary::
   :nosignatures:
   :toctree: generated/api
   :template: class.rst

   skactiveml.classifier.PWC
   skactiveml.classifier.CMM
   skactiveml.classifier.SklearnClassifier
   skactiveml.classifier.MultiAnnotClassifier
   skactiveml.classifier.LogisticRegressionRY

:mod:`skactiveml.visualization`: Visualization functions
========================================================

.. automodule:: skactiveml.visualization
    :no-members:
    :no-inherited-members:

.. currentmodule:: skactiveml

.. autosummary::
   :nosignatures:
   :toctree: generated/api
   :template: function.rst

   skactiveml.visualization.plot_utility
   skactiveml.visualization.plot_decision_boundary

:mod:`skactiveml.utils`: Utility classes, functions and constants
=================================================================

.. automodule:: skactiveml.utils
    :no-members:
    :no-inherited-members:

Classes
-------
.. currentmodule:: skactiveml

.. autosummary::
   :nosignatures:
   :toctree: generated/api
   :template: class.rst

   skactiveml.utils.ExtLabelEncoder

Functions
---------
.. currentmodule:: skactiveml

.. autosummary::
   :nosignatures:
   :toctree: generated/api
   :template: function.rst

   skactiveml.utils.rand_argmax
   skactiveml.utils.rand_argmin
   skactiveml.utils.compute_vote_vectors
   skactiveml.utils.is_unlabeled
   skactiveml.utils.is_labeled
   skactiveml.utils.check_classes
   skactiveml.utils.check_missing_label
   skactiveml.utils.check_cost_matrix
   skactiveml.utils.check_scalar
   skactiveml.utils.check_classifier_params
   skactiveml.utils.check_X_y
   skactiveml.utils.check_random_state
   skactiveml.utils.call_func
   skactiveml.utils.simple_batch
   skactiveml.utils.check_class_prior
   skactiveml.utils.ext_confusion_matrix
   skactiveml.utils.fit_if_not_fitted
   skactiveml.utils.labeled_indices
   skactiveml.utils.unlabeled_indices
   skactiveml.utils.check_type

Constants
---------
.. currentmodule:: skactiveml

.. autosummary::
   :nosignatures:
   :toctree: generated/api
   :template: module.rst

   skactiveml.utils.MISSING_LABEL

.. currentmodule:: skactiveml

.. autodata:: skactiveml.utils.MISSING_LABEL

.. list-table::

   * - :doc:`skactiveml.utils.MISSING_LABEL <../skactiveml.utils.MISSING_LABEL>`
     - Define constant for missing label used throughout the package.
