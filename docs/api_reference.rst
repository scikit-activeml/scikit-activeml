API Reference
=============

.. toctree::

This is an overview of the API.

.. currentmodule:: skactiveml

Pool:
-----

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: class.rst

   pool.RandomSampler
   pool.McPAL
   pool.UncertaintySampling
   pool.EpistemicUncertainty
   pool.ExpectedErrorReduction
   pool.QBC
   pool.FourDS
   pool.ALCE

Classifier:
-----------

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: class.rst

   classifier.PWC
   classifier.CMM
   classifier.SklearnClassifier
   classifier.MultiAnnotClassifier
   classifier.LogisticRegressionRY

Utils:
------

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: class.rst

   utils.rand_argmax
   utils.rand_argmin
   utils.compute_vote_vectors
   utils.is_unlabeled
   utils.is_labeled
   utils.ExtLabelEncoder
   utils.check_classes
   utils.check_missing_label
   utils.check_cost_matrix
   utils.check_scalar
   utils.check_classifier_params
   utils.check_X_y
   utils.check_random_state
   utils.MISSING_LABEL
   utils.call_func
   utils.simple_batch
   utils.ext_confusion_matrix

