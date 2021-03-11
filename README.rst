|Doc|_ |Codecov|_ |PythonVersion|_ |PyPi|_ |Paper|_

.. |Doc| image:: https://img.shields.io/badge/readthedocs.io-latest-green
.. _Doc: https://scikit-activeml.readthedocs.io/en/latest/

.. |Codecov| image:: https://codecov.io/gh/scikit-activeml/scikit-activeml/branch/master/graph/badge.svg
.. _Codecov: https://app.codecov.io/gh/scikit-activeml/scikit-activeml

.. |PythonVersion| image:: https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9-blue
.. _PythonVersion: https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9-blue

.. |PyPi| image:: https://badge.fury.io/py/scikit-activeml.svg
.. _PyPi: https://badge.fury.io/py/scikit-activeml

.. |Paper| image:: https://img.shields.io/badge/paper-10.20944/preprints202103.0194.v1-blue
.. _Paper: https://www.preprints.org/manuscript/202103.0194/v1


scikit-activeml
===============

*scikit-activeml* is a Python module for active learning on top of SciPy and scikit-learn. It is distributed under the 3-Clause BSD licence.

The project was initiated in 2020 by the Intelligent Embedded Systems Group at University Kassel.

Installation
============

The easiest way of installing scikit-activeml is using ``pip``   ::

    pip install -U scikit-activeml


Example
=======

The following code implements an Active Learning Cycle with 20 iterations using a Logistic Regression Classifier and Uncertainty Sampling. To use other classifiers, you can simply wrap classifiers from ``scikit-learn`` or use classifiers provided by ``scikit-activeml``. Note that the main difficulty using active learning with ``scikit-learn`` is the ability to handle unlabeled data which we denote as a specific value (``MISSING_LABEL``) in the label vector ``y_true``. More query strategies can be found in the documentation.     ::

    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import make_classification
    from skactiveml.pool import UncertaintySampling
    from skactiveml.utils import is_unlabeled, MISSING_LABEL
    from skactiveml.classifier import SklearnClassifier 

    X, y_true = make_classification(random_state=0)
    y = np.full(shape=y_true.shape, fill_value=MISSING_LABEL)

    clf = SklearnClassifier(LogisticRegression(),
                            classes=np.unique(y_true))
    qs = UncertaintySampling(clf, method='entropy')

    n_cycles = 20
    for c in range(n_cycles):
         unlbld_idx = np.where(is_unlabeled(y))[0]
         X_cand = X[unlbld_idx]
         query_idx = unlbld_idx[qs.query(X_cand=X_cand, X=X, y=y)]
         y[query_idx] = y_true[query_idx]
         clf.fit(X, y)

Development
===========

More information are available in the `Developer's Guide
<https://scikit-activeml.readthedocs.io/en/latest/developers_guide.html>`_.

Documentation
=============

The doumentation is available here:
https://scikit-activeml.readthedocs.io
