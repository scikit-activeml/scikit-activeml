.. intro_start

|Doc|_ |Codecov|_ |PythonVersion|_ |PyPi|_ |Paper|_

.. |Doc| image:: https://img.shields.io/badge/docs-latest-green
.. _Doc: https://scikit-activeml.github.io/scikit-activeml/

.. |Codecov| image:: https://codecov.io/gh/scikit-activeml/scikit-activeml/branch/master/graph/badge.svg
.. _Codecov: https://app.codecov.io/gh/scikit-activeml/scikit-activeml

.. |PythonVersion| image:: https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9-blue
.. _PythonVersion: https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9-blue

.. |PyPi| image:: https://badge.fury.io/py/scikit-activeml.svg
.. _PyPi: https://badge.fury.io/py/scikit-activeml

.. |Paper| image:: https://img.shields.io/badge/paper-10.20944/preprints202103.0194.v1-blue
.. _Paper: https://www.preprints.org/manuscript/202103.0194/v1

|

.. image:: https://raw.githubusercontent.com/scikit-activeml/scikit-activeml/master/docs/logos/scikit-activeml-logo.png
   :width: 200

|

*scikit-activeml* is a Python module for active learning on top of SciPy and scikit-learn. It is distributed under the 3-Clause BSD licence.

The project was initiated in 2020 by the Intelligent Embedded Systems Group at University Kassel.

.. intro_end

.. install_start

Installation
============

The easiest way of installing scikit-activeml is using ``pip``   ::

    pip install -U scikit-activeml

.. install_end

.. examples_start

Example
=======

The following code implements an active learning cycle with 20 iterations using a logistic regression classifier and uncertainty sampling. To use other classifiers, you can simply wrap classifiers from ``scikit-learn`` or use classifiers provided by ``scikit-activeml``. Note that the main difficulty using active learning with ``scikit-learn`` is the ability to handle unlabeled data, which we denote as a specific value (``MISSING_LABEL``) in the label vector ``y``. More query strategies can be found in the documentation.     ::

    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import make_classification
    from skactiveml.pool import UncertaintySampling
    from skactiveml.utils import unlabeled_indices, MISSING_LABEL
    from skactiveml.classifier import SklearnClassifier 

    # Generate data set.
    X, y_true = make_classification(random_state=0)
    y = np.full(shape=y_true.shape, fill_value=MISSING_LABEL)

    # Create classifier and query strategy.
    clf = SklearnClassifier(LogisticRegression(), classes=np.unique(y_true))
    qs = UncertaintySampling(method='entropy')

    # Execute active learning cycle.
    n_cycles = 20
    for c in range(n_cycles):
         clf.fit(X, y)
         unlbld_idx = unlabeled_indices(y)
         X_cand = X[unlbld_idx]
         query_idx = unlbld_idx[qs.query(X_cand=X_cand, clf=clf)]
         y[query_idx] = y_true[query_idx]
    print(f'Accuracy: {clf.fit(X, y).score(X, y_true)}')

.. examples_end

.. dev_start

Development
===========

More information are available in the `Developer's Guide
<https://scikit-activeml.readthedocs.io/en/latest/developers_guide.html>`_.

.. dev_end

Documentation
=============

The documentation is available here:
https://scikit-activeml.readthedocs.io
