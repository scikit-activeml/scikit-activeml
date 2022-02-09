.. intro_start

|Doc|_ |Codecov|_ |PythonVersion|_ |PyPi|_ |Paper|_

.. |Doc| image:: https://img.shields.io/badge/docs-latest-green
.. _Doc: https://scikit-activeml.github.io/scikit-activeml-docs/

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

Machine learning applications often need large amounts of training data to
perform well. Whereas unlabeled data can be easily gathered, the labeling process
is difficult, time-consuming, or expensive in most applications. Active learning can help solve
this problem by querying labels for those data samples that will improve the performance
the most. Thereby, the goal is that the learning algorithm performs sufficiently well with
fewer labels

With this goal in mind, **scikit-activeml** has been developed as a Python module for active learning
on top of `scikit-learn <https://scikit-learn.org/stable/>`_. The project was initiated in 2020 by the
`Intelligent Embedded Systems Group <https://www.uni-kassel.de/eecs/en/sections/intelligent-embedded-systems/home>`_
at the University of Kassel and is distributed under the `3-Clause BSD licence
<https://github.com/scikit-activeml/scikit-activeml/blob/master/LICENSE.txt>`_.

.. intro_end

.. overview_start

Overview
========

Our philosophy is to extend the ``sklearn`` eco-system with the most relevant
query strategies for active learning and to implement tools for working with partially
unlabeled data. An overview of our repository's structure is given in the image below.
Each node represents a class or interface. The arrows illustrate the inheritance
hierarchy among them. The functionality of a dashed node is not yet available in our library.

|

.. image:: https://raw.githubusercontent.com/scikit-activeml/scikit-activeml/master/docs/logos/scikit-activeml-structure.png
   :width: 1000

|

In our package ``skactiveml``, there three major components, i.e., ``SkactivemlClassifier``,
``QueryStrategy``, and the not yet supported ``SkactivemlRegressor``.
The classifier and regressor modules are necessary to deal with partially unlabeled
data and to implement active-learning specific estimators. This way, an active learning
cycle can be easily implemented to start with zero initial labels. Regarding the
active learning query strategies, we currently differ between
the pool-based (a large pool of unlabeled samples is available) and stream-based
(unlabeled samples arrive sequentially, i.e., as a stream) paradigm.
On top of both paradigms, we also distinguish the single- and multi-annotator
setting. In the latter setting, multiple error-prone annotators are queried
to provide labels. As a result, an active learning query strategy not only decides
which samples but also which annotators should be queried.

.. overview_end

.. user_installation_start

User Installation
=================

The easiest way of installing scikit-activeml is using ``pip``:

::

    pip install -U scikit-activeml

.. install_end

.. examples_start

Examples
========
In the following, there are two simple examples illustrating the straightforwardness
of implementing active learning cycles with our Python package ``skactiveml``.
For more in-depth examples, we refer to our
`tutorial section <https://scikit-activeml.github.io/scikit-activeml-docs/>`_.

Pool-based Active Learning
##########################

The following code implements an active learning cycle with 20 iterations using a logistic regression
classifier and uncertainty sampling. To use other classifiers, you can simply wrap classifiers from
``sklearn`` or use classifiers provided by ``skactiveml``. Note that the main difficulty using
active learning with ``sklearn`` is the ability to handle unlabeled data, which we denote as a specific value
(``MISSING_LABEL``) in the label vector ``y``. More query strategies can be found in the documentation.

.. code-block:: python
    
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import make_classification
    from skactiveml.pool import UncertaintySampling
    from skactiveml.utils import unlabeled_indices, MISSING_LABEL
    from skactiveml.classifier import SklearnClassifier 

    # Generate data set.
    X, y_true = make_classification(random_state=0)
    y = np.full(shape=y_true.shape, fill_value=MISSING_LABEL)

    # LogisticRegression needs initial training data otherwise a warning will 
    # be raised by SklearnClassifier. Therfore, the first 10 instances are used as
    # training data.
    y[:10] = y_true[:10]

    # Create classifier and query strategy.
    clf = SklearnClassifier(LogisticRegression(), classes=np.unique(y_true))
    qs = UncertaintySampling(method='entropy')

    # Execute active learning cycle.
    n_cycles = 20
    for c in range(n_cycles):
        query_idx = qs.query(X=X, y=y, clf=clf)
        y[query_idx] = y_true[query_idx]
    print(f'Accuracy: {clf.fit(X, y).score(X, y_true)}')

.. examples_end

Stream-based Active Learning
############################

Citing
======
If you use ``scikit-activeml`` in one of your research projects and find it helpful,
please cite the following:

::

    @article{skactiveml2021,
        title={scikitactiveml: {A} {L}ibrary and {T}oolbox for {A}ctive {L}}earning {A}lgorithms},
        author={Daniel Kottke and Marek Herde and Tuan Pham Minh and Alexander Benz and Pascal Mergard and Atal Roghman and Christoph Sandrock and Bernhard Sick},
        journal={Preprints},
        doi={10.20944/preprints202103.0194.v1},
        year={2021},
        url={https://github.com/scikit-activeml/scikit-activeml}
    }