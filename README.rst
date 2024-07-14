.. intro_start

|Doc| |Codecov| |PythonVersion| |PyPi| |Paper| |Black|

.. |Doc| image:: https://img.shields.io/badge/docs-latest-green
   :target: https://scikit-activeml.github.io/scikit-activeml-docs/

.. |Codecov| image:: https://codecov.io/gh/scikit-activeml/scikit-activeml/branch/master/graph/badge.svg
   :target: https://app.codecov.io/gh/scikit-activeml/scikit-activeml

.. |PythonVersion| image:: https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-blue.svg
   :target: https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-blue

.. |PyPi| image:: https://badge.fury.io/py/scikit-activeml.svg
   :target: https://badge.fury.io/py/scikit-activeml

.. |Paper| image:: https://img.shields.io/badge/paper-10.20944/preprints202103.0194.v1-blue.svg
   :target: https://www.preprints.org/manuscript/202103.0194/v1

.. |Black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black

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

.. user_installation_start

User Installation
=================

The easiest way of installing scikit-activeml is using ``pip``:

::

    pip install -U scikit-activeml

.. note::
    pip installs based on the minimum requirements. If you encounter any incompatibility issues use:
    ::
        pip install -r requirements_max.txt
.. user_installation_end

.. examples_start

Examples
========
We provide a broad overview of different use-cases in our `tutorial section <https://scikit-activeml.github.io/scikit-activeml-docs/tutorials.html>`_ offering

- `Pool-based Active Learning - Getting Started <https://scikit-activeml.github.io/scikit-activeml-docs/generated/tutorials/00_pool_getting_started.html>`_,
- `Deep Pool-based Active Learning - scikit-activeml with Skorch <https://scikit-activeml.github.io/scikit-activeml-docs/generated/tutorials/01_deep_pool_al_with_skorch.html>`_,
- `Pool-based Active Learning for Regression - Getting Started <https://scikit-activeml.github.io/scikit-activeml-docs/generated/tutorials/02_pool_regression_getting_started.html>`_,
- `Pool-based Active Learning - Sample Annotating <https://scikit-activeml.github.io/scikit-activeml-docs/generated/tutorials/03_pool_oracle_annotations.html>`_,
- `Pool-based Active Learning - Simple Evaluation Study <https://scikit-activeml.github.io/scikit-activeml-docs/generated/tutorials/04_pool_simple_evaluation_study.html>`_,
- `Multi-annotator Pool-based Active Learning - Getting Started <https://scikit-activeml.github.io/scikit-activeml-docs/generated/tutorials/10_multiple_annotators_getting_started.html>`_,
- `Stream-based Active Learning - Getting Started <https://scikit-activeml.github.io/scikit-activeml-docs/generated/tutorials/20_stream_getting_started.html>`_,
- `Batch Stream-based Active Learning with Pool Query Strategies <https://scikit-activeml.github.io/scikit-activeml-docs/generated/tutorials/21_stream_batch_with_pool_al.html>`_,
- and `Stream-based Active Learning With River <https://scikit-activeml.github.io/scikit-activeml-docs/generated/tutorials/22_river_classifier.html>`_.

Two simple examples illustrating the straightforwardness of implementing active learning cycles with our Python package ``skactiveml`` are given in the following.

Pool-based Active Learning
##########################

The following code snippet implements an active learning cycle with 20 iterations using a Gaussian process
classifier and uncertainty sampling. To use other classifiers, you can simply wrap classifiers from
``sklearn`` or use classifiers provided by ``skactiveml``. Note that the main difficulty using
active learning with ``sklearn`` is the ability to handle unlabeled data, which we denote as a specific value
(``MISSING_LABEL``) in the label vector ``y``. More query strategies can be found in the documentation.

.. code-block:: python

    import numpy as np
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.datasets import make_blobs
    from skactiveml.pool import UncertaintySampling
    from skactiveml.utils import unlabeled_indices, MISSING_LABEL
    from skactiveml.classifier import SklearnClassifier

    # Generate data set.
    X, y_true = make_blobs(n_samples=200, centers=4, random_state=0)
    y = np.full(shape=y_true.shape, fill_value=MISSING_LABEL)

    # Use the first 10 instances as initial training data.
    y[:10] = y_true[:10]

    # Create classifier and query strategy.
    clf = SklearnClassifier(
        GaussianProcessClassifier(random_state=0),
        classes=np.unique(y_true),
        random_state=0
    )
    qs = UncertaintySampling(method='entropy')

    # Execute active learning cycle.
    n_cycles = 20
    for c in range(n_cycles):
        query_idx = qs.query(X=X, y=y, clf=clf)
        y[query_idx] = y_true[query_idx]

    # Fit final classifier.
    clf.fit(X, y)

As a result, we obtain an actively trained Gaussian process classifier.
A corresponding visualization of its decision boundary (black line) and the
sample utilities (greenish contours) is given below.

.. image:: https://raw.githubusercontent.com/scikit-activeml/scikit-activeml/master/docs/logos/pal-example-output.png
   :width: 400

Stream-based Active Learning
############################

The following code snippet implements an active learning cycle with 200 data points and
the default budget of 10% using a pwc classifier and split uncertainty sampling. 
Like in the pool-based example you can wrap other classifiers from ``sklearn``,
``sklearn`` compatible classifiers or like the example classifiers provided by ``skactiveml``.

.. code-block:: python

    import numpy as np
    from sklearn.datasets import make_blobs
    from skactiveml.classifier import ParzenWindowClassifier
    from skactiveml.stream import Split
    from skactiveml.utils import MISSING_LABEL

    # Generate data set.
    X, y_true = make_blobs(n_samples=200, centers=4, random_state=0)

    # Create classifier and query strategy.
    clf = ParzenWindowClassifier(random_state=0, classes=np.unique(y_true))
    qs = Split(random_state=0)

    # Initializing the training data as an empty array.
    X_train = []
    y_train = []

    # Initialize the list that stores the result of the classifier's prediction.
    correct_classifications = []

    # Execute active learning cycle.
    for x_t, y_t in zip(X, y_true):
        X_cand = x_t.reshape([1, -1])
        y_cand = y_t
        clf.fit(X_train, y_train)
        correct_classifications.append(clf.predict(X_cand)[0] == y_cand)
        sampled_indices = qs.query(candidates=X_cand, clf=clf)
        qs.update(candidates=X_cand, queried_indices=sampled_indices)
        X_train.append(x_t)
        y_train.append(y_cand if len(sampled_indices) > 0 else MISSING_LABEL)

As a result, we obtain an actively trained Parzen window classifier.
A corresponding visualization of its accuracy curve accross the active learning
cycle is given below.

.. image:: https://raw.githubusercontent.com/scikit-activeml/scikit-activeml/master/docs/logos/stream-example-output.png
   :width: 400

Query Strategy Overview
#######################

For better orientation, we provide an `overview <https://scikit-activeml.github.io/scikit-activeml-docs/generated/strategy_overview.html>`_
(incl. paper references and `visualizations <https://scikit-activeml.github.io/scikit-activeml-docs/generated/sphinx_gallery_examples/index.html>`_)
of the query strategies implemented by ``skactiveml``.

|Overview| |Visualization|

.. |Overview| image:: https://raw.githubusercontent.com/scikit-activeml/scikit-activeml/master/docs/logos/strategy-overview.gif
   :width: 400
   
.. |Visualization| image:: https://raw.githubusercontent.com/scikit-activeml/scikit-activeml/master/docs/logos/example-overview.gif
   :width: 400

.. examples_end

.. citing_start

Citing
======
If you use ``skactiveml`` in one of your research projects and find it helpful,
please cite the following:

::

    @article{skactiveml2021,
        title={scikit-activeml: {A} {L}ibrary and {T}oolbox for {A}ctive {L}earning {A}lgorithms},
        author={Daniel Kottke and Marek Herde and Tuan Pham Minh and Alexander Benz and Pascal Mergard and Atal Roghman and Christoph Sandrock and Bernhard Sick},
        journal={Preprints},
        doi={10.20944/preprints202103.0194.v1},
        year={2021},
        url={https://github.com/scikit-activeml/scikit-activeml}
    }

.. citing_end
