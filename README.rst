.. intro_start

|

.. image:: https://raw.githubusercontent.com/scikit-activeml/scikit-activeml/master/docs/logos/scikit-activeml-logo.png
   :class: dark-light
   :align: center
   :width: 40%

|

=====================================================================
scikit-activeml: A Library and Toolbox for Active Learning Algorithms
=====================================================================
|Doc| |Codecov| |PythonVersion| |PyPi| |Black| |Downloads| |Paper|

.. |Doc| image:: https://img.shields.io/badge/docs-latest-green
   :target: https://scikit-activeml.github.io/latest/

.. |Codecov| image:: https://codecov.io/gh/scikit-activeml/scikit-activeml/branch/master/graph/badge.svg
   :target: https://app.codecov.io/gh/scikit-activeml/scikit-activeml

.. |PythonVersion| image:: https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue.svg
   :target: https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue

.. |PyPi| image:: https://badge.fury.io/py/scikit-activeml.svg
   :target: https://badge.fury.io/py/scikit-activeml

.. |Paper| image:: https://img.shields.io/badge/paper-10.20944/preprints202103.0194.v1-blue.svg
   :target: https://www.preprints.org/manuscript/202103.0194/v1

.. |Black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black

.. |Downloads| image:: https://static.pepy.tech/badge/scikit-activeml
   :target: https://www.pepy.tech/projects/scikit-activeml

Machine learning models often need large amounts of training data to perform well.
While unlabeled data can be gathered with relative ease, labeling is typically difficult,
time-consuming, or expensive. Active learning addresses this challenge by querying labels
for the most informative samples, enabling high performance with fewer labeled examples.
With this goal in mind, **scikit-activeml** has been developed as a Python library for active
learning on top of `scikit-learn <https://scikit-learn.org/stable/>`_.

.. intro_end

.. user_installation_start

User Installation
=================

The easiest way to install scikit-activeml is using ``pip``:

::

    pip install -U scikit-activeml

This installation via `pip` includes only the minimum requirements to avoid
potential package downgrades within your installation. If you encounter any incompatibility issues,
you can install the `maximum requirements <https://github.com/scikit-activeml/scikit-activeml/blob/master/requirements_max.txt>`_,
which have been tested for the current package release:

::

    pip install -r requirements_max.txt

.. user_installation_end

.. examples_start

Examples
========

We provide a broad overview of different use cases in our `tutorial section <https://scikit-activeml.github.io/latest/tutorials.html>`_,
including:

- `Pool-based Active Learning - Getting Started <https://scikit-activeml.github.io/latest/generated/tutorials/00_pool_getting_started.html>`_
- `Deep Pool-based Active Learning - scikit-activeml with Skorch <https://scikit-activeml.github.io/latest/generated/tutorials/01_deep_pool_al_with_skorch.html>`_
- `Pool-based Active Learning for Regression - Getting Started <https://scikit-activeml.github.io/latest/generated/tutorials/02_pool_regression_getting_started.html>`_
- `Pool-based Active Learning - Sample Annotating <https://scikit-activeml.github.io/latest/generated/tutorials/03_pool_oracle_annotations.html>`_
- `Pool-based Active Learning - Simple Evaluation Study <https://scikit-activeml.github.io/latest/generated/tutorials/04_pool_simple_evaluation_study.html>`_
- `Active Image Classification via Self-supervised Learning <https://scikit-activeml.github.io/latest/generated/tutorials/05_pool_al_with_self_supervised_learning.html>`_
- `Multi-annotator Pool-based Active Learning - Getting Started <https://scikit-activeml.github.io/latest/generated/tutorials/10_multiple_annotators_getting_started.html>`_
- `Stream-based Active Learning - Getting Started <https://scikit-activeml.github.io/latest/generated/tutorials/20_stream_getting_started.html>`_
- `Batch Stream-based Active Learning with Pool Query Strategies <https://scikit-activeml.github.io/latest/generated/tutorials/21_stream_batch_with_pool_al.html>`_
- `Stream-based Active Learning With River <https://scikit-activeml.github.io/latest/generated/tutorials/22_river_classifier.html>`_

Below are two code snippets illustrating how straightforward it is to implement active learning cycles using our Python package ``skactiveml``.

Pool-based Active Learning
--------------------------

The following snippet implements an active learning cycle with 20 iterations using a Gaussian process
classifier and uncertainty sampling. You can substitute other classifiers from ``sklearn`` or those
provided by ``skactiveml``. Note that when using active learning with ``sklearn``, unlabeled data
is represented by the value ``MISSING_LABEL`` in the label vector ``y``. Additional query strategies
are available in our documentation.

.. code-block:: python

    import numpy as np
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.datasets import make_blobs
    from skactiveml.pool import UncertaintySampling
    from skactiveml.utils import MISSING_LABEL
    from skactiveml.classifier import SklearnClassifier

    # Generate data set.
    X, y_true = make_blobs(n_samples=200, centers=4, random_state=0)
    y = np.full(shape=y_true.shape, fill_value=MISSING_LABEL)

    # Use the first 10 samples as initial training data.
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

As a result, an actively trained Gaussian process classifier is obtained.
A visualization of its decision boundary (black line) along with sample utilities (greenish contours) is shown below.

.. image:: https://raw.githubusercontent.com/scikit-activeml/scikit-activeml/master/docs/logos/pal-example-output.png
   :width: 400

Stream-based Active Learning
----------------------------

The following snippet implements an active learning cycle with 200 data points and a default budget of 10%
using a Parzen window classifier and split uncertainty sampling.
Similar to the pool-based example, you can wrap classifiers from ``sklearn``, use sklearn-compatible classifiers,
or choose from the example classifiers provided by ``skactiveml``.

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

    # Initialize training data as empty lists.
    X_train = []
    y_train = []

    # Initialize a list to store prediction results.
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

As a result, an actively trained Parzen window classifier is obtained.
A visualization of its accuracy curve across the active learning cycle is shown below.

.. image:: https://raw.githubusercontent.com/scikit-activeml/scikit-activeml/master/docs/logos/stream-example-output.png
   :width: 400

Query Strategy Overview
=======================
For better orientation, we provide an `overview <https://scikit-activeml.github.io/latest/generated/strategy_overview.html>`_
(including paper references and `visual examples <https://scikit-activeml.github.io/latest/generated/sphinx_gallery_examples/index.html>`_)
of the query strategies implemented by ``skactiveml``.

|Overview| |Visualization|

.. |Overview| image:: https://raw.githubusercontent.com/scikit-activeml/scikit-activeml/master/docs/logos/strategy-overview.gif
   :width: 365

.. |Visualization| image:: https://raw.githubusercontent.com/scikit-activeml/scikit-activeml/master/docs/logos/example-overview.gif
   :width: 365

.. examples_end

.. citing_start

Citing
======
If you use ``skactiveml`` in your research projects and find it helpful, please cite the following:

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
