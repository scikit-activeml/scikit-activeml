Contributing Guide
==================

**scikit-activeml** is a library that implements the most important query
strategies for active learning. It is built upon the well-known machine
learning framework `scikit-learn <https://scikit-learn.org/stable/>`__.

Overview
--------

Our philosophy is to extend the ``sklearn`` ecosystem with the most relevant
query strategies for active learning and to implement tools for working with
partially unlabeled data. An overview of our repository’s structure is provided
in the image below. Each node represents a class or interface, and the arrows
illustrate the inheritance hierarchy among them. Dashed nodes indicate
functionality that is not yet available in our library.

.. image:: https://raw.githubusercontent.com/scikit-activeml/scikit-activeml/master/docs/logos/scikit-activeml-structure.png
   :width: 1000

In our package ``skactiveml``, there are three major components:
``SkactivemlClassifier``, ``SkactivemlRegressor``, and ``QueryStrategy``.
The classifier and regressor modules are necessary to handle partially unlabeled
data and to implement active-learning–specific estimators. This way, an active
learning cycle can be easily implemented starting with zero initial labels.
Regarding active learning query strategies, we currently differentiate between
the pool-based paradigm (a large pool of unlabeled samples is available) and the
stream-based paradigm (unlabeled samples arrive sequentially, i.e., as a stream).
Furthermore, we distinguish between the single-annotator and multi-annotator
settings. In the latter case, multiple error-prone annotators are queried to
provide labels. As a result, an active learning query strategy not only decides
which samples to query but also which annotators should be queried.

Thank You, Contributors!
~~~~~~~~~~~~~~~~~~~~~~~~

A big thank you to all contributors who provide the **scikit-activeml**
project with new enhancements and bug fixes.

Getting Help
~~~~~~~~~~~~

If you have any questions, please reach out to other developers via the
following channels:

-  `GitHub Issues <https://github.com/scikit-activeml/scikit-activeml/issues>`__

Roadmap
~~~~~~~

Our roadmap is summarized in the issue
`Upcoming Features <https://github.com/scikit-activeml/scikit-activeml/issues/145>`__.

Get Started
-----------

Before you contribute to this project, please follow the steps below.

Setup Development Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are several ways to create a local Python environment, such as
`virtualenv <https://www.google.com/search?client=safari&rls=en&q=virtualenv&ie=UTF-8&oe=UTF-8>`__,
`pipenv <https://pipenv.pypa.io/en/latest/>`__, or
`miniconda <https://docs.conda.io/en/latest/miniconda.html>`__. One possible
workflow is to install ``miniconda`` and use it to create a Python environment.

Example with miniconda
^^^^^^^^^^^^^^^^^^^^^^

Create a new Python environment named **scikit-activeml**:

.. code:: bash

   conda create -n scikit-activeml

To ensure that the correct environment is active:

.. code:: bash

   conda activate scikit-activeml

Then install ``pip``:

.. code:: bash

   conda install pip

Install Dependencies
~~~~~~~~~~~~~~~~~~~~

Now, install the required project dependencies, which are defined in the
``requirements.txt`` and ``requirements_extra.txt`` (for development) files.

.. code:: bash

   # Make sure your scikit-activeml Python environment is active!
   cd <project-root>
   pip install -r requirements.txt
   pip install -r requirements_extra.txt

After the pip installation is successful, you must install ``pandoc`` and
``ghostscript`` if they are not already installed.

Example with macOS (Homebrew)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: bash

   brew install pandoc ghostscript

Contributing Code
-----------------

General Coding Conventions
~~~~~~~~~~~~~~~~~~~~~~~~~~

This library follows the conventions of
`scikit-learn <https://scikit-learn.org/stable/developers/develop.html#coding-guidelines>`__
and should conform to the `PEP 8 <https://www.python.org/dev/peps/pep-0008/>`__
Style Guide for Python code. For linting, the use of
`flake8 <https://flake8.pycqa.org/en/latest/>`__ is recommended. The Python
package `black <https://black.readthedocs.io/en/stable/>`__ provides a simple
solution for code formatting. For example, you can install it and format your
code using the following commands:

.. code:: bash

   pip install black
   black --line-length 79 example_file.py

Example for Code Contribution Cycle (C3) and Pull Requests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Fork the repository using the GitHub
   `Fork <https://github.com/scikit-activeml/scikit-activeml/fork>`__ button.

2. Clone your fork to your local machine:

.. code:: bash

   git clone https://github.com/<your-username>/scikit-activeml.git

3. Create a new branch for your changes from the ``development`` branch:

.. code:: bash

   git checkout -b <branch-name>

4. After you have finished implementing the feature, ensure that all tests pass.
   You can run the tests using:

.. code:: bash

   $ pytest

Make sure you have covered all lines with tests.

.. code:: bash

   $ pytest --cov=./skactiveml

5. Commit and push your changes.

.. code:: bash

   git add <modified-files>
   git commit -m "<commit-message>"
   git push

6. Create a pull request.

Query Strategies
----------------

All query strategies inherit from the abstract superclass
``skactiveml.base.QueryStrategy``, which is implemented in ``skactiveml/base.py``.
This superclass inherits from ``sklearn.base.Estimator``. By default, its
``__init__`` method requires a ``random_state`` parameter, and the abstract
``query`` method enforces the implementation of the sample selection logic.

Single-annotator Pool-based Query Strategies
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _general-1:

General
^^^^^^^

Single-annotator pool-based query strategies are stored in the file
``skactiveml/pool/*.py`` and inherit from
``skactiveml.base.SingleAnnotatorPoolQueryStrategy``.

The class must implement the following methods:

+-------------------+---------------------------------------------------------+
| Method            | Description                                             |
+===================+=========================================================+
| ``__init__``      | Method for initialization.                              |
+-------------------+---------------------------------------------------------+
| ``query``         | Select the samples whose labels are to be queried.      |
+-------------------+---------------------------------------------------------+

.. _init-1:

``__init__`` Method
^^^^^^^^^^^^^^^^^^^^

For typical class parameters, we use standard names:

+------------------------------+----------------------------------------------+
| Parameter                    | Description                                  |
+==============================+==============================================+
| ``random_state``             | An integer or a np.random.RandomState,       |
|                              | similar to scikit-learn.                      |
+------------------------------+----------------------------------------------+
| ``prior``, optional          | Prior probabilities for the distribution     |
|                              | in probabilistic strategies.                 |
+------------------------------+----------------------------------------------+
| ``method``, optional         | A string for classes that implement multiple |
|                              | methods.                                     |
+------------------------------+----------------------------------------------+
| ``cost_matrix``, optional    | A cost matrix defining the cost of           |
|                              | misclassifying samples.                      |
+------------------------------+----------------------------------------------+

.. _query-1:

``query`` Method
^^^^^^^^^^^^^^^^^^

Required Parameters:

+-----------------------------------+----------------------------------------------+
| Parameter                         | Description                                  |
+===================================+==============================================+
| ``X``                             | Training dataset, usually complete (i.e.,    |
|                                   | including both labeled and unlabeled samples).|
+-----------------------------------+----------------------------------------------+
| ``y``                             | Labels of the training dataset. (May include   |
|                                   | unlabeled samples, indicated by a            |
|                                   | MISSING_LABEL.)                              |
+-----------------------------------+----------------------------------------------+
| ``candidates``, optional          | If ``candidates`` is None, the unlabeled         |
|                                   | samples from (X, y) are considered as          |
|                                   | candidates. If ``candidates`` is an array of     |
|                                   | integers with shape (n_candidates,), it is     |
|                                   | considered as indices of the samples in        |
|                                   | (X, y). If it is an array with shape           |
|                                   | (n_candidates, n_features), the candidates     |
|                                   | are directly provided (and may not be contained  |
|                                   | in X). This is not supported by all query        |
|                                   | strategies.                                  |
+-----------------------------------+----------------------------------------------+
| ``batch_size``, optional          | Number of samples to be selected in one AL     |
|                                   | cycle.                                       |
+-----------------------------------+----------------------------------------------+
| ``return_utilities``, optional    | If True, additionally return the utilities     |
|                                   | computed by the query strategy.              |
+-----------------------------------+----------------------------------------------+

Returns:

+-----------------------------------+----------------------------------------------+
| Parameter                         | Description                                  |
+===================================+==============================================+
| ``query_indices``                 | Indices indicating which candidate sample’s   |
|                                   | label is to be queried. For example,         |
|                                   | ``query_indices[0]`` indicates the first      |
|                                   | selected sample. Depending on the shape of     |
|                                   | ``candidates``, the indexing refers either to  |
|                                   | samples in X or directly to candidates.      |
+-----------------------------------+----------------------------------------------+
| ``utilities``, optional           | Utilities of the samples after selection.    |
|                                   | For example, ``utilities[0]`` indicates the    |
|                                   | utility for selecting the first sample. For    |
|                                   | labeled samples, the utility will be set to    |
|                                   | np.nan.                                      |
+-----------------------------------+----------------------------------------------+

.. _general-advice-1:

General Advice
''''''''''''''

Use the ``self._validate_data`` method (implemented in the superclass)
to check the inputs ``X`` and ``y`` only once. Fit the classifier or
regressor if it is not yet fitted (using ``fit_if_not_fitted`` from ``utils``).
Calculate utilities via an extra public function. Use the
``simple_batch`` function from ``utils`` to determine the query indices and set
the utilities in naive batch query strategies.

.. _testing-1:

Testing
^^^^^^^

The test classes in ``skactiveml.pool.test.TestQueryStrategy`` for
single-annotator pool-based query strategies must inherit from the test
template ``skactiveml.tests.template_query_strategy.TemplateSingleAnnotatorPoolQueryStrategy``.
As a result, many required functionalities will be automatically tested.
You must specify the parameters of ``qs_class`` and ``init_default_params`` in
the ``__init__`` accordingly. Depending on whether the query strategy can handle
regression, classification, or both, you also need to define the parameters
``query_default_params_reg`` or ``query_default_params_clf``. Once the parameters
are set, adjust the tests until all errors are resolved. Please refer to the test
template for more detailed information.

Classifiers
-----------

Standard classifier implementations are part of the subpackage
``skactiveml.classifier``, and classifiers learning from multiple
annotators are implemented in the subpackage
``skactiveml.classifier.multiannotator``. Every classifier inherits from
``skactiveml.base.SkactivemlClassifier`` and must implement the following methods:

+-------------------+---------------------------------------------------------+
| Method            | Description                                             |
+===================+=========================================================+
| ``__init__``      | Method for initialization.                              |
+-------------------+---------------------------------------------------------+
| ``fit``           | Method to fit the classifier for given training data.   |
+-------------------+---------------------------------------------------------+
| ``predict_proba`` | Method predicting class-membership probabilities for    |
|                   | samples.                                                |
+-------------------+---------------------------------------------------------+
| ``predict``       | Method predicting class labels for samples. The super   |
|                   | implementation uses ``predict_proba``.                  |
+-------------------+---------------------------------------------------------+

.. _init-2:

``__init__`` Method (Classifiers)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

+-----------------------------------+----------------------------------------------+
| Parameter                         | Description                                  |
+===================================+==============================================+
| ``classes``, optional             | Holds the label for each class. If None, the   |
|                                   | classes are determined during fitting.       |
+-----------------------------------+----------------------------------------------+
| ``missing_label``, optional       | Value representing a missing label.          |
+-----------------------------------+----------------------------------------------+
| ``cost_matrix``, optional         | A cost matrix where ``cost_matrix[i,j]``       |
|                                   | indicates the cost of predicting class         |
|                                   | ``classes[j]`` for a sample of class           |
|                                   | ``classes[i]``. Only set if ``classes`` is not   |
|                                   | None.                                        |
+-----------------------------------+----------------------------------------------+
| ``random_state``, optional        | Ensures reproducibility (cf. scikit-learn).    |
+-----------------------------------+----------------------------------------------+

.. _fit-1:

``fit`` Method (Classifiers)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

+-----------------------------------+----------------------------------------------+
| Parameter                         | Description                                  |
+===================================+==============================================+
| ``X``                             | Matrix of feature values representing the    |
|                                   | samples.                                     |
+-----------------------------------+----------------------------------------------+
| ``y``                             | Contains the class labels of the training      |
|                                   | samples. Missing labels are represented by     |
|                                   | the attribute ``missing_label``. Usually,      |
|                                   | ``y`` is a column array except for multi-      |
|                                   | annotator classifiers, which expect a matrix   |
|                                   | with columns for each annotator.             |
+-----------------------------------+----------------------------------------------+
| ``sample_weight``, optional       | Contains weights for the training samples’     |
|                                   | class labels. Must have the same shape as ``y``. |
+-----------------------------------+----------------------------------------------+

.. _predict-proba-1:

``predict_proba`` Method
^^^^^^^^^^^^^^^^^^^^^^^^^

+-----------------------------------+----------------------------------------------+
| Parameter                         | Description                                  |
+===================================+==============================================+
| ``X``                             | Matrix of feature values representing the    |
|                                   | samples for which predictions are made.      |
+-----------------------------------+----------------------------------------------+

.. _predict-1:

``predict`` Method
^^^^^^^^^^^^^^^^^^^^^

+-----------------------------------+----------------------------------------------+
| Parameter                         | Description                                  |
+===================================+==============================================+
| ``X``                             | Matrix of feature values representing the    |
|                                   | samples for which predictions are made.      |
+-----------------------------------+----------------------------------------------+

.. _score-1:

``score`` Method
^^^^^^^^^^^^^^^^^^^^^

+-----------------------------------+----------------------------------------------+
| Parameter                         | Description                                  |
+===================================+==============================================+
| ``X``                             | Matrix of feature values representing the    |
|                                   | samples for which predictions are made.      |
+-----------------------------------+----------------------------------------------+
| ``y``                             | Contains the true labels for each sample.      |
+-----------------------------------+----------------------------------------------+
| ``sample_weight``, optional       | Defines the importance of each sample when     |
|                                   | computing accuracy.                          |
+-----------------------------------+----------------------------------------------+

Regressors
----------

Standard regressor implementations are part of the subpackage
``skactiveml.regressor``. Every regressor inherits from
``skactiveml.base.SkactivemlRegressor`` and must implement the following methods:

+-------------------+---------------------------------------------------------+
| Method            | Description                                             |
+===================+=========================================================+
| ``__init__``      | Method for initialization.                              |
+-------------------+---------------------------------------------------------+
| ``fit``           | Method to fit the regressor for given training data.    |
+-------------------+---------------------------------------------------------+
| ``predict``       | Method predicting the target values for samples.        |
+-------------------+---------------------------------------------------------+

.. _init-3:

``__init__`` Method (Regressors)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

+-----------------------------------+----------------------------------------------+
| Parameter                         | Description                                  |
+===================================+==============================================+
| ``random_state``, optional        | Ensures reproducibility (cf. scikit-learn).    |
+-----------------------------------+----------------------------------------------+
| ``missing_label``, optional       | Value representing a missing label.          |
+-----------------------------------+----------------------------------------------+

.. _fit-2:

``fit`` Method (Regressors)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

+-----------------------------------+----------------------------------------------+
| Parameter                         | Description                                  |
+===================================+==============================================+
| ``X``                             | Matrix of feature values representing the    |
|                                   | samples.                                     |
+-----------------------------------+----------------------------------------------+
| ``y``                             | Contains the target values of the training      |
|                                   | samples. Missing labels are represented by     |
|                                   | the attribute ``missing_label``. Usually,      |
|                                   | ``y`` is a column array except for multi-target  |
|                                   | regressors, which expect a matrix with columns |
|                                   | for each target type.                          |
+-----------------------------------+----------------------------------------------+
| ``sample_weight``, optional       | Contains weights for the training samples’     |
|                                   | targets. Must have the same shape as ``y``.     |
+-----------------------------------+----------------------------------------------+

.. _predict-2:

``predict`` Method (Regressors)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

+-----------------------------------+----------------------------------------------+
| Parameter                         | Description                                  |
+===================================+==============================================+
| ``X``                             | Matrix of feature values representing the    |
|                                   | samples for which predictions are made.      |
+-----------------------------------+----------------------------------------------+

.. _score-2:

``score`` Method (Regressors)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

+-----------------------------------+----------------------------------------------+
| Parameter                         | Description                                  |
+===================================+==============================================+
| ``X``                             | Matrix of feature values representing the    |
|                                   | samples for which predictions are made.      |
+-----------------------------------+----------------------------------------------+
| ``y``                             | Contains the true target values for each sample.|
+-----------------------------------+----------------------------------------------+
| ``sample_weight``, optional       | Defines the importance of each sample when     |
|                                   | computing the R2 score.                      |
+-----------------------------------+----------------------------------------------+

Annotator Models
----------------

Annotator models implement the interface
``skactiveml.base.AnnotatorModelMixin``. These models can estimate the
performance of annotators for given samples. Each annotator model must implement
the ``predict_annotator_perf`` method, which estimates the performance per
sample for each annotator as a proxy for the quality of the provided annotations.

.. _predict-annotator-perf-1:

``predict_annotator_perf`` Method
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Required Parameters:

+-------------+-----------------------------------------------------------+
| Parameter   | Description                                               |
+=============+===========================================================+
| ``X``       | Matrix of feature values representing the samples.      |
+-------------+-----------------------------------------------------------+

Returns:

+-------------+-----------------------------------------------------------+
| Parameter   | Description                                               |
+=============+===========================================================+
| ``P_annot`` | The estimated performance per sample-annotator pair.      |
+-------------+-----------------------------------------------------------+

Examples
--------

Two of our main goals are to make active learning more understandable and
improve our framework’s usability. Therefore, we require an example for each
query strategy. To do so, create a file named
``scikit-activeml/docs/examples/query_strategy.json``. Currently, we support
examples for single-annotator pool-based and stream-based query strategies.

The JSON file supports the following entries:

+------------------+----------------------------------------------------------+
| Entry            | Description                                              |
+==================+==========================================================+
| ``class``        | Query strategy’s class name.                             |
+------------------+----------------------------------------------------------+
| ``package``      | Name of the sub-package (e.g., pool).                    |
+------------------+----------------------------------------------------------+
| ``method``       | Query strategy’s official name.                          |
+------------------+----------------------------------------------------------+
| ``category``     | The methodological category of this query strategy, e.g.,  |
|                  | Expected Error Reduction, Model Change, Query-by-Committee,|
|                  | Random Sampling, Uncertainty Sampling, or Others.        |
+------------------+----------------------------------------------------------+
| ``template``     | Defines the general setup/setting of the example.        |
|                  | Supported templates include:                             |
|                  | ``examples/template_pool.py``,                           |
|                  | ``examples/template_pool_regression.py``,                |
|                  | ``examples/template_stream.py``, and                      |
|                  | ``examples/template_pool_batch.py``                      |
+------------------+----------------------------------------------------------+
| ``tags``         | Search categories. Supported tags include ``pool``,      |
|                  | ``stream``, ``single-annotator``, ``multi-annotator``,     |
|                  | ``classification``, and ``regression``.                 |
+------------------+----------------------------------------------------------+
| ``title``        | Title of the example, usually named after the query      |
|                  | strategy.                                                |
+------------------+----------------------------------------------------------+
| ``text_0``       | Placeholder for additional explanations.               |
+------------------+----------------------------------------------------------+
| ``refs``         | References (BibTeX keys) to the paper(s) describing the    |
|                  | query strategy.                                          |
+------------------+----------------------------------------------------------+
| ``sequence``     | Order in which content is displayed, usually           |
|                  | ["title", "text_0", "plot", "refs"].                      |
+------------------+----------------------------------------------------------+
| ``import_misc``  | Python code for imports (e.g.,                        |
|                  | "from skactiveml.pool import RandomSampling").            |
+------------------+----------------------------------------------------------+
| ``n_samples``    | Number of samples in the example dataset.              |
+------------------+----------------------------------------------------------+
| ``init_qs``      | Python code to initialize the query strategy object, e.g.,|
|                  | "RandomSampling()".                                      |
+------------------+----------------------------------------------------------+
| ``query_params`` | Python code for parameters passed to the query method,   |
|                  | e.g., "X=X, y=y".                                        |
+------------------+----------------------------------------------------------+
| ``preproc``      | Python code for preprocessing before executing the AL    |
|                  | cycle, e.g., "X = (X-X.min())/(X.max()-X.min())".         |
+------------------+----------------------------------------------------------+
| ``n_cycles``     | Number of active learning cycles.                      |
+------------------+----------------------------------------------------------+
| ``init_clf``     | Python code to initialize the classifier object, e.g.,   |
|                  | "ParzenWindowClassifier(classes=[0, 1])". (Only supported  |
|                  | for certain templates.)                                  |
+------------------+----------------------------------------------------------+
| ``init_reg``     | Python code to initialize the regressor object, e.g.,    |
|                  | "NICKernelRegressor()". (Only supported for the          |
|                  | regression template.)                                    |
+------------------+----------------------------------------------------------+

Testing and Code Coverage
-------------------------

Please ensure test coverage is close to 100%. The current code coverage can be
viewed `here <https://app.codecov.io/gh/scikit-activeml/scikit-activeml>`__.

Documentation
-------------

Guidelines for writing documentation in ``scikit-activeml`` adopt the
`scikit-learn guidelines <https://scikit-learn.org/stable/developers/contributing.html#guidelines-for-writing-documentation>`__
used by scikit-learn.

Building the Documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To ensure your documentation is well formatted, build it using Sphinx:

.. code:: bash

   sphinx-build -b html docs docs/_build

Issue Tracking
--------------

We use `GitHub Issues <https://github.com/scikit-activeml/scikit-activeml/issues>`__
as our issue tracker. If you believe you have found a bug in
``scikit-activeml``, please report it there. Documentation bugs can also be reported.

Checking If a Bug Already Exists
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before filing an issue, please check whether the problem has already been reported.
This will help determine if the problem is resolved or fixed in an upcoming release, save
time, and provide guidance on how to fix it. Search the issue database using the search box
at the top of the issue tracker page (filter by the ``bug`` label).

Reporting an Issue
~~~~~~~~~~~~~~~~~~

Use the following labels when reporting an issue:

+------------------+-------------------------------------------+
| Label            | Use Case                                  |
+==================+===========================================+
| ``bug``          | Something isn’t working                   |
+------------------+-------------------------------------------+
| ``enhancement``  | Request for a new feature                 |
+------------------+-------------------------------------------+
| ``documentation``| Improvement or additions to documentation |
+------------------+-------------------------------------------+
| ``question``     | General questions                         |
+------------------+-------------------------------------------+
