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

   pytest

Make sure you have covered all lines with tests.

.. code:: bash

   pytest --cov=./skactiveml

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

``__init__``
^^^^^^^^^^^^

For typical class parameters, we use standard names:

+------------------------------+----------------------------------------------+
| Parameter                    | Description                                  |
+==============================+==============================================+
| ``random_state``             | An integer or a np.random.RandomState,       |
|                              | similar to scikit-learn.                     |
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

``query``
^^^^^^^^^

Required Parameters:

+-----------------------------------+----------------------------------------------+
| Parameter                         | Description                                  |
+===================================+==============================================+
| ``X``                             | Training dataset, usually complete (i.e.,    |
|                                   | including both labeled and unlabeled         |
|                                   | samples).                                    |
+-----------------------------------+----------------------------------------------+
| ``y``                             | Labels of the training dataset. (May include |
|                                   | unlabeled samples, indicated by a            |
|                                   | MISSING_LABEL.)                              |
+-----------------------------------+----------------------------------------------+
| ``candidates``, optional          | If ``candidates`` is None, the unlabeled     |
|                                   | samples from (X, y) are considered as        |
|                                   | candidates. If ``candidates`` is an array    |
|                                   | of integers with shape (n_candidates,),      |
|                                   | it is considered as indices of the samples   |
|                                   | in (X, y). If it is an array with shape      |
|                                   | (n_candidates, n_features), the candidates   |
|                                   | are directly provided (and may not be        |
|                                   | contained in X). This is not supported by    |
|                                   | all query strategies.                        |
+-----------------------------------+----------------------------------------------+
| ``batch_size``, optional          | Number of samples to be selected in one AL   |
|                                   | cycle.                                       |
+-----------------------------------+----------------------------------------------+
| ``return_utilities``, optional    | If True, additionally return the utilities   |
|                                   | computed by the query strategy.              |
+-----------------------------------+----------------------------------------------+

Returns:

+-----------------------------------+----------------------------------------------+
| Parameter                         | Description                                  |
+===================================+==============================================+
| ``query_indices``                 | Indices indicating which candidate sample’s  |
|                                   | label is to be queried. For example,         |
|                                   | ``query_indices[0]`` indicates the first     |
|                                   | selected sample. Depending on the shape of   |
|                                   | ``candidates``, the indexing refers either   |
|                                   | to samples in X or directly to candidates.   |
+-----------------------------------+----------------------------------------------+
| ``utilities``, optional           | Utilities of the samples after selection.    |
|                                   | For example, ``utilities[0]`` indicates the  |
|                                   | utility for selecting the first sample. For  |
|                                   | labeled samples, the utility will be set to  |
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

Single-annotator Stream-based Query Strategies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _general-2:

General
^^^^^^^

All query strategies are stored in a file ``skactivml/stream/*.py``.
Every query strategy inherits from
``skactiveml.base.SingleAnnotatorStreamQueryStrategy``. Every query strategy has
either an internal budget handling or an outsourced ``budget_manager``.

For typical class parameters we use standard names:

+------------------------------+------------------------------------------+
| Parameter                    | Description                              |
+==============================+==========================================+
| ``random_state``             | Integer that acts as random seed         |
|                              | or ``np.random.RandomState`` like        |
|                              | sklearn.                                 |
+------------------------------+------------------------------------------+
| ``budget``                   | The share of labels that the strategy is |
|                              | allowed to query.                        |
+------------------------------+------------------------------------------+
| ``budget_manager``, optional | Enforces the budget constraint.          |
+------------------------------+------------------------------------------+

The class must implement the following methods:

+------------+-----------------------------------------------------------------+
| Function   | Description                                                     |
+============+=================================================================+
| ``init``   | Function for initialization.                                    |
+------------+-----------------------------------------------------------------+
| ``query``  | Identify the instances whose labels to select without adapting  |
|            | the internal state.                                             |
+------------+-----------------------------------------------------------------+
| ``update`` | Adapting the budget monitoring according to the queried labels. |
+------------+-----------------------------------------------------------------+

.. _query-method-2:

``query``
^^^^^^^^^

Required Parameters:

+------------------------------+-------------------------------------------------------------+
| Parameter                    | Description                                                 |
+==============================+=============================================================+
| ``candidates``               | Set of candidate instances,                                 |
|                              | inherited from                                              |
|                              | ``SingleAnnotatorStreamBasedQueryStrategy``.                |
+------------------------------+-------------------------------------------------------------+
| ``clf``, optional            | The classifier used by the                                  |
|                              | strategy.                                                   |
+------------------------------+-------------------------------------------------------------+
| ``X``, optional              | Set of labeled and unlabeled                                |
|                              | instances.                                                  |
+------------------------------+-------------------------------------------------------------+
| ``y``, optional              | Labels of ``X`` (it may be set to                           |
|                              | ``MISSING_LABEL`` if ``y`` is                               |
|                              | unknown).                                                   |
+------------------------------+-------------------------------------------------------------+
| ``sample_weight``, optional  | Weights for each instance in                                |
|                              | ``X`` or ``None`` if all are                                |
|                              | equally weighted.                                           |
+------------------------------+-------------------------------------------------------------+
| ``fit_clf``, optional        | Uses ``X`` and ``y`` to fit the classifier.                 |
+------------------------------+-------------------------------------------------------------+
| ``return_utilities``         | Whether to return the candidates' utilities,                |
|                              | inherited from ``SingleAnnotatorStreamBasedQueryStrategy``. |
+------------------------------+-------------------------------------------------------------+

Returns:

+-----------------------------------+-----------------------------------+
| Parameter                         | Description                       |
+===================================+===================================+
| ``queried_indices``               | Indices of the best instances     |
|                                   | from ``X_Cand``.                  |
+-----------------------------------+-----------------------------------+
| ``utilities``                     | Utilities of all candidate        |
|                                   | instances, only if                |
|                                   | ``return_utilities`` is ``True``. |
+-----------------------------------+-----------------------------------+

.. _general-advice-2:

General advice
''''''''''''''

The ``query`` method must not change the internal state of the ``query``
strategy (``budget``, ``budget_manager`` and ``random_state`` included) to allow
for assessing multiple instances with the same state. Update the internal state
in the ``update()`` method. If the class implements a classifier (``clf``) the
optional attributes need to be implement. Use ``self._validate_data`` method
(is implemented in superclass). Check the input ``X`` and ``y`` only once. Fit
classifier if ``fit_clf`` is set to ``True``.

.. _update-1:

``update``
^^^^^^^^^^

Required Parameters:

+-------------------------------+----------------------------------------------+
| Parameter                     | Description                                  |
+===============================+==============================================+
| ``candidates``                | Set of candidate instances,                  |
|                               | inherited from                               |
|                               | ``SingleAnnotatorStreamBasedQueryStrategy``. |
+-------------------------------+----------------------------------------------+
| ``queried_indices``           | Typically the return value of                |
|                               | ``query``.                                   |
+-------------------------------+----------------------------------------------+
| ``budget_manager_param_dict`` | Provides additional parameters to            |
|                               | the ``update`` method of the                 |
|                               | ``budget_manager`` (only include             |
|                               | if a ``budget_manager`` is used).            |
+-------------------------------+----------------------------------------------+

.. _general-advice-3:

General advice
''''''''''''''

Use ``self._validate_data`` in the case the strategy is used without using
the ``query`` method (if parameters need to be initialized before the
update). If a ``budget_manager`` is used forward the update call to the
``budget_manager.update`` method.

.. _testing-2:

Testing
^^^^^^^
The test classes in ``skactiveml.stream.test.TestQueryStrategy`` for
single-annotator stream-based query strategies must inherit from the test
template ``skactiveml.tests.template_query_strategy.TemplateSingleAnnotatorStreamQueryStrategy``.
As a result, many required functionalities will be automatically tested.
You must specify the parameters of ``qs_class`` and ``init_default_params`` in
the ``__init__`` accordingly. Depending on whether the query strategy can handle
regression, classification, or both, you also need to define the parameters
``query_default_params_reg`` or ``query_default_params_clf``. Once the parameters
are set, adjust the tests until all errors are resolved. Please refer to the test
template for more detailed information.

.. _general-advice-4:

``budget_manager``
^^^^^^^^^^^^^^^^^^

All budget managers are stored in
``skactivml/stream/budget_manager/*.py``. The class must implement the
following methods:

+-----------------------------------+-----------------------------------+
| Parameter                         | Description                       |
+===================================+===================================+
| ``__init__``                      | Function for initialization       |
+-----------------------------------+-----------------------------------+
| ``query_by_utilities``            | Identify which instances to query |
|                                   | based on the assessed utility     |
+-----------------------------------+-----------------------------------+
| ``update``                        | Adapting the budget monitoring    |
|                                   | according to the queried labels   |
+-----------------------------------+-----------------------------------+

.. _update-2:

``update``
^^^^^^^^^^

The update method of the budget manager has the same functionality as
the query strategy update.

Required Parameters:

+-----------------------------------+-----------------------------------+
| Parameter                         | Description                       |
+===================================+===================================+
| ``budget``                        | % of labels that the strategy is  |
|                                   | allowed to query                  |
+-----------------------------------+-----------------------------------+
| ``random_state``                  | Integer that acts as random seed  |
|                                   | or ``np.random.RandomState`` like |
|                                   | sklearn                           |
+-----------------------------------+-----------------------------------+

.. _query-by-utilities-1:

``query_by_utilities``
^^^^^^^^^^^^^^^^^^^^^^

Required Parameters:

+-----------------------------------+------------------------------------+
| Parameter                         | Description                        |
+===================================+====================================+
| ``utilities``                     | The ``utilities`` of ``candidates``|
|                                   | calculated by the query strategy,  |
|                                   | inherited from ``BudgetManager``   |
+-----------------------------------+------------------------------------+

Returns:

+-----------------------------------+------------------------------------+
| Parameter                         | Description                        |
+===================================+====================================+
| ``queried_indices``               | The indices of samples in          |
|                                   | candidates whose labels are        |
|                                   | queried, with                      |
|                                   | ``0 <= queried_indices <=          |
|                                   | n_candidates``.                    |
+-----------------------------------+------------------------------------+


.. _general-advice-5:

General advice for working with a ``budget_manager``:
'''''''''''''''''''''''''''''''''''''''''''''''''''''

If a ``budget_manager`` is used, the ``_validate_data`` of the query
strategy needs to be adapted accordingly:

-  If only a ``budget`` is given use the default ``budget_manager`` with
   the given budget
-  If only a ``budget_manager`` is given use the ``budget_manager``
-  If both are not given use the default ``budget_manager`` with the
   default budget
-  If both are given and the budget differs from
   ``budget_manager.budget`` throw an error

.. _testing-3:

Testing
^^^^^^^
The test classes ``skactiveml.stream.budgetmanager.test.TestBudgetManager``
of budget managers need to inherit from the test template
``skactiveml.tests.template_budget_manager.TemplateBudgetManager``.
As a result, many required functionalities will be automatically tested.
As a requirement, one needs to specify the parameters of ``bm_class``,
``init_default_params`` and ``query_by_utility_params`` of the ``__init__``
accordingly. Once, the parameters are set, the developer needs to adjust the
test until all errors are resolved. We refer to the test template for more
detailed information.


Multi-Annotator Pool-based Query Strategies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All query strategies are stored in a file
``skactiveml/pool/multi/*.py`` and inherit
``skactiveml.base.MultiAnnotatorPoolQueryStrategy``.

The class must implement the following methods:

+------------+----------------------------------------------------------------+
| Method     | Description                                                    |
+============+================================================================+
| ``init``   | Method for initialization.                                     |
+------------+----------------------------------------------------------------+
| ``query``  | Select the annotator-sample pairs to decide which sample's     |
|            | class label is to be queried from which annotator.             |
+------------+----------------------------------------------------------------+

.. _query-method-3:

``query``
^^^^^^^^^

Required Parameters:

+-----------------------------------+-----------------------------------+
| Parameter                         | Description                       |
+===================================+===================================+
| ``X``                             | Training data set, usually        |
|                                   | complete, i.e. including the      |
|                                   | labeled and unlabeled samples.    |
+-----------------------------------+-----------------------------------+
| ``y``                             | Labels of the training data set   |
|                                   | for each annotator (possibly      |
|                                   | including unlabeled ones          |
|                                   | indicated by self.MISSING_LABEL), |
|                                   | meaning that ``y[i, j]`` contains |
|                                   | the label annotated by annotator  |
|                                   | ``i`` for sample ``j``.           |
+-----------------------------------+-----------------------------------+
| ``candidates``, optional          | If ``candidates`` is ``None``,    |
|                                   | the samples from ``(X, y)``, for  |
|                                   | which an annotator exists such    |
|                                   | that the annotator sample pair is |
|                                   | unlabeled are considered as       |
|                                   | sample candidates.                |
|                                   | If ``candidates`` is of shape     |
|                                   | ``(n_candidates,)`` and of type   |
|                                   | int, ``candidates`` is considered |
|                                   | as the indices of the sample      |
|                                   | candidates in ``(X, y)``. If      |
|                                   | ``candidates`` is of shape        |
|                                   | ``(n_candidates, n_features)``,   |
|                                   | the sample candidates are         |
|                                   | directly given in ``candidates``  |
|                                   | (not necessarily contained in     |
|                                   | ``X``). This is not supported by  |
|                                   | all query strategies.             |
+-----------------------------------+-----------------------------------+
| ``annotators``, optional          | If ``annotators`` is ``None``,    |
|                                   | all annotators are considered as  |
|                                   | available annotators. If          |
|                                   | ``annotators`` is of shape        |
|                                   | (n_avl_annotators), and of type   |
|                                   | int, ``annotators`` is considered |
|                                   | as the indices of the available   |
|                                   | annotators. If candidate samples  |
|                                   | and available annotators are      |
|                                   | specified: The annotator-sample   |
|                                   | pairs, for which the sample is a  |
|                                   | candidate sample and the          |
|                                   | annotator is an available         |
|                                   | annotator are considered as       |
|                                   | candidate annotator-sample-pairs. |
|                                   | If ``annotators`` is a boolean    |
|                                   | array of shape (n_candidates,     |
|                                   | n_avl_annotators) the             |
|                                   | annotator-sample pairs, for which |
|                                   | the sample is a candidate sample  |
|                                   | and the boolean matrix has entry  |
|                                   | ``True`` are considered as        |
|                                   | candidate annotator-sample pairs. |
+-----------------------------------+-----------------------------------+
| ``batch_size``, optional          | The number of annotator-sample    |
|                                   | pairs to be selected in one AL    |
|                                   | cycle.                            |
+-----------------------------------+-----------------------------------+
| ``return_utilities``, optional    | If ``True``, also return the      |
|                                   | utilities based on the query      |
|                                   | strategy.                         |
+-----------------------------------+-----------------------------------+

Returns:

+-----------------------------------+-----------------------------------+
| Parameter                         | Description                       |
+===================================+===================================+
| ``query_indices``                 | The ``query_indices`` indicate    |
|                                   | for which candidate sample a      |
|                                   | label is to be queried, e.g.,     |
|                                   | ``query_indices[0]`` indicates    |
|                                   | the first selected sample. If     |
|                                   | candidates is None or of shape    |
|                                   | (n_candidates), the indexing      |
|                                   | refers to samples in ``X``. If    |
|                                   | candidates is of shape            |
|                                   | (n_candidates, n_features), the   |
|                                   | indexing refers to samples in     |
|                                   | candidates.                       |
+-----------------------------------+-----------------------------------+
| ``utilities``                     | The utilities of samples after    |
|                                   | each selected sample of the       |
|                                   | batch, e.g., ``utilities[0]``     |
|                                   | indicates the utilities used for  |
|                                   | selecting the first sample (with  |
|                                   | index ``query_indices[0]``) of    |
|                                   | the batch. Utilities for labeled  |
|                                   | samples will be set to np.nan. If |
|                                   | candidates is None or of shape    |
|                                   | (n_candidates), the indexing      |
|                                   | refers to samples in ``X``. If    |
|                                   | candidates is of shape            |
|                                   | (n_candidates, n_features), the   |
|                                   | indexing refers to samples in     |
|                                   | candidates.                       |
+-----------------------------------+-----------------------------------+

.. _general-advice-6:

General advice
''''''''''''''

Use ``self._validate_data method`` (is implemented in superclass).
Check the input ``X`` and ``y`` only once. Fit classifier if it is not
yet fitted (may use ``fit_if_not_fitted`` form ``utils``). If the
strategy combines a single annotator query strategy with a performance
estimate:

-  define an aggregation function,
-  evaluate the performance for each sample-annotator pair,
-  use the ``SingleAnnotatorWrapper``.

If the strategy is a ``greedy`` method regarding the utilities:

-  calculate utilities (in an extra function),
-  use ``skactiveml.utils.simple_batch`` function for returning values.

.. _testing-4:

Testing
^^^^^^^

The test classes ``skactiveml.pool.multiannotator.test.TestQueryStrategy`` of
multi-annotator pool-based query strategies need inherit form
``unittest.TestCase``. In this class, each parameter ``a`` of the
``__init__`` method needs to be tested via a method ``test_init_param_a``.
This applies also for a parameter ``a`` of the ``query`` method, which is
tested via a method ``test_query_param_a``. The main logic of the query
strategy is test via the method ``test_query``.


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

``__init__``
~~~~~~~~~~~~

+-----------------------------------+----------------------------------------------+
| Parameter                         | Description                                  |
+===================================+==============================================+
| ``classes``, optional             | Holds the label for each class. If None, the |
|                                   | classes are determined during fitting.       |
+-----------------------------------+----------------------------------------------+
| ``missing_label``, optional       | Value representing a missing label.          |
+-----------------------------------+----------------------------------------------+
| ``cost_matrix``, optional         | A cost matrix where ``cost_matrix[i,j]``     |
|                                   | indicates the cost of predicting class       |
|                                   | ``classes[j]`` for a sample of class         |
|                                   | ``classes[i]``. Only set if ``classes`` is   |
|                                   | not None.                                    |
+-----------------------------------+----------------------------------------------+
| ``random_state``, optional        | Ensures reproducibility (cf. scikit-learn).  |
+-----------------------------------+----------------------------------------------+

.. _fit-1:

``fit``
~~~~~~~

Required Parameters:

+-----------------------------------+----------------------------------------------+
| Parameter                         | Description                                  |
+===================================+==============================================+
| ``X``                             | Matrix of feature values representing the    |
|                                   | samples.                                     |
+-----------------------------------+----------------------------------------------+
| ``y``                             | Contains the class labels of the training    |
|                                   | samples. Missing labels are represented by   |
|                                   | the attribute ``missing_label``. Usually,    |
|                                   | ``y`` is a column array except for multi-    |
|                                   | annotator classifiers, which expect a matrix |
|                                   | with columns for each annotator.             |
+-----------------------------------+----------------------------------------------+
| ``sample_weight``, optional       | Contains weights for the training samples’   |
|                                   | class labels. Must have the same shape as    |
|                                   | ``y``.                                       |
+-----------------------------------+----------------------------------------------+

Returns:

+-----------------------------------+-----------------------------------+
| Parameter                         | Description                       |
+===================================+===================================+
|``self``                           | The fitted classifier object.     |
+-----------------------------------+-----------------------------------+

.. _general-advice-7:

General advice
^^^^^^^^^^^^^^

Use ``self._validate_data`` method (is implemented in superclass) to
check standard parameters of ``__init__`` and ``fit`` method. If the
``classes`` parameter was provided, the classifier can be fitted with
training sample of which each was assigned a ``missing_label``.
In this case, the classifier should  make random predictions, i.e.,
outputting uniform class-membership probabilities when calling
``predict_proba``. Ensure that the classifier can handle ``missing labels``
also in other cases.

.. _predict-proba-1:

``predict_proba``
~~~~~~~~~~~~~~~~~

Required Parameters:

+-----------------------------------+----------------------------------------------+
| Parameter                         | Description                                  |
+===================================+==============================================+
| ``X``                             | Matrix of feature values representing the    |
|                                   | samples for which predictions are made.      |
+-----------------------------------+----------------------------------------------+

Returns:

+-----------------------------------+-----------------------------------+
| Parameter                         | Description                       |
+===================================+===================================+
| ``P``                             | The estimated class-membership    |
|                                   | probabilities per sample.         |
+-----------------------------------+-----------------------------------+

.. _general-advice-8:

General advice
^^^^^^^^^^^^^^

Check parameter ``X`` regarding its shape, i.e., use superclass method
``self._check_n_features`` to ensure a correct number of features. Check
that the classifier has been fitted. If the classifier is a
``skactiveml.base.ClassFrequencyEstimator``, this method is already
implemented in the superclass.

.. _predict-1:

``predict``
~~~~~~~~~~~

Required Parameters:

+-----------------------------------+----------------------------------------------+
| Parameter                         | Description                                  |
+===================================+==============================================+
| ``X``                             | Matrix of feature values representing the    |
|                                   | samples for which predictions are made.      |
+-----------------------------------+----------------------------------------------+

Returns:

+-----------------------------------+-----------------------------------+
| Parameter                         | Description                       |
+===================================+===================================+
| ``y_pred``                        | The estimated class label         |
|                                   | of each per sample.               |
+-----------------------------------+-----------------------------------+

.. _general-advice-9:

General advice
^^^^^^^^^^^^^^

Usually, this method is already implemented by the superclass through
calling the ``predict_proba`` method. If the superclass method is
overwritten, ensure that it can handle imbalanced costs and missing
labels.

.. _score-1:

``score``
~~~~~~~~~

Required Parameters:

+-----------------------------------+----------------------------------------------+
| Parameter                         | Description                                  |
+===================================+==============================================+
| ``X``                             | Matrix of feature values representing the    |
|                                   | samples for which predictions are made.      |
+-----------------------------------+----------------------------------------------+
| ``y``                             | Contains the true labels for each sample.    |
+-----------------------------------+----------------------------------------------+
| ``sample_weight``, optional       | Defines the importance of each sample when   |
|                                   | computing accuracy.                          |
+-----------------------------------+----------------------------------------------+

Returns:

+-----------------------------------+-----------------------------------+
| Parameter                         | Description                       |
+===================================+===================================+
| ``score``                         | Mean accuracy of                  |
|                                   | ``self.predict(X)`` regarding     |
|                                   | ``y``.                            |
+-----------------------------------+-----------------------------------+

.. _general-advice-10:

General advice
^^^^^^^^^^^^^^

Usually, this method is already implemented by the superclass. If the
superclass method is overwritten, ensure that it checks the parameters
and that the classifier has been fitted.

.. _testing-5:

Testing
~~~~~~~

The test classes ``skactiveml.classifier.TestClassifier``
of classifiers need to inherit from the test template
``skactiveml.tests.template_estimators.TemplateSkactivemlClassifier``.
As a result, many required functionalities will be automatically tested.
As a requirement, one needs to specify the parameters of ``estimator_class``,
``init_default_params``, ``fit_default_params``, and ``predict_default_params``
of the ``__init__`` accordingly. Once, the parameters are set, the developer
needs to adjust the test until all errors are resolved. We refer to the test
template for more detailed information.

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

``__init__``
~~~~~~~~~~~~

Required Parameters:

+-----------------------------------+----------------------------------------------+
| Parameter                         | Description                                  |
+===================================+==============================================+
| ``random_state``, optional        | Ensures reproducibility (cf. scikit-learn).  |
+-----------------------------------+----------------------------------------------+
| ``missing_label``, optional       | Value representing a missing label.          |
+-----------------------------------+----------------------------------------------+

.. _fit-2:

``fit``
~~~~~~~

Required Parameters:

+-----------------------------------+----------------------------------------------+
| Parameter                         | Description                                  |
+===================================+==============================================+
| ``X``                             | Matrix of feature values representing the    |
|                                   | samples.                                     |
+-----------------------------------+----------------------------------------------+
| ``y``                             | Contains the target values of the training   |
|                                   | samples. Missing labels are represented by   |
|                                   | the attribute ``missing_label``. Usually,    |
|                                   | ``y`` is a column array except for multi-    |
|                                   | target regressors, which expect a matrix     |
|                                   | with columns for each target type.           |
+-----------------------------------+----------------------------------------------+
| ``sample_weight``, optional       | Contains weights for the training samples’   |
|                                   | targets. Must have the same shape as ``y``.  |
+-----------------------------------+----------------------------------------------+

Returns:

+-----------------------------------+-----------------------------------+
| Parameter                         | Description                       |
+===================================+===================================+
|``self``                           | The fitted regressor object.      |
+-----------------------------------+-----------------------------------+

.. _general-advice-11:

General advice
^^^^^^^^^^^^^^

Use ``self._validate_data`` method (is implemented in superclass) to
check standard parameters of ``__init__`` and ``fit`` method. If the regressor
was fitted on training sample of which each was assigned a ``missing_label``,
the regressor should predict a default value of zero when calling ``predict``.
Ensure that the regressor can handle ``missing labels`` also in other cases.


.. _predict-2:

``predict``
~~~~~~~~~~~

Required Parameters:

+-----------------------------------+----------------------------------------------+
| Parameter                         | Description                                  |
+===================================+==============================================+
| ``X``                             | Matrix of feature values representing the    |
|                                   | samples for which predictions are made.      |
+-----------------------------------+----------------------------------------------+

Returns:

+-----------------------------------+-----------------------------------+
| Parameter                         | Description                       |
+===================================+===================================+
| ``y_pred``                        | The estimated targets per sample. |
+-----------------------------------+-----------------------------------+

.. _general-advice-12:

General advice
^^^^^^^^^^^^^^

Check parameter ``X`` regarding its shape, i.e., use method
``skactiveml.utils.check_n_features`` to ensure a correct number of
features. Check that the regressor has been fitted. If the classifier is a
``skactiveml.base.ProbabilisticRegressor``, this method is already
implemented in the superclass.

.. _score-2:

``score``
~~~~~~~~~

Required Parameters:

+-----------------------------------+----------------------------------------------+
| Parameter                         | Description                                  |
+===================================+==============================================+
| ``X``                             | Matrix of feature values representing the    |
|                                   | samples for which predictions are made.      |
+-----------------------------------+----------------------------------------------+
| ``y``                             | Contains the true target values for each     |
|                                   | sample.                                      |
+-----------------------------------+----------------------------------------------+
| ``sample_weight``, optional       | Defines the importance of each sample when   |
|                                   | computing the R2 score.                      |
+-----------------------------------+----------------------------------------------+

Returns:

+-----------------------------------+-----------------------------------+
| Parameter                         | Description                       |
+===================================+===================================+
| ``score``                         | R2 score of ``self.predict(X)``   |
|                                   |  regarding ``y``.                 |
+-----------------------------------+-----------------------------------+

General advice
^^^^^^^^^^^^^^

Usually, this method is already implemented by the superclass. If the
superclass method is overwritten, ensure that it checks the parameters
and that the regressor has been fitted.

.. _testing-6:

Testing
~~~~~~~

The test classes ``skactiveml.classifier.TestRegressor``
of regressors need to inherit from the test template
``skactiveml.tests.template_estimators.TemplateSkactivemlRegressor``.
As a result, many required functionalities will be automatically tested.
As a requirement, one needs to specify the parameters of ``estimator_class``,
``init_default_params``, ``fit_default_params``, and ``predict_default_params``
of the ``__init__`` accordingly. Once, the parameters are set, the developer
needs to adjust the test until all errors are resolved. We refer to the test
template for more detailed information.


Annotator Models
----------------

Annotator models implement the interface
``skactiveml.base.AnnotatorModelMixin``. These models can estimate the
performance of annotators for given samples. Each annotator model must implement
the ``predict_annotator_perf`` method, which estimates the performance per
sample for each annotator as a proxy for the quality of the provided annotations.

.. _predict-annotator-perf-1:

``predict_annotator_perf``
~~~~~~~~~~~~~~~~~~~~~~~~~~

Required Parameters:

+-------------+-----------------------------------------------------------+
| Parameter   | Description                                               |
+=============+===========================================================+
| ``X``       | Matrix of feature values representing the samples.        |
+-------------+-----------------------------------------------------------+

Returns:

+-------------+-----------------------------------------------------------+
| Parameter   | Description                                               |
+=============+===========================================================+
| ``P_annot`` | The estimated performance per sample-annotator pair.      |
+-------------+-----------------------------------------------------------+

.. _general-advice-14:

General advice
^^^^^^^^^^^^^^

Check parameter ``X`` regarding its shape and check that the annotator
model has been fitted. If no samples or class labels were provided
during the previous call of the ``fit`` method, the maximum value of
annotator performance should be outputted for each sample-annotator
pair.


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
| ``category``     | The methodological category of this query strategy, e.g.,|
|                  | Expected Error Reduction, Model Change,                  |
|                  | Query-by-Committee, Random Sampling,                     |
|                  | Uncertainty Sampling, or Others.                         |
+------------------+----------------------------------------------------------+
| ``template``     | Defines the general setup/setting of the example.        |
|                  | Supported templates include:                             |
|                  | ``examples/template_pool.py``,                           |
|                  | ``examples/template_pool_regression.py``,                |
|                  | ``examples/template_stream.py``, and                     |
|                  | ``examples/template_pool_batch.py``.                     |
+------------------+----------------------------------------------------------+
| ``tags``         | Search categories. Supported tags include ``pool``,      |
|                  | ``stream``, ``single-annotator``, ``multi-annotator``,   |
|                  | ``classification``, and ``regression``.                  |
+------------------+----------------------------------------------------------+
| ``title``        | Title of the example, usually named after the query      |
|                  | strategy.                                                |
+------------------+----------------------------------------------------------+
| ``text_0``       | Placeholder for additional explanations.                 |
+------------------+----------------------------------------------------------+
| ``refs``         | References (BibTeX keys) to the paper(s) describing      |
|                  | the query strategy.                                      |
+------------------+----------------------------------------------------------+
| ``sequence``     | Order in which content is displayed, usually             |
|                  | ``["title", "text_0", "plot", "refs"]``.                 |
+------------------+----------------------------------------------------------+
| ``import_misc``  | Python code for imports (e.g.,                           |
|                  | ``from skactiveml.pool import RandomSampling``).         |
+------------------+----------------------------------------------------------+
| ``n_samples``    | Number of samples in the example dataset.                |
+------------------+----------------------------------------------------------+
| ``init_qs``      | Python code to initialize the query strategy object,     |
|                  | e.g., ``RandomSampling()``.                              |
+------------------+----------------------------------------------------------+
| ``query_params`` | Python code for parameters passed to the query method,   |
|                  | e.g., ``X=X, y=y``.                                      |
+------------------+----------------------------------------------------------+
| ``preproc``      | Python code for preprocessing before executing the AL    |
|                  | cycle, e.g., ``X = (X-X.min())/(X.max()-X.min())``.      |
+------------------+----------------------------------------------------------+
| ``n_cycles``     | Number of active learning cycles.                        |
+------------------+----------------------------------------------------------+
| ``init_clf``     | Python code to initialize the classifier object, e.g.,   |
|                  | ``ParzenWindowClassifier(classes=[0, 1])``. (Only        |
|                  | supported for certain templates.)                        |
+------------------+----------------------------------------------------------+
| ``init_reg``     | Python code to initialize the regressor object, e.g.,    |
|                  | ``NICKernelRegressor()``. (Only supported for the        |
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
~~~~~~~~~~~~~~~~~~~~~~~~~~

To ensure your documentation is well formatted, build it using Sphinx:

.. code:: bash

   sphinx-build -b html docs docs/_build

Issue Tracking
--------------

We use `GitHub Issues <https://github.com/scikit-activeml/scikit-activeml/issues>`__
as our issue tracker. If you believe you have found a bug in
``scikit-activeml``, please report it there. Documentation bugs can also be reported.

Checking If a Bug Already Exists
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before filing an issue, please check whether the problem has already been reported.
This will help determine if the problem is resolved or fixed in an upcoming release,
save time, and provide guidance on how to fix it. Search the issue database using
the search box at the top of the issue tracker page (filter by the ``bug`` label).

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
