Developer Guide
===============

**Scikit-ActiveML** is a library that implements the most important
query strategies of active learning. It is built upon the well-known
machine learning framework
`scikit-learn <https://scikit-learn.org/stable/>`__.

Introduction
------------

Thank you, contributors!
~~~~~~~~~~~~~~~~~~~~~~~~

A big thank you to all contributors who provide the **Scikit-ActiveML**
project with new enhancements and bug fixes.

Getting Help
~~~~~~~~~~~~

If you have any questions, please reach out to other developers via the
following channels:

-  `Github
   Issues <https://github.com/scikit-activeml/scikit-activeml/issues>`__

Roadmap
~~~~~~~

Our Roadmap is summarized in the issue `Upcoming
Features <https://github.com/scikit-activeml/scikit-activeml/issues/145>`__.

Get Started
-----------

Before you can contribute to this project, you might execute the
following steps.

Setup Development Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are several ways to create a local Python environment, such as
`virtualenv <https://www.google.com/search?client=safari&rls=en&q=virtualenv&ie=UTF-8&oe=UTF-8>`__,
`pipenv <https://pipenv.pypa.io/enz/latest/>`__,
`miniconda <https://docs.conda.io/en/latest/miniconda.html>`__, etc. One
possible workflow is to install ``miniconda`` and use it to create a
Python environment. And use ``pip`` to install packages in this
environment.

Example With miniconda
^^^^^^^^^^^^^^^^^^^^^^

Create a new Python environment named **scikit-activeml**:

.. code:: bash

   conda create -n scikit-activeml

To be sure that the correct env is active:

.. code:: bash

   conda activate scikit-activeml

Then install ``pip``:

.. code:: bash

   conda install pip

Install Dependencies
~~~~~~~~~~~~~~~~~~~~

Now we can install some required project dependencies, which are defined
in the ``requirements.txt`` file.

.. code:: bash

   # Make sure your scikit-activeml python env is active!
   cd <project-root>
   pip install -r requirements.txt
   pip install -r requirements_extra.txt

After the pip installation was successful, we have to install ``pandoc``
and ``ghostscript`` if it is not already installed.

Example with MacOS (Homebrew)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: bash

   brew install pandoc ghostscript

Contributing Code
-----------------

General Coding Conventions
~~~~~~~~~~~~~~~~~~~~~~~~~~

As this library conforms to the convention of
`scikit-learn <https://scikit-learn.org/stable/developers/develop.html#coding-guidelines>`__,
the code should conform to `PEP
8 <https://www.python.org/dev/peps/pep-0008/>`__ Style Guide for Python
Code. For linting, the use of
`flake8 <https://flake8.pycqa.org/en/latest/>`__ is recommended.

Example for C3 (Code Contribution Cycle) and Pull Requests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Fork the repository using the Github ``Fork`` button.

2. Then, clone your fork to your local machine:

.. code:: bash

   git clone https://github.com/<your-username>/scikit-activeml.git

3. Create a new branch for your changes from the ``master`` branch:

.. code:: bash

   git checkout -b <branch-name>

4. After you have finished implementing the feature, make sure that all
   the tests pass. The tests can be run as

.. code:: bash

   $ pytest

Make sure, you covered all lines by tests.

.. code:: bash

   $ pytest --cov=./skactiveml

5. Commit and push the changes.

.. code:: bash

   $ git add <modified-files>
   $ git commit -m "<commit-message>"
   $ git push

Query Strategies
----------------

Pool-based Query Strategies
~~~~~~~~~~~~~~~~~~~~~~~~~~~

General
^^^^^^^

All query strategies are stored in a file
``skactiveml/pool/_query_strategy.py``. Every class inherits from
``SingleAnnotatorPoolQueryStrategy``. The class must implement the
``__init__`` function for initialization and a ``query`` function.

``__init__`` function
^^^^^^^^^^^^^^^^^^^^^

For typical class parameters we use standard names:

+-----------------------------------+-----------------------------------+
| Parameter                         | Description                       |
+===================================+===================================+
| ``prior``                         | Prior probabilities for the       |
|                                   | distribution of probabilistic     |
|                                   | strategies                        |
+-----------------------------------+-----------------------------------+
| ``random_state``                  | Number or np.random.RandomState   |
|                                   | like sklearn                      |
+-----------------------------------+-----------------------------------+
| ``method``                        | String for classes that implement |
|                                   | multiple methods                  |
+-----------------------------------+-----------------------------------+
| ``cost_matrix``                   | Cost matrix defining the cost of  |
|                                   | predicting instances wrong        |
+-----------------------------------+-----------------------------------+

``query`` function
^^^^^^^^^^^^^^^^^^

Required Parameters:

+-----------------------------------+-----------------------------------+
| Parameter                         | Description                       |
+===================================+===================================+
| ``X_cand``                        | Set of candidate instances,       |
|                                   | inherited from                    |
|                                   | ``Single                          |
|                                   | AnnotatorPoolBasedQueryStrategy`` |
+-----------------------------------+-----------------------------------+
| ``clf``                           | The classifier used by the        |
|                                   | strategy                          |
+-----------------------------------+-----------------------------------+
| ``X``                             | Set of labeled and unlabeled      |
|                                   | instances                         |
+-----------------------------------+-----------------------------------+
| ``y``                             | (unknown) labels of ``X``         |
+-----------------------------------+-----------------------------------+
| ``sample_weight``                 | Weights of training samples in    |
|                                   | ``X``                             |
+-----------------------------------+-----------------------------------+
| ``sample_weight_cand``            | Weights of samples in ``X_cand``  |
+-----------------------------------+-----------------------------------+
| ``batch_size``                    | Number of instances for batch     |
|                                   | querying, inherited from          |
|                                   | ``Single                          |
|                                   | AnnotatorPoolBasedQueryStrategy`` |
+-----------------------------------+-----------------------------------+
| ``return_utilities``              | Inherited from                    |
|                                   | ``Single                          |
|                                   | AnnotatorPoolBasedQueryStrategy`` |
+-----------------------------------+-----------------------------------+

Returns:

+-----------------------------------+-----------------------------------+
| Parameter                         | Description                       |
+===================================+===================================+
| ``query_indices``                 | Indices of the best instances     |
+-----------------------------------+-----------------------------------+
| ``utilities``                     | Utilities of all candidate        |
|                                   | instances, only if                |
|                                   | ``return_utilities`` is ``True``  |
+-----------------------------------+-----------------------------------+

General advice
''''''''''''''

Use ``self._validate_data`` function (Is implemented in the superclass).
Check the input ``X`` and ``y`` only once. Fit classifier if it is not
yet fitted (May use ``fit_if_not_fitted`` form utils). Calculate
utilities (In an extra function. Use ``simple_batch`` function from
utils for return value.

Testing
^^^^^^^

All query strategies are tested by a general unittest
(``test_pool.py``). Querying of every method is tested with standard
configurations with 0, 1, and 5 initial labels. For every class
``ExampleQueryStrategy`` that inherits from
``SingleAnnotatorPoolQueryStrategy`` (stored in ``_example.py``), it is
automatically tested if there exists a file ``test/test_example.py``. It
is necessary that both filenames are the same. Moreover, the test class
must be called ``TestExampleQueryStrategy(unittest.TestCase)``. Every
parameter in ``__init__()`` will be tested if it is written the same as
a class variable. Every parameter arg in ``__init__()`` will be
evaluated if there exists a method in the testclass
``TestExampleQueryStrategy`` that is called ``test_init_param_arg()``.
Every parameter arg in ``query()`` will be evaluated if there exists a
method in the testclass ``TestExampleQueryStrategy`` that is called
``test_query_param_arg()``. Standard parameters ``random_state``,
``X_cand``, ``batch_size`` and ``return_utilities`` are tested and do
not have to be tested in the specific tests.

Stream-based Query Strategies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _general-1:

General
^^^^^^^

All query strategies are stored in a file ``skactivml/stream/*.py``.
Every query strategy inherits from
``SingleAnnotatorStreamQueryStrategy``. Every query strategy has
either an internal budget handling or an outsourced ``budget_manager``.

For typical class parameters we use standard names:

+-----------------------------------+-----------------------------------+
| Parameter                         | Description                       |
+===================================+===================================+
| ``random_state``                  | Integer that acts as random seed  |
|                                   | or ``np.random.RandomState`` like |
|                                   | sklearn                           |
+-----------------------------------+-----------------------------------+
| ``budget``                        | % of labels that the strategy is  |
|                                   | allowed to query                  |
+-----------------------------------+-----------------------------------+
| ``budget_manager``                | Enforces the budget constraint    |
+-----------------------------------+-----------------------------------+

The class must implement the following functions:

+------------+----------------------------------------------------------------+
| Function   | Description                                                    |
+============+================================================================+
| ``init``   | Function for initialization                                    |
+------------+----------------------------------------------------------------+
| ``query``  | Identify the instances whose labels to select                  |
+------------+----------------------------------------------------------------+
| ``update`` | Adapting the budget monitoring according to the queried labels |
+------------+----------------------------------------------------------------+

.. _query-function-1:

``query`` function
^^^^^^^^^^^^^^^^^^

Required Parameters:

+-----------------------------------+-----------------------------------+
| Parameter                         | Description                       |
+===================================+===================================+
| ``X_cand``                        | Set of candidate instances,       |
|                                   | inherited from                    |
|                                   | ``SingleAn                        |
|                                   | notatorStreamBasedQueryStrategy`` |
+-----------------------------------+-----------------------------------+
| ``clf``                           | The classifier used by the        |
|                                   | strategy                          |
+-----------------------------------+-----------------------------------+
| ``X``                             | Set of labeled and unlabeled      |
|                                   | instances                         |
+-----------------------------------+-----------------------------------+
| ``y``                             | Labels of ``X`` (it may be set to |
|                                   | ``MISSING_LABEL`` if ``y`` is     |
|                                   | unknown)                          |
+-----------------------------------+-----------------------------------+
| ``sample_weight``                 | Weights for each instance in      |
|                                   | ``X`` or ``None`` if all are      |
|                                   | equally weighted                  |
+-----------------------------------+-----------------------------------+
| ``return_utilities``              | Inherited from                    |
|                                   | Single                            |
|                                   | AnnotatorStreamBasedQueryStrategy |
+-----------------------------------+-----------------------------------+

Returns:

+-----------------------------------+-----------------------------------+
| Parameter                         | Description                       |
+===================================+===================================+
| ``queried_indices``               | Indices of the best instances     |
|                                   | from ``X_Cand``                   |
+-----------------------------------+-----------------------------------+
| ``utilities``                     | Utilities of all candidate        |
|                                   | instances, only if                |
|                                   | ``return_utilities`` is ``True``  |
+-----------------------------------+-----------------------------------+

.. _general-advice-1:

General advice
''''''''''''''

The ``query`` function must not change the internal state of the
``query`` strategy (``budget`` and ``random_state`` included) to allow
for assessing multiple instances with the same state. Update the the
internal state in the ``update()`` function. Use ``self._validate_data``
function (is implemented in superclass). Check the input ``X`` and ``y``
only once. Fit classifier if it is not yet fitted (may use
``fit_if_not_fitted`` from ``utils``).

``update`` function
^^^^^^^^^^^^^^^^^^^

Required Parameters:

+-----------------------------------+-----------------------------------+
| Parameter                         | Description                       |
+===================================+===================================+
| ``X_cand``                        | Set of candidate instances,       |
|                                   | inherited from                    |
|                                   | ``SingleAn                        |
|                                   | notatorStreamBasedQueryStrategy`` |
+-----------------------------------+-----------------------------------+
| ``queried_indices``               | Typically the return value of     |
|                                   | ``query``                         |
+-----------------------------------+-----------------------------------+
| ``budget_manager_param_dict``     | Provides additional parameters to |
|                                   | the ``update`` function of the    |
|                                   | ``budget_manager`` (only include  |
|                                   | if a ``budget_manager`` is used)  |
+-----------------------------------+-----------------------------------+

.. _general-advice-2:

General advice
''''''''''''''

Use ``self._validate_data`` in case the strategy is used without using
the ``query`` method (if parameters need to be initialized before the
update). If a ``budget_manager`` is used forward the update call to the
``budget_manager.update`` method.

.. _testing-1:

Testing
^^^^^^^

All stream query strategies are tested by a general unittest
(``stream/tests/test_stream.py``) -For every class
``ExampleQueryStrategy`` that inherits from
``SingleAnnotatorStreamQueryStrategy`` (stored in ``_example.py``), it
is automatically tested if there exists a file ``test/test_example.py``.
It is necessary that both filenames are the same. Moreover, the test
class must be called ``TestExampleQueryStrategy`` and inherit from
``unittest.TestCase``. Every parameter in ``init()`` will be tested if
it is written the same as a class variable. Every parameter arg in
``init()`` will be evaluated if there exists a method in the testclass
``TestExampleQueryStrategy`` that is called ``test_init_param_arg()``.
Every parameter arg in ``query()`` will be evaluated if there exists a
method in the testclass ``TestExampleQueryStrategy`` that is called
``test_query_param_arg()``.

General advice for the ``budget_manager``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All budget managers are stored in
``skactivml/stream/budget_manager/\*.py``. The class must implement the
following functions:

+-----------------------------------+-----------------------------------+
| Parameter                         | Description                       |
+===================================+===================================+
| ``__init__``                      | Function for initialization       |
+-----------------------------------+-----------------------------------+
| ``update``                        | Adapting the budget monitoring    |
|                                   | according to the queried labels   |
+-----------------------------------+-----------------------------------+
| ``query_by_utilities``            | Identify which instances to query |
|                                   | based on the assessed utility     |
+-----------------------------------+-----------------------------------+

.. _update-function-1:

``update`` function
^^^^^^^^^^^^^^^^^^^

The update function of the budget manager has the same functionality as
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

``query_by_utilities`` function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Required Parameters:

+-----------------------------------+-----------------------------------+
| Parameter                         | Description                       |
+===================================+===================================+
| ``utilities``                     | The ``utilities`` of ``X_cand``   |
|                                   | calculated by the query strategy, |
|                                   | inherited from ``BudgetManager``  |
+-----------------------------------+-----------------------------------+

General advice for working with a ``budget_manager``:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If a ``budget_manager`` is used, the ``_validate_data`` of the query
strategy needs to be adapted accordingly:

-  If only a ``budget`` is given use the default ``budget_manager`` with
   the given budget
-  If only a ``budget_manager`` is given use the ``budget_manager``
-  If both are not given use the default ``budget_manager`` with the
   default budget
-  If both are given and the budget differs from
   ``budget_manager.budget`` throw an error

All budget managers are tested by a general unittest
(``stream/budget_manager/tests/test_budget_manager.py``). For every
class ``ExampleBudgetManager`` that inherits from ``BudgetManager``
(stored in ``_example.py``), it is automatically tested if there exists
a file ``test/test_example.py``. It is necessary that both filenames are
the same.

Moreover, the test class must be called ``TestExampleBudgetManager`` and
inheriting from ``unittest.TestCase``. Every parameter in ``__init__()``
will be tested if it is written the same as a class variable. Every
parameter ``arg`` in ``__init__()`` will be evaluated if there exists a
method in the testclass ``TestExampleQueryStrategy`` that is called
``test_init_param_arg()``. Every parameter ``arg`` in
``query_by_utility()`` will be evaluated if there exists a method in the
testclass ``TestExampleQueryStrategy`` that is called
``test_query_by_utility`` ``_param_arg()``.

Multi-Annotator Pool-based Query Strategies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All query strategies are stored in a file
``skactiveml/pool/multi/_query_strategy.py``. Every class inherits from
``MultiAnnotatorPoolQueryStrategy``. The class must implement the
following functions:

+--------------+--------------------------------------------------------------+
| Parameter    | Description                                                  |
+==============+==============================================================+
| ``__init__`` | Function for initialization of hyperparameters               |
+--------------+--------------------------------------------------------------+
| ``query``    | Identify the instance annotator pairs whose labels to select |
+--------------+--------------------------------------------------------------+

For typical class parameters we use standard names:

================ ================================================
Parameter        Description
================ ================================================
``random_state`` Number or ``np.random.RandomState`` like sklearn
================ ================================================

.. _query-function-2:

``query`` function
^^^^^^^^^^^^^^^^^^

Required Parameters:

+-----------------------------------+-----------------------------------+
| Parameter                         | Description                       |
+===================================+===================================+
| ``X_cand``                        | Sequence of candidate instances   |
|                                   | to be queried, inherited from     |
|                                   | ``Multi                           |
|                                   | AnnotatorPoolBasedQueryStrategy`` |
+-----------------------------------+-----------------------------------+
| ``A_cand``                        | Boolean mask further specifying   |
|                                   | which annotator can be queried    |
|                                   | for which candidate instance,     |
|                                   | inherited from                    |
|                                   | ``Multi                           |
|                                   | AnnotatorPoolBasedQueryStrategy`` |
+-----------------------------------+-----------------------------------+
| ``clf``                           | The classifier used by the        |
|                                   | strategy                          |
+-----------------------------------+-----------------------------------+
| ``X``                             | Sequence of labeled and unlabeled |
|                                   | instances                         |
+-----------------------------------+-----------------------------------+
| ``y``                             | (unknown) Labels of ``X`` for     |
|                                   | each annotator                    |
+-----------------------------------+-----------------------------------+
| ``sample_weight``                 | Weights of the prediction of a    |
|                                   | sample from an annotator (used    |
|                                   | for predictions of labels)        |
+-----------------------------------+-----------------------------------+
| ``A_perf``                        | Performance of an annotators for  |
|                                   | a given sample, usually the       |
|                                   | accuracy (used for estimating the |
|                                   | best annotator to query for a     |
|                                   | given candidate sample)           |
+-----------------------------------+-----------------------------------+
| ``ybatch_size``                   | Number of instances for batch     |
|                                   | querying, inherited from          |
|                                   | ``Multi                           |
|                                   | AnnotatorPoolBasedQueryStrategy`` |
+-----------------------------------+-----------------------------------+
| ``return_utilities``              | Inherited from                    |
|                                   | ``Multi                           |
|                                   | AnnotatorPoolBasedQueryStrategy`` |
+-----------------------------------+-----------------------------------+

Returns:

+-----------------------------------+-----------------------------------+
| Parameter                         | Description                       |
+===================================+===================================+
| ``query_indices``                 | Indices of the best candidate     |
|                                   | instance annotator pair           |
+-----------------------------------+-----------------------------------+
| ``utilities``                     | Utilities of all candidate        |
|                                   | instances annotator pairs, only   |
|                                   | if ``return_utilities`` is        |
|                                   | ``True``                          |
+-----------------------------------+-----------------------------------+

.. _general-advice-3:

General advice
''''''''''''''

Use ``self._validate_data function`` (is implemented in superclass).
Check the input ``X`` and ``y`` only once. Fit classifier if it is not
yet fitted (may use ``fit_if_not_fitted`` form ``utils``). If the
strategy combines a single annotator query strategy with a performance
estimate:

-  Define an aggregation function
-  Evaluate the performance for each annotator sample pair
-  Use the ``SingleAnnotatorWrapper``

If the strategy is a ``greedy`` method regarding the utilities:

-  Calculate utilities (in an extra function)
-  Use ``simple_batch`` function from utils for return value

Classifiers
-----------

Standard classifier implementations are part of the subpackage
``skactiveml.classifier`` and classifiers learning from multiple
annotators are implemented in its subpackage
``skactiveml.classifier.multi``. Every class of a classifier inherits
from ``skactiveml.base.SkactivemlClassifier`` The class of a classifier
must implement the ``__init__`` method for initialization, a ``fit``
method for training, and a ``predict_proba`` method predicting class
membership probabilities for samples. A ``predict`` method is already
implemented in the superclass by using the outputs of the
``predict_proba`` method. Additionally, a ``score`` method is
implemented by the superclass to evaluate the accuracy of a fitted
classifier. A commonly used subclass of
``skactiveml.base.SkactivemlClassifier`` is the
sk\ ``activeml.base.ClassFrequencyEstimator``, which requires an
implementation of the method ``predict_freq``, which can be interpreted
as prior parameters of a Dirichlet distribution over the class
membership probabilities of a sample.

``init`` function
~~~~~~~~~~~~~~~~~

Required Parameters:

+-----------------------------------+-----------------------------------+
| Parameter                         | Description                       |
+===================================+===================================+
| ``classes``                       | Holds the label for each class.   |
|                                   | If ``None``, the classes are      |
|                                   | determined during the fit         |
+-----------------------------------+-----------------------------------+
| ``missing_label``                 | Value to represent a missing      |
|                                   | label                             |
+-----------------------------------+-----------------------------------+
| ``cost_matrix``                   | Cost matrix with                  |
|                                   | ``cost_matrix[i,j]`` indicating   |
|                                   | cost of predicting class          |
|                                   | ``classes[j]`` for a sample of    |
|                                   | class ``classes[i]``. Can be only |
|                                   | set, if classes is not ``None``   |
+-----------------------------------+-----------------------------------+
| ``random_state``                  | Ensures reproducibility           |
|                                   | (cf. scikit-learn)                |
+-----------------------------------+-----------------------------------+
| ``class_prior``                   | HA                                |
|                                   | ``skactive                        |
|                                   | ml.base.ClassFrequencyEstimator`` |
|                                   | requires additionally this        |
|                                   | parameter as prior observations   |
|                                   | of the class frequency estimates  |
+-----------------------------------+-----------------------------------+

``fit`` function
~~~~~~~~~~~~~~~~

Required Parameters:

+-----------------------------------+-----------------------------------+
| Parameter                         | Description                       |
+===================================+===================================+
| ``X``                             | Is a matrix of feature values     |
|                                   | representing the samples          |
+-----------------------------------+-----------------------------------+
| ``y``                             | Contains the class labels of the  |
|                                   | training samples. Missing labels  |
|                                   | are represented through the       |
|                                   | attribute ‘missing_label’.        |
|                                   | Usually, ``y`` is a column array  |
|                                   | except for multi-annotator        |
|                                   | classifiers which expect a matrix |
|                                   | with columns containing the class |
|                                   | labels provided by a specific     |
|                                   | annotator                         |
+-----------------------------------+-----------------------------------+
| ``sample_weight``                 | ontains the weights of the        |
|                                   | training samples’ class labels.   |
|                                   | It must have the same shape as    |
|                                   | ``y``                             |
+-----------------------------------+-----------------------------------+

Returns:

========= ============================
Parameter Description
========= ============================
``self``  The fitted classifier object
========= ============================

.. _general-advice-4:

General advice
^^^^^^^^^^^^^^

Use ``self._validate_data`` method (is implemented in superclass) to
check standard parameters of ``__init__`` and ``fit`` method. If
``self.n_features_`` is None, no samples were provided as training data.
In this case, the classifier should still be fitted but only for the
purpose to make random predictions, i.e., outputting uniform class
membership probabilities when calling ``predict_proba``. Ensure that the
classifier can handle missing labels.

``predict_proba`` function
~~~~~~~~~~~~~~~~~~~~~~~~~~

Required Parameters:

+-----------------------------------+-----------------------------------+
| Parameter                         | Description                       |
+===================================+===================================+
| ``X``                             | Is a matrix of feature values     |
|                                   | representing the samples, for     |
|                                   | which the classifier will make    |
|                                   | predictions                       |
+-----------------------------------+-----------------------------------+

Returns:

========= =======================================================
Parameter Description
========= =======================================================
``P``     The estimated class membership probabilities per sample
========= =======================================================

.. _general-advice-5:

General advice
^^^^^^^^^^^^^^

Check parameter ``X`` regarding its shape, i.e., use superclass method
``self._check_n_features`` to ensure a correct number of features. Check
that the classifier has been fitted. If the classifier is a
``skactiveml.base.ClassFrequencyEstimator``, this method is already
implemented in the superclass. If no samples or class labels were
provided during the previous call of the ``fit`` method, uniform class
membership probabilities are to be outputted.

``predict_freq`` function
~~~~~~~~~~~~~~~~~~~~~~~~~

Required Parameters:

+-----------------------------------+-----------------------------------+
| Parameter                         | Description                       |
+===================================+===================================+
| ``X``                             | Is a matrix of feature values     |
|                                   | representing the samples, for     |
|                                   | which the classifier will make    |
|                                   | predictions                       |
+-----------------------------------+-----------------------------------+

Returns:

+-----------------------------------+-----------------------------------+
| Parameter                         | Description                       |
+===================================+===================================+
| ``F``                             | The estimated class frequency     |
|                                   | estimates (excluding the prior    |
|                                   | observations)                     |
+-----------------------------------+-----------------------------------+

.. _general-advice-6:

General advice
^^^^^^^^^^^^^^

Check parameter X regarding its shape, i.e., use superclass method
``self._check_n_features`` to ensure a correct number of features. Check
that the classifier has been fitted. If no samples or class labels were
provided during the previous call of the ``fit`` method, a matrix of
zeros is to be outputted.

``predict`` function
~~~~~~~~~~~~~~~~~~~~

Required Parameters:

+-----------------------------------+-----------------------------------+
| Parameter                         | Description                       |
+===================================+===================================+
| ``X``                             | Is a matrix of feature values     |
|                                   | representing the samples, for     |
|                                   | which the classifier will make    |
|                                   | predictions                       |
+-----------------------------------+-----------------------------------+

Returns:

========== ========================================
Parameter  Description
========== ========================================
``y_pred`` The estimated class label of each sample
========== ========================================

.. _general-advice-7:

General advice
^^^^^^^^^^^^^^

Usually, this method is already implemented by the superclass through
calling the ``predict_proba`` method. If the superclass method is
overwritten, ensure that it can handle imbalanced costs and missing
labels. If no samples or class labels were provided during the previous
call of the ``fit`` method, random class label predictions are to be
outputted.

``score`` function
~~~~~~~~~~~~~~~~~~

Required Parameters:

+-----------------------------------+-----------------------------------+
| Parameter                         | Description                       |
+===================================+===================================+
| ``X``                             | Is a matrix of feature values     |
|                                   | representing the samples, for     |
|                                   | which the classifier will make    |
|                                   | predictions                       |
+-----------------------------------+-----------------------------------+
| ``y``                             | Contains the true label of each   |
|                                   | sample                            |
+-----------------------------------+-----------------------------------+
| ``sample_weight``                 | Defines the importance of each    |
|                                   | sample when computing the         |
|                                   | accuracy of the classifier        |
+-----------------------------------+-----------------------------------+

Returns:

========= ====================================================
Parameter Description
========= ====================================================
``score`` Mean accuracy of ``self.predict(X)`` regarding ``y``
========= ====================================================

.. _general-advice-8:

General advice
^^^^^^^^^^^^^^

Usually, this method is already implemented by the superclass. If the
superclass method is overwritten, ensure that it checks the parameters
and that the classifier has been fitted.

.. _testing-2:

Testing
~~~~~~~

All classifiers are tested by a general unittest
(``skactiveml/classifier/tests/test_classifier.py``). For every class
``ExampleClassifier`` that inherits from
``skactiveml.base.SkactivemlClassifier`` (stored in
``_example_classifier.py``), it is automatically tested if there exists
a file ``tests/test_example_classifier.py``. It is necessary that both
filenames are the same. Moreover, the test class must be called
``TestExampleClassifier`` and inherit from ``unittest.TestCase``. For
each parameter of an implemented method, there must be a test method
called ``test_methodname_parametername`` in the Python file
``_example_classifier.py``. It is to check whether invalid parameters
are handled correctly. For each implemented method, there must be a test
method called ``test_methodname`` in the Python file
``_example_classifier.py``. It is to check whether the method works as
intended.

Annotators Models
-----------------

Annotator models are marked by implementing the interface
``skactiveml.base.AnnotMixing``. These models can estimate the
performances of annotators for given samples. Every class of a
classifier inherits from ``skactiveml.base.SkactivemlClassifier``. The
class of an annotator model must implement the ``predict_annotator_perf``
method estimating the performances per sample of each annotator as
proxies of the provided annotation’s qualities.

``predict_annotator_perf`` function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Required Parameters:

========= ======================================================
Parameter Description
========= ======================================================
``X``     Is a matrix of feature values representing the samples
========= ======================================================

Returns:

=========== ====================================================
Parameter   Description
=========== ====================================================
``P_annot`` The estimated performances per sample-annotator pair
=========== ====================================================

.. _general-advice-9:

General advice
^^^^^^^^^^^^^^

Check parameter ``X`` regarding its shape and check that the annotator
model has been fitted. If no samples or class labels were provided
during the previous call of the ``fit`` method, the maximum value of
annotator performance should be outputted for each sample-annotator
pair.

Testing and code coverage
-------------------------

Please ensure test coverage is close to 100%. The current code coverage
can be viewed
`here <https://app.codecov.io/gh/scikit-activeml/scikit-activeml>`__.

Documentation (User guide and Developer guide)
----------------------------------------------

Guidelines for writing documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In ``Scikit-ActiveML``, the
`guidelines <https://scikit-learn.org/stable/developers/contributing.html#guidelines-for-writing-documentation>`__
for writing the documentation are adopted from
`scikit-learn <https://scikit-learn.org/stable/>`__.

Building the documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~

   TODO: How to build the user guide and developer guide?

Issue Tracking
--------------

We use `Github
Issues <https://github.com/scikit-activeml/scikit-activeml/issues>`__ as
our issue tracker. If you think you have found a bug in
``Scikit-ActiveML``, you can report it to the issue tracker.
Documentation bugs can also be reported there.

Checking If A Bug Already Exists
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The first step before filing an issue report is to see whether the
problem has already been reported. Checking if the problem is an
existing issue will:

1. Help you see if the problem has already been resolved or has been
   fixed for the next release
2. Save time for you and the developers
3. Help you learn what needs to be done to fix it
4. Determine if additional information, such as how to replicate the
   issue, is needed

To see if the issue already exists, search the issue database (``bug``
label) using the search box on the top of the issue tracker page.

Reporting an issue
~~~~~~~~~~~~~~~~~~

Use the following labels to report an issue:

================= ====================================
Label             Usecase
================= ====================================
``bug``           Something isn’t working
``enhancement``   New feature
``documentation`` Improvement or additions to document
``question``      General questions
================= ====================================
