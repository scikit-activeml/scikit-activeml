Contributing
============

**scikit-activeml** is a library that implements the most important query
strategies for active learning. It is built upon the well-known machine
learning framework `scikit-learn <https://scikit-learn.org/stable/>`__.

Overview
--------

Our philosophy is to extend the scikit-learn ecosystem with the most relevant
query strategies for active learning and to implement tools for working with
partially unlabeled data. An overview of our repository's structure is provided
in the image below. Each node represents a class or interface, and the arrows
illustrate the inheritance hierarchy among them. Dashed nodes indicate
functionality that is not yet available in our library. scikit-learn
is used in this image for identification only and does not imply endorsement.

.. image:: https://raw.githubusercontent.com/scikit-activeml/scikit-activeml/master/docs/logos/scikit-activeml-structure.png
   :width: 1000

In our package ``skactiveml``, there are three major components:
``SkactivemlClassifier``, ``SkactivemlRegressor``, and ``QueryStrategy``.
The classifier and regressor modules are necessary to handle partially unlabeled
data and to implement active-learning-specific estimators. This way, an active
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
workflow is to install miniconda and use it to create a Python environment.

Example with miniconda
^^^^^^^^^^^^^^^^^^^^^^

Create a new Python environment named ``scikit-activeml``:

.. code:: bash

   conda create -n scikit-activeml

To ensure that the correct environment is active:

.. code:: bash

   conda activate scikit-activeml

Then install pip:

.. code:: bash

   conda install pip

Install Dependencies
~~~~~~~~~~~~~~~~~~~~

Now, install the required project dependencies for development. A standard
editable contributor setup uses the ``dev`` extra:

.. code:: bash

   # Make sure your scikit-activeml Python environment is active!
   cd <project-root>
   pip install -e '.[dev]'

If you also need the optional dependencies used by deep learning features,
streaming integrations, or tutorials, install the ``opt`` extra alongside the
development dependencies:

.. code:: bash

   pip install -e '.[dev,opt]'

This uses the same ``opt`` extra described in the installation section of the
README and pulls in the packages listed in ``requirements_opt.txt``. If you
need the maximum tested dependency set for local reproduction of the CI
environment, you can additionally use:

.. code:: bash

   pip install -e '.[dev,max,opt_max]'

As in the README, the ``opt`` extra does **not** install PyTorch automatically.
Please install the appropriate PyTorch build separately for your hardware and
CUDA setup if your contribution depends on PyTorch/skorch features.
If your contribution uses CapyMOA, you also need a Java runtime. One
simple option with conda is:

.. code:: bash

   conda install conda-forge::openjdk

After the pip installation is successful, you must install pandoc and
ghostscript, and ffmpeg if they are not already installed. For example,
via conda you can do this as follows:

.. code:: bash

   conda install -c conda-forge pandoc ghostscript ffmpeg


We also provide a `pre-commit <https://pre-commit.com/>`__
configuration that offers automatic linting checks using
`black <https://black.readthedocs.io/en/stable/>`__ and
`flake8 <https://flake8.pycqa.org/en/latest/>`__. To use these pre-commit
hooks, you need to install them:

.. code:: bash

  pre-commit install

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
solution for code formatting. For example, you can format your
code using the following commands:

.. code:: bash

   black skactiveml
   flake8 skactiveml/ --ignore=E203,W503

For a new strategy module centered on one public strategy class, derive the
private module name from the class name in snake case and use the same stem for
its test module. A conceptual or acronym-based filename is appropriate when a
module intentionally groups a strategy family.

Where Helpers Live
~~~~~~~~~~~~~~~~~~

Keep helper functions close to the code that uses them. A helper used in one
file should stay in that file. If several files in one subpackage use it, put
it in a private file in that subpackage. If several subpackages use a general
helper and it does not import from ``skactiveml.base``, put it in a clearly
named private file under ``skactiveml/utils``.

Do not import helper functions or constants from ``skactiveml.base``; that
file is for base classes. If a helper must import a base class, keep it in the
subpackage that uses it. ``skactiveml/pool/_target.py`` is an example. Put
exception classes in ``skactiveml/exceptions.py``.

Add a query-strategy base class only when it contains behavior shared by its
subclasses. Do not add one only to share a constant or to help tests find
strategies.

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

   python -m pytest --no-cov -n 6

Make sure you have covered all lines with tests.

.. code:: bash

   python -m pytest -n 6

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
This superclass inherits from ``sklearn.base.Estimator``. Its optional
``random_state`` parameter controls reproducibility, and the abstract ``query``
method enforces the implementation of the sample-selection logic.

Target Semantics and Compatibility
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Labels and target semantics have an explicit public contract. Read the
:doc:`target_semantics` reference before changing estimators, query strategies,
label helpers, or multi-annotator components. In particular:

- resolve target meaning with
  :func:`~skactiveml.utils.resolve_target_spec`; do not infer semantics from
  array shape or values alone;
- declare the exact target combinations a component supports through its
  ``_target_capabilities``;
- preserve the fitted ``target_spec_`` on estimators and wrappers that retain a
  fitted target specification; query strategies resolve each query and do not
  retain a last-query specification;
- add tests for every supported target combination and for invalid or
  unsupported combinations, including ambiguous two-dimensional labels and
  complete-row requirements for multi-label data;
- operate on encoded labels for multi-label data, because each label output
  may declare its own binary class vocabulary, e.g.
  ``[["no", "yes"], ["off", "on"]]``. A pool strategy test supplying
  ``query_default_params_clf_multilabel`` runs the shared custom class
  vocabulary contract automatically. If its acquisition path needs auxiliary
  inputs, override ``_multilabel_custom_vocabulary_params`` and record the
  override in ``skactiveml/pool/tests/test_multilabel_contracts.py``
  instead of skipping the contract.

When adding multi-label support to a pool strategy, update the inventory in
``skactiveml/pool/tests/test_multilabel_contracts.py`` and the
:ref:`multilabel-strategy-inventory`. The tests group strategies by their use of
probabilities; the reference groups them by their use of multi-label data.

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
| ``query_indices``                 | Indices indicating which candidate sample's  |
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

Use the ``self._validate_data`` method (implemented in the superclass) exactly
once for structural validation of the query inputs ``X`` and ``y``. Target
resolution is a separate semantic step and may intentionally precede
``self._validate_data`` so that invalid or unsupported target semantics fail
before estimator fitting or query-state initialization. Fit the classifier or
regressor if it is not yet fitted (using ``fit_if_not_fitted`` from ``utils``).
Calculate utilities via an extra public function. Use the ``simple_batch``
function from ``utils`` to determine the query indices and set the utilities in
naive batch query strategies.

.. _testing-1:

Testing
^^^^^^^

Pool-strategy tests live in ``skactiveml/pool/tests``. Reuse the test template
``skactiveml.tests.template_query_strategy.TemplateSingleAnnotatorPoolQueryStrategy``.
Specify ``qs_class`` and ``init_default_params`` in the template setup and,
depending on the supported task, ``query_default_params_reg`` or
``query_default_params_clf``. Refer to the template for the complete contract.

Keep test helpers in the narrowest module that owns them. A helper used by one
test module stays in that module. Put broadly reused assertions and fixture
helpers in ``skactiveml.tests.utils``. Put inherited query-strategy contract
behavior in ``skactiveml.tests.template_query_strategy``.

Single-annotator Stream-based Query Strategies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _general-2:

General
^^^^^^^

All query strategies are stored in ``skactiveml/stream/*.py``.
Every query strategy inherits from
``skactiveml.base.SingleAnnotatorStreamQueryStrategy``. Every query strategy has
either an internal budget handling or an outsourced ``budget_manager``.

For typical class parameters we use standard names:

+------------------------------+------------------------------------------+
| Parameter                    | Description                              |
+==============================+==========================================+
| ``random_state``             | Integer that acts as random seed         |
|                              | or ``np.random.RandomState`` like        |
|                              | scikit-learn.                            |
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
Stream-strategy tests live in ``skactiveml/stream/tests``. Reuse the test
template
``skactiveml.tests.template_query_strategy.TemplateSingleAnnotatorStreamQueryStrategy``.
Specify ``qs_class`` and ``init_default_params`` in the template setup and,
depending on the supported task, ``query_default_params_reg`` or
``query_default_params_clf``. Refer to the template for the complete contract.

.. _general-advice-4:

``budget_manager``
^^^^^^^^^^^^^^^^^^

All budget managers are stored in
``skactiveml/stream/budgetmanager/*.py``. The class must implement the
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

Once initialized, ``query_by_utility`` must leave the budget manager's
acquisition state unchanged, including counters, thresholds, histories, and
random state. Repeated queries with the same utilities and parameters must
return the same indices until ``update`` advances that state. Both
``query_by_utility`` and ``update`` must initialize the state needed for
querying when called first. The shared budget-manager test template checks
this contract for both initialization paths.

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
|                                   | scikit-learn                      |
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
Budget-manager tests live in ``skactiveml/stream/budgetmanager/tests`` and
share the template
``skactiveml.tests.template_budget_manager.TemplateBudgetManager``. Specify
``bm_class``, ``init_default_params``, and ``query_by_utility_params`` in the
template setup, then adjust the generated tests as needed.


Multi-annotator Pool-based Query Strategies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All query strategies are stored in
``skactiveml/pool/multiannotator/*.py`` and inherit
``skactiveml.base.MultiAnnotatorPoolQueryStrategy``.

The class must implement the following methods:

+------------+----------------------------------------------------------------+
| Method     | Description                                                    |
+============+================================================================+
| ``__init__`` | Method for initialization.                                   |
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

Use ``self._validate_data`` (implemented in the superclass).
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

Multi-annotator strategy tests live in
``skactiveml/pool/multiannotator/tests`` and use ``unittest.TestCase``. Test
constructor and query parameters explicitly and keep the query-result contract
for sample-annotator pairs covered by the existing neighboring tests.


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
| ``sample_weight``, optional       | Contains weights for the training samples'   |
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

General Advice
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

General Advice
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

General Advice
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

General Advice
^^^^^^^^^^^^^^

Usually, this method is already implemented by the superclass. If the
superclass method is overwritten, ensure that it checks the parameters
and that the classifier has been fitted.

.. _testing-5:

Testing
~~~~~~~

Classifier tests live in ``skactiveml/classifier/tests`` and reuse the template
``skactiveml.tests.template_estimator.TemplateSkactivemlClassifier``. Specify
``estimator_class``, ``init_default_params``, ``fit_default_params``, and
``predict_default_params`` in the template setup, then adjust the generated
tests as needed.

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
| ``predict``       | Method predicting the labels for samples.               |
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
| ``y``                             | Contains the labels of the training          |
|                                   | samples. Missing labels are represented by   |
|                                   | the attribute ``missing_label``. Usually,    |
|                                   | ``y`` is a column array except for multi-    |
|                                   | target regressors, which expect a matrix     |
|                                   | with columns for each target type.           |
+-----------------------------------+----------------------------------------------+
| ``sample_weight``, optional       | Contains weights for the training samples'   |
|                                   | labels. Must have the same shape as ``y``.   |
+-----------------------------------+----------------------------------------------+

Returns:

+-----------------------------------+-----------------------------------+
| Parameter                         | Description                       |
+===================================+===================================+
|``self``                           | The fitted regressor object.      |
+-----------------------------------+-----------------------------------+

.. _general-advice-11:

General Advice
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
| ``y_pred``                        | The estimated labels per sample,  |
|                                   | as an array of shape              |
|                                   | ``(n_samples,)``.                 |
+-----------------------------------+-----------------------------------+

.. _general-advice-12:

General Advice
^^^^^^^^^^^^^^

Check parameter ``X`` regarding its shape, i.e., use method
``skactiveml.utils.check_n_features`` to ensure a correct number of
features. Check that the regressor has been fitted. If the regressor is a
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
| ``y``                             | Contains the true labels for each            |
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

General Advice
^^^^^^^^^^^^^^

Usually, this method is already implemented by the superclass. If the
superclass method is overwritten, ensure that it checks the parameters
and that the regressor has been fitted.

.. _testing-6:

Testing
~~~~~~~

Regressor tests live in ``skactiveml/regressor/tests`` and reuse the template
``skactiveml.tests.template_estimator.TemplateSkactivemlRegressor``. Specify
``estimator_class``, ``init_default_params``, ``fit_default_params``, and
``predict_default_params`` in the template setup, then adjust the generated
tests as needed.


Examples
--------

Two of our main goals are to make active learning more understandable and
improve our framework's usability. Therefore, we require an example for each
query strategy. Add a JSON file to an existing example directory under
``docs/examples/``. Each directory must contain a ``README.rst``.
The documentation generator recursively scans these directories and creates
the example pages, notebooks, and strategy overview. Existing directory names
are retained for stable published paths. Reader-facing groups use Pool or
Stream, then Classification or Regression.

Each JSON file contains a list of example entries with the following fields:

+------------------+----------------------------------------------------------+
| Entry            | Description                                              |
+==================+==========================================================+
| ``class``        | Query strategy's class name.                             |
+------------------+----------------------------------------------------------+
| ``package``      | Name of the sub-package (e.g., pool).                    |
+------------------+----------------------------------------------------------+
| ``method``       | Query strategy's official name.                          |
+------------------+----------------------------------------------------------+
| ``category``     | The strategy's methodological category. Examples are     |
|                  | Informativeness, Representativeness, Hybrid, Baseline,   |
|                  | and Wrapper.                                             |
+------------------+----------------------------------------------------------+
| ``template``     | Selects the rendering code for the example.              |
|                  | Supported templates include:                             |
|                  | ``examples/template_pool.py``,                           |
|                  | ``examples/template_pool_multilabel.py``,                |
|                  | ``examples/template_pool_regression.py``,                |
|                  | ``examples/template_stream.py``, and                     |
|                  | ``examples/template_pool_batch.py``.                     |
+------------------+----------------------------------------------------------+
| ``tags``         | Supported setups and batching behavior for the           |
|                  | documented method variant. Tags include ``pool``,        |
|                  | ``stream``, ``single-annotator``, ``multi-annotator``,   |
|                  | ``classification``, ``multi-label``, and ``regression``. |
+------------------+----------------------------------------------------------+
| ``example_setup``| The task, target type, and annotation type shown.        |
|                  | Pool examples also specify the demonstrated selection    |
|                  | rule. See the field values below.                        |
+------------------+----------------------------------------------------------+
| ``batch_size``   | Pool query batch size, stored as a string, e.g., ``"3"``.|
|                  | Shared by the executable example and its batch caption.  |
|                  | Omit for native stream examples.                         |
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

The strategy overview obtains complete task, target type, and annotation type
combinations from ``_target_capabilities``. The ``tags`` restrict these to the
documented method variant. They describe supported use cases. The ``template``
field selects the rendering code, and ``example_setup`` describes the setup
actually illustrated by the demonstration. For variants whose constructor
parameters change their capabilities, add ``capability_params``
with those parameters as a JSON object, for example
``{"method": "expected_average_precision"}``. These parameters are used for
capability inspection, independently of the example code in ``init_qs``.

Each ``example_setup`` is a small JSON object using these fields:

* ``task``: ``"classification"`` or ``"regression"``.
* ``target_type``: ``"single-output"`` or ``"multi-label"``.
* ``annotation_type``: ``"single-annotator"`` or ``"multi-annotator"``.
* ``selection`` for pool examples: ``"top-k-batch"``, ``"diverse-batch"``,
  ``"random-batch"``, or ``"single-sample"`` for the strategy's batch
  selection rule. Omit this field for native stream examples.

For example, a pool demonstration of diverse multi-label selection uses:

.. code-block:: json

   {
       "example_setup": {
           "task": "classification",
           "target_type": "multi-label",
           "annotation_type": "single-annotator",
           "selection": "diverse-batch"
       },
       "batch_size": "1"
   }

Keep these values consistent with the template, estimator, and query arguments.
The pool ``batch_size`` is used for execution and, if it is larger than one,
for the card caption. Do not maintain a separate caption count. It counts
selected samples for a single annotator, including complete multi-label
vectors, or sample–annotator
pairs for multiple annotators. If a demonstration collects stream arrivals
before querying a pool strategy, distinguish its arrival batch size from the
number of acquired samples. Native stream examples describe their budget and
update workflow and omit pool selection and batch-size metadata.
Choose a count compatible with the selected template's plotting layout.
For example, the ordinary and multi-label pool templates show one acquisition
per cycle, while the classification batch template shows four.

Pool entries must specify their batching behavior using ``top-k-batch``,
``diverse-batch``, or ``random-batch``. Use ``single-sample`` for an entry
restricted to one sample per cycle. It cannot be combined with batch tags.
Stream entries do not carry pool batching tags. An optional ``batch_note``
explains qualifications such as random selection within tree leaves.
The overview shows that generic pool wrappers' target types depend on the
wrapped strategy. The sub-sampling wrapper also inherits its batching rule.

The visualization landing page groups examples using ``example_setup``,
independently of the capabilities listed in ``tags``. An example with
``"target_type": "single-output"`` remains a single-output demonstration even
if its strategy also supports the multi-label target type. The strategy
overview describes
supported capabilities and links to the API class page, whose gallery lists
every associated demonstration. The layout lives in
``docs/_templates/visualizations.rst.jinja``. Its mini-galleries use the
existing generated scripts and keep example URLs stable. Cards are sorted
alphabetically by their displayed example titles within each task block.

Testing and Code Coverage
-------------------------

Please ensure test coverage is close to 100%. The current code coverage can be
viewed `here <https://app.codecov.io/gh/scikit-activeml/scikit-activeml>`__.

Documentation
-------------

Guidelines for writing documentation in scikit-activeml adopt the
`scikit-learn guidelines <https://scikit-learn.org/stable/developers/contributing.html#guidelines-for-writing-documentation>`__
used by scikit-learn.

Use Pool and Stream for sample availability and Classification and Regression
for the prediction task. Name the two setup dimensions target type
(single-output or multi-label) and annotation type (single-annotator or
multi-annotator), as ``TargetSpec`` does. They qualify a task and are not
prediction tasks themselves. Call every value of ``y`` a label. Use
annotation only for the act of labeling, as in annotation request or
annotation session, and annotator for the person who provides labels. Use
``multi-label`` and ``multi-annotator`` in prose and headings. Preserve literal
Python identifiers and package names such as ``multilabel`` and
``multiannotator``.

Use Sphinx ``:doc:`` and ``:ref:`` links within the documentation so readers
stay in the version they are browsing. Notebook links to project
documentation use relative document-source paths, such as
``../../target_semantics.rst``. Sphinx renders them as HTML links; Colab copies
resolve them to HTML URLs for their documentation version. Preserve public page
paths, notebook/example filenames, and section fragments. When changing a heading,
retain its previous fragment using an explicit target. Tutorials keep a direct
hidden toctree for their notebooks, and example navigation retains its existing
generated URLs.

In-depth Tutorial Design
~~~~~~~~~~~~~~~~~~~~~~~~

Tutorial notebooks help a machine learning audience design experiments and
annotation workflows with scikit-activeml. They demonstrate the framework's
capabilities through concise, adaptable code. Assume familiarity with Python,
NumPy, supervised learning, and the scikit-learn estimator interface. Explain
active learning concepts where they help readers use the framework.

Use a flexible layout suited to one scenario or study question:

* **Opening:** briefly state the goal, the capabilities demonstrated, and
  expected knowledge. The reader-facing roadmap belongs in ``tutorials.rst``.
* **Main example:** keep classifier and strategy construction, querying,
  annotation, fitting, and relevant evaluation easy to locate and adapt.
  Explain framework-specific choices beside their use. Include necessary
  support code within the notebook, grouping longer helpers before the
  relevant workflow. Put experiment settings together and method parameters
  beside their construction. Use explicit variations or straightforward
  comparison loops rather than a general experiment runner.
* **Outlook:** briefly suggest useful adaptations and selected next steps.

Results should serve the notebook's goal. A workflow demonstration may use
qualitative outputs; a comparative study needs appropriate evidence and a
brief explanation of its particular evaluation protocol. Interpret results
within that experiment's scope. Equations, reference sections, quantitative
comparisons, and summary tables are not required elements.

For published studies, save the reference outputs with the configuration
that produced them. Expensive notebooks default to ``QUICK_RUN=False`` and
offer cheaper execution through ``QUICK_RUN=True``. Keep their strategy lists
visible so readers can comment out methods. Simpler notebooks use ordinary
editable settings. Preserve each substantive comparison and interface
variation in the saved results, and explain that changed settings produce
different outputs. Keep installation guidance short. Activate cuML before
estimator imports where used. Use throttled progress displays, with elapsed
time confined to those displays. Keep environment inventories, stage timings,
and validation code outside the notebook. Saved outputs show no warnings.
Fix the cause of a warning that signals a mistake, such as a mismatched
``missing_label``. Filter an expected, informational warning individually
by its message, with a comment that gives the reason. Do not hide all
warnings or disable logging.

Use small, self-contained plotting settings: 11-point axes and ticks,
12-point titles, and 10-point legends. Where colors identify query strategies,
use gray for random sampling and blue for uncertainty sampling. Use complete
legends and layouts suited to the scenario and documentation width. Keep class
colors consistent across scatter plots. Let metric-axis limits follow the
displayed curves and bands with modest padding. For repeated learning curves,
use light bands showing one sample standard deviation (``ddof=1``); single
runs need no variation bands. Numerical legend summaries can replace separate
score tables. State what those values average and avoid repeating them in
printed output. Label counts according to what was acquired, distinguishing
samples, multi-label rows, repeated labels, and stream arrivals. Explain
whether initial labels are included. Inspect the published rendering for
clipping and overlap.

Save annotation widget state when it reliably renders in the built HTML.
Check it in a clean browser without a kernel, including refresh, model/output
buffers, assets, and rendering errors. If embedding fails, capture an accurate
screenshot of the current interface. Caption the actual recorded state and
explain that callbacks require a live notebook. Check callback behavior
separately from unattended execution.

Before publishing executable changes that affect results or saved outputs,
run the published configuration from a fresh kernel. Check notebook validity,
code/output consistency, and rendered figures outside the tutorial. Use
smaller configurations during development and proportionate checks for prose
changes. Sphinx builds display saved tutorial outputs; they do not execute
these notebooks.

Building the Documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~

To ensure your documentation examples execute successfully and the pages are
well formatted, activate the ``scikit-activeml`` environment and build them
using Sphinx. The configuration queries release tags and external inventories,
so a network connection is required:

.. code:: bash

   python -m sphinx -b doctest docs docs/_build/doctest
   python -m sphinx -b html docs docs/_build

Issue Tracking
--------------

We use `GitHub Issues <https://github.com/scikit-activeml/scikit-activeml/issues>`__
as our issue tracker. If you believe you have found a bug in
scikit-activeml, please report it there. Documentation bugs can also be reported.

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
| ``bug``          | Something isn't working                   |
+------------------+-------------------------------------------+
| ``enhancement``  | Request for a new feature                 |
+------------------+-------------------------------------------+
| ``documentation``| Improvement or additions to documentation |
+------------------+-------------------------------------------+
| ``question``     | General questions                         |
+------------------+-------------------------------------------+
