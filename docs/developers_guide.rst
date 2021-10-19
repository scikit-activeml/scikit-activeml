Developer's Guide
=================

.. toctree::

This page will be used to document the discussed guidelines and should help new contributors to adhere to the same set guidelines.

Conformity with scikit-learn
----------------------------
As the selection streategies are inheriting from BaseEstimator, the selection strategies should conform to the initialization scheme proposed by scikit-learn, i.e., there shall be no code in the __init__ function besides storing the attributes. All verification and transformation of inputs shall be done in the query (i.e., fit in scikit-learn) function.

Code style and linting
----------------------
As this library conforms to the convention of scikit-learn, the code should conform to `PEP 8 Sytle Guide for Python Code <https://www.python.org/dev/peps/pep-0008/>`_. For linting, the use of `flake8 <https://flake8.pycqa.org/en/latest/>`_ is recommended.

Homogeneous naming scheme
-------------------------
To simplify the use of this library, a homogeneous naming scheme is important. Thus, the following attribute and parameter names are to be used when appropriately:

* Parameters of `__init__()`:
    * all optional
    * `clf`
    * `classes`
    * `missing_label`
    * `random_state`
    * `metric`
    * `cost_matrix`
    * `budget_manager` (stream)
* Parameters of `query()`:
    * `X_cand`
    * `X`
    * `y`
    * `X_eval`
    * `y_eval`
    * `A_cand` (multiannot) - matrix wer was noch labeln kann
    * (kernel kram...)
    * `sample_weight_cand`
    * `sample_weight` (same dim as y)
    * `sample_weight_eval`
    * `simulate` (esp stream)
    * `batch_size`
    * `return_utilities`

General Unittest for Query Strategies (`test_pool.py`):
-------------------------------------------------------

1. Querying of every method is tested with standard configurations with `0`, `1`, and `5` initial labels.

2. For every class `ExampleQueryStrategy` that inherits from `SingleAnnotPoolBasedQueryStrategy` (stored in `_example.py`), it is automatically tested if there exists a file `test/test_example.py`. It is necessary that both filenames are the same. Moreover, the test class must be called `TestExampleQueryStrategy(unittest.TestCase)`

3. Every parameter in `__init__()` will be tested if it is written the same as a class variable.

4. Every parameter `arg` in `__init__()` will be evaluated if there exists a method in the testclass `TestExampleQueryStrategy` that is called `test_init_param_arg()`.

5. Every parameter `arg` in `query()` will be evaluated if there exists a method in the testclass `TestExampleQueryStrategy` that is called `test_query_param_arg()`.

Handling of unlabeled instances
-------------------------------
Active learning generally uses labeled and unlabeled instances. To simplify the data handling, the SkactivemlClassifier is able to handle unlabeled data. The unlabeled data is marked as such by setting corresponding entry in y (the label) during fitting to missing_label which is set during the initialization of the classifier. All classifier and the wrappers (e.g. for scikit-learn classifiers) are compatible with unlabeled instances.

Handling of Batch / Non-Batch scenarios
---------------------------------------
All query strategies, except the stream based approaches, support the batch scenario. All strategies that are not explicitly designed to support the batch scenario shall employ a greedy strategy to iteratively select instances to fill the queried batch. The query methods have a batch_size parameter to specify the number of instances to queried instances. If the batch size is dynamic, the batch_size parameter shall be set to 'adaptive'. The utilities that are returned, when return_utilities is set to true, have the following shape: batch_size x n_cand, reflect the utilities for each individual acquisition.

Handling of pool-based, stream-based AL and membership query synthesis
----------------------------------------------------------------------
* separate packages and classes follow \*PoolBasedQueryStrategy, \*StreamBasedQueryStrategy, \*MembershipQuerySynthesis

Handling of active learning with multiple annotators
----------------------------------------------------
* create MultiAnnotPoolBasedQueryStrategy()
    * def query(self, X_cand, \*args, A_cand=None, return_utilities=False, \*\*kwargs)
    * is independent, not compatible with SAPBQS
    * return_utilities shape: batch_size x \*shape(A_cand)

Handling of uncertain oracles
-----------------------------
* will be considered when labels are added to y
* oracles with errors (oracles with confidences can be used with sample_weight)
* e.g. using annotlib
* evtl. sample_weight for X, y in query()
* CAUTION: sample_weight_cand for X_cand and sample_weight for X

Transductive and inductive active learning
------------------------------------------
* possible with distinct X_cand list independent X, y, E_eval, etc.
* explain with notebooks

Classification and regression
-----------------------------
* separate methods for classifiers and regressors, identify with tags (see sklearn) [estimator_type]

Evaluation
----------
* extra package evaluation with some extra functions (visualization, etc.)

Stopping criteria
-----------------
* extra package stopping
* separate inductive from transductive?

Json example file structure
---------------------------
* The example can be modified by modifying the template.py file
* One json file for each module.
* Organized in the same folder and naming structure as the packages.
* The json file should contain a list with one entry for each example.
* Each example entry it self is a dictionary with the following keys:
    * "class": The name of the class for which the example is intended. Multiple use is possible.
    * "method": The method for which the example is intended. Each method-class combination should only be used once.
    * "refs" (optional): A list of references to bibliographic entries in the 'refs.bib' file.
    * "title": The title of the example page.
    * "text": Every key that starts with 'text' will be formatted as a paragraph in the example.
    * "code": Every key that starts with 'code' will be formatted as a python code block in the example.
    * "bp": Each key that starts with 'bp' is formatted as Python code and added on a specific line in the example. You need to define the line by adding the key starting with '#_' to the appropriate line in the template.py file.
    * "sequence": A list that contains the order in which the blocks are shown. Possible blocks are:
        * 'title': Shows the title.
        * 'text': Shows the specified text paragraph.
        * 'code': Shows the specified code block.
        * 'refs': Shows the references.
        * 'plot': Shows the example plot including the generated code of the AL cycle.
    * "categories": A dictionary with a survey as key and the depending category as value.
    * "init_params": A dictionary which contains the parameters used to initialize the query strategy.
    * "query_params": A dictionary which contains the parameters used to call the 'query' method.
    * "clf" (optional): The initialization string of the classifier used as model.