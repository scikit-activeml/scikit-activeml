Developing Components
=====================

This page covers the code in ``skactiveml``. Query strategies decide which
samples, and with multiple annotators which annotators, to ask for labels.
Classifiers and regressors learn from partially labeled data, so that an
active learning cycle can start with zero labels. :doc:`/concepts` explains
the pool-based and stream-based scenarios and the single-annotator and
multi-annotator settings.

The figure shows the base classes in ``skactiveml.base``. Lines connect each
class to the class it inherits from. Gray boxes are scikit-learn classes,
which appear for identification only and do not imply endorsement. Select a
class to open its API reference.

.. The tree is embedded inline so that its text follows the light and dark
   theme. Its title and description provide the alternative text.

.. container:: structure-figure

   .. raw:: html
      :file: ../logos/base-classes.svg

General Coding Conventions
--------------------------

This library follows the conventions of
`scikit-learn <https://scikit-learn.org/stable/developers/develop.html#coding-guidelines>`__
and the `PEP 8 <https://www.python.org/dev/peps/pep-0008/>`__ style guide.
The pre-commit hooks check the formatting with black and lint the code with
flake8. To format and lint by hand:

.. code:: bash

   black skactiveml
   flake8 skactiveml/ --ignore=E203,W503

.. _contributing-adding-a-component:

.. _contributing-adding-a-query-strategy:

Adding a Component
------------------

A new query strategy, budget manager, classifier, or regressor needs these
steps:

1. Implement the class in a private module of its subpackage, such as
   ``skactiveml/pool/_my_strategy.py``. Derive the module name from the class
   name in snake case and use the same stem for its test module. A module
   that intentionally groups a family of classes may have a conceptual or
   acronym-based name.
2. Import the class in the subpackage's ``__init__.py`` and add it to
   ``__all__``. The API reference, the example generator, and the tests find
   components through ``__all__``.
3. If the class supports other target combinations than its base class
   declares, declare them in ``_target_capabilities``, as
   :ref:`contributing-target-semantics` describes.
4. Write the docstrings in the numpydoc style. Cite the method's papers in a
   ``References`` section with numbered entries, such as ``.. [1]``.
5. Add tests. Reuse the test template of the component's kind where one
   exists, as the sections below describe. A template requires a test named
   ``test_<method>_param_<name>`` for each parameter of the methods it
   checks, such as ``test_init_param_random_state`` or
   ``test_query_param_batch_size``. For most of these methods, it also
   requires a test named after the method, such as ``test_query`` or
   ``test_fit``. The template provides the tests for the shared parameters.
   Add the tests for your own. The sections below name the template tests
   that you must override.
6. Query strategies only: add the papers' BibTeX entries to
   ``docs/refs.bib`` and an example entry, as :ref:`contributing-examples`
   describes.
7. Run the tests, check the coverage as :ref:`contributing-testing`
   describes, and build the documentation as
   :ref:`contributing-building-docs` describes.

Where Helpers Live
------------------

Keep helper functions close to the code that uses them. A helper used in one
file should stay in that file. If several files in one subpackage use it, put
it in a private file in that subpackage. If several subpackages use a general
helper and it does not import from ``skactiveml.base``, put it in a clearly
named private file under ``skactiveml/utils``.

Do not import helper functions or constants from ``skactiveml.base``. That
file is for base classes. If a helper must import a base class, keep it in the
subpackage that uses it. ``skactiveml/pool/_target.py`` and
``skactiveml/stream/_target.py`` are examples. Put exception classes in
``skactiveml/exceptions.py``.

A base class contains these parts:

* Its public contract.
* The methods it calls itself.
* Declarations with conservative defaults, such as ``_target_capabilities``.
* Behavior that all its subclasses share.
* Methods that work with the state it defines, such as the label state of
  ``SkactivemlClassifier``.

Put other helpers, such as those that only check passed arguments or that
need attributes only some subclasses define, in functions placed by the rules
above. Add a query-strategy base class only when it contains behavior shared
by its subclasses. Do not add one only to share a constant or to help tests
find strategies.

.. _contributing-target-semantics:

Target Semantics and Compatibility
----------------------------------

Labels and target semantics have an explicit public contract. Read the
:doc:`/label_formats` pages before changing estimators, query strategies,
label helpers, or multi-annotator components. In particular:

- Resolve the target meaning with
  :func:`~skactiveml.utils.resolve_target_spec`. Do not infer it from the
  array shape or the values alone. Code that only needs the number of samples
  may use the shape, such as a wrapper that treats each row of a
  two-dimensional ``y`` as one sample.
- Declare the exact target combinations a component supports in its
  ``_target_capabilities``.
- Override ``_declared_classes`` in a classifier wrapper that takes its
  ``classes`` from a wrapped estimator, as ``SlidingWindowClassifier`` does,
  so that query strategies see this vocabulary before fitting.
- Keep the fitted ``target_spec_`` on estimators and wrappers that retain a
  fitted target specification. Query strategies resolve the specification in
  each query and do not retain it.
- Add tests for every supported target combination and for invalid or
  unsupported combinations, including ambiguous two-dimensional labels and
  complete-row requirements for multi-label data.
- Operate on encoded labels for multi-label data, because each label output
  may declare its own binary class vocabulary, such as
  ``[["no", "yes"], ["off", "on"]]``.

When adding multi-label support to a pool strategy:

- Set ``query_default_params_clf_multilabel`` in its test setup. The template
  then runs the shared contract for custom class vocabularies.
- In ``test_multilabel_contracts.py`` in ``skactiveml/pool/tests/``, add the
  strategy to the inventory group that matches its use of probabilities, and
  add its test class to ``DEFAULT_VOCABULARY_FIXTURE``. If the strategy needs
  auxiliary inputs for the vocabulary contract, override
  ``_multilabel_custom_vocabulary_params`` in its test class and list the
  test class with the reason in ``SPECIALIZED_VOCABULARY_FIXTURE`` instead.
- A strategy that uses the classifier's probabilities needs a
  ``SklearnClassifier`` in its multi-label test parameters, so that the tests
  can vary its ``proba_format``. It must reject probabilities with the wrong
  number of outputs, for example with ``_canonicalize_multilabel_probas`` from
  ``skactiveml/utils/_validation.py``.
- Tag at least one of its example entries ``multi-label``, as
  :ref:`contributing-tags` describes.
- Add the strategy to the :ref:`multi-label strategy inventory
  <multilabel-strategy-inventory>`, which groups the strategies by their use
  of multi-label data.

Query Strategies
----------------

All query strategies inherit from :class:`~skactiveml.base.QueryStrategy`,
which inherits from scikit-learn's ``BaseEstimator``. Its optional
``random_state`` parameter controls reproducibility, and the abstract
``query`` method contains the sample selection.

* Draw all random numbers from ``self.random_state_``, including the seeds of
  sub-estimators. ``_validate_data`` sets it from ``random_state``.
* ``__init__`` only stores its arguments. The documentation build and the
  tests create each strategy with ``None`` for every required argument and
  read its ``_target_capabilities``, so this property must accept ``None``
  values.

.. _contributing-strategy-criteria:

Criteria for New Query Strategies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We add a query strategy if it meets these criteria:

* A peer-reviewed publication, such as a conference or journal paper,
  describes the method. An arXiv preprint with more than 30 citations on
  Google Scholar also qualifies.
* Its implementation comes with tests that cover every line.
* It comes with an example for the documentation.

Open a feature request before you start, so that the maintainers can confirm
that the strategy fits the library. For other preprints, the feature request
explains why the method belongs in the library, for example with comparisons
against strategies in scikit-activeml. The maintainers decide.

Single-annotator Pool-based Query Strategies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These strategies live in ``skactiveml/pool/``, inherit from
:class:`~skactiveml.base.SingleAnnotatorPoolQueryStrategy`, and implement
``__init__`` and ``query``. The base class documents the full signature of
``query``. ``UncertaintySampling`` and ``RandomSampling`` show the complete
flow with and without an estimator.

* Use the standard parameter names where they apply: ``missing_label``
  (``np.nan`` by default), ``random_state``, ``prior`` for prior
  probabilities, ``method`` for a class that implements several methods, and
  ``cost_matrix``.
* ``query`` takes ``X`` and ``y``, with ``missing_label`` for the unlabeled
  samples, and optionally ``candidates``, ``batch_size``, and
  ``return_utilities``. The ``candidates`` are ``None`` for the unlabeled
  samples in ``X``, indices of samples in ``X``, or a separate feature
  matrix.
* Get the candidate samples with ``self._transform_candidates``. It also
  returns their indices in ``X``, or ``None`` for a separate feature matrix.
  If the strategy needs candidates from ``X``, pass
  ``enforce_mapping=True``, so that it raises a ``MappingError`` for a
  separate feature matrix.
* Pass the resolved target type to ``self._validate_data`` and
  ``self._transform_candidates``. A strategy with an estimator passes
  ``target_spec.target_type``.
* ``query`` returns the indices of the selected samples. They index ``X`` if
  ``candidates`` is ``None`` or an index array, and ``candidates`` if it is a
  feature matrix. If requested, ``query`` also returns the utilities with one
  row per selection of the batch. Samples that are not candidates, including
  the labeled ones, get the utility ``np.nan``.
* Call ``self._validate_data`` exactly once for the structural validation of
  ``X`` and ``y``. Target resolution is a separate step and may come first.
* If the strategy takes one estimator that it may refit, use
  ``_fit_and_resolve_estimator_target_spec`` from
  ``skactiveml/pool/_target.py``. It validates the estimator and returns
  ``(estimator, target_spec)``. If refitting is requested, such as with
  ``fit_clf=True``, the estimator is a fitted clone.
* A strategy that uses no estimator for ``y`` takes the target type from a
  ``target_type`` parameter if it supports multi-label data, as
  :doc:`/label_formats/resolution` describes. It resolves the parameter with
  ``self._resolve_query_target_type(y)``.
* Compute the utilities in a separate public function. A naive batch strategy
  uses ``simple_batch`` from ``skactiveml.utils`` to select the batch and set
  the utilities.

The tests live in ``skactiveml/pool/tests`` and reuse
``TemplateSingleAnnotatorPoolQueryStrategy`` from
``skactiveml.tests.template_query_strategy``. Set ``qs_class``,
``init_default_params``, and, depending on the task,
``query_default_params_clf`` or ``query_default_params_reg`` in the template
setup. Then add these tests:

* ``test_query_param_clf`` or ``test_query_param_reg``, if ``query`` takes
  ``clf`` or ``reg``. Override the template's test and call it with the
  strategy's invalid estimators as ``test_cases``, or with
  ``test_cases=[]`` if the strategy accepts every estimator. Otherwise, the
  template raises a ``NotImplementedError``.
* ``test_query``, which checks the strategy's results. The template requires
  it but does not provide it.

Single-annotator Stream-based Query Strategies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These strategies live in ``skactiveml/stream/``, inherit from
:class:`~skactiveml.base.SingleAnnotatorStreamQueryStrategy`, and implement
``__init__``, ``query``, and ``update``. Each strategy handles its budget
internally or through a budget manager.

* Use the standard parameter names ``random_state``, ``budget`` for the share
  of labels the strategy may query, and ``budget_manager``.
* A strategy that takes ``clf`` declares single-output classification as its
  only target combination. A strategy that uses no labels declares the three
  task-agnostic combinations: single-output and multi-label classification,
  and single-output regression.
* ``query`` takes the ``candidates`` and returns the indices of the queried
  candidates and, if ``return_utilities`` is ``True``, the utilities of all
  candidates. A strategy that uses a classifier also takes ``clf`` and
  optionally ``X``, ``y``, ``sample_weight``, and ``fit_clf``.
* ``query`` must not change the state of the strategy, including ``budget``,
  ``budget_manager``, and ``random_state``, so that several samples can be
  assessed with the same state. Only its first call creates the state it
  needs, such as ``random_state_`` and ``budget_manager_``. ``update``
  advances the state.
* In ``query``, call ``self._validate_data`` and check ``X`` and ``y`` only
  once. Call ``_validate_X_y_sample_weight`` from
  ``skactiveml/stream/_validation.py``, followed by
  ``_fit_and_resolve_clf_target_spec`` from ``skactiveml/stream/_target.py``.
  The target helper returns ``(clf, target_spec)``, with a fitted clone when
  ``fit_clf=True``.
* ``update`` takes the ``candidates``, the ``queried_indices``, typically
  returned by ``query``, and ``budget_manager_param_dict`` with further
  parameters for the budget manager. A strategy without a budget manager
  accepts and ignores this dictionary.
* ``update`` must also work before the first ``query``. It validates its
  input before it changes the state, so that invalid input leaves the state
  unchanged. Check ``candidates`` with ``check_array``, then call
  ``_validate_budget_update`` from ``skactiveml/utils/_validation.py`` with
  ``candidates`` and ``queried_indices``. A strategy with a budget manager
  also passes ``budget_manager_param_dict``. A strategy without one
  validates its budget with ``self._validate_budget()``. ``update`` must not
  set ``n_features_in_``.

A query strategy with a budget manager calls ``_validate_budget_manager``
from ``skactiveml/stream/_validation.py`` in every ``_validate_data`` and
``update`` call. On first use, ``_validate_budget_manager`` creates the
working copy ``budget_manager_`` with ``check_budget_manager`` from
``skactiveml.utils``:

- With only a ``budget``, it creates the default budget manager with this
  budget.
- With only a ``budget_manager``, it uses a copy of it.
- With neither, it creates the default budget manager with its default
  budget.
- With both, it uses the budget manager and warns if the budgets differ.

Later calls apply changes of ``budget`` to a default ``budget_manager_``. The
strategy's ``update`` forwards the call with ``call_func`` from
``skactiveml.utils``, which passes only the arguments that the budget manager
accepts:

.. code:: python

   if budget_manager_param_dict is None:
       budget_manager_param_dict = {}
   call_func(
       self.budget_manager_.update,
       candidates=candidates,
       queried_indices=queried_indices,
       **budget_manager_param_dict,
   )

The tests live in ``skactiveml/stream/tests`` and reuse
``TemplateSingleAnnotatorStreamQueryStrategy`` from
``skactiveml.tests.template_query_strategy``. Set ``qs_class``,
``init_default_params``, and ``query_default_params_clf`` in the template
setup. The stream tests use ``query_default_params_clf`` for every strategy,
including those that support regression. Then override these tests:

* ``test_query_param_clf``, if ``query`` takes ``clf``, as for pool
  strategies.
* ``test_query``, which calls the template's test with the
  ``expected_output`` and ``expected_utilities`` of the strategy.

If the default budget manager's ``update`` needs more arguments, such as the
``utilities``, set them in ``self.update_params["budget_manager_param_dict"]``
after the template setup, as ``test_stream_probabilistic_al.py`` does.

Multi-annotator Pool-based Query Strategies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These strategies live in ``skactiveml/pool/multiannotator/``, inherit from
:class:`~skactiveml.base.MultiAnnotatorPoolQueryStrategy`, and implement
``__init__`` and ``query``. They select sample–annotator pairs.

* ``query`` takes ``X`` and ``y``, where ``y[i, j]`` holds the label of
  annotator ``j`` for sample ``i``, and optionally ``candidates``,
  ``annotators``, ``batch_size``, and ``return_utilities``.
* The ``candidates`` work as in pool strategies. Without them, every sample
  with an unlabeled sample–annotator pair is a candidate. The ``annotators``
  are ``None`` for all annotators, indices of the available annotators, or a
  boolean array of shape ``(n_candidates, n_annotators)`` that marks the
  candidate pairs.
* ``query`` returns one row per selected pair with the index of the sample,
  indexed as in pool strategies, and the index of the annotator. If
  requested, it also returns the utilities with one row per selection of the
  batch. Labeled pairs get the utility ``np.nan``.
* Call ``self._validate_data`` and check ``X`` and ``y`` only once. Fit a
  clone of the classifier if ``fit_clf`` is ``True``.
* A strategy that combines a single-annotator strategy with an estimate of
  the annotators' performance defines an aggregation function, evaluates the
  performance for each sample–annotator pair, and uses the
  ``SingleAnnotatorWrapper``.
* A greedy strategy computes the utilities in a separate function and uses
  ``simple_batch`` from ``skactiveml.utils`` to select the batch.

The tests live in ``skactiveml/pool/multiannotator/tests`` and use
``unittest.TestCase``. Test the constructor and query parameters explicitly
and keep the query-result contract for sample–annotator pairs covered by the
existing neighboring tests.

Budget Managers
---------------

Budget managers live in ``skactiveml/stream/budgetmanager/``, inherit from
:class:`~skactiveml.base.BudgetManager`, and implement ``__init__``,
``query_by_utility``, and ``update``. Stream strategies use them as the
previous section describes.

* ``query_by_utility`` takes the candidates' ``utilities`` from the query
  strategy and returns the indices of the candidates to query as a list. It
  checks the ``utilities`` and the ``budget`` with ``self._validate_data``.
* ``update`` takes the ``candidates`` and the ``queried_indices`` and
  advances the state, as the ``update`` of a query strategy does. Before it
  changes the state, it checks them with ``_validate_budget_update`` from
  ``skactiveml/utils/_validation.py``, without ``check_array``, and validates
  the ``budget`` with ``self._validate_data(np.array([]))``. Some budget
  managers take more arguments, such as the
  ``utilities`` of ``BalancedIncrementalQuantileFilter``.
* Both ``query_by_utility`` and ``update`` create the state needed for
  querying when called first. After that, ``query_by_utility`` leaves the
  acquisition state unchanged, including counters, thresholds, histories,
  and the random state. Repeated queries with the same utilities and
  parameters return the same indices until ``update`` advances the state.

The tests live in ``skactiveml/stream/budgetmanager/tests`` and reuse
``TemplateBudgetManager`` from ``skactiveml.tests.template_budget_manager``.
Set ``bm_class``, ``init_default_params``, and ``query_by_utility_params`` in
the template setup. Then override ``test_query_by_utility``, which calls the
template's test with the ``expected_output`` of the budget manager. The
template checks the state contract for both initialization paths.

Classifiers
-----------

Classifiers live in ``skactiveml.classifier``, and classifiers that learn from
multiple annotators in ``skactiveml.classifier.multiannotator``. They inherit
from :class:`~skactiveml.base.SkactivemlClassifier` and implement
``__init__``, ``fit``, ``predict_proba``, and, if needed, ``predict``.
``ParzenWindowClassifier`` shows the complete flow.

* Use the standard parameter names ``classes``, ``missing_label``,
  ``cost_matrix``, which requires ``classes``, and ``random_state``. Only
  classifiers that support multi-label classification take ``target_type``,
  as :doc:`/label_formats/multilabel` describes.
* ``fit`` takes ``X``, ``y`` with ``missing_label`` for missing labels, and
  optionally ``sample_weight`` with the shape of ``y``. ``y`` has the shape
  ``(n_samples,)`` for single-output classification and
  ``(n_samples, n_outputs)`` for multi-label classification. Multi-annotator
  classifiers expect ``(n_samples, n_annotators)``.
* In ``fit``, use ``self._validate_data`` to check the standard parameters of
  ``__init__`` and ``fit``. It returns the labels encoded as class indices,
  with ``-1`` for missing labels, and sets ``classes_``, ``target_spec_``,
  ``cost_matrix_``, and ``random_state_``. If ``classes`` is given, the
  classifier can be fitted on samples that all have missing labels. It then
  predicts uniform class-membership probabilities. Handle missing labels in
  all other cases too.
* In ``predict_proba``, check that the classifier is fitted and check the
  number of features with ``check_n_features`` from ``skactiveml.utils``. A
  :class:`~skactiveml.base.ClassFrequencyEstimator` inherits
  ``predict_proba`` and implements ``predict_freq`` instead.
* The base class implements ``predict`` through ``predict_proba`` and
  ``score`` as the mean accuracy. An override of ``predict`` must handle
  imbalanced costs and missing labels. An override of ``score`` checks its
  parameters and that the classifier is fitted.

The tests live in ``skactiveml/classifier/tests`` and reuse a template from
``skactiveml.tests.template_estimator``: ``TemplateSkactivemlClassifier``, or
``TemplateClassFrequencyEstimator`` for a ``ClassFrequencyEstimator``. Set
``estimator_class``, ``init_default_params``, ``fit_default_params``, and
``predict_default_params`` in the template setup, then adjust the generated
tests as needed. The tests of multi-annotator classifiers live in
``skactiveml/classifier/multiannotator/tests`` and reuse
``TemplateMultiAnnotatorClassifier``.

Regressors
----------

Regressors live in ``skactiveml.regressor``. They inherit from
:class:`~skactiveml.base.SkactivemlRegressor` and implement ``__init__``,
``fit``, and ``predict``.

* Use the standard parameter names ``missing_label`` and ``random_state``.
* ``fit`` takes ``X``, ``y`` of shape ``(n_samples,)`` with ``missing_label``
  for missing labels, and optionally ``sample_weight`` with the shape of
  ``y``. Regressors reject labels with several columns.
* In ``fit``, use ``self._validate_data`` to check the standard parameters of
  ``__init__`` and ``fit``. A regressor fitted on samples that all have
  missing labels predicts zero. Handle missing labels in all other cases too.
* In ``predict``, check that the regressor is fitted and check the number of
  features with ``check_n_features`` from ``skactiveml.utils``. ``predict``
  returns an array of shape ``(n_samples,)``. A
  :class:`~skactiveml.base.ProbabilisticRegressor` inherits ``predict`` and
  implements ``predict_target_distribution`` instead.
* The base class implements ``score`` as the R² score. An override checks its
  parameters and that the regressor is fitted.

The tests live in ``skactiveml/regressor/tests`` and reuse
``TemplateSkactivemlRegressor`` from ``skactiveml.tests.template_estimator``,
or ``TemplateProbabilisticRegressor`` for a ``ProbabilisticRegressor``.
Set ``estimator_class``, ``init_default_params``, ``fit_default_params``, and
``predict_default_params`` in the template setup, then adjust the generated
tests as needed.

.. _contributing-testing:

Testing and Code Coverage
-------------------------

CI requires that the tests cover every line of ``skactiveml``, both with the
oldest supported and with the newest available versions of the optional
dependencies. Code that depends on a library version must be covered under
both. To list the lines that the tests miss, run:

.. code:: bash

   OMP_NUM_THREADS=1 python -m pytest -n 6 --cov-report=term-missing:skip-covered

CI checks this coverage with the optional dependencies, PyTorch, and a Java
runtime installed. Without the optional dependencies, the tests leave out the
skorch, River, and CapyMOA integrations, so check that the lines you changed
are covered and let CI check the rest. The coverage of the published branches
is on `Codecov <https://app.codecov.io/gh/scikit-activeml/scikit-activeml>`__.

Keep test helpers in the narrowest module that owns them. A helper used by one
test module stays in that module. Put broadly reused assertions and fixture
helpers in ``skactiveml.tests.utils``. Put inherited query-strategy contract
behavior in ``skactiveml.tests.template_query_strategy``.
