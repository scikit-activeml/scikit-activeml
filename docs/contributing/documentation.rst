Documentation and Examples
==========================

The documentation consists of the pages in ``docs/``, the API reference from
the docstrings, the example pages generated from JSON entries, and the
tutorial notebooks. This page explains how to write and build them.

Writing Style
-------------

Follow the
`scikit-learn guidelines <https://scikit-learn.org/stable/developers/contributing.html#guidelines-for-writing-documentation>`__
for writing documentation, with these additions:

* Use *pool* and *stream* for how samples arrive, and *classification* and
  *regression* for the prediction task.
* Name the two setup dimensions target type (single-output or multi-label)
  and annotation type (single-annotator or multi-annotator), as
  ``TargetSpec`` does. They qualify a task and are not prediction tasks
  themselves.
* Call every value of ``y`` a label. Use annotation only for the act of
  labeling, as in annotation request or annotation session, and annotator for
  the person who provides labels.
* Write ``multi-label`` and ``multi-annotator`` in prose and headings. Keep
  literal Python identifiers and package names, such as ``multilabel`` and
  ``multiannotator``.
* Capitalize headings as ``README.rst`` does. Capitalize the major words,
  and lowercase articles, coordinating conjunctions, and prepositions such as
  of, to, and with. In a hyphenated compound, capitalize only the first part,
  as in Single-annotator Pool-based Query Strategies. Method names keep their
  published form, such as Query-by-Committee.
* Write docstrings in the numpydoc style and cite papers in a numbered
  ``References`` section, as :ref:`contributing-adding-a-component`
  describes.

Links
-----

Use Sphinx ``:doc:`` and ``:ref:`` links within the documentation, so that
readers stay in the version they are browsing. Notebooks link to
documentation pages with relative source paths, such as
``../../label_formats.rst``. These paths start from
``docs/generated/tutorials/``, where the build copies the notebooks. Sphinx
turns them into links, and the Colab copies get links to the same
documentation version if the paths start with ``../`` and end in ``.rst`` or
``.html``, optionally with an anchor. Notebooks link to other tutorials by
their file name, such as ``00_pool_getting_started.ipynb``.

Once a page is published for the latest release or the development version,
links to it should keep working:

* Keep page paths and the file names of notebooks and examples. A removed or
  moved page gets a redirect page at its old path in ``docs/_extra/``, which
  the build copies to the website as it is. A moved tutorial needs two
  redirects, because the build also publishes its Colab copy. For an example,
  see ``22_river_classifier.html`` in ``docs/_extra/generated/tutorials/`` and
  ``docs/_extra/generated/tutorials_colab/``.
* RST pages keep their section anchors. When you rename a heading, put a
  label with the old anchor above it:

  .. code-block:: rst

     .. _old-heading:

     New Heading
     -----------

  When a section moves to another page, add its old anchor to the
  forwarding script of the old page, as ``docs/contributing.rst`` does. The
  Strategy Overview keeps old anchors through ``OVERVIEW_LEGACY_IDS`` in
  ``docs/generate.py``. The Visualizations pages keep them through the
  ``anchor()`` macro in ``visualization_macros.rst.jinja``.
* Notebook pages do not keep section anchors. A link to a renamed notebook
  heading opens the top of the page.
* The documentation of earlier releases and their READMEs on PyPI load some
  images in ``docs/logos/`` from ``master`` through
  ``raw.githubusercontent.com`` URLs, such as
  ``scikit-activeml-structure.png``. Keep these files unchanged, and give a
  replacement a new file name.

.. _contributing-examples:

Examples
--------

Each query strategy needs an example. An example is an entry in a JSON file. The documentation
build turns each entry into an example page with an animation and a
notebook. It also adds a card to the Visualizations pages and a row to the
Strategy Overview.

Put the entry in the directory under ``docs/examples/`` that matches its
``package`` and the setup it shows. The tests check part of this placement.

* ``1-pool-classification/``: pool classification, including multi-label
  classification.
* ``2-pool-regression/``: pool regression.
* ``3-pool-multi-annotator/``: pool classification with multiple annotators.
* ``4-stream/``: stream classification.

The JSON files are usually named after the strategy's module, such as
``_badge.json``. The build requires a ``README.rst`` in each directory. The
introduction of the Visualizations pages comes from
``docs/examples/README.rst``. The file name and URL of an example contain its
directory, ``class``, and ``method``. Keep them, so that the URLs keep
working. Two entries of the same class need different ``method`` values.

Each JSON file holds a list of entries. This is ``_badge.json`` in
``docs/examples/1-pool-classification/``, with a shortened ``text_0``:

.. code-block:: json

   [
     {
       "class": "Badge",
       "package": "pool",
       "method": "Batch Active Learning by Diverse Gradient Embedding (BADGE)",
       "category": "Hybrid",
       "template": "examples/template_pool_batch.py",
       "tags": ["pool", "classification", "single-annotator", "multi-label",
                "diverse-batch"],
       "example_setup": {
         "task": "classification",
         "target_type": "single-output",
         "annotation_type": "single-annotator",
         "selection": "diverse-batch"
       },
       "batch_size": "4",
       "title": "Batch Active Learning by Diverse Gradient Embedding (BADGE)",
       "refs": ["ash2020deep"],
       "sequence": ["title", "text_0", "plot", "refs"],
       "text_0": "**Idea:** BADGE selects a batch by running k-means++ ...",
       "import_misc": "from skactiveml.pool import Badge",
       "init_qs": "Badge(random_state=random_state)",
       "query_params": "X=X, y=y, clf=clf"
     }
   ]

Each entry needs these fields, except that only pool entries have
``batch_size`` and only stream entries have ``update_params``:

``class``
   The query strategy's class name. It must be in the ``__all__`` list of
   its package.

``package``
   The subpackage: ``pool``, ``pool.multiannotator``, or ``stream``. It sets
   the scenario.

``method``
   The method's name in the Strategy Overview. The example's URL contains
   it.

``category``
   ``Informativeness``, ``Representativeness``, ``Hybrid``, ``Baseline``, or
   ``Wrapper``. The Strategy Overview filters by this selection principle.
   Use ``Wrapper`` for a generic wrapper, such as ``SubSamplingWrapper``,
   whose principle depends on the strategy it wraps. These entries appear
   only when the Selection Principle filter is set to Any. An entry that
   wraps a specific strategy, such as ``SingleAnnotatorWrapper`` with
   ``CoreSet``, uses the principle of that strategy. A new generic wrapper
   also needs an entry in ``docs/generate.py``, as
   :ref:`contributing-generator` describes.

``template``
   The template that renders the example, as a path relative to ``docs/``,
   such as ``examples/template_pool.py``. See `Templates`_.

``tags``
   The scenario, the setups that the documented variant supports, and the
   batch selection of a pool strategy. See `Tags and Example Setup`_.

``example_setup``
   The setup that the example shows. See `Tags and Example Setup`_.

``batch_size``
   Pool entries only. The number of samples that each cycle acquires, as a
   string such as ``"4"``. With multiple annotators, it counts
   sample–annotator pairs. ``IntervalEstimationThreshold`` also accepts
   ``"adaptive"`` to determine the number of annotators from its performance
   threshold.

``title``
   The title of the example page, usually the method's name.

``refs``
   BibTeX keys from ``docs/refs.bib``.

``sequence``
   The order of the page's parts, usually
   ``["title", "text_0", "plot", "refs"]``. It may also contain further
   ``subtitle_N``, ``text_N``, and ``code_N`` parts with fields of the same
   name.

``text_0``
   The explanation above the animation, usually starting with
   ``**Idea:**``.

``import_misc``
   Python code that imports the query strategy, such as
   ``from skactiveml.pool import Badge``.

``init_qs``
   Python code that creates the query strategy, such as
   ``Badge(random_state=random_state)``. Pass the template's ``random_state``
   to the strategy and to every other random component of the entry, so that
   each build shows the same animation.

``query_params``
   Python code for the arguments of ``query``, such as ``X=X, y=y, clf=clf``.

``update_params``
   Stream entries only. Python code for the arguments of ``update``, usually
   ``candidates=X_cand, queried_indices=sampled_indices``.

A template written for one strategy may leave out some of these fields.
These fields are optional:

``import_clf``, ``init_clf``
   Python code that imports and creates the classifier. The default is a
   ``ParzenWindowClassifier``. The multi-label template uses its own
   classifier and ignores these fields.

``import_reg``, ``init_reg``
   Python code that imports and creates the regressor. The default is a
   ``NICKernelRegressor``.

``preproc``
   Python code that runs before the cycle, as a string or a list of lines.

``plot_utility_params``
   Python code for more arguments of ``plot_utilities``, such as
   ``candidates=X``. Only ``template_pool.py`` uses it.

``install_dependencies``
   The commented installation command of the notebook. The default is
   ``# !pip install scikit-activeml``.

``capability_params``
   A JSON object with the constructor parameters that change the strategy's
   capabilities, such as ``{"method": "expected_average_precision"}``. The
   Strategy Overview creates the strategy with them. The example code does
   not use them.

``batch_note``
   A short note that the Strategy Overview shows below the entry's
   properties, such as ``"Random selection within tree leaves"``.

Check new or changed entries with:

.. code:: bash

   python -m pytest --no-cov skactiveml/tests/test_examples.py

These tests run every example. They also require an example for each class
in ``skactiveml.pool.__all__`` and ``skactiveml.stream.__all__``. A class
there that is not a query strategy goes into
``QUERY_STRATEGY_EXCEPTIONS_LIST`` in ``test_examples.py``. The tests run the
examples at a reduced size and do not check the ``refs`` or the RST in the
text fields, so also build the documentation. To run the examples at full
size, as the website does:

.. code:: bash

   FULLEXAMPLES=1 python -m pytest --no-cov skactiveml/tests/test_examples.py -k test_example_files

.. _contributing-tags:

Tags and Example Setup
~~~~~~~~~~~~~~~~~~~~~~

The ``tags`` describe what the documented variant supports. The
``example_setup`` describes what its example shows.

The tags name the scenario (``pool`` or ``stream``), the tasks
(``classification``, ``regression``), the target type ``multi-label``, and the
annotation types (``single-annotator``, ``multi-annotator``). Single-output
support needs no tag. The strategy's ``_target_capabilities`` list the
complete combinations it supports. The tags can only restrict these
combinations. The Strategy Overview shows the combinations that remain. For a
variant whose constructor parameters change its capabilities, add
``capability_params``. Each class that supports multi-label classification
needs at least one entry tagged ``multi-label``, and other classes need none.

The tags of a pool entry also name its batch selection: ``top-k-batch``,
``diverse-batch``, or ``random-batch``. Use ``single-sample`` alone for a
strategy that selects one sample per cycle. Stream entries have no batch tags.

The ``example_setup`` has these fields:

* ``task``: ``"classification"`` or ``"regression"``.
* ``target_type``: ``"single-output"`` or ``"multi-label"``.
* ``annotation_type``: ``"single-annotator"`` or ``"multi-annotator"``.
* ``selection``, for pool entries only: the batch selection that the example
  uses.

The build stops with an error if an entry breaks one of these rules, among
others. The error names the method, or the class if it is missing from
``__all__``:

* ``class`` is in the ``__all__`` list of its package.
* ``category`` is one of the categories above.
* The documented variant supports every capability tag.
* A pool entry has batch tags, and a stream entry has none.
* ``example_setup`` has exactly these fields, with these values.
* The documented variant supports the setup.
* The setup has a group on the Visualizations pages. A stream example shows
  single-output classification with one annotator.
* The selection is one of the entry's batch tags.
* ``batch_size`` is a positive whole number in a string, or ``"adaptive"``
  for ``IntervalEstimationThreshold``. A ``single-sample`` entry uses ``"1"``.
* A shared template matches the setup and the batch size. For example,
  ``template_pool.py`` shows one sample per cycle, and
  ``template_pool_batch.py`` shows batches of four.

The Visualizations pages group the examples by their ``example_setup``, not by
their tags. An example with ``"target_type": "single-output"`` stays a
single-output example, even if its strategy also supports multi-label
classification. A batch caption, such as "Diverse batch of 4 samples", comes
from ``selection`` and ``batch_size``. Cards are sorted by title within each
group.

Templates
~~~~~~~~~

A template is a Python script with placeholders, such as ``"$init_qs"`` or
``"$n_cycles|20"``. The build replaces each placeholder with the entry's field
of the same name, or with the default after ``|``. Each value goes into the
script as Python code, which is why numeric ``batch_size`` values are strings.
The special ``batch_size`` value ``"adaptive"`` is inserted as a quoted Python
string. Entries may also set the placeholders with defaults, such as
``n_samples``, ``n_cycles``, or ``res``. Local builds and the tests replace
these three with smaller values, unless the environment variable
``FULLEXAMPLES`` is set.

These templates in ``docs/examples/`` are shared:

* ``template_pool.py``: pool classification, one sample per cycle.
* ``template_pool_batch.py``: pool classification, batches of four samples.
* ``template_pool_multilabel.py``: pool multi-label classification.
* ``template_pool_multi_annotator.py``: pool classification with multiple
  annotators.
* ``template_pool_regression.py``: pool regression, one sample per cycle.
  Its regressor must support ``predict(X, return_std=True)``.
* ``template_pool_regression_batch.py``: pool regression with batches. It
  expects ``query`` to return the utilities, so the entry's
  ``query_params`` include ``return_utilities=True``.
* ``template_stream.py``: stream classification with a single-output
  classifier.

A strategy that needs a different animation can have its own template in its
directory, such as ``template_DUAL.py`` in
``docs/examples/1-pool-classification/``. The file name must start with
``template``, so that the build does not copy the template as an example.
Start it with the header of a shared template, which contains the
``"$colab_link"`` and ``"$install_dependencies|..."`` placeholders. The build
appends the legend of the example's group and a final ``plt.show()`` cell to
each template. In the notebooks, it replaces this cell with a player for the
animation. Therefore, a template must end with a code block, import
``matplotlib.pyplot as plt``, and name its figure ``fig`` and its animation
``ani``. The legends come from ``VISUALIZATION_LEGENDS`` in
``docs/generate.py``. Since a custom template gets the legend of its group,
its plot must match this legend. When a template changes its plot, update the
legend too.

Tutorials
---------

Tutorial notebooks show a machine learning audience how to design experiments
and annotation workflows with scikit-activeml. Keep their code concise and
easy to adapt. Assume familiarity with Python,
NumPy, supervised learning, and the scikit-learn estimator interface. Explain
active learning concepts where they help readers use the framework.

Required Parts
~~~~~~~~~~~~~~

A new tutorial needs these parts:

* A notebook ``tutorials/NN_name.ipynb``. The first digit of ``NN`` groups
  the scenario: 0 for pool-based, 1 for multi-annotator, 2 for
  stream-based, and 3 for multi-label tutorials.
* A title cell, followed by the Colab badge. Copy the badge cell from an
  existing tutorial. The build replaces its placeholder ``<colab_link>`` with
  the notebook's link for the documentation version.
* A code cell with the installation command as a comment, such as
  ``# !pip install scikit-activeml tqdm``. The Colab copy uncomments it.
* A card in ``docs/tutorials.rst`` and an entry in its hidden toctree, in the
  same order as the cards, because this order sets the previous and next
  links. An introductory tutorial for pool-based or stream-based active
  learning or for a learning setting goes to Getting Started with a
  pictogram ``docs/logos/tutorial-*.svg``. Other tutorials go to Go Further
  with a topic line and a ``topic-*`` class from
  ``docs/_static/notebooks.css``. Copy an existing card in the same section
  to keep its classes. Its first badge names the data, such as
  ``:bdg-primary-line:`Tabular```. Add ``:bdg-secondary-line:`PyTorch``` if
  the tutorial assumes PyTorch knowledge, i.e., its code uses PyTorch beyond
  selecting the device.

Structure
~~~~~~~~~

Use a flexible layout suited to one scenario or study question:

* **Opening:** briefly state the goal, the capabilities demonstrated, and
  expected knowledge. A Go Further tutorial names the getting-started
  tutorial to read first. The overview of all tutorials belongs in
  ``tutorials.rst``.
* **Main example:** keep classifier and strategy construction, querying,
  annotation, fitting, and relevant evaluation easy to locate and adapt.
  Explain framework-specific choices beside their use. Include necessary
  support code within the notebook, grouping longer helpers before the
  relevant workflow. Put experiment settings together and method parameters
  beside their construction. Use explicit variations or straightforward
  comparison loops rather than a general experiment runner.
* **Outlook:** briefly suggest useful adaptations and selected next steps.

Results should serve the notebook's goal. A workflow demonstration may use
qualitative outputs. A comparative study needs appropriate evidence and a
brief explanation of its evaluation protocol. Interpret results within that
experiment's scope. Equations, reference sections, quantitative comparisons,
and summary tables are not required.

Settings and Outputs
~~~~~~~~~~~~~~~~~~~~

* Expensive notebooks default to ``QUICK_RUN = False`` and offer a cheaper
  run through ``QUICK_RUN = True``. Keep their strategy lists visible, so
  that readers can shorten them. Simpler notebooks use ordinary editable
  settings.
* Keep installation guidance short. A notebook that uses cuML activates it
  before it imports estimators.
* Show progress with tqdm progress bars that update rarely, such as with
  ``mininterval=20``. Runtimes appear only in these bars, not in the text.
* Keep environment inventories, stage timings, and validation code outside
  the notebook. Keep downloaded data and caches outside ``tutorials/``,
  because the build copies this directory twice.

Sphinx shows the saved outputs and does not run the notebooks. So commit a
tutorial with the outputs of a complete run with its published settings.
Saved outputs show no warnings. Fix the cause of a warning that signals a
mistake, such as a mismatched ``missing_label``. Filter an expected,
informational warning individually by its message, with a comment that gives
the reason. Do not hide all warnings or disable logging. A notebook with
widgets also saves the widget state, so that the website can show the
widgets. In JupyterLab, turn on *Save Widget State Automatically* in the
*Settings* menu. The notebook's text says that the widgets respond only in a
running notebook.

Plots
~~~~~

* Use small, self-contained plotting settings: 11-point axes and ticks,
  12-point titles, and 10-point legends.
* Where colors identify query strategies, use gray for random sampling and
  blue for uncertainty sampling. Keep class colors consistent across scatter
  plots.
* Use complete legends and layouts that fit the documentation width.
* Let metric-axis limits follow the displayed curves and bands with modest
  padding.
* For repeated learning curves, use light bands that show one sample
  standard deviation (``ddof=1``). Single runs need no bands.
* Numerical legend summaries can replace separate score tables. State what
  those values average, and do not repeat them in printed output.
* Name the x-axis after what was acquired, such as "Number of labeled
  samples" or "Number of queried labels". Distinguish samples, multi-label
  rows, repeated labels, and stream arrivals, and state whether initial
  labels are included.

.. _contributing-building-docs:

Building the Documentation
--------------------------

Build the documentation to check that the examples run and the pages render.
The build needs pandoc, ghostscript, and ffmpeg, for example from conda:

.. code:: bash

   conda install -c conda-forge pandoc ghostscript ffmpeg

Also install the optional dependencies and PyTorch, as
:ref:`contributing-install` describes for running all tests.
Otherwise, the API reference lacks the skorch, River, and CapyMOA wrappers and
the PyTorch-based multi-annotator classifiers. The configuration queries
release tags and external inventories, so a network connection is required.
With the ``scikit-activeml`` environment active, run:

.. code:: bash

   python -m sphinx -b doctest docs docs/_build/doctest
   python -m sphinx -b html docs docs/_build

The build generates and runs the visualization examples on all CPU cores. Set
the environment variable ``num_cpus`` to use fewer, such as ``num_cpus=4``.
Local builds run the examples with 50 samples and 5 cycles. Set the
environment variable ``FULLEXAMPLES``, such as ``FULLEXAMPLES=1``, to run them
at full size, as the website does.
