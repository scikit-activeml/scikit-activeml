Contributing
============

This page explains how to set up scikit-activeml for development and how a
change gets from an issue to a merged pull request. Three pages go into
detail:

* :doc:`contributing/components`: conventions, query strategies, budget
  managers, classifiers, regressors, and their tests.
* :doc:`contributing/documentation`: writing style, links, examples, and
  tutorials.
* :doc:`contributing/maintainers`: the documentation generator, the README,
  and releases.

.. toctree::
   :hidden:

   contributing/components
   contributing/documentation
   contributing/maintainers

.. raw:: html

   <script>
     // Links to sections that moved to another page open them there.
     (function () {
       var moved = {
         "overview": "contributing/components.html",
         "contributing-code": "contributing/components.html",
         "general-coding-conventions": "contributing/components.html#general-coding-conventions",
         "where-helpers-live": "contributing/components.html#where-helpers-live",
         "target-semantics-and-compatibility": "contributing/components.html#target-semantics-and-compatibility",
         "query-strategies": "contributing/components.html#query-strategies",
         "adding-a-query-strategy": "contributing/components.html#adding-a-component",
         "single-annotator-pool-based-query-strategies": "contributing/components.html#single-annotator-pool-based-query-strategies",
         "single-annotator-stream-based-query-strategies": "contributing/components.html#single-annotator-stream-based-query-strategies",
         "budget-manager": "contributing/components.html#budget-managers",
         "multi-annotator-pool-based-query-strategies": "contributing/components.html#multi-annotator-pool-based-query-strategies",
         "classifiers": "contributing/components.html#classifiers",
         "regressors": "contributing/components.html#regressors",
         "testing-and-code-coverage": "contributing/components.html#testing-and-code-coverage",
         "documentation": "contributing/documentation.html",
         "links": "contributing/documentation.html#links",
         "examples": "contributing/documentation.html#examples",
         "templates": "contributing/documentation.html#templates",
         "tags-and-example-setup": "contributing/documentation.html#tags-and-example-setup",
         "tutorials": "contributing/documentation.html#tutorials",
         "building-the-documentation": "contributing/documentation.html#building-the-documentation",
         "general": "contributing/components.html#single-annotator-pool-based-query-strategies",
         "init": "contributing/components.html#single-annotator-pool-based-query-strategies",
         "query": "contributing/components.html#single-annotator-pool-based-query-strategies",
         "general-advice": "contributing/components.html#single-annotator-pool-based-query-strategies",
         "testing": "contributing/components.html#single-annotator-pool-based-query-strategies",
         "update": "contributing/components.html#single-annotator-stream-based-query-strategies",
         "query-by-utility": "contributing/components.html#budget-managers",
         "query-by-utilities": "contributing/components.html#budget-managers",
         "general-advice-for-working-with-a-budget-manager": "contributing/components.html#budget-managers",
         "fit": "contributing/components.html#classifiers",
         "predict-proba": "contributing/components.html#classifiers",
         "predict": "contributing/components.html#classifiers",
         "score": "contributing/components.html#classifiers"
       };
       var target = moved[window.location.hash.slice(1)];
       if (target) {
         window.location.replace(target);
       }
     })();
   </script>

.. _thank-you-contributors:

Ways to Contribute
------------------

* **Report a bug or request a feature:** open an issue, as
  :ref:`contributing-getting-help` describes.
* **Fix a bug:** follow the :ref:`contributing-pull-requests` and add a test
  that fails without the fix. If no issue describes the bug, describe it in
  the pull request.
* **Add a query strategy:** check the
  :ref:`contributing-strategy-criteria` and open a feature request first.
  Then follow :ref:`contributing-adding-a-component`.
* **Add a classifier, regressor, or budget manager:** follow
  :ref:`contributing-adding-a-component`.
* **Improve the documentation, an example, or a tutorial:** see
  :doc:`contributing/documentation`.
* **Maintain the build or publish a release:** see
  :doc:`contributing/maintainers`.

The `contributors page
<https://github.com/scikit-activeml/scikit-activeml/graphs/contributors>`__
lists everyone who has contributed code. Thank you for your work!

By contributing, you agree to follow our
`Code of Conduct <https://github.com/scikit-activeml/scikit-activeml/blob/master/CODE_OF_CONDUCT.md>`__.
Contributions are licensed under the project's
`BSD 3-Clause License <https://github.com/scikit-activeml/scikit-activeml/blob/master/LICENSE.txt>`__.

.. _contributing-getting-help:

.. _getting-help:

.. _issue-tracking:

.. _checking-if-a-bug-already-exists:

.. _reporting-an-issue:

Get Help and Report Issues
--------------------------

Ask questions in
`GitHub Discussions <https://github.com/scikit-activeml/scikit-activeml/discussions>`__.
Report bugs, documentation problems, and feature requests in
`GitHub Issues <https://github.com/scikit-activeml/scikit-activeml/issues>`__
with the matching form: *Bug report*, *Documentation improvement*, or
*Feature request*. Search the open and closed issues first, because the
problem may already be fixed or planned.

Get Started
-----------

Get the Code
~~~~~~~~~~~~

``master`` holds the latest release, and ``development`` collects the changes
for the next one. Work on ``development``.

Fork the repository with the GitHub
`Fork <https://github.com/scikit-activeml/scikit-activeml/fork>`__ button and
clone your fork. Add the main repository as ``upstream``. A clone of your fork
starts on ``master``, so switch to the latest ``development`` before you
install the dependencies:

.. code:: bash

   git clone https://github.com/<your-username>/scikit-activeml.git
   cd scikit-activeml
   git remote add upstream https://github.com/scikit-activeml/scikit-activeml.git
   git fetch upstream
   git switch -c development upstream/development

.. _example-with-miniconda:

.. _setup-development-environment:

Set Up the Environment
~~~~~~~~~~~~~~~~~~~~~~

scikit-activeml supports Python 3.11 to 3.14. There are several ways to create
a local Python environment, such as
`virtualenv <https://virtualenv.pypa.io/>`__,
`pipenv <https://pipenv.pypa.io/en/latest/>`__, or
`miniconda <https://docs.conda.io/en/latest/miniconda.html>`__. With
miniconda, create and activate an environment named ``scikit-activeml``:

.. code:: bash

   conda create -n scikit-activeml python=3.13
   conda activate scikit-activeml

.. _contributing-install:

Install Dependencies
~~~~~~~~~~~~~~~~~~~~

With the environment active, install the package in editable mode with the
development dependencies of the ``dev`` extra. Run this in the cloned
repository:

.. code:: bash

   pip install -e '.[dev]'

Run this command again when the ``requirements*.txt`` files change on
``development``. Install more only if you need it:

* **To run all tests as CI does:** add the ``opt`` extra, PyTorch, and a Java
  runtime. The ``opt`` extra installs the packages of
  ``requirements_opt.txt``, which the skorch, River, and CapyMOA integrations
  use. Without them, the tests of these integrations are skipped. The ``opt``
  extra does not install PyTorch, so install the PyTorch build for your
  hardware separately, such as the CPU build that CI uses. CapyMOA needs a
  Java runtime, such as OpenJDK from conda-forge:

  .. code:: bash

     pip install -e '.[dev,opt]'
     pip install torch --index-url https://download.pytorch.org/whl/cpu
     conda install conda-forge::openjdk

* **If a newer dependency release breaks your setup:** the ``max`` and
  ``opt_max`` extras cap the dependencies at the newest versions known to
  work. CI tests the oldest supported and the newest available versions.

  .. code:: bash

     pip install -e '.[dev,max,opt_max]'

* **To build the documentation:** see
  :ref:`contributing-building-docs`.

Install the `pre-commit <https://pre-commit.com/>`__ hooks. Before each
commit, they check the formatting with
`black <https://black.readthedocs.io/en/stable/>`__ and lint with
`flake8 <https://flake8.pycqa.org/en/latest/>`__:

.. code:: bash

   pre-commit install

The hooks use black and flake8 from the active environment, so activate it
before you commit. If the black hook fails, run ``black skactiveml``, then add
and commit the files again. With Python 3.11 or 3.12, black warns that it
cannot parse code formatted for Python 3.13. You can ignore this warning.

.. _contributing-pull-requests:

.. _example-for-code-contribution-cycle-c3-and-pull-requests:

Pull Request Workflow
---------------------

1. Start a branch from the latest ``development`` branch:

   .. code:: bash

      git fetch upstream
      git switch -c <branch-name> upstream/development

2. Implement your changes, as :doc:`contributing/components` or
   :doc:`contributing/documentation` describes. Put the tests of a module in
   the test module with the same stem, such as
   ``skactiveml/pool/tests/test_my_strategy.py`` for
   ``skactiveml/pool/_my_strategy.py``. While you work, run the tests of this
   module:

   .. code:: bash

      python -m pytest --no-cov skactiveml/pool/tests/test_my_strategy.py

   Then run all tests. Set ``-n`` to the number of parallel workers, and
   limit the threads of each worker with ``OMP_NUM_THREADS=1``. Otherwise,
   the tests may run much slower:

   .. code:: bash

      OMP_NUM_THREADS=1 python -m pytest --no-cov -n 6

3. Check that the tests cover every line, as
   :ref:`contributing-testing` describes.

4. Commit and push your changes. Write one-line commit messages in the
   imperative mood, such as "Fix CapyMOAClassifier for capymoa 0.15".

   .. code:: bash

      git add <modified-files>
      git commit -m "<commit-message>"
      git push -u origin <branch-name>

5. Open a pull request with ``development`` as its base branch. Its template
   asks for the issue it closes, a one-line release note, and the
   :ref:`contributing-review-checklist`. Open it as a draft until the
   checklist holds, then mark it as ready for review. The pull request runs
   these checks:

   * The tests with Python 3.11 to 3.14, with the oldest supported and the
     newest available dependencies, each with and without the optional ones.
   * Full coverage in the tests with the optional dependencies.
   * ``black --check``, and ``flake8`` for syntax errors and undefined names.
     The pre-commit hooks run the complete flake8 checks.
   * The doctest and HTML builds of the documentation.

   For a first pull request, a maintainer may need to approve the checks
   before they run.

6. A maintainer reviews the pull request and may ask for changes. Push them
   as further commits to the same branch. If your branch conflicts with
   ``development``, merge ``upstream/development`` into it. If you get no
   response, comment on the pull request. After the approval, a maintainer
   merges the pull request into ``development``, and the next release from
   ``development`` includes it.

.. _contributing-review-checklist:

Review Checklist
~~~~~~~~~~~~~~~~

Before you mark the pull request as ready for review, check these items:

* The tests pass and cover every line.
* A bug fix comes with a test that fails without it.
* black and flake8 pass.
* New public classes and functions have numpydoc docstrings and are listed
  in ``__all__``.
* A new query strategy has an :ref:`example entry <contributing-examples>`
  and BibTeX entries.
* The documentation builds without warnings if the pull request changes it.
* I can explain every change, including parts written with AI tools.

AI-assisted Contributions
-------------------------

You may use AI tools for code, tests, and documentation. You are responsible
for every change you submit: understand it, run the tests, and be ready to
explain it in the review. Check that generated references exist and match the
cited papers, and that generated text is accurate. Maintainers may close pull
requests whose authors cannot explain their changes.
