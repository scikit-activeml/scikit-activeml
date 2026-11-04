Maintainer Notes
================

These notes describe the parts of the build and release process that most
contributions do not touch.

.. _contributing-generator:

Documentation Generator
-----------------------

``docs/conf.py`` calls the functions in ``docs/generate.py`` before Sphinx
reads the sources. They write the API reference, the example scripts from the
JSON entries, the Visualizations pages, the Strategy Overview, the tutorial
pages with their Colab copies, and a copy of the README for the start page to
``docs/generated/``. They also write the version switcher to
``docs/_static/switcher.json``.

* The Visualizations pages come from ``visualizations.rst.jinja`` and
  ``visualization_section.rst.jinja`` in ``docs/_templates/``, which share the
  macros in ``visualization_macros.rst.jinja``. The legend of each group comes
  from ``VISUALIZATION_LEGENDS`` in ``docs/generate.py``. The build shows it
  on these pages and between each example's code and its animation.
* The Strategy Overview notes that the target types of ``SubSamplingWrapper``
  and ``ParallelUtilityEstimationWrapper`` depend on the wrapped strategy.
  The batch selection of ``SubSamplingWrapper`` does too.
  ``get_strategy_metadata`` and ``json_data_to_strategy_table`` in
  ``docs/generate.py`` name these wrappers. Add a new generic wrapper there.
* The filter groups of the Strategy Overview come from ``OVERVIEW_FILTERS`` in
  ``docs/generate.py``. ``docs/_static/filter_examples.js`` matches each group
  against the metadata of the entries, so a new group needs a rule there too.
  Each option shows a pictogram from ``docs/_static/strategy_filters/``. The
  README figure ``docs/logos/query-strategy-properties.svg`` repeats these
  pictograms, so update it together with them.
* After the build, ``docs/_ext/copy_sphinx_gallery_notebooks.py`` prepares the
  example notebooks offered for download. It renders their legends as HTML
  and replaces the final ``plt.show()`` cell with a player for the animation.
  Its copies for Colab in ``generated/sphinx_gallery_notebooks/`` of the
  build output also uncomment the installation command.

README
------

The README is the project's front page on GitHub and PyPI, and the start page
of the documentation includes it.

* Its images use paths relative to the repository, such as
  ``docs/logos/readme_pool.gif``, so that GitHub shows the images of the
  branch or tag a reader views. For PyPI, ``docs/clean_readme_for_pypi.py``
  turns them into raw GitHub URLs at the release tag. So commit changed
  images before you tag a release. For the documentation,
  ``generate_documentation_readme`` in ``docs/generate.py`` maps them to the
  build's own files and turns the links to the latest documentation into
  links within the build.
* PyPI cannot collapse the two code snippets of the animations. Between the
  ``code_snippets_start`` and ``code_snippets_end`` markers,
  ``docs/clean_readme_for_pypi.py`` replaces them with links named by their
  summaries, which lead to the README on GitHub at the release tag.
* The two animations come from ``docs/readme_pool.py`` and
  ``docs/readme_stream.py``. Each script runs the README's code snippet with
  added plotting code, so that the animation shows the snippet's run. When a
  snippet changes, change its script in the same way. The scripts need the
  packages of the README's installation commands: PyTorch, torchvision, the
  ``opt`` extra, datasets, sentence-transformers, and transformers. They
  download datasets and models from Hugging Face, and a GPU makes them fast.
  Run them from ``docs/``:

  .. code:: bash

     cd docs
     python readme_pool.py
     python readme_stream.py

.. _contributing-releases:

Releases
--------

``master`` holds the latest release, and ``development`` collects the changes
for the next one. The changelog is the
`Releases page <https://github.com/scikit-activeml/scikit-activeml/releases>`__.
To publish a release from ``development``:

1. On ``development``, merge the open dependabot pull requests, which raise
   the caps in ``requirements_max.txt`` and ``requirements_opt_max.txt``.
   Update the ``version`` in ``CITATION.cff`` and ``__version__`` in
   ``skactiveml/__init__.py``. If the authors change, update
   ``CITATION.cff``, the ``author`` in ``docs/conf.py``, and the citation in
   the README together.
2. Check the package that PyPI will receive. These commands write
   ``README-pypi.rst`` and ``dist/``, which you delete afterwards:

   .. code:: bash

      pip install build twine
      python docs/clean_readme_for_pypi.py README.rst README-pypi.rst
      python -m build
      twine check dist/*

3. Merge ``development`` into ``master`` and wait until the checks of the
   merge commit pass.
4. Draft a release on GitHub with ``master`` as its target. Create the tag
   ``X.Y.Z`` in the release form, without a prefix such as ``v`` or a suffix
   such as ``rc1``. Write the release notes as described below. Publish the
   release, not as a pre-release, as soon as the checks pass, and push
   nothing else to ``master`` until the documentation build has cloned it.
5. Merge ``master`` back into ``development``, so that both branches contain
   the release.
6. Check that the publishing run succeeded, that PyPI shows ``X.Y.Z`` with
   the README's images, that the release workflow of the documentation
   repository succeeded, and that the version switcher lists ``X.Y``.

Start the release notes with *Generate release notes* in the release form,
which lists the merged pull requests by their titles. Replace each title with
the one-line release note of its pull request, and leave out the notes
"None". Group the notes by category, such as Bug Fixes, Documentation, and
Requirements, and start with breaking changes. Credit the author and the pull
request of each note, as the earlier releases do.

A patch release can also start from ``master``. Branch from ``master``, add
the fixes and update the versions as in step 1, and check the package as in
step 2. Then merge the branch into ``master`` through a pull request, wait
until the checks of the merge commit pass, and continue with step 4. Step 5
brings the fixes into ``development``. The documentation workflow always
builds ``master``, so it cannot publish the documentation of a patch release
for an older version line.

Publishing the release runs ``.github/workflows/python-publish.yml``. It sets
``__version__`` from the tag, writes ``README-pypi.rst`` with the images
pinned to the tag, and uploads the package to PyPI. PyPI accepts each version
only once. If the run fails before the upload, rerun it instead of creating
the release again. If a wrong package was uploaded, yank it on PyPI and
publish a new version.

Then the run asks the documentation repository
`scikit-activeml.github.io <https://github.com/scikit-activeml/scikit-activeml.github.io>`__
to build the documentation of the release. Its workflow ``release.yml``
clones ``master``, not the tag. The workflow names the documentation after
the highest tag of the repository and deploys it to ``latest`` and to
``X.Y``. It and the version switcher in ``docs/generate.py`` handle only
lightweight tags of the form ``X.Y.Z`` with single-digit ``Y`` and ``Z``. The
release form creates lightweight tags.

The workflows ``main.yml`` and ``development.yml`` of the documentation
repository rebuild ``latest`` from ``master`` and ``development`` from
``development``. They run on the ``repository_dispatch`` events
``build_docs_latest`` and ``build_docs_development``, which
``.github/workflows/main.yml`` of this repository sends on each push to
``master`` or ``development``, without waiting for the tests. ``release.yml``
runs on the event ``release``. The deploy jobs of the three workflows share
one concurrency group. When a deploy queues while another one waits, GitHub
cancels the waiting one, so check the Actions tab of the documentation
repository after merges and releases. To rebuild, rerun a run there, or send
an event with a token that has write access to the documentation repository:

.. code:: bash

   REPO=scikit-activeml/scikit-activeml.github.io
   curl --fail -H "Authorization: token <token>" --request POST \
     https://api.github.com/repos/$REPO/dispatches \
     -d '{"event_type": "build_docs_development"}'

The workflows of this repository use three secrets:

* ``PYPI_API_TOKEN`` uploads the package to PyPI.
* ``DOCS_PAT`` sends the events to the documentation repository and needs
  write access to it. The workflows send the events without ``--fail``, so
  an expired token leaves the runs green while the documentation stops
  updating.
* ``CODECOV_TOKEN`` uploads the coverage. A failed upload fails the run.
