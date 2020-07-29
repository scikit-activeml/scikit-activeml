==============
scikit-example
==============

This is a simple scikit. The main intent is to serve as a template for new
scikits.


Installation from sources
=========================

In the root directory of the package, just do::

    python setup.py install


Distribution
============

A scikit is a standard Python package, which can be distributed by a number of
different means:

Source distribution
-------------------

To prepare a source distribution of the package::

    python setup.py sdist

Eggs
----

Eggs are a format for easy distribution of pre-built packages. It is
cross-platform for packages without any C code, and platform specific
otherwise. To build an egg::

    python setup.py bdist_egg

Binary installers
-----------------

Binary installers are platform specific. On Windows, you can do::

    python setup.py bdist_wininst

On Mac OS X (this requires an extension, bdist_mpkg, available on Pypi)::

    python setup.py bdist_mpkg


Registration onto PyPi
======================

A Scikit should be registered to PyPi, the Python package index.
This will make it easier for people to find and download the package,
and moreover it will list the package in the Scikits index:
http://scikits.appspot.com/scikits

For more information, see the `PyPi tutorial
<https://python-packaging-user-guide.readthedocs.io>`__

To register a package on PyPi and upload the sources at the same time::

    python setup.py register sdist upload

You can also upload the files manually using the forms on the PyPi web page:
https://pypi.python.org/

Binary distributions as eggs can also be uploaded to pypi. For example::

    python setup.py bdist_egg upload

Once a source or binary distribution is uploaded to PyPi, people can simply
install it with either with pip or with easy_install::

    pip scikit-example
    easy_install scikit-example

If you don't want to install as an egg, but from the sources::

    easy_install -eNb example scikit-example

Will download the most recent sources, and extract them into the example
directory.

