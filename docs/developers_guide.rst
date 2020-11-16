Developer's Guide
=================

.. toctree::

This page will be used to document the discussed guidelines and should help new contributors to adhere to the same set guidelines.

Conformity with scikit-learn
----------------------------
As the selection streategies are inheriting from BaseEstimator, the selection strategies should conform to the initialization scheme proposed by scikit-learn, i.e., there shall be no code in the __init__ function besides storing the attributes. All verification and transformation of inputs shall be done in the query (i.e., fit in scikit-learn) function.

Code style and Linting
----------------------
As this library conforms to the convention of scikit-learn, the code should conform to `PEP 8 Sytle Guide for Python Code <https://www.python.org/dev/peps/pep-0008/>`_. For linting, the use of `flake8 <https://flake8.pycqa.org/en/latest/>`_ is recommended.

Homogeneous naming scheme
-------------------------
To simplify the use of this library, a homogeneous naming scheme is important. Thus, the following attribute and parameter names are to be used when appropriately:

* Attributes:
    * clf
    * classes
    * missing_label
* Parameters
    * X_cand
    * X
    * y
    * return_utilities
    * batch_size

Handling of unlabeled instances
-------------------------------

Active learning generally uses labeled and unlabeled instances. To simplify the data handling, the SkactivemlClassifier is able to handle unlabeled data. The unlabeled data is marked as such by setting corresponding entry in y (the label) during fitting to missing_label which is set during the initialization of the classifier. All classifier and the wrappers (e.g. for scikit-learn classifiers) are compatible with unlabeled instances.

