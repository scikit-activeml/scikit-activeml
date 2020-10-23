Developer's Guide
=================

.. toctree::

This page will be used to document the discussed guidelines and should help new contributors to adhere to the same set guidelines.

Conformity with scikit-learn
----------------------------
As the selection streategies are inheriting from BaseEstimator, the selection strategies should conform to the initialization scheme proposed by scikit-learn, i.e., there shall be no code in the __init__ function besides storing the attributes. All verification and transformation of inputs shall be done in the query (i.e., fit in scikit-learn) function.

Subtitle
--------
test  
test2  
test3  

Subtitle
--------
test

test2

test3