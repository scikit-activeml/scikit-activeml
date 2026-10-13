Licenses
========

scikit-activeml is primarily licensed under BSD-3-Clause. Code adapted
from third-party projects retains the licenses and notices given below.
Copies of these texts are also part of the source distribution and wheel.

scikit-activeml
---------------

.. literalinclude:: ../LICENSE.txt
   :language: text

.. _license-batchbald-redux:

batchbald_redux
---------------

The ``batch_bald`` computations and entropy helpers used by
``BatchBALD`` and ``GreedyBALD`` are adapted from
`batchbald_redux <https://github.com/BlackHC/batchbald_redux>`__.
The source module describes the modifications made by the scikit-activeml
developers.

.. literalinclude:: ../LICENSES/batchbald_redux/NOTICE
   :language: text

.. literalinclude:: ../LICENSES/batchbald_redux/LICENSE
   :language: text

.. _license-libact:

libact
------

Parts of ``CostEmbeddingAL``, including its partial multidimensional
scaling implementation, are adapted from
`libact <https://github.com/ntucllab/libact>`__.

.. literalinclude:: ../LICENSES/libact/LICENSE
   :language: text

.. _license-scikit-learn:

scikit-learn
------------

The partial multidimensional scaling implementation used by
``CostEmbeddingAL`` also contains code adapted from scikit-learn at
`commit 14031f6
<https://github.com/scikit-learn/scikit-learn/tree/14031f6>`__.

.. literalinclude:: ../LICENSES/scikit-learn/COPYING-14031f6
   :language: text

The logistic loss helpers used by ``EpistemicUncertaintySampling`` are
adapted from `scikit-learn 1.0.X
<https://github.com/scikit-learn/scikit-learn/tree/1.0.X>`__.

.. literalinclude:: ../LICENSES/scikit-learn/COPYING-1.0.X
   :language: text
