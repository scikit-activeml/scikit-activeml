{% if fullname in (
    "skactiveml.pool.BatchBALD",
    "skactiveml.pool.GreedyBALD",
    "skactiveml.pool.batch_bald",
) %}
.. note::

   This implementation uses computations adapted from ``batchbald_redux``
   under the Apache-2.0 license. See :ref:`license-batchbald-redux` for
   the license and notices.
{% elif fullname == "skactiveml.pool.CostEmbeddingAL" %}
.. note::

   This implementation includes code adapted from ``libact`` under the
   BSD-2-Clause license and ``scikit-learn`` under the BSD-3-Clause license.
   See :ref:`license-libact` and :ref:`license-scikit-learn` for the license
   texts and notices.
{% elif fullname == "skactiveml.pool.EpistemicUncertaintySampling" %}
.. note::

   The logistic loss helpers used by this implementation are adapted from
   ``scikit-learn`` under the BSD-3-Clause license. See
   :ref:`license-scikit-learn` for the license text and notices.
{% endif %}
