.. _release-notes:

=============
Release Notes
=============

Version 1.1
===========

Resolved target semantics
-------------------------

Version 1.1 introduces the public immutable
:class:`~skactiveml.utils.TargetSpec` and the public
:func:`~skactiveml.utils.resolve_target_spec` resolver.  Estimators and pool
query strategies now declare target intent with ``target_type``; fitted
classifiers and regressors expose their concrete specification as
``target_spec_``.  The public label mask and index helpers accept a resolved
``target_type`` so multi-label calls operate on complete sample vectors.

Explicit multi-label classification is supported by demonstrated components,
including :class:`~skactiveml.classifier.SklearnClassifier` with
``classes=None`` when every output exposes both binary classes.  Automatic
resolution rejects ambiguous bare two-dimensional classification targets
instead of guessing from their values.  Valid future multi-output semantics
are distinguished from invalid input and rejected by version 1.1 components
that do not declare those capabilities.

Released defaults are preserved: unambiguous single-label classification and
single-output regression continue to resolve under ``target_type="auto"``;
regression column vectors remain compatible; and existing multi-annotator
components continue to interpret matrix columns as annotators and query
sample-annotator pairs.  Multi-output classification, multi-output regression,
partial-label querying, and multi-label multi-annotator querying remain outside
the version 1.1 execution scope.
