"""Observability: LangSmith tracing with taint-based redaction (Phase 5,
plan §3.9).

The package is on the framework allow-list (CLAUDE.md constraint #6), but
only ``langsmith_tracing`` imports ``langsmith``. ``guard`` and ``taint``
are framework-free, so a composition root can check the environment and
build a taint set without loading LangSmith — and this ``__init__``
imports nothing, so importing ``tessera.observability.guard`` stays
framework-free too.
"""
