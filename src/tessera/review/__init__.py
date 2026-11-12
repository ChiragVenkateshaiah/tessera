"""The human-review workflow (Phase 5, plan §3.8): ``tessera corpus review``.

A LangGraph graph with ``interrupt()`` and a SQLite checkpointer that walks
the documents held back for review (``review_status: pending``), shows a
reviewer the evidence, and applies the reviewer's ruling to the corpus,
both stacks' indexes and an audit log.

Exempt from the no-I/O rule the way ``ingestion/loader.py`` is (CLAUDE.md
constraint #6): reading and writing the corpus, the audit log and the
checkpoint database is this package's job. Every path arrives as a
parameter. ``decisions`` is pure; LangGraph is imported only by ``graph``.
"""
