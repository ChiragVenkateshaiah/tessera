"""Factories that build framework objects (Phase 5 plan §3.2.2).

The composition roots call these lazily — only when ``--stack lc`` or
LangSmith is selected — so a native-only run never imports a framework.
On the framework allow-list (CLAUDE.md constraint #6).
"""
