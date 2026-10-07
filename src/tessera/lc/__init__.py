"""The LangChain stack (Phase 5, ADR 0006): a second ``Pipeline`` built
beside the native core, one switchable layer at a time
(``docs/Tessera_Phase5_Plan.md`` §3.2).

On the framework allow-list (CLAUDE.md constraint #6). Imported only when
the LangChain stack is selected; the native stack never loads it.
"""
