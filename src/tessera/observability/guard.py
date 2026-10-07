"""The env-var tracing guard (Phase 5 plan §3.9.1).

LangSmith and LangChain switch tracing on from environment variables, and
their default client has none of Tessera's redaction hooks. Tessera
enables tracing only from its own config (``TESSERA_LANGSMITH_*``), so the
composition roots refuse to start while any variable that would trace —
or would weaken redaction — is set. Refused, not honoured (CLAUDE.md):
they are refused even when Tessera's own tracing is on, so no code path
can fall back on the SDK's environment-driven client.

Framework-free on purpose: the check runs before anything imports
``langsmith``.
"""

from __future__ import annotations

from collections.abc import Mapping

# The SDK reads each name under both prefixes (langsmith.utils.get_env_var).
_PREFIXES = ("LANGSMITH", "LANGCHAIN")
_NAMES = (
    # Switch tracing on with the SDK's default, unredacted client.
    "TRACING",
    "TRACING_V2",
    # Send runs through OpenTelemetry instead of the client's run pipeline.
    "OTEL_ENABLED",
    # Copy every run to further endpoints.
    "RUNS_ENDPOINTS",
    # Trace raw inputs/outputs when a process_inputs/outputs hook fails.
    "ALLOW_UNPROCESSED_PAYLOADS",
)
REFUSED_ENV_VARS: tuple[str, ...] = tuple(f"{p}_{n}" for p in _PREFIXES for n in _NAMES)

_OFF_VALUES = frozenset({"", "false", "0", "no", "off"})


class TracingEnvError(RuntimeError):
    """A refused LangSmith/LangChain environment variable is set."""


def refused_tracing_env(environ: Mapping[str, str]) -> list[str]:
    """The refused variables set to anything but an explicit "off", in a
    stable order; empty when the environment is clean."""
    return [
        name
        for name in REFUSED_ENV_VARS
        if environ.get(name) is not None and environ[name].strip().lower() not in _OFF_VALUES
    ]


def check_tracing_env(environ: Mapping[str, str]) -> None:
    """Raise TracingEnvError naming every refused variable that is set."""
    found = refused_tracing_env(environ)
    if found:
        raise TracingEnvError(
            f"Refusing to start: {', '.join(found)} {'is' if len(found) == 1 else 'are'} "
            "set. Tessera never lets LangSmith/LangChain trace from environment "
            "variables — their default client has no redaction. Unset "
            f"{'it' if len(found) == 1 else 'them'}, and turn tracing on with "
            "TESSERA_LANGSMITH_TRACING=true instead (docs/Tessera_Phase5_Plan.md §3.9.1)."
        )
