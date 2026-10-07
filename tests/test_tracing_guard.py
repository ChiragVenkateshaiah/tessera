"""The env-var tracing guard (Phase 5 plan §3.9.1, P5-3 acceptance).

LangSmith and LangChain trace from environment variables with a client
that has no redaction. Tessera refuses to start while any of them is set
— with its own tracing config off, the case the plan names, and on — and
the refusal comes before anything could send a request.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from tessera import cli
from tessera.ingestion.access_loader import Walls
from tessera.observability.guard import (
    REFUSED_ENV_VARS,
    TracingEnvError,
    check_tracing_env,
    refused_tracing_env,
)

runner = CliRunner()


def test_every_tracing_variable_is_refused_under_both_prefixes() -> None:
    for name in ("TRACING", "TRACING_V2", "OTEL_ENABLED", "RUNS_ENDPOINTS", "ALLOW_UNPROCESSED_PAYLOADS"):
        assert f"LANGSMITH_{name}" in REFUSED_ENV_VARS
        assert f"LANGCHAIN_{name}" in REFUSED_ENV_VARS


@pytest.mark.parametrize("value", ["true", "TRUE", "1", "local", '{"https://x": "k"}'])
def test_a_set_variable_is_refused(value: str) -> None:
    assert refused_tracing_env({"LANGCHAIN_TRACING_V2": value}) == ["LANGCHAIN_TRACING_V2"]
    with pytest.raises(TracingEnvError, match="LANGCHAIN_TRACING_V2"):
        check_tracing_env({"LANGCHAIN_TRACING_V2": value})


@pytest.mark.parametrize("value", ["", "  ", "false", "False", "0", "no", "off"])
def test_an_explicit_off_or_empty_value_is_allowed(value: str) -> None:
    check_tracing_env({"LANGSMITH_TRACING": value, "LANGSMITH_API_KEY": "k", "PATH": "/bin"})


def test_the_message_names_every_variable_and_the_tessera_switch() -> None:
    with pytest.raises(TracingEnvError) as err:
        check_tracing_env({"LANGSMITH_TRACING": "true", "LANGCHAIN_TRACING_V2": "true"})
    message = str(err.value)
    assert "LANGSMITH_TRACING, LANGCHAIN_TRACING_V2 are set" in message
    assert "TESSERA_LANGSMITH_TRACING=true" in message


@pytest.fixture
def outbound(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Every HTTP request any library tries to send (requests and httpx)."""
    import httpx
    import requests

    sent: list[Any] = []
    monkeypatch.setattr(requests.Session, "send", lambda self, req, **kw: sent.append(req))
    monkeypatch.setattr(httpx.Client, "send", lambda self, req, **kw: sent.append(req))
    return sent


@pytest.fixture
def cli_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    monkeypatch.delenv("TESSERA_LANGSMITH_TRACING", raising=False)
    monkeypatch.setattr(cli, "_load_walls", lambda settings: Walls(cleared={}))

    def must_not_build(*args: object, **kwargs: object) -> None:
        pytest.fail("the command got past the guard")

    monkeypatch.setattr(cli, "ChromaVectorStore", must_not_build)
    monkeypatch.setattr(cli, "LocalEmbedder", must_not_build)


@pytest.mark.parametrize("command", [["query", "x"], ["chat"], ["serve"], ["eval"], ["ingest"]])
@pytest.mark.parametrize("variable", ["LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2"])
def test_commands_refuse_to_start_with_tessera_tracing_off(
    command: list[str],
    variable: str,
    cli_env: None,
    outbound: list[Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(variable, "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2-test")

    result = runner.invoke(cli.app, command)

    assert result.exit_code == 1
    assert "Refusing to start" in result.output
    assert variable in result.output
    assert outbound == []


def test_tessera_tracing_on_does_not_make_the_variables_acceptable(
    cli_env: None, outbound: list[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("TESSERA_LANGSMITH_TRACING", "true")
    monkeypatch.setenv("TESSERA_LANGSMITH_API_KEY", "lsv2-test")

    result = runner.invoke(cli.app, ["query", "x"])

    assert result.exit_code == 1
    assert "Refusing to start" in result.output
    assert outbound == []


def test_tracing_off_stops_env_driven_tracing(
    outbound: list[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The variables really do trace (the P5-1 spike posted to
    api.smith.langchain.com), and ``tracing_off()`` — which the LangChain
    stack runs inside when Tessera's tracing is off — stops it."""
    ls = pytest.importorskip("langsmith")
    from langsmith import utils as ls_utils
    from langsmith.run_trees import get_cached_client

    from tessera.observability.langsmith_tracing import tracing_off

    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2-test")
    monkeypatch.setenv("LANGSMITH_ENDPOINT", "https://langsmith.invalid")
    ls_utils.get_env_var.cache_clear()
    try:
        run_trees: list[object] = []

        @ls.traceable
        def step(x: str) -> str:
            run_trees.append(ls.get_current_run_tree())
            return x.upper()

        with tracing_off():
            step("secret")
        # No run exists, so there is nothing to send (and no client is
        # built — building one would itself call GET /info).
        assert run_trees == [None]
        assert outbound == []

        step("secret")  # the control: the environment alone traces
        assert run_trees[-1] is not None
        get_cached_client().flush()
        posts = [r for r in outbound if r.method == "POST" and "/runs" in r.url]
        assert posts, "env-driven tracing sent no run — the control is broken"
    finally:
        monkeypatch.delenv("LANGSMITH_TRACING")
        ls_utils.get_env_var.cache_clear()
