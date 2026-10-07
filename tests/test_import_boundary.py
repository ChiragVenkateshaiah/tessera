"""The framework import boundary (Phase 5 plan §3.2.2, CLAUDE.md #6).

- Only ``tessera/lc/``, ``tessera/review/``, ``tessera/integrations/``,
  ``tessera/observability/`` and ``evals/langsmith_sync.py`` may import
  ``langchain*``, ``langgraph*`` or ``langsmith`` (an allow-list, walked
  over every module).
- No agents or tool calling anywhere in ``src/``: no ``create_agent``,
  ``langgraph.prebuilt``, ``ToolNode`` or ``add_messages`` (user,
  2026-10-03).
- A native-only run of ``pipeline``, ``cli`` and ``api`` loads none of the
  frameworks — checked in a fresh interpreter, because ``langsmith`` is a
  hard dependency of ``langchain-core`` and an in-process check would see
  whatever earlier tests imported.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src" / "tessera"
FRAMEWORKS = ("langchain", "langgraph", "langsmith")
ALLOWED = (
    "src/tessera/lc/",
    "src/tessera/review/",
    "src/tessera/integrations/",
    "src/tessera/observability/",
    "evals/langsmith_sync.py",
)
FORBIDDEN_NAMES = {"create_agent", "ToolNode", "add_messages"}


def _modules() -> list[Path]:
    return sorted([*SRC.rglob("*.py"), *(REPO / "evals").rglob("*.py")])


def _imported(tree: ast.AST) -> list[str]:
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.append(node.module)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, (ast.Name, ast.Attribute))
            and getattr(node.func, "id", getattr(node.func, "attr", None))
            in ("import_module", "__import__")
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            names.append(node.args[0].value)
    return names


def _is_framework(module: str) -> bool:
    return module.split(".")[0].startswith(FRAMEWORKS)


def test_only_allow_listed_modules_import_the_frameworks() -> None:
    offenders = []
    for path in _modules():
        rel = path.relative_to(REPO).as_posix()
        if rel.startswith(ALLOWED):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
        offenders += [f"{rel}: {m}" for m in _imported(tree) if _is_framework(m)]
    assert offenders == []


def test_the_allow_list_check_catches_an_import(tmp_path: Path) -> None:
    tree = ast.parse("import langsmith\nfrom langchain_core.runnables import Runnable\n")
    assert [m for m in _imported(tree) if _is_framework(m)] == ["langsmith", "langchain_core.runnables"]


def test_no_agents_or_tool_calling_in_src() -> None:
    offenders = []
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(REPO).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
        for module in _imported(tree):
            if module.startswith("langgraph.prebuilt"):
                offenders.append(f"{rel}: imports {module}")
        for node in ast.walk(tree):
            name = (
                node.id
                if isinstance(node, ast.Name)
                else node.attr
                if isinstance(node, ast.Attribute)
                else None
            )
            if isinstance(node, ast.alias):
                name = node.asname or node.name.split(".")[-1]
            if name in FORBIDDEN_NAMES:
                offenders.append(f"{rel}:{getattr(node, 'lineno', '?')}: {name}")
    assert offenders == []


NATIVE_RUN = r"""
import json, sys
from tessera.generation.base import LLMClient
from tessera.pipeline import NativePipeline
from tessera.store.base import SearchResult, VectorStore
from tessera.embedding.base import Embedder
import tessera.cli, tessera.api
import tessera.observability.guard, tessera.observability.taint

class LLM(LLMClient):
    def complete(self, system, user, temperature=0.0):
        if "archetype" in system and "reasoning" in system:
            return '{"archetype": "A", "reasoning": "r"}'
        return "We have it [1]."

class Emb(Embedder):
    dimension = 1
    def embed_documents(self, texts): return [[1.0] for _ in texts]
    def embed_query(self, text): return [1.0]

class Store(VectorStore):
    def add(self, chunks, embeddings): pass
    def count(self): return 1
    def query(self, embedding, k, where=None):
        return [SearchResult(chunk_id="m::0", text="Framework text.", score=0.9,
            document_path="data/corpus/methodology/m.md", document_title="M",
            doc_type="methodology", industry="x", topics=["t"], date="2024-01-01",
            heading_path=("Overview",))]

run = NativePipeline(LLM(), Emb(), Store()).run("do we have a framework?")
assert run.answer.answer

from fastapi.testclient import TestClient
app = tessera.api.create_app(LLM(), Emb(), Store(), None, llm_name="fake")
assert TestClient(app).post("/api/ask", json={"question": "framework?"}).status_code == 200

from typer.testing import CliRunner
assert CliRunner().invoke(tessera.cli.app, ["query", "--help"]).exit_code == 0

print(json.dumps(sorted(m for m in sys.modules if m.split(".")[0].startswith(("langchain", "langgraph", "langsmith")))))
"""


def test_a_native_only_run_loads_no_framework() -> None:
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("LANGSMITH_", "LANGCHAIN_", "TESSERA_LANGSMITH_"))
    }
    env["TESSERA_STACK"] = "native"
    result = subprocess.run(
        [sys.executable, "-c", NATIVE_RUN],
        capture_output=True,
        text=True,
        cwd=REPO,
        env=env,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    loaded = json.loads(result.stdout.strip().splitlines()[-1])
    assert loaded == []
