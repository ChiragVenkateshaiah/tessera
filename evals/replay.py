"""Recorded LLM responses, replayed (Phase 5 plan §3.7, P5-7).

The orchestration-equivalence check runs the native pipeline and the
LangGraph pipeline on the **same real responses**: every native NIM
completion for every eval case is recorded once, and both stacks then
replay them. A replay that asks for a prompt that was never recorded
raises, so a drifted index or prompt can't pass silently.

Determinism: both the recording and the test build their indexes from
the corpus and the expertise dataset with exact search (``search_ef``
above the collection size), so retrieval — and so every prompt — is the
same in both. ``build_exact_indexes`` is that one recipe.

Record (live NIM, ~2 calls per case, free tier)::

    set -a; source .env; set +a
    python -m evals.replay --out tests/fixtures/replay/native-nim.json

Native only: no framework is imported here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any

from tessera.embedding.base import Embedder
from tessera.generation.base import Completion, LLMClient, Usage

REPO = Path(__file__).resolve().parents[1]
EXACT_EF = 2000  # above the collection sizes (336 chunks, 600 people)


def prompt_key(system: str, user: str, temperature: float) -> str:
    """One key per distinct request."""
    raw = json.dumps([system, user, temperature], ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class RecordingLLM(LLMClient):
    """Delegates to ``inner`` and keeps every completion by prompt key."""

    def __init__(self, inner: LLMClient) -> None:
        self._inner = inner
        self._lock = threading.Lock()
        self.recorded: dict[str, dict[str, Any]] = {}

    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        return self.complete_with_usage(system, user, temperature).text

    def complete_with_usage(self, system: str, user: str, temperature: float = 0.0) -> Completion:
        completion = self._inner.complete_with_usage(system, user, temperature)
        with self._lock:
            self.recorded[prompt_key(system, user, temperature)] = {
                "text": completion.text,
                "usage": asdict(completion.usage) if completion.usage is not None else None,
            }
        return completion


class ReplayMiss(KeyError):
    """A request that was never recorded: the index, a prompt or the
    routing changed since the recording."""


class ReplayLLM(LLMClient):
    """Answers only from a recording; counts what it served."""

    def __init__(self, recorded: dict[str, dict[str, Any]]) -> None:
        self._recorded = recorded
        self.served = 0

    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        return self.complete_with_usage(system, user, temperature).text

    def complete_with_usage(self, system: str, user: str, temperature: float = 0.0) -> Completion:
        key = prompt_key(system, user, temperature)
        if key not in self._recorded:
            raise ReplayMiss(f"no recorded response for this request (system starts {system[:60]!r})")
        entry = self._recorded[key]
        self.served += 1
        usage = Usage(**entry["usage"]) if entry["usage"] is not None else None
        return Completion(entry["text"], usage)


def load_recording(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_exact_indexes(root: Path, embedder: Embedder) -> tuple[Any, Any]:
    """The document and people indexes of the repo's corpus and dataset,
    with exact search, under ``root``."""
    from tessera.ingestion.chunker import chunk_corpus, chunk_embedding_text
    from tessera.ingestion.expertise_loader import load_expertise, profile_summary_text
    from tessera.ingestion.loader import indexable, load_corpus
    from tessera.store.chroma import ChromaVectorStore
    from tessera.store.chroma_expertise import ChromaExpertiseStore

    chunks = chunk_corpus(indexable(load_corpus(REPO / "data" / "corpus")))
    store = ChromaVectorStore(persist_dir=root, search_ef=EXACT_EF)
    store.add(chunks, embedder.embed_documents([chunk_embedding_text(c) for c in chunks]))
    people = load_expertise(REPO / "data" / "expertise" / "people")
    experts = ChromaExpertiseStore(persist_dir=root, search_ef=EXACT_EF)
    experts.add(people, embedder.embed_documents([profile_summary_text(p) for p in people]))
    return store, experts


def load_walls_for_repo() -> Any:
    from tessera.ingestion.access_loader import load_walls
    from tessera.ingestion.expertise_loader import load_expertise
    from tessera.ingestion.loader import load_corpus

    documents = load_corpus(REPO / "data" / "corpus")
    return load_walls(
        REPO / "data" / "access" / "walls.yaml",
        person_ids=[p.person_id for p in load_expertise(REPO / "data" / "expertise" / "people")],
        engagements={d.engagement for d in documents if d.engagement},
    )


def record(out: Path) -> int:
    """Run every eval case through the native pipeline on live NIM and
    save every completion."""
    import subprocess
    import tempfile

    from evals.harness import case_principal, load_cases
    from tessera.config import Settings
    from tessera.embedding.local import LocalEmbedder
    from tessera.generation.nvidia import NvidiaClient
    from tessera.generation.resilient import Pacer, RetryingLLMClient
    from tessera.pipeline import NativePipeline

    settings = Settings()  # type: ignore[call-arg]
    embedder = LocalEmbedder()
    walls = load_walls_for_repo()
    cases = load_cases(REPO / "evals" / "cases")
    live = RetryingLLMClient(
        NvidiaClient(api_key=settings.nvidia_api_key, model=settings.nvidia_model, sdk_max_retries=0),
        pacer=Pacer(3.0, max_interval=30.0),
        on_retry=lambda attempt, delay, error: print(f"  retry {attempt} in {delay:.0f}s: {error}", file=sys.stderr),
    )
    recorder = RecordingLLM(live)
    failed: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        store, experts = build_exact_indexes(Path(tmp), embedder)
        pipeline = NativePipeline(recorder, embedder, store, experts)
        for i, case in enumerate(cases, start=1):
            try:
                pipeline.run(case.query, case_principal(case, walls))
                print(f"[{i}/{len(cases)}] {case.id} ok", file=sys.stderr, flush=True)
            except Exception as exc:  # recorded as missing; the test reports it
                failed.append(case.id)
                print(f"[{i}/{len(cases)}] {case.id} FAILED: {exc}", file=sys.stderr, flush=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO, check=False
    ).stdout.strip()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "meta": {"commit": commit, "model": settings.nvidia_model, "cases": len(cases), "failed": failed},
                "responses": recorder.recorded,
            },
            indent=1,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Recorded {len(recorder.recorded)} responses for {len(cases) - len(failed)} cases -> {out}", file=sys.stderr)
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    return record(args.out)


if __name__ == "__main__":
    raise SystemExit(main())
