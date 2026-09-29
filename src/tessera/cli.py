"""`tessera ingest` / `tessera query` / `tessera eval`."""

from __future__ import annotations

import sys
from pathlib import Path

import typer
from pydantic import ValidationError

from tessera.config import Settings
from tessera.embedding.local import LocalEmbedder
from tessera.generation.nvidia import NvidiaClient
from tessera.ingestion.chunker import chunk_corpus, chunk_embedding_text
from tessera.ingestion.expertise_loader import load_expertise, profile_summary_text
from tessera.ingestion.loader import load_corpus
from tessera.pipeline import answer_query
from tessera.store.chroma import ChromaVectorStore
from tessera.store.chroma_expertise import ChromaExpertiseStore

# evals/ sits alongside src/, not inside it, so it isn't shipped as part
# of the installed tessera package or resolvable from the console-script
# entry point's own sys.path (which points at the venv's bin/ dir, not
# the repo — confirmed empirically: `tessera eval` raises
# ModuleNotFoundError without this, regardless of cwd, even from the
# repo root). Only `eval_command` below needs `evals` importable, so the
# repo root is added to sys.path lazily, right before that import, not
# as an unconditional module-level side effect every `tessera` command
# would otherwise pay for.
REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_CASES_DIR = REPO_ROOT / "evals" / "cases"

app = typer.Typer(help="Tessera — internal knowledge assistant (Phase 1 CLI).")


def _load_settings() -> Settings:
    try:
        return Settings()
    except ValidationError as exc:
        typer.echo(
            "Missing or invalid configuration — copy .env.example to .env "
            "and fill in NVIDIA_API_KEY (get one at "
            "https://build.nvidia.com/nvidia/nemotron-3-ultra-550b-a55b), "
            "then `set -a; source .env; set +a` before running this "
            "command.\n",
            err=True,
        )
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc


def _require_index(store: ChromaVectorStore) -> None:
    if store.count() == 0:
        typer.echo(
            "No index found — run `tessera ingest` first.", err=True
        )
        raise typer.Exit(code=1)


@app.command()
def ingest() -> None:
    """Load the corpus, chunk it, embed it, and persist the index."""
    settings = _load_settings()

    docs = load_corpus(settings.corpus_dir)
    chunks = chunk_corpus(docs)
    typer.echo(f"Loaded {len(docs)} documents, {len(chunks)} chunks.")

    embedder = LocalEmbedder()
    embeddings = embedder.embed_documents([chunk_embedding_text(c) for c in chunks])

    store = ChromaVectorStore(persist_dir=settings.vectorstore_dir)
    store.add(chunks, embeddings)
    typer.echo(f"Indexed {store.count()} chunks at {settings.vectorstore_dir}.")


@app.command(name="index-people")
def index_people() -> None:
    """Load the expertise dataset, embed each profile, and persist the
    people index (a separate collection from the document index).
    """
    settings = _load_settings()

    people = load_expertise(settings.expertise_dir, corpus_dir=settings.corpus_dir)
    typer.echo(f"Loaded {len(people)} people.")

    embedder = LocalEmbedder()
    embeddings = embedder.embed_documents([profile_summary_text(p) for p in people])

    store = ChromaExpertiseStore(persist_dir=settings.vectorstore_dir)
    store.add(people, embeddings)
    typer.echo(f"Indexed {store.count()} people at {settings.vectorstore_dir}.")


@app.command()
def query(text: str) -> None:
    """Answer a query against the persisted index, with citations."""
    settings = _load_settings()
    store = ChromaVectorStore(persist_dir=settings.vectorstore_dir)
    _require_index(store)

    expertise_store = ChromaExpertiseStore(persist_dir=settings.vectorstore_dir)
    if expertise_store.count() == 0:
        # Not fatal: only archetype B needs it, and the pipeline answers a
        # B query with a plain "index not built" message.
        expertise_store = None

    embedder = LocalEmbedder()
    llm = NvidiaClient(api_key=settings.nvidia_api_key, model=settings.nvidia_model)

    result = answer_query(text, llm, embedder, store, expertise_store)

    typer.echo(f"\n[{result.archetype.value}] {result.answer}\n")
    if result.experts:
        typer.echo("People:")
        for i, m in enumerate(result.experts, start=1):
            p = m.person
            flag = "" if m.is_evidenced else "  [self-reported only]"
            typer.echo(
                f"  [{i}] {p.name} — {p.title}, {p.practice}, {p.office} "
                f"(updated {p.last_updated.isoformat()}){flag}"
            )
    if result.citations:
        typer.echo("Sources:")
        for citation in result.citations:
            heading = " > ".join(citation.heading_path)
            typer.echo(
                f"  [{citation.marker}] {citation.document_title} — "
                f"{heading} ({citation.document_path})"
            )


@app.command(name="eval")
def eval_command(
    check: bool = typer.Option(
        False,
        "--check",
        help="Exit non-zero if any gated quality-bar threshold fails "
        "(evals/QUALITY_BAR.md). The report is printed either way.",
    ),
) -> None:
    """Run the eval harness against the persisted index and print a report."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    try:
        from evals.harness import format_report, load_cases, run_harness
    except ModuleNotFoundError as exc:
        typer.echo(
            "`evals` isn't importable — `tessera eval` only runs from a "
            "source checkout (evals/ ships alongside src/, not inside the "
            "installed package). Clone the repo and run this from its "
            "root.",
            err=True,
        )
        raise typer.Exit(code=1) from exc

    settings = _load_settings()
    store = ChromaVectorStore(persist_dir=settings.vectorstore_dir)
    _require_index(store)

    embedder = LocalEmbedder()
    llm = NvidiaClient(api_key=settings.nvidia_api_key, model=settings.nvidia_model)

    cases = load_cases(EVAL_CASES_DIR)
    report = run_harness(cases, llm, embedder, store, settings.corpus_dir)

    typer.echo(format_report(report))

    if check:
        from evals.harness import evaluate_bar

        result = evaluate_bar(report)
        if not result.passed:
            failed = ", ".join(t.name for t in result.gated_failures)
            typer.echo(f"\nQuality bar FAILED: {failed}", err=True)
            raise typer.Exit(code=1)
