"""`tessera ingest` / `tessera query` / `tessera eval`."""

from __future__ import annotations

import sys
from pathlib import Path

import typer
from pydantic import ValidationError

from tessera.config import Settings
from tessera.embedding.local import LocalEmbedder
from tessera.generation.gemini import GeminiClient
from tessera.ingestion.chunker import chunk_corpus
from tessera.ingestion.loader import load_corpus
from tessera.pipeline import answer_query
from tessera.store.chroma import ChromaVectorStore

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
            "and fill in GEMINI_API_KEY (get one at "
            "https://aistudio.google.com/apikey), then `set -a; source "
            ".env; set +a` before running this command.\n",
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
    embeddings = embedder.embed_documents([c.text for c in chunks])

    store = ChromaVectorStore(persist_dir=settings.vectorstore_dir)
    store.add(chunks, embeddings)
    typer.echo(f"Indexed {store.count()} chunks at {settings.vectorstore_dir}.")


@app.command()
def query(text: str) -> None:
    """Answer a query against the persisted index, with citations."""
    settings = _load_settings()
    store = ChromaVectorStore(persist_dir=settings.vectorstore_dir)
    _require_index(store)

    embedder = LocalEmbedder()
    llm = GeminiClient(api_key=settings.gemini_api_key, model=settings.gemini_model)

    result = answer_query(text, llm, embedder, store)

    typer.echo(f"\n[{result.archetype.value}] {result.answer}\n")
    if result.citations:
        typer.echo("Sources:")
        for citation in result.citations:
            heading = " > ".join(citation.heading_path)
            typer.echo(
                f"  [{citation.marker}] {citation.document_title} — "
                f"{heading} ({citation.document_path})"
            )


@app.command(name="eval")
def eval_command() -> None:
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
    llm = GeminiClient(api_key=settings.gemini_api_key, model=settings.gemini_model)

    cases = load_cases(EVAL_CASES_DIR)
    report = run_harness(cases, llm, embedder, store, settings.corpus_dir)

    typer.echo(format_report(report))
