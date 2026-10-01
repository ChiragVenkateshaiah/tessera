"""`tessera ingest` / `index-people` / `query` / `chat` / `serve` / `eval`."""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

import typer
from pydantic import ValidationError

from tessera.config import Settings
from tessera.embedding.local import LocalEmbedder
from tessera.generation.nvidia import NvidiaClient
from tessera.generation.resilient import RetryingLLMClient
from tessera.ingestion.chunker import chunk_corpus, chunk_embedding_text
from tessera.ingestion.expertise_loader import load_expertise, profile_summary_text
from tessera.ingestion.loader import load_corpus
from tessera.labels import ARCHETYPE_LABELS
from tessera.pipeline import AnswerResult, answer_query
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

# `tessera eval` runs 100+ LLM calls; NVIDIA NIM throttled far below its
# documented limit on 2026-09-29 (checkpoint.md Notes), so a sweep spaces
# its calls out. Interactive `tessera query` doesn't need spacing, only the
# retry/backoff.
EVAL_MIN_CALL_INTERVAL_SECONDS = 3.0

CHAT_EXIT_WORDS = frozenset({"exit", "quit", ":q"})

app = typer.Typer(help="Tessera — internal knowledge assistant (local CLI).")


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


def _build_llm(settings: Settings, *, min_interval: float = 0.0) -> RetryingLLMClient:
    """NVIDIA NIM behind retry/backoff. The SDK's own fast retries are
    turned off so they don't multiply with ours.
    """

    def announce(attempt: int, delay: float, error: BaseException) -> None:
        code = getattr(error, "status_code", "error")
        typer.echo(
            f"  LLM returned {code}; retrying in {delay:.0f}s (attempt {attempt})",
            err=True,
        )

    inner = NvidiaClient(
        api_key=settings.nvidia_api_key,
        model=settings.nvidia_model,
        sdk_max_retries=0,
    )
    return RetryingLLMClient(inner, min_interval=min_interval, on_retry=announce)


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


def _open_stores(
    settings: Settings,
) -> tuple[ChromaVectorStore, ChromaExpertiseStore | None]:
    """The document index (required) and the people index (optional)."""
    store = ChromaVectorStore(persist_dir=settings.vectorstore_dir)
    _require_index(store)

    expertise_store: ChromaExpertiseStore | None = ChromaExpertiseStore(
        persist_dir=settings.vectorstore_dir
    )
    if expertise_store.count() == 0:
        # Not fatal: only archetype B needs it, and the pipeline answers a
        # B query with a plain "index not built" message.
        expertise_store = None
    return store, expertise_store


def render_answer(result: AnswerResult) -> str:
    """The answer, then its People and/or Sources, as printed by `query`
    and `chat` (and written to a chat transcript).
    """
    lines = [f"[{result.archetype.value}] {result.answer}", ""]
    if result.experts:
        lines.append("People:")
        for i, m in enumerate(result.experts, start=1):
            p = m.person
            flag = "" if m.is_evidenced else "  [self-reported only]"
            lines.append(
                f"  [{i}] {p.name} — {p.title}, {p.practice}, {p.office} "
                f"(updated {p.last_updated.isoformat()}){flag}"
            )
    if result.citations:
        lines.append("Sources:")
        for citation in result.citations:
            heading = " > ".join(citation.heading_path)
            lines.append(
                f"  [{citation.marker}] {citation.document_title} — "
                f"{heading} ({citation.document_path})"
            )
    return "\n".join(lines).rstrip()


@app.command()
def query(text: str) -> None:
    """Answer a query against the persisted index, with citations."""
    settings = _load_settings()
    store, expertise_store = _open_stores(settings)

    embedder = LocalEmbedder()
    llm = _build_llm(settings)

    result = answer_query(text, llm, embedder, store, expertise_store)

    typer.echo(f"\n{render_answer(result)}")


@app.command()
def chat(
    transcript: Path | None = typer.Option(
        None,
        "--transcript",
        help="Append every question and answer to this Markdown file.",
    ),
) -> None:
    """Ask questions one after another in an interactive session.

    Loads the indexes and embedding model once, then answers each question
    the same way `tessera query` does. Each question is answered on its
    own — earlier questions are not used as context. Type `exit` (or
    Ctrl-D) to leave.
    """
    settings = _load_settings()
    store, expertise_store = _open_stores(settings)

    typer.echo("Loading the embedding model…")
    embedder = LocalEmbedder()
    llm = _build_llm(settings)

    if transcript is not None:
        with transcript.open("a", encoding="utf-8") as f:
            f.write(f"# Tessera chat — {datetime.now():%Y-%m-%d %H:%M}\n\n")

    people = "on" if expertise_store is not None else "off (run `tessera index-people`)"
    typer.echo(
        f"Tessera — {store.count()} document chunks indexed, people search {people}.\n"
        "Ask a question, or type `exit` to leave."
    )

    asked = 0
    while True:
        try:
            text = input("\ntessera> ").strip()
        except (EOFError, KeyboardInterrupt):
            typer.echo("")
            break
        if not text:
            continue
        if text.lower() in CHAT_EXIT_WORDS:
            break

        start = time.perf_counter()
        try:
            result = answer_query(text, llm, embedder, store, expertise_store)
        except KeyboardInterrupt:
            typer.echo("  (cancelled)")
            continue
        except Exception as exc:  # one failed question shouldn't end the session
            typer.echo(f"  Couldn't answer that: {exc}", err=True)
            continue
        elapsed = time.perf_counter() - start
        asked += 1

        label = ARCHETYPE_LABELS[result.archetype]
        rendered = render_answer(result)
        typer.echo(f"\n── {result.archetype.value} · {label} · {elapsed:.1f}s\n{rendered}")

        if transcript is not None:
            with transcript.open("a", encoding="utf-8") as f:
                f.write(
                    f"## Q{asked}. {text}\n\n"
                    f"*Archetype {result.archetype.value} ({label}), {elapsed:.1f}s*\n\n"
                    f"```text\n{rendered}\n```\n\n"
                )

    typer.echo(f"Answered {asked} question{'s' if asked != 1 else ''}.")
    if transcript is not None and asked:
        typer.echo(f"Transcript: {transcript}")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Interface to listen on."),
    port: int = typer.Option(8000, help="Port to listen on."),
) -> None:
    """Serve the HTTP API (and, from P4-2, the chat page) locally.

    Builds the same dependencies `chat` does, once, then hands them to
    `tessera.api.create_app` — the API never reads config itself.
    """
    # Imported here so the other commands don't pay for loading the web stack.
    import uvicorn

    from tessera.api import create_app

    settings = _load_settings()
    store, expertise_store = _open_stores(settings)

    typer.echo("Loading the embedding model…")
    embedder = LocalEmbedder()
    llm = _build_llm(settings)

    api = create_app(
        llm, embedder, store, expertise_store, llm_name=f"nvidia:{settings.nvidia_model}"
    )
    typer.echo(f"Tessera on http://{host}:{port}  (Ctrl-C to stop)")
    uvicorn.run(api, host=host, port=port)


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

    expertise_store = ChromaExpertiseStore(persist_dir=settings.vectorstore_dir)
    if expertise_store.count() == 0:
        typer.echo(
            "No people index found — run `tessera index-people` first "
            "(archetype B cases can't be scored without it).",
            err=True,
        )
        raise typer.Exit(code=1)

    embedder = LocalEmbedder()
    llm = _build_llm(settings, min_interval=EVAL_MIN_CALL_INTERVAL_SECONDS)

    def show_progress(done: int, total: int, result: object) -> None:
        error = getattr(result, "error", None)
        status = "ERROR" if error else "ok"
        typer.echo(f"  [{done}/{total}] {getattr(result, 'case_id', '?')} {status}", err=True)

    cases = load_cases(EVAL_CASES_DIR)
    report = run_harness(
        cases,
        llm,
        embedder,
        store,
        settings.corpus_dir,
        expertise_store=expertise_store,
        on_case_complete=show_progress,
    )

    typer.echo(format_report(report))

    if check:
        from evals.harness import evaluate_bar

        result = evaluate_bar(report)
        if not result.passed:
            failed = ", ".join(t.name for t in result.gated_failures)
            typer.echo(f"\nQuality bar FAILED: {failed}", err=True)
            raise typer.Exit(code=1)
