"""`tessera ingest` / `index-people` / `query` / `chat` / `serve` / `eval`."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import typer
from pydantic import ValidationError

from tessera.config import MODEL_PRICES, Settings
from tessera.embedding.local import LocalEmbedder
from tessera.generation.base import LLMClient
from tessera.generation.bedrock import BedrockClient
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


def _announce_retry(attempt: int, delay: float, error: BaseException) -> None:
    code = getattr(error, "status_code", "error")
    typer.echo(
        f"  LLM returned {code}; retrying in {delay:.0f}s (attempt {attempt})",
        err=True,
    )


def _build_nvidia(settings: Settings, *, min_interval: float = 0.0) -> RetryingLLMClient:
    """NVIDIA NIM behind retry/backoff. The SDK's own fast retries are
    turned off so they don't multiply with ours.
    """
    inner = NvidiaClient(
        api_key=settings.nvidia_api_key,
        model=settings.nvidia_model,
        sdk_max_retries=0,
    )
    return RetryingLLMClient(inner, min_interval=min_interval, on_retry=_announce_retry)


def _require_aws_profile(profile: str) -> None:
    """Fail before the first question, not on it: without this a missing
    profile turns every case of an eval sweep into an ERROR row.
    """
    from botocore.exceptions import ProfileNotFound
    from botocore.session import Session

    try:
        credentials = Session(profile=profile).get_credentials()
    except ProfileNotFound:
        credentials = None
    if credentials is None:
        typer.echo(
            f"AWS profile '{profile}' not found or has no credentials. Create it "
            f"with `aws configure --profile {profile}` (it needs "
            "bedrock-mantle:CreateInference), or point BEDROCK_AWS_PROFILE at "
            "an existing one. TESSERA_LLM_PROVIDER=nvidia runs without AWS.",
            err=True,
        )
        raise typer.Exit(code=1)


def _build_bedrock(
    settings: Settings, model: str, *, effort: str | None
) -> RetryingLLMClient:
    inner = BedrockClient(
        model,
        aws_region=settings.bedrock_region,
        aws_profile=settings.bedrock_aws_profile,
        effort=effort,
        sdk_max_retries=0,
    )
    return RetryingLLMClient(inner, on_retry=_announce_retry)


@dataclass(frozen=True)
class LLMs:
    """The clients one command needs: ``answer`` writes answers, ``router``
    classifies the question (the same client on NIM), and ``name`` says
    which models those are, for display.
    """

    answer: LLMClient
    router: LLMClient
    name: str


def _build_llms(settings: Settings, *, min_interval: float = 0.0) -> LLMs:
    """Per TESSERA_LLM_PROVIDER. min_interval paces NIM calls (an eval
    sweep); Bedrock isn't paced.
    """
    if settings.llm_provider == "bedrock":
        _require_aws_profile(settings.bedrock_aws_profile)
        return LLMs(
            answer=_build_bedrock(
                settings,
                settings.bedrock_answer_model,
                effort=settings.bedrock_answer_effort,
            ),
            # Haiku 4.5 rejects output_config.effort, so the router gets none.
            router=_build_bedrock(settings, settings.bedrock_router_model, effort=None),
            name=(
                f"bedrock:{settings.bedrock_answer_model} "
                f"(router {settings.bedrock_router_model})"
            ),
        )
    nvidia = _build_nvidia(settings, min_interval=min_interval)
    return LLMs(answer=nvidia, router=nvidia, name=f"nvidia:{settings.nvidia_model}")


def usage_line(result: AnswerResult) -> str:
    """Tokens and, when every model is priced, cost — e.g.
    "1,234 in / 210 out tokens · $0.0091".
    """
    usage = result.usage
    line = f"{usage.input_tokens:,} in / {usage.output_tokens:,} out tokens"
    cost = usage.cost_usd(MODEL_PRICES)
    return line if cost is None else f"{line} · ${cost:.4f}"


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
    llms = _build_llms(settings)

    result = answer_query(
        text, llms.answer, embedder, store, expertise_store, router_llm=llms.router
    )

    typer.echo(f"\n{render_answer(result)}\n\n({usage_line(result)})")


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
    llms = _build_llms(settings)

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
            result = answer_query(
                text, llms.answer, embedder, store, expertise_store, router_llm=llms.router
            )
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
        typer.echo(
            f"\n── {result.archetype.value} · {label} · {elapsed:.1f}s · "
            f"{usage_line(result)}\n{rendered}"
        )

        if transcript is not None:
            with transcript.open("a", encoding="utf-8") as f:
                f.write(
                    f"## Q{asked}. {text}\n\n"
                    f"*Archetype {result.archetype.value} ({label}), {elapsed:.1f}s, "
                    f"{usage_line(result)}*\n\n"
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
    """Serve the HTTP API locally.

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
    llms = _build_llms(settings)

    api = create_app(
        llms.answer,
        embedder,
        store,
        expertise_store,
        llm_name=llms.name,
        router_llm=llms.router,
        prices=MODEL_PRICES,
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
    llms = _build_llms(settings, min_interval=EVAL_MIN_CALL_INTERVAL_SECONDS)
    # The judge stays on Nemotron so bar numbers stay comparable across
    # providers (plan §3.1.4). On NIM it is the same paced client as the
    # answers, so pacing covers every call.
    judge = (
        llms.answer
        if settings.llm_provider == "nvidia"
        else _build_nvidia(settings, min_interval=EVAL_MIN_CALL_INTERVAL_SECONDS)
    )
    typer.echo(f"Answers: {llms.name} · judge: nvidia:{settings.nvidia_model}", err=True)

    def show_progress(done: int, total: int, result: object) -> None:
        error = getattr(result, "error", None)
        status = "ERROR" if error else "ok"
        typer.echo(f"  [{done}/{total}] {getattr(result, 'case_id', '?')} {status}", err=True)

    cases = load_cases(EVAL_CASES_DIR)
    report = run_harness(
        cases,
        llms.answer,
        embedder,
        store,
        settings.corpus_dir,
        expertise_store=expertise_store,
        on_case_complete=show_progress,
        router_llm=llms.router,
        judge_llm=judge,
        prices=MODEL_PRICES,
    )

    typer.echo(format_report(report))

    if check:
        from evals.harness import evaluate_bar

        result = evaluate_bar(report)
        if not result.passed:
            failed = ", ".join(t.name for t in result.gated_failures)
            typer.echo(f"\nQuality bar FAILED: {failed}", err=True)
            raise typer.Exit(code=1)
