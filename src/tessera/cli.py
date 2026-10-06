"""`tessera ingest` / `index-people` / `data-report` / `query` / `chat` /
`serve` / `eval` / `feedback`."""

from __future__ import annotations

import json
import sys
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import typer
from pydantic import ValidationError

from tessera.config import MODEL_PRICES, Settings
from tessera.embedding.local import LocalEmbedder
from tessera.feedback.base import Feedback
from tessera.feedback.candidates import candidate_cases, render_candidates
from tessera.feedback.local import JsonlFeedbackStore, JsonlTraceLog
from tessera.generation.base import LLMClient
from tessera.generation.bedrock import BedrockClient
from tessera.generation.gemini import GeminiClient
from tessera.generation.nvidia import NvidiaClient
from tessera.generation.resilient import RetryingLLMClient
from tessera.ingestion.access_loader import Walls, load_walls
from tessera.ingestion.chunker import chunk_corpus, chunk_embedding_text
from tessera.ingestion.expertise_loader import load_expertise, profile_summary_text
from tessera.ingestion.data_quality import (
    NEAR_DUPLICATE_THRESHOLD,
    STALE_AFTER_YEARS,
    build_report,
    format_data_report,
)
from tessera.ingestion.loader import indexable, load_corpus, scan_corpus
from tessera.labels import ARCHETYPE_LABELS
from tessera.pipeline import AnswerResult, NativePipeline, answer_query
from tessera.principal import Principal
from tessera.store.chroma import ChromaVectorStore
from tessera.store.chroma_expertise import ChromaExpertiseStore
from tessera.trace import trace_record

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
    code = getattr(error, "status_code", None) or getattr(error, "code", "error")
    typer.echo(
        f"  LLM returned {code}; retrying in {delay:.0f}s (attempt {attempt})",
        err=True,
    )


STACK_HELP = (
    "Which pipeline answers: native, or lc (the Phase 5 LangChain stack, "
    "not built yet). Default: TESSERA_STACK, else native."
)
LC_NOT_BUILT_MESSAGE = (
    "The LangChain stack (--stack lc) isn't built yet: Phase 5 builds it in "
    "P5-4 to P5-7 (docs/Tessera_Phase5_Plan.md). Use --stack native."
)


def _resolve_stack(settings: Settings, stack: str | None) -> str:
    """The stack to run (the option, else TESSERA_STACK). Only native
    exists so far; asking for lc stops with a clear message."""
    chosen = stack or settings.stack
    if chosen not in ("native", "lc"):
        typer.echo(f"Unknown stack {chosen!r}: use native or lc.", err=True)
        raise typer.Exit(code=2)
    if chosen == "lc":
        typer.echo(LC_NOT_BUILT_MESSAGE, err=True)
        raise typer.Exit(code=2)
    return chosen


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


def _require_gcp(settings: Settings) -> str:
    """Fail before the first question, not on it (as for AWS): a missing
    project or missing Application Default Credentials would otherwise
    turn every case of an eval sweep into an ERROR row.
    """
    import google.auth
    from google.auth.exceptions import DefaultCredentialsError

    try:
        google.auth.default()
    except DefaultCredentialsError:
        typer.echo(
            "No Google Cloud credentials found. Run `gcloud auth "
            "application-default login` (the account needs the Agent Platform "
            "User role on the project). TESSERA_LLM_PROVIDER=nvidia runs "
            "without Google Cloud.",
            err=True,
        )
        raise typer.Exit(code=1)
    if not settings.gcp_project:
        typer.echo(
            "GOOGLE_CLOUD_PROJECT is not set: put the project id in .env.",
            err=True,
        )
        raise typer.Exit(code=1)
    return settings.gcp_project


def _build_gemini(
    settings: Settings, project: str, model: str, *, thinking: str | None
) -> RetryingLLMClient:
    inner = GeminiClient(
        model,
        project=project,
        location=settings.gcp_location,
        thinking_level=thinking,
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
    sweep); Gemini and Bedrock aren't paced.
    """
    if settings.llm_provider == "gemini":
        project = _require_gcp(settings)
        return LLMs(
            answer=_build_gemini(
                settings,
                project,
                settings.gemini_answer_model,
                thinking=settings.gemini_answer_thinking,
            ),
            router=_build_gemini(
                settings,
                project,
                settings.gemini_router_model,
                thinking=settings.gemini_router_thinking,
            ),
            name=(
                f"gemini:{settings.gemini_answer_model} "
                f"(router {settings.gemini_router_model})"
            ),
        )
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

    loaded = load_corpus(settings.corpus_dir)
    # Quarantined documents (pending human review) are never embedded.
    docs = indexable(loaded)
    chunks = chunk_corpus(docs)
    held = len(loaded) - len(docs)
    quarantine = f" ({held} quarantined, not indexed)" if held else ""
    typer.echo(f"Loaded {len(loaded)} documents{quarantine}, {len(chunks)} chunks.")

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


@app.command(name="data-report")
def data_report(
    stale_years: int = typer.Option(
        STALE_AFTER_YEARS, help="Flag current documents older than this many years."
    ),
    threshold: float = typer.Option(
        NEAR_DUPLICATE_THRESHOLD, help="Cosine similarity for a near-duplicate pair."
    ),
    show_known: bool = typer.Option(
        False, "--show-known", help="Also list the known Related Frameworks pairs."
    ),
) -> None:
    """Report corpus data quality: missing metadata, near-duplicate chunks,
    stale, superseded and quarantined documents. Reads the corpus directly
    (no index needed) and makes zero LLM calls.
    """
    settings = _load_settings()
    documents, problems = scan_corpus(settings.corpus_dir)
    chunks = chunk_corpus(indexable(documents))
    embeddings = LocalEmbedder().embed_documents([c.text for c in chunks])
    report = build_report(
        documents,
        problems,
        chunks,
        embeddings,
        settings.corpus_dir,
        datetime.now(timezone.utc).date(),
        near_duplicate_threshold=threshold,
        stale_after_years=stale_years,
    )
    typer.echo(format_data_report(report, show_known=show_known))


def _load_walls(settings: Settings) -> Walls:
    """The ethical walls, validated against the corpus's restricted
    engagements and the people dataset.
    """
    engagements = {d.engagement for d in load_corpus(settings.corpus_dir) if d.engagement}
    people = [p.person_id for p in load_expertise(settings.expertise_dir)]
    return load_walls(settings.access_file, person_ids=people, engagements=engagements)


def _principal_resolver(settings: Settings) -> Callable[[str], Principal | None]:
    """person_id -> Principal with the engagements they are cleared for,
    or None for someone not in the people dataset. A DEMO identity: the
    caller is taken at their word (Phase 4 plan §6).
    """
    walls = _load_walls(settings)
    people = {p.person_id for p in load_expertise(settings.expertise_dir)}

    def resolve(person_id: str) -> Principal | None:
        if person_id not in people:
            return None
        return Principal(person_id, walls.engagements_for(person_id))

    return resolve


def _require_principal(settings: Settings, person_id: str | None) -> Principal | None:
    if person_id is None:
        return None
    principal = _principal_resolver(settings)(person_id)
    if principal is None:
        typer.echo(f"No person {person_id!r} in {settings.expertise_dir}.", err=True)
        raise typer.Exit(code=1)
    return principal


def principal_line(principal: Principal | None) -> str:
    """Who answers are scoped to, as `query` and `chat` print it."""
    if principal is None:
        return "(internal documents only — pass --as PERSON_ID to ask as someone; demo identity)"
    cleared = ", ".join(sorted(principal.engagements)) or "no restricted engagements"
    return f"(asking as {principal.person_id} — demo identity; cleared for: {cleared})"


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
            section = f" — {heading}" if heading else ""
            lines.append(
                f"  [{citation.marker}] {citation.document_title}{section} "
                f"({citation.document_path})"
            )
    return "\n".join(lines).rstrip()


def _record_trace(
    settings: Settings, result: AnswerResult, latency_s: float, llm_name: str
) -> str:
    """Write the answer's trace to the trace log; return its trace_id. A
    write failure is reported, not raised — the answer still stands.
    """
    trace_id = uuid.uuid4().hex
    record = trace_record(
        trace_id,
        result,
        timestamp=datetime.now(timezone.utc),
        latency_s=latency_s,
        prices=MODEL_PRICES,
        llm=llm_name,
    )
    try:
        JsonlTraceLog(settings.trace_log).append(record)
    except OSError as exc:
        typer.echo(f"  (couldn't write the trace to {settings.trace_log}: {exc})", err=True)
    return trace_id


@app.command()
def query(
    text: str,
    as_person: str | None = typer.Option(
        None,
        "--as",
        help="Ask as this person_id (e.g. c0014). A DEMO identity, not "
        "authentication: restricted engagement documents are searched only "
        "for the engagements that person is cleared for. Default: internal "
        "documents only.",
    ),
    stack: str | None = typer.Option(None, "--stack", help=STACK_HELP),
) -> None:
    """Answer a query against the persisted index, with citations."""
    settings = _load_settings()
    _resolve_stack(settings, stack)
    principal = _require_principal(settings, as_person)
    store, expertise_store = _open_stores(settings)

    embedder = LocalEmbedder()
    llms = _build_llms(settings)

    start = time.perf_counter()
    result = answer_query(
        text,
        llms.answer,
        embedder,
        store,
        expertise_store,
        router_llm=llms.router,
        principal=principal,
    )
    trace_id = _record_trace(settings, result, time.perf_counter() - start, llms.name)

    typer.echo(
        f"\n{render_answer(result)}\n\n({usage_line(result)} · trace {trace_id})"
        f"\n{principal_line(principal)}"
    )


@app.command()
def chat(
    transcript: Path | None = typer.Option(
        None,
        "--transcript",
        help="Append every question and answer to this Markdown file.",
    ),
    as_person: str | None = typer.Option(
        None,
        "--as",
        help="Ask as this person_id (e.g. c0014). A DEMO identity, not "
        "authentication: restricted engagement documents are searched only "
        "for the engagements that person is cleared for. Default: internal "
        "documents only.",
    ),
    stack: str | None = typer.Option(None, "--stack", help=STACK_HELP),
) -> None:
    """Ask questions one after another in an interactive session.

    Loads the indexes and embedding model once, then answers each question
    the same way `tessera query` does. Each question is answered on its
    own — earlier questions are not used as context. Type `exit` (or
    Ctrl-D) to leave.
    """
    settings = _load_settings()
    _resolve_stack(settings, stack)
    principal = _require_principal(settings, as_person)
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
    typer.echo(principal_line(principal))

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
                text,
                llms.answer,
                embedder,
                store,
                expertise_store,
                router_llm=llms.router,
                principal=principal,
            )
        except KeyboardInterrupt:
            typer.echo("  (cancelled)")
            continue
        except Exception as exc:  # one failed question shouldn't end the session
            typer.echo(f"  Couldn't answer that: {exc}", err=True)
            continue
        elapsed = time.perf_counter() - start
        asked += 1
        trace_id = _record_trace(settings, result, elapsed, llms.name)

        label = ARCHETYPE_LABELS[result.archetype]
        rendered = render_answer(result)
        typer.echo(
            f"\n── {result.archetype.value} · {label} · {elapsed:.1f}s · "
            f"{usage_line(result)} · trace {trace_id}\n{rendered}"
        )

        if transcript is not None:
            with transcript.open("a", encoding="utf-8") as f:
                f.write(
                    f"## Q{asked}. {text}\n\n"
                    f"*Archetype {result.archetype.value} ({label}), {elapsed:.1f}s, "
                    f"{usage_line(result)}, trace `{trace_id}`*\n\n"
                    f"```text\n{rendered}\n```\n\n"
                )

    typer.echo(f"Answered {asked} question{'s' if asked != 1 else ''}.")
    if transcript is not None and asked:
        typer.echo(f"Transcript: {transcript}")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Interface to listen on."),
    port: int = typer.Option(8000, help="Port to listen on."),
    stack: str | None = typer.Option(None, "--stack", help=STACK_HELP),
) -> None:
    """Serve the HTTP API locally.

    Builds the same dependencies `chat` does, once, then hands them to
    `tessera.api.create_app` — the API never reads config itself.
    """
    # Imported here so the other commands don't pay for loading the web stack.
    import uvicorn

    from tessera.api import create_app

    settings = _load_settings()
    _resolve_stack(settings, stack)
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
        trace_log=JsonlTraceLog(settings.trace_log),
        feedback_store=JsonlFeedbackStore(settings.feedback_file),
        resolve_principal=_principal_resolver(settings),
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
    json_path: Path | None = typer.Option(
        None,
        "--json",
        help="Also write every case's result and the aggregates as JSON here.",
    ),
    stack: str | None = typer.Option(None, "--stack", help=STACK_HELP),
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
    chosen_stack = _resolve_stack(settings, stack)
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
    typer.echo(
        f"Stack: {chosen_stack} · answers: {llms.name} · judge: nvidia:{settings.nvidia_model}",
        err=True,
    )
    pipeline = NativePipeline(
        llms.answer, embedder, store, expertise_store, router_llm=llms.router
    )

    def show_progress(done: int, total: int, result: object) -> None:
        error = getattr(result, "error", None)
        status = "ERROR" if error else "ok"
        typer.echo(f"  [{done}/{total}] {getattr(result, 'case_id', '?')} {status}", err=True)

    cases = load_cases(EVAL_CASES_DIR)
    walls = _load_walls(settings)
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
        walls=walls,
        pipeline=pipeline,
    )

    typer.echo(format_report(report))
    if json_path is not None:
        from evals.harness import report_to_dict

        meta = {
            "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "commit": _git_commit(),
            "provider": settings.llm_provider,
            "stack": chosen_stack,
            "answers": llms.name,
            "judge": f"nvidia:{settings.nvidia_model}",
        }
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(report_to_dict(report, meta), indent=1) + "\n")
        typer.echo(f"Wrote {json_path}", err=True)

    if check:
        from evals.harness import evaluate_bar

        result = evaluate_bar(report)
        if not result.passed:
            failed = ", ".join(t.name for t in result.gated_failures)
            typer.echo(f"\nQuality bar FAILED: {failed}", err=True)
            raise typer.Exit(code=1)


def _git_commit() -> str | None:
    """The checked-out commit, for an eval export's provenance; None
    outside a git checkout. A trailing "+dirty" flags uncommitted changes.
    """
    import subprocess

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=REPO_ROOT
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True, text=True, check=True, cwd=REPO_ROOT,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return f"{commit}+dirty" if dirty else commit


feedback_app = typer.Typer(
    help="Rate answers and turn thumbs-down into candidate eval cases."
)
app.add_typer(feedback_app, name="feedback")


@feedback_app.command("add")
def feedback_add(
    trace_id: str = typer.Argument(..., help="The trace id printed with the answer."),
    rating: str = typer.Option(..., "--rating", help="up or down."),
    reason: str | None = typer.Option(None, help="Short reason, e.g. wrong-source."),
    comment: str | None = typer.Option(None, help="Free-text detail."),
) -> None:
    """Rate one answer (what `POST /api/feedback` does, from the terminal)."""
    if rating not in ("up", "down"):
        typer.echo("--rating must be 'up' or 'down'.", err=True)
        raise typer.Exit(code=1)
    settings = _load_settings()
    if JsonlTraceLog(settings.trace_log).get(trace_id) is None:
        typer.echo(f"No answer with trace id {trace_id!r} in {settings.trace_log}.", err=True)
        raise typer.Exit(code=1)
    JsonlFeedbackStore(settings.feedback_file).add(
        Feedback(
            trace_id=trace_id,
            rating=rating,  # type: ignore[arg-type]
            created_at=datetime.now(timezone.utc),
            reason=reason,
            comment=comment,
        )
    )
    typer.echo(f"Recorded thumbs-{rating} for {trace_id}.")


@feedback_app.command("review")
def feedback_review() -> None:
    """List thumbs-down answers with what their traces show."""
    settings = _load_settings()
    traces = JsonlTraceLog(settings.trace_log)
    cases, missing = candidate_cases(
        JsonlFeedbackStore(settings.feedback_file).list(), traces.get, settings.corpus_dir
    )
    if not cases and not missing:
        typer.echo("No thumbs-down feedback.")
        return
    for case in cases:
        seen = case["observed"]
        trace = traces.get(seen["trace_id"]) or {}
        shown = seen["sources_shown"] or seen["people_shown"]
        why = " — ".join(x for x in (seen["reason"], seen["comment"]) if x) or "(no reason given)"
        typer.echo(
            f"\n{seen['trace_id']}  [{case['archetype']}]  {case['query']}\n"
            f"  feedback: {why}\n"
            f"  route: {trace.get('route_reasoning', '')}\n"
            f"  shown to the model: {', '.join(shown) if shown else 'nothing (fixed response)'}\n"
            f"  cost: {trace.get('cost_usd')}  latency: {trace.get('latency_s')}s"
        )
    for trace_id in missing:
        typer.echo(f"\n{trace_id}  (thumbs-down, but no trace record found)")


@feedback_app.command("to-cases")
def feedback_to_cases(
    out: Path | None = typer.Option(
        None, "--out", help="Staging file (default: TESSERA_FEEDBACK_CANDIDATES)."
    ),
) -> None:
    """Write thumbs-down answers as CANDIDATE eval cases for a human to
    label. Never writes into evals/cases/ — unlabelled cases would corrupt
    the quality bar.
    """
    settings = _load_settings()
    target = out or settings.feedback_candidates
    if EVAL_CASES_DIR.resolve() in target.resolve().parents:
        typer.echo(
            "Refusing to write candidates into evals/cases/ — they must be "
            "labelled first. Write them elsewhere and promote by hand.",
            err=True,
        )
        raise typer.Exit(code=1)
    cases, missing = candidate_cases(
        JsonlFeedbackStore(settings.feedback_file).list(),
        JsonlTraceLog(settings.trace_log).get,
        settings.corpus_dir,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_candidates(cases), encoding="utf-8")
    typer.echo(f"Wrote {len(cases)} candidate case(s) to {target}.")
    if missing:
        typer.echo(f"Skipped {len(missing)} with no trace record: {', '.join(missing)}", err=True)
