"""Time to first token under streaming (Phase 5 plan §3.7, P5-7): the
LangGraph orchestrator answers A/C eval cases through ``stream_run`` and
reports, per case, when the first answer token arrived, when the answer
finished, and each node's wall time.

    set -a; source .env; set +a
    TESSERA_LC_RETRY=lc python -m evals.ttft --cases ql001,ql003,ql011,...

Tokens stream only when the answer chain calls a LangChain chat model
directly: ``TESSERA_LC_MODEL_CLIENT=lc``, ``TESSERA_LC_PROMPT_CHAIN=lc``
and ``TESSERA_LC_RETRY=lc``. With the native retry layer the chain goes
through the native port, which returns the whole answer at once — the
script reports that instead of a number. Live calls (2 per case on NIM).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Time to first token through the LangGraph orchestrator.")
    parser.add_argument("--cases", required=True, help="comma-separated A/C case ids")
    parser.add_argument("--json", type=Path, default=None, help="also write the measurements here")
    args = parser.parse_args(argv)

    from evals.harness import case_principal, load_cases
    from tessera import cli
    from tessera.generation.resilient import Pacer
    from tessera.lc.graph import LangGraphPipeline, stream_run
    from tessera.store.chroma_expertise import ChromaExpertiseStore

    settings = cli._load_settings()
    pacer = Pacer(3.0, max_interval=30.0)
    llms = cli._build_llms(settings, pacer=pacer)
    lc = cli._lc_retrieval(settings, llms, pacer=pacer)
    pipeline = cli._native_pipeline(
        lc.llms or llms, lc.embedder, lc.store, ChromaExpertiseStore(persist_dir=settings.vectorstore_dir), None,
        steps=lc.pipeline_steps(), orchestration="graph",
    )
    assert isinstance(pipeline, LangGraphPipeline)
    walls = cli._load_walls(settings)
    wanted = args.cases.split(",")
    cases = [c for c in load_cases(REPO / "evals" / "cases") if c.id in wanted]
    print(f"switches: {lc.switches}", file=sys.stderr)
    rows = []
    for case in cases:
        streamed = stream_run(pipeline, case.query, case_principal(case, walls))
        nodes = dict(streamed.node_seconds)
        rows.append(
            {
                "case": case.id,
                "archetype": streamed.run.answer.archetype.value,
                "first_token_s": streamed.first_token_s,
                "total_s": streamed.total_s,
                "tokens": streamed.tokens,
                "nodes_s": nodes,
            }
        )
        ttft = f"{streamed.first_token_s:.2f}" if streamed.first_token_s is not None else "no tokens streamed"
        print(f"{case.id:<8} {rows[-1]['archetype']}  first token {ttft:>18}  total {streamed.total_s:6.2f}  "
              f"generate {nodes.get('generate', 0):6.2f}  tokens {streamed.tokens}")
    timed = [r for r in rows if r["first_token_s"] is not None]
    if timed:
        ttfts = [r["first_token_s"] for r in timed]
        totals = [r["total_s"] for r in timed]
        print(f"median first token {statistics.median(ttfts):.2f} s · median total {statistics.median(totals):.2f} s "
              f"({len(timed)} cases)")
    if args.json:
        args.json.write_text(json.dumps({"switches": lc.switches, "rows": rows}, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
