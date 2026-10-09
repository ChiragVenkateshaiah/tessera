"""Compare eval sweeps (``tessera eval --json`` exports) against a baseline,
on the cases scored in both (Phase 5 plan §4.1; the ERROR-row rule in
CLAUDE.md: never merge re-runs into an export, compare only on cases
scored in both).

For each candidate export, against the baseline:
- **deterministic fields**, case by case: routing, the retrieved documents
  (A/C) and people (B), recall / reciprocal rank, leaks, injection;
- **judge means** over the cases both judged: A/C groundedness and
  relevance, B groundedness and relevance — read against the judge's
  noise floor (P5-0: A/C relevance ±0.03, B relevance ±0.11,
  groundedness unchanged);
- **latency and tokens** (routing + answer calls; the judge is excluded).

**The noise floor.** P5-0 measured it on one pair of native sweeps. With
``--floor-from``, it is instead the largest judge-mean difference over
every pair of the given native exports (on the cases each pair scored in
both) — the spread the judge shows when generation hasn't changed. The
P5-0 native sweeps through P5-5 ran different retrieval code in places
(P5-4/P5-5 changed shared retrieval), so this floor is slightly generous;
the deterministic fields say which cases that touched.

Pure over the exports: zero LLM calls. Run from the repo root:
    python -m evals.compare_sweeps evals/baselines/p5-5-native.json \\
        evals/baselines/p5-6-lc-router.json ... --out evals/reports/p5-6-generation.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

NOISE_FLOOR = {"ac_relevance": 0.03, "b_relevance": 0.11, "ac_groundedness": 0.0, "b_groundedness": 0.0}
JUDGE_METRICS = tuple(NOISE_FLOOR)


@dataclass
class Comparison:
    """One candidate against the baseline, over the cases scored in both."""

    name: str
    switches: dict[str, Any]
    commit: str | None
    scored_in_both: int
    errors: list[str]  # the candidate's own ERROR rows (kept in its export)
    routing_changed: list[str] = field(default_factory=list)
    documents_changed: list[str] = field(default_factory=list)
    people_changed: list[str] = field(default_factory=list)
    recall_changed: list[str] = field(default_factory=list)
    leaks: list[str] = field(default_factory=list)
    injection_failed: list[str] = field(default_factory=list)
    judge: dict[str, tuple[float | None, float | None]] = field(default_factory=dict)  # metric -> (base, cand)
    judge_changed: dict[str, list[str]] = field(default_factory=dict)  # metric -> "id a→b"
    latency: tuple[float, float] = (0.0, 0.0)
    tokens: tuple[tuple[int, int], tuple[int, int]] = ((0, 0), (0, 0))  # (in, out) base, cand

    def outside_noise_floor(self, floor: dict[str, float] = NOISE_FLOOR) -> list[str]:
        """Judge means that moved by more than the noise floor."""
        out = []
        for metric, (base, cand) in self.judge.items():
            if base is not None and cand is not None and abs(cand - base) > floor[metric] + 1e-9:
                out.append(metric)
        return out


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _scores(row: dict[str, Any]) -> dict[str, int]:
    """The judge's scores for a row, keyed like NOISE_FLOOR."""
    if row.get("judge"):
        return {"ac_groundedness": row["judge"]["groundedness"], "ac_relevance": row["judge"]["relevance"]}
    if row.get("expertise_judge"):
        return {"b_groundedness": row["expertise_judge"]["groundedness"], "b_relevance": row["expertise_judge"]["relevance"]}
    return {}


def compare(baseline: dict[str, Any], candidate: dict[str, Any], name: str) -> Comparison:
    """Compare two exports on the cases neither has as an ERROR row."""
    base = {r["case_id"]: r for r in baseline["case_results"]}
    cand = {r["case_id"]: r for r in candidate["case_results"]}
    errors = sorted(i for i, r in cand.items() if r.get("error"))
    both = [i for i in base if i in cand and not base[i].get("error") and not cand[i].get("error")]
    meta = candidate.get("meta", {})
    c = Comparison(
        name=name,
        switches=meta.get("lc_switches", {"stack": meta.get("stack")}),
        commit=meta.get("commit"),
        scored_in_both=len(both),
        errors=errors,
    )
    judged: dict[str, tuple[list[int], list[int]]] = {m: ([], []) for m in NOISE_FLOOR}
    lat_b: list[float] = []
    lat_c: list[float] = []
    tok = [[0, 0], [0, 0]]
    for i in both:
        b, k = base[i], cand[i]
        if b["actual_archetype"] != k["actual_archetype"]:
            c.routing_changed.append(f"{i} {b['actual_archetype']}→{k['actual_archetype']}")
        if b["retrieved_documents"] != k["retrieved_documents"]:
            c.documents_changed.append(i)
        if b["retrieved_people"] != k["retrieved_people"]:
            c.people_changed.append(i)
        for metric in ("recall", "reciprocal_rank_score", "person_recall", "person_reciprocal_rank"):
            if b.get(metric) != k.get(metric):
                c.recall_changed.append(f"{i} {metric} {b.get(metric)}→{k.get(metric)}")
        if k.get("leaked"):
            c.leaks.append(i)
        if k.get("access_set") == "injection" and (k.get("leaked") or not k.get("contract_held")):
            c.injection_failed.append(i)
        sb, sk = _scores(b), _scores(k)
        for metric in NOISE_FLOOR:
            if metric in sb and metric in sk:
                judged[metric][0].append(sb[metric])
                judged[metric][1].append(sk[metric])
                if sb[metric] != sk[metric]:
                    c.judge_changed.setdefault(metric, []).append(f"{i} {sb[metric]}→{sk[metric]}")
        lat_b.append(b["latency_seconds"])
        lat_c.append(k["latency_seconds"])
        for side, row in ((0, b), (1, k)):
            tok[side][0] += row.get("input_tokens") or 0
            tok[side][1] += row.get("output_tokens") or 0
    c.judge = {m: (_mean(bs), _mean(ks)) for m, (bs, ks) in judged.items()}
    c.latency = (_mean(lat_b) or 0.0, _mean(lat_c) or 0.0)
    c.tokens = ((tok[0][0], tok[0][1]), (tok[1][0], tok[1][1]))
    return c


def empirical_floor(exports: list[dict[str, Any]]) -> dict[str, float]:
    """The largest judge-mean difference over every pair of exports."""
    floor = dict.fromkeys(JUDGE_METRICS, 0.0)
    for i, a in enumerate(exports):
        for b in exports[i + 1 :]:
            for metric, (x, y) in compare(a, b, "pair").judge.items():
                if x is not None and y is not None:
                    floor[metric] = max(floor[metric], abs(x - y))
    return floor


def _fmt(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.2f}"


def format_markdown(
    comparisons: list[Comparison], baseline_name: str, floor: dict[str, float] = NOISE_FLOOR, floor_source: str = "P5-0"
) -> str:
    """A markdown table, one row per candidate, plus the changed cases."""
    lines = [
        f"Baseline: `{baseline_name}`. Noise floor ({floor_source}): "
        + ", ".join(f"{m} ±{v:.2f}" for m, v in floor.items())
        + ".",
        "",
        "| Sweep | Scored in both | Routing Δ | Docs Δ | People Δ | Recall/RR Δ "
        "| A/C ground. | A/C relev. | B ground. | B relev. | Leaks | Injection fails "
        "| Latency (s) | Tokens in/out | Outside floor |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c in comparisons:
        j = {m: f"{_fmt(b)}→{_fmt(k)}" for m, (b, k) in c.judge.items()}
        (bi, bo), (ki, ko) = c.tokens
        lines.append(
            f"| `{c.name}` | {c.scored_in_both} | {len(c.routing_changed)} | {len(c.documents_changed)} "
            f"| {len(c.people_changed)} | {len(c.recall_changed)} | {j['ac_groundedness']} | {j['ac_relevance']} "
            f"| {j['b_groundedness']} | {j['b_relevance']} | {len(c.leaks)} | {len(c.injection_failed)} "
            f"| {c.latency[0]:.1f}→{c.latency[1]:.1f} | {bi}/{bo}→{ki}/{ko} "
            f"| {', '.join(c.outside_noise_floor(floor)) or 'none'} |"
        )
    lines.append("")
    for c in comparisons:
        lines.append(f"**`{c.name}`** — ERROR rows kept: {', '.join(c.errors) or 'none'}.")
        for label, items in (
            ("routing", c.routing_changed),
            ("documents", c.documents_changed),
            ("people", c.people_changed),
            ("recall/RR", c.recall_changed),
            ("leaks", c.leaks),
            ("injection", c.injection_failed),
            *((f"judge {m}", v) for m, v in sorted(c.judge_changed.items())),
        ):
            if items:
                lines.append(f"- {label}: {', '.join(items)}")
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidates", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, help="Also write the comparisons as JSON here.")
    parser.add_argument(
        "--floor-from", type=Path, nargs="+", default=None,
        help="Native exports to take the noise floor from (max pairwise judge-mean difference).",
    )
    args = parser.parse_args(argv)
    baseline = json.loads(args.baseline.read_text())
    comparisons = [compare(baseline, json.loads(p.read_text()), p.stem) for p in args.candidates]
    floor, source = NOISE_FLOOR, "P5-0"
    if args.floor_from:
        floor = empirical_floor([json.loads(p.read_text()) for p in args.floor_from])
        source = f"max over {len(args.floor_from)} native sweeps, pairwise"
    print(format_markdown(comparisons, args.baseline.name, floor, source))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        payload = {"baseline": args.baseline.name, "noise_floor": floor, "noise_floor_source": source,
                   "comparisons": [asdict(c) for c in comparisons]}
        args.out.write_text(json.dumps(payload, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
