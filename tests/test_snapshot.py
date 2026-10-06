"""The golden retrieval snapshot (evals/snapshot.py) on fakes: what it
records per archetype, that it never makes a routing call, and that the
diff names exactly what changed. The real snapshot runs against the corpus
via `python -m evals.snapshot` and is committed; it isn't rebuilt here.
"""

from __future__ import annotations

import json
from pathlib import Path

from evals.harness import EvalCase
from evals.snapshot import (
    SCRIPTED_ANSWER,
    ScriptedLLM,
    build_snapshot,
    diff_snapshots,
    snapshot_case,
)
from tessera.ingestion.expertise_loader import Skill
from tessera.retrieval.router import Archetype
from tessera.store.base import SearchResult
from tests.test_expertise_retrieval import BagEmbedder, FixedScoreStore, person, proj
from tests.test_harness import FakeEmbedder, FakeVectorStore

CORPUS = Path("/corpus")


def _chunk(name: str, index: int, score: float, **kw: object) -> SearchResult:
    return SearchResult(
        chunk_id=f"{name}::{index}",
        text=f"text {name} {index}",
        score=score,
        document_path=str(CORPUS / "methodology" / f"{name}.md"),
        document_title=name,
        doc_type="methodology",
        industry="cross-industry",
        topics=["t"],
        date="2024-01-01",
        heading_path=("Overview",),
        **kw,  # type: ignore[arg-type]
    )


def _case(case_id: str, archetype: Archetype, **kw: object) -> EvalCase:
    return EvalCase(
        id=case_id,
        query="pharma pricing",
        archetype=archetype,
        relevant_sources=[],
        ideal_answer="",
        **kw,  # type: ignore[arg-type]
    )


def _people_store() -> FixedScoreStore:
    doer = person("doer", projects=[proj("pharma-pricing")])
    claimer = person("claimer", skills=[Skill("pharma-pricing", 5, "self_reported")])
    return FixedScoreStore([(doer, 0.61234), (claimer, 0.7)])


def test_scripted_llm_cites() -> None:
    # Calls are counted by the pipeline's generation recorder, not here.
    assert ScriptedLLM().complete("s", "u") == SCRIPTED_ANSWER
    assert "[1]" in SCRIPTED_ANSWER


def test_lookup_case_records_retrieval_with_rounded_scores_and_one_generation_call() -> None:
    store = FakeVectorStore([_chunk("pricing", 0, 0.712345), _chunk("pricing", 1, 0.20001)])

    record = snapshot_case(
        _case("a1", Archetype.LOOKUP), FakeEmbedder(), store, _people_store(), CORPUS, None
    )

    assert record["archetype"] == "A"
    assert record["documents"] == ["methodology/pricing.md"]
    assert {"chunk_id": "pricing::0", "score": 0.7123} in record["chunks"]
    assert "pricing::0" in record["over_floor"]
    assert "pricing::1" not in record["over_floor"]  # below the relevance floor
    assert record["restricted_seen"] == []
    assert record["generation_calls"] == 1
    json.dumps(record)  # plain data only


def test_restricted_chunk_is_recorded_for_a_cleared_principal_and_denied_otherwise() -> None:
    from tessera.ingestion.access_loader import Walls

    store = FakeVectorStore(
        [_chunk("halcyon-summary", 0, 0.8, sensitivity="restricted", engagement="halcyon")]
    )
    walls = Walls(cleared={"halcyon": frozenset({"c0048"})})

    def seen(principal: str | None) -> list[str]:
        case = _case("a2", Archetype.LOOKUP, principal=principal)
        record = snapshot_case(case, FakeEmbedder(), store, _people_store(), CORPUS, walls)
        return record["restricted_seen"]

    assert seen("c0048") == ["halcyon"]
    assert seen("c0001") == []  # walled: deny by default
    assert seen(None) == []


def test_expertise_case_records_the_shortlist_without_document_fields() -> None:
    record = snapshot_case(
        _case("b1", Archetype.EXPERTISE),
        BagEmbedder(),
        FakeVectorStore([]),
        _people_store(),
        CORPUS,
        None,
    )

    ids = [m["person_id"] for m in record["shortlist"]]
    assert ids[0] == "doer"  # evidence outranks a self-reported claim
    assert record["shortlist"][0]["score"] == 0.6123
    assert ["project", False] in record["shortlist"][0]["evidence"]
    assert "documents" not in record
    assert record["generation_calls"] in (0, 1)
    assert record["generation_calls"] == (1 if record["presented"] else 0)


def test_comparative_case_is_terminal_with_no_calls() -> None:
    record = snapshot_case(
        _case("d1", Archetype.COMPARATIVE),
        FakeEmbedder(),
        FakeVectorStore([]),
        _people_store(),
        CORPUS,
        None,
    )

    assert record == {"archetype": "D", "generation_calls": 0, "context_marker_hits": []}


def test_build_is_keyed_by_case_id_and_deterministic() -> None:
    store = FakeVectorStore([_chunk("pricing", 0, 0.7)])
    cases = [_case("z", Archetype.LOOKUP), _case("a", Archetype.COMPARATIVE)]

    first = build_snapshot(cases, FakeEmbedder(), store, _people_store(), CORPUS, None)
    second = build_snapshot(cases[::-1], FakeEmbedder(), store, _people_store(), CORPUS, None)

    assert list(first["cases"]) == ["a", "z"]
    assert first == second


def test_diff_names_each_changed_field_and_missing_or_new_cases() -> None:
    before = {"version": 1, "k": 5, "score_decimals": 4, "cases": {
        "a": {"archetype": "A", "documents": ["x.md"]},
        "gone": {"archetype": "D"},
    }}
    after = {"version": 1, "k": 5, "score_decimals": 4, "cases": {
        "a": {"archetype": "A", "documents": ["y.md"]},
        "new": {"archetype": "D"},
    }}

    lines = diff_snapshots(before, after)

    assert "a.documents: ['x.md'] -> ['y.md']" in lines
    assert "gone: missing" in lines
    assert "new: new case" in lines
    assert diff_snapshots(before, before) == []


def test_a_field_added_since_the_older_snapshot_is_accepted_only_while_empty() -> None:
    from evals.snapshot import diff_snapshots

    old = {"version": 1, "k": 5, "score_decimals": 4, "cases": {"a": {"generation_calls": 1}}}
    new_empty = {**old, "cases": {"a": {"generation_calls": 1, "context_marker_hits": []}}}
    new_hit = {**old, "cases": {"a": {"generation_calls": 1, "context_marker_hits": ["x:y"]}}}

    assert diff_snapshots(old, new_empty) == ["a.context_marker_hits: None -> []"]
    assert diff_snapshots(old, new_empty, allow_added={"context_marker_hits"}) == []
    assert diff_snapshots(old, new_hit, allow_added={"context_marker_hits"}) == [
        "a.context_marker_hits: None -> ['x:y']"
    ]


def test_context_marker_hits_are_recorded_for_an_uncleared_engagement() -> None:
    from dataclasses import replace

    store = FakeVectorStore([_chunk("pricing", 0, 0.7)])
    (chunk,) = store.query([1.0], k=1)
    case = replace(
        _case("a1", Archetype.LOOKUP),
        principal="c0014",
        forbidden_markers={"halcyon": [chunk.text.split()[0]]},
    )

    record = snapshot_case(case, FakeEmbedder(), store, _people_store(), CORPUS, None)

    assert record["context_marker_hits"] == [f"halcyon:{chunk.text.split()[0]}"]
