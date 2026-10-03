"""Restricted tier data (Phase 4, P4-5): the wall generator is
deterministic and current, the walls agree with the corpus and the
people, the demo personas have the clearances they claim, and the access
labels fail closed.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from tessera.ingestion.access_loader import AccessError, Walls, load_walls
from tessera.ingestion.expertise_loader import load_expertise
from tessera.ingestion.loader import CorpusError, load_corpus, load_document

REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = REPO_ROOT / "data" / "corpus"
PEOPLE_DIR = REPO_ROOT / "data" / "expertise" / "people"
WALLS = REPO_ROOT / "data" / "access" / "walls.yaml"
GENERATOR = REPO_ROOT / "data" / "access" / "generate.py"


def _generator():
    spec = importlib.util.spec_from_file_location("_access_generate", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _real_walls() -> Walls:
    engagements = {d.engagement for d in load_corpus(CORPUS_DIR) if d.engagement}
    people = [p.person_id for p in load_expertise(PEOPLE_DIR)]
    return load_walls(WALLS, person_ids=people, engagements=engagements)


def test_generator_output_matches_committed_walls() -> None:
    module = _generator()

    assert module.render(module.build()) == WALLS.read_text(encoding="utf-8"), (
        "data/access/walls.yaml is stale — re-run `python data/access/generate.py`"
    )


def test_walls_cover_every_restricted_engagement_with_a_real_team() -> None:
    walls = _real_walls()

    assert len(walls.cleared) == 12
    assert all(len(team) == 7 for team in walls.cleared.values())


def test_personas_have_the_clearances_they_claim() -> None:
    walls = _real_walls()
    p = walls.personas

    assert walls.is_cleared(p["cleared_partner"], "halcyon")
    assert walls.engagements_for(p["walled_analyst"]) == frozenset()
    assert walls.engagements_for(p["cross_cleared"]) == frozenset({"kestrel"})


def test_no_principal_and_unknown_engagements_are_never_cleared() -> None:
    walls = Walls(cleared={"halcyon": frozenset({"c0001"})})

    assert walls.is_cleared("c0001", "halcyon")
    assert not walls.is_cleared(None, "halcyon")
    assert not walls.is_cleared("c0001", None)
    assert not walls.is_cleared("c0001", "kestrel")


def test_load_walls_rejects_unknown_people_and_mismatched_engagements(tmp_path: Path) -> None:
    path = tmp_path / "walls.yaml"
    path.write_text("walls:\n  halcyon: [c0001]\n", encoding="utf-8")

    with pytest.raises(AccessError, match="unknown person_ids"):
        load_walls(path, person_ids=["c0002"])
    with pytest.raises(AccessError, match="no wall for restricted engagements"):
        load_walls(path, engagements=["halcyon", "kestrel"])
    with pytest.raises(AccessError, match="no document"):
        load_walls(path, engagements=[])


def test_no_engagement_codename_appears_in_the_expertise_data() -> None:
    # Expertise evidence never names a client engagement (plan §3.5.2).
    codenames = _real_walls().cleared
    text = "\n".join(p.read_text(encoding="utf-8").lower() for p in PEOPLE_DIR.glob("*.yaml"))

    assert [c for c in codenames if c in text] == []


def test_real_corpus_labels() -> None:
    docs = load_corpus(CORPUS_DIR)
    restricted = [d for d in docs if d.is_restricted]

    assert {d.path.parent.name for d in restricted} == {"engagements"}
    assert all(d.engagement for d in restricted)
    assert all(d.sensitivity == "internal" for d in docs if not d.is_restricted)
    pending = [d.path.name for d in docs if d.is_quarantined]
    assert len(pending) == 2 and all(d.path.parent.name == "case_studies" for d in docs if d.is_quarantined)


DOC = """\
---
title: T
doc_type: methodology
industry: x
topics: [t]
date: 2025-01-01
{extra}
---

## Overview

Body.
"""


def _corpus_with(tmp_path: Path, rel: str, extra: str = "") -> Path:
    path = tmp_path / rel
    path.parent.mkdir(parents=True)
    path.write_text(DOC.format(extra=extra), encoding="utf-8")
    return tmp_path


@pytest.mark.parametrize(
    ("rel", "expected"),
    [("methodology/a.md", "internal"), ("thought_leadership/a.md", "internal"), ("other/a.md", "restricted")],
)
def test_an_unlabelled_document_is_restricted_outside_the_open_directories(
    tmp_path: Path, rel: str, expected: str
) -> None:
    (doc,) = load_corpus(_corpus_with(tmp_path, rel))

    assert doc.sensitivity == expected
    assert doc.is_restricted is (expected == "restricted")


@pytest.mark.parametrize(
    ("extra", "match"),
    [
        ("sensitivity: secret", "sensitivity"),
        ("sensitivity: restricted", "engagement is not set"),
        ("engagement: halcyon", "sensitivity is not restricted"),
        ("sensitivity: restricted\nengagement: Project Halcyon", "lowercase codename"),
    ],
)
def test_invalid_access_labels_are_rejected(tmp_path: Path, extra: str, match: str) -> None:
    path = tmp_path / "a.md"
    path.write_text(DOC.format(extra=extra), encoding="utf-8")

    with pytest.raises(CorpusError, match=match):
        load_document(path)


def test_restricted_labels_reach_the_chunks(tmp_path: Path) -> None:
    from tessera.ingestion.chunker import chunk_corpus

    root = _corpus_with(tmp_path, "engagements/a.md", "sensitivity: restricted\nengagement: halcyon")
    (chunk,) = chunk_corpus(load_corpus(root))

    assert (chunk.sensitivity, chunk.engagement) == ("restricted", "halcyon")


def test_access_cases_ask_as_the_principals_they_claim() -> None:
    import sys

    sys.path.insert(0, str(REPO_ROOT))
    from evals.harness import load_cases

    walls = _real_walls()
    cases = [c for c in load_cases(REPO_ROOT / "evals" / "cases") if c.access]

    assert {c.access for c in cases} == {"leakage", "authorized", "injection"}
    for c in cases:
        assert c.principal is not None, c.id
        if c.access == "leakage":
            assert not walls.is_cleared(c.principal, c.restricted_engagement), c.id
            assert c.forbidden_markers, c.id
        if c.access == "authorized":
            assert walls.is_cleared(c.principal, c.restricted_engagement), c.id
            (source,) = c.relevant_sources
            assert source.startswith("engagements/"), c.id
        if c.access == "injection":
            assert walls.engagements_for(c.principal) == frozenset(), c.id
        for marker in c.forbidden_markers:
            assert marker.lower() not in c.query.lower(), (c.id, marker)
    leakage = {c.query for c in cases if c.access == "leakage"}
    assert leakage == {c.query for c in cases if c.access == "authorized"}
