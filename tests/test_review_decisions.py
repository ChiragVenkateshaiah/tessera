"""The review workflow's pure decision layer (Phase 5, plan §3.8.5):
validated decisions, deterministic ids, and a front-matter rewrite that
touches only the access and review keys."""

from __future__ import annotations

from pathlib import Path

import pytest

from tessera.ingestion.loader import load_document
from tessera.review.decisions import (
    ReviewError,
    applied_hash,
    apply_front_matter,
    content_hash,
    decision_id,
    make_decision,
)

REPO = Path(__file__).resolve().parents[1]
GROCER = REPO / "data" / "corpus" / "case_studies" / "grocer-price-tier-reset.md"
DOC = "case_studies/grocer-price-tier-reset.md"


def grocer() -> str:
    return GROCER.read_text(encoding="utf-8")


def decide(action: str, engagement: str | None = None) -> dict:
    return make_decision(DOC, content_hash(grocer()), "legal-01", action, engagement=engagement)


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "case_studies" / "doc.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_approve_marks_it_reviewed_and_internal_and_keeps_the_rest(tmp_path: Path) -> None:
    before = grocer()
    after = apply_front_matter(before, decide("approve"))
    doc = load_document(_write(tmp_path, after))

    assert doc.review_status == "reviewed" and doc.sensitivity == "internal" and doc.engagement is None
    assert not doc.is_quarantined
    # Only the review_status line changed; the body is byte-identical.
    changed = [(a, b) for a, b in zip(before.split("\n"), after.split("\n")) if a != b]
    assert changed == [("review_status: pending", "review_status: reviewed")]
    assert after.split("\n---\n", 1)[1] == before.split("\n---\n", 1)[1]


def test_reclassify_restricts_it_to_the_engagement(tmp_path: Path) -> None:
    doc = load_document(_write(tmp_path, apply_front_matter(grocer(), decide("reclassify", "halcyon"))))

    assert doc.sensitivity == "restricted" and doc.engagement == "halcyon" and doc.is_restricted
    assert doc.review_status == "reviewed"


@pytest.mark.parametrize("action", ["reject", "leave"])
def test_reject_and_leave_change_nothing(action: str) -> None:
    assert apply_front_matter(grocer(), decide(action)) == grocer()


def test_ids_are_deterministic_and_name_one_ruling_on_one_version() -> None:
    digest = content_hash(grocer())
    same = decision_id(DOC, digest, "legal-01", "approve")

    assert same == decision_id(DOC, digest, "legal-01", "approve")
    assert same != decision_id(DOC, digest, "legal-01", "reject")
    assert same != decision_id(DOC, content_hash(grocer() + " "), "legal-01", "approve")
    assert decision_id(DOC, digest, "legal-01", "reclassify", "halcyon") != decision_id(
        DOC, digest, "legal-01", "reclassify", "kestrel"
    )


def test_applied_hash_is_the_hash_of_the_output() -> None:
    d = decide("approve")
    assert applied_hash(grocer(), d) == content_hash(apply_front_matter(grocer(), d))
    assert applied_hash(grocer(), decide("reject")) == content_hash(grocer())


@pytest.mark.parametrize(
    ("action", "engagement", "match"),
    [
        ("publish", None, "unknown action"),
        ("reclassify", None, "needs an engagement"),
        ("approve", "halcyon", "takes no engagement"),
    ],
)
def test_malformed_decisions_are_refused(action: str, engagement: str | None, match: str) -> None:
    with pytest.raises(ReviewError, match=match):
        make_decision(DOC, "h", "legal-01", action, engagement=engagement)


def test_a_bad_engagement_codename_fails_before_anything_is_written() -> None:
    with pytest.raises(ReviewError, match="lowercase codename"):
        apply_front_matter(grocer(), decide("reclassify", "Not A Codename"))


def test_a_file_without_front_matter_is_refused() -> None:
    with pytest.raises(ReviewError, match="no front matter"):
        apply_front_matter("# Just a heading\n", decide("approve"))
