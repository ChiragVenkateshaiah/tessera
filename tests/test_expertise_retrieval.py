"""P3-3: evidence-strength re-rank in retrieval/expertise.py.

Deterministic unit tests run against a fake Embedder / ExpertiseStore (no
model, no disk); one integration test runs the acceptance query over the
real dataset with the real embedder.
"""

from datetime import date
from pathlib import Path

import pytest

from tessera.embedding.base import Embedder
from tessera.embedding.local import LocalEmbedder
from tessera.ingestion.expertise_loader import (
    Person,
    ProjectEntry,
    Skill,
    load_expertise,
    profile_summary_text,
)
from tessera.retrieval.expertise import (
    CANDIDATE_K,
    LEAD_ROLES,
    QueryIntent,
    find_experts,
    parse_intent,
)
from tessera.store.base import ExpertiseStore, PersonMatch
from tessera.store.chroma_expertise import ChromaExpertiseStore

ROOT = Path(__file__).resolve().parents[1]
VOCAB = ["pharma", "pricing", "retail", "cost", "supply"]


class BagEmbedder(Embedder):
    """Bag-of-words over VOCAB: identical word sets → cosine 1, disjoint → ~0."""

    @property
    def dimension(self) -> int:
        return len(VOCAB)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)

    @staticmethod
    def _vec(text: str) -> list[float]:
        return [float(w in text.lower()) + 1e-6 for w in VOCAB]


class FixedScoreStore(ExpertiseStore):
    """Returns its people in a fixed order with fixed semantic scores, and
    records the where filter it was called with."""

    def __init__(self, scored: list[tuple[Person, float]]) -> None:
        self._scored = scored
        self.last_where: object = "unset"

    def add(self, people, embeddings) -> None:  # pragma: no cover
        raise NotImplementedError

    def search(self, embedding, k, where=None) -> list[PersonMatch]:
        self.last_where = where
        return [PersonMatch(person=p, score=s) for p, s in self._scored[:k]]

    def count(self) -> int:
        return len(self._scored)


def person(
    pid: str,
    skills=(),
    projects=(),
    authored=(),
    practice: str = "pricing",
) -> Person:
    return Person(
        person_id=pid,
        name=pid,
        title="Consultant",
        office="London",
        practice=practice,
        skills=tuple(skills),
        project_history=tuple(projects),
        authored=tuple(authored),
        languages=("English",),
        last_updated=date(2026, 6, 30),
    )


def proj(
    topic: str, year: int = 2025, industry: str = "energy", role: str = "workstream lead"
) -> ProjectEntry:
    return ProjectEntry(industry=industry, topic=topic, role=role, year=year)


def rank(people_scores: list[tuple[Person, float]], query: str = "pharma pricing", **kw):
    store = FixedScoreStore(people_scores)
    result = find_experts(query, BagEmbedder(), store, **kw)
    return [m.person.person_id for m in result.matches], result, store


def test_evidenced_project_outranks_self_reported_claim_even_with_lower_semantic_score() -> None:
    claimer = person("claimer", skills=[Skill("pharma-pricing", 5, "self_reported")])
    doer = person("doer", projects=[proj("pharma-pricing")])
    ids, _, _ = rank([(claimer, 0.70), (doer, 0.55)])
    assert ids == ["doer", "claimer"]


def test_one_evidenced_project_beats_many_self_reported_claims() -> None:
    many = person(
        "many",
        skills=[
            Skill("pharma-pricing", 5, "self_reported"),
            Skill("pricing-strategy", 5, "self_reported"),
            Skill("value-based-pricing", 5, "self_reported"),
        ],
    )
    one = person("one", projects=[proj("pharma-pricing", year=2018)])
    ids, _, _ = rank([(many, 0.6), (one, 0.6)])
    assert ids[0] == "one"


def test_authored_document_counts_as_evidence() -> None:
    author = person("author", authored=["methodology/pharma-pricing-guide.md"])
    nothing = person("nothing", skills=[Skill("supply-chain", 3, "evidenced")])
    ids, result, _ = rank([(nothing, 0.6), (author, 0.5)])
    assert ids[0] == "author"
    assert result.matches[0].evidence[0].kind == "authored"


def test_recent_project_outranks_old_one() -> None:
    old = person("old", projects=[proj("pharma-pricing", year=2015)])
    new = person("new", projects=[proj("pharma-pricing", year=2026)])
    ids, _, _ = rank([(old, 0.6), (new, 0.6)])
    assert ids == ["new", "old"]


def test_industry_named_in_query_boosts_matching_projects() -> None:
    other = person("other", projects=[proj("pricing", industry="energy")])
    match = person("match", projects=[proj("pricing", industry="retail")])
    ids, _, _ = rank([(other, 0.6), (match, 0.6)], query="retail pricing")
    assert ids == ["match", "other"]


def test_where_filter_is_passed_through_to_the_store() -> None:
    p = person("a", projects=[proj("pharma-pricing")])
    _, _, store = rank([(p, 0.6)], where={"practice": "pricing"})
    assert store.last_where == {"practice": "pricing"}


def test_evidence_is_attached_and_self_reported_is_flagged() -> None:
    p = person(
        "a",
        skills=[Skill("pharma-pricing", 4, "self_reported")],
        projects=[proj("pharma-pricing", year=2024)],
    )
    _, result, _ = rank([(p, 0.6)])
    match = result.matches[0]
    kinds = {e.kind: e for e in match.evidence}
    assert set(kinds) == {"project", "skill"}
    assert "2024" in kinds["project"].description
    assert kinds["skill"].self_reported and not kinds["project"].self_reported
    assert match.is_evidenced
    assert match.evidence_score == sum(e.strength for e in match.evidence)
    assert match.score == 0.6  # raw semantic similarity preserved


def test_self_reported_only_match_is_not_evidenced() -> None:
    p = person("a", skills=[Skill("pharma-pricing", 5, "self_reported")])
    _, result, _ = rank([(p, 0.6)])
    assert not result.matches[0].is_evidenced


def test_irrelevant_evidence_is_ignored() -> None:
    p = person("a", projects=[proj("supply-chain")], skills=[Skill("supply-chain", 5, "evidenced")])
    _, result, _ = rank([(p, 0.4)], query="pharma pricing")
    assert result.matches[0].evidence == ()
    assert result.matches[0].evidence_score == 0.0


def test_k_limits_results_and_ties_break_by_person_id() -> None:
    people = [(person(f"p{i}", projects=[proj("pharma-pricing")]), 0.5) for i in (3, 1, 2)]
    ids, _, _ = rank(people, k=2)
    assert ids == ["p1", "p2"]


def test_no_candidates_returns_empty_result() -> None:
    ids, result, _ = rank([])
    assert ids == [] and result.query == "pharma pricing"


def test_candidate_pool_is_wider_than_top_k() -> None:
    assert CANDIDATE_K > 5


@pytest.fixture(scope="module")
def real_index(tmp_path_factory: pytest.TempPathFactory):
    """The real 600-person dataset, indexed once with the real embedder."""
    people = load_expertise(ROOT / "data/expertise/people", ROOT / "data/corpus")
    embedder = LocalEmbedder()
    store = ChromaExpertiseStore(tmp_path_factory.mktemp("people-index"))
    store.add(people, embedder.embed_documents([profile_summary_text(p) for p in people]))
    return people, embedder, store


def test_acceptance_pharma_pricing_evidenced_people_outrank_self_taggers(real_index) -> None:
    """Plan §5 P3-3 acceptance, on the real dataset and embedder."""
    people, embedder, store = real_index

    topic = "pharma-pricing"

    def has_real_evidence(p: Person) -> bool:
        return any(e.topic == topic for e in p.project_history) or any(
            topic in a for a in p.authored
        )

    result = find_experts("who knows pharma pricing", embedder, store, k=CANDIDATE_K)
    ranked = [m.person for m in result.matches]

    assert all(has_real_evidence(p) for p in ranked[:5])
    # People resting on self-reported skills alone (no project, authorship
    # or evidenced skill behind any relevant topic) rank below everyone
    # with real pharma-pricing project history or authored docs. People
    # evidenced on *adjacent* topics may sit anywhere between — that is
    # intended, not a violation.
    real_pos = [i for i, m in enumerate(result.matches) if has_real_evidence(m.person)]
    self_only_pos = [i for i, m in enumerate(result.matches) if not m.is_evidenced]
    assert real_pos and self_only_pos, "pool should contain both kinds of person"
    assert min(self_only_pos) > max(real_pos)


# --- query intent: "led" and "recently" ---


@pytest.mark.parametrize(
    "query,lead,recent",
    [
        ("Who's led our M&A integration engagements recently?", True, True),
        ("who ran the latest ZBB engagement", True, True),
        ("Who is our lead on pricing?", True, False),
        ("Who has managed decarbonization programs?", True, False),
        ("Who has been working recently on pricing", False, True),
        ("who knows about pharma pricing", False, False),
        ("Who has deep scenario planning experience?", False, False),
    ],
)
def test_parse_intent_reads_lead_and_recent_cues(query, lead, recent) -> None:
    assert parse_intent(query) == QueryIntent(lead=lead, recent=recent)


def test_lead_intent_prefers_a_lead_role_over_an_author_with_more_raw_evidence() -> None:
    author = person(
        "author",
        authored=["methodology/pricing-guide.md"],
        projects=[proj("pricing", role="workstream member")],
    )
    leader = person("leader", projects=[proj("pricing", role="engagement lead")])

    neutral, _, _ = rank([(author, 0.6), (leader, 0.6)], query="who knows about pricing")
    led, _, _ = rank([(author, 0.6), (leader, 0.6)], query="who led pricing")

    assert neutral[0] == "author"  # authorship + a project outweighs one lead project
    assert led[0] == "leader"  # ...until the question is specifically who LED


def test_lead_intent_lowers_rank_but_not_the_topical_evidence_score() -> None:
    """The generation floors compare evidence_score, so phrasing a question
    as "who LED" must not shrink it — that made the system answer "no
    obvious expert" for a query it had good experts for (P3 sweep,
    2026-09-30)."""
    author = person("author", authored=["methodology/pricing-guide.md"])
    _, led, _ = rank([(author, 0.6)], query="who led pricing")
    _, neutral, _ = rank([(author, 0.6)], query="who knows pricing")

    assert led.matches[0].evidence_score == neutral.matches[0].evidence_score > 0
    assert 0 < led.matches[0].rank_score < neutral.matches[0].rank_score


def test_a_lead_query_still_clears_the_generation_floor_when_experts_exist() -> None:
    from tessera.generation.expertise import filter_qualified

    leader = person(
        "leader",
        projects=[
            proj("pricing", year=2026, role="engagement lead"),
            proj("pricing", year=2025, role="engagement lead"),
        ],
    )
    _, result, _ = rank([(leader, 0.6)], query="who led pricing recently")

    assert [m.person.person_id for m in filter_qualified(result.matches)] == ["leader"]


def test_recent_intent_makes_old_projects_count_much_less() -> None:
    old = person("old", projects=[proj("pricing", year=2021, role="engagement lead")])
    new = person("new", projects=[proj("pricing", year=2026, role="engagement lead")])

    def old_share(query: str) -> float:
        _, result, _ = rank([(old, 0.6), (new, 0.6)], query=query)
        # rank_score = intent-adjusted evidence + semantic similarity; strip
        # the semantic part to compare the evidence that recency reshapes.
        scores = {m.person.person_id: m.rank_score - m.score for m in result.matches}
        return scores["old"] / scores["new"]

    assert old_share("who knows pricing") > old_share("who worked on pricing recently") * 3


def test_queries_without_cues_are_scored_exactly_as_before() -> None:
    p = person(
        "a",
        authored=["methodology/pricing-guide.md"],
        skills=[Skill("pricing", 4, "evidenced")],
        projects=[proj("pricing", year=2019, role="analyst")],
    )
    _, plain, _ = rank([(p, 0.6)], query="who knows about pricing")
    from tessera.retrieval import expertise as mod

    # The same query with the intent forced to neutral is identical.
    original = mod.parse_intent
    mod.parse_intent = lambda q: QueryIntent()
    try:
        _, forced, _ = rank([(p, 0.6)], query="who knows about pricing")
    finally:
        mod.parse_intent = original
    assert plain.matches[0].evidence_score == forced.matches[0].evidence_score


def test_lead_roles_are_the_engagement_owning_roles() -> None:
    assert LEAD_ROLES == {
        "engagement partner", "engagement lead", "engagement manager", "workstream lead"
    }
    assert "advisor" not in LEAD_ROLES and "analyst" not in LEAD_ROLES


def test_real_data_lead_recent_queries_surface_recent_leads(real_index) -> None:
    """Held-out regression on the real dataset: before intent handling the
    top 5 for these had 0-1 recent lead-role project holders on the topic;
    now most do. Checked against the raw records, not eval labels.
    """
    people, embedder, store = real_index
    by_id = {p.person_id: p for p in people}

    def recent_leads(query: str, topics: set[str]) -> int:
        top = find_experts(query, embedder, store, k=5).matches
        return sum(
            any(
                e.topic in topics and e.role in LEAD_ROLES and e.year >= 2023
                for e in by_id[m.person.person_id].project_history
            )
            for m in top
        )

    assert recent_leads("Who has led pricing strategy engagements recently?", {"pricing-strategy"}) >= 2
    assert recent_leads("Who has managed decarbonization programs recently?", {"decarbonization"}) >= 2
    assert recent_leads("Who's led our M&A integration engagements recently?", {"ma-integration"}) >= 3


def test_real_data_lead_query_is_not_gated_out_by_the_generation_floor(real_index) -> None:
    """The bug the sweep found: intent discounts pushed every score under
    EXPERTISE_QUERY_FLOOR, so the answer was "no obvious expert"."""
    from tessera.generation.expertise import filter_qualified

    _, embedder, store = real_index
    result = find_experts("Who's led our M&A integration engagements recently?", embedder, store)

    assert len(filter_qualified(result.matches)) >= 3
