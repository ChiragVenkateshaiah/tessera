"""Archetype B retrieval: find people, ranked by strength of evidence.

Structured-evidence ranking over the expertise dataset via ExpertiseStore —
not document retrieval, and not a refusal (CLAUDE.md constraint #3).

Two stages (plan §3.2):
1. Semantic candidate pool — embed the query, search profile embeddings.
2. Evidence re-rank — score each candidate by what actually backs the
   match: projects on relevant topics (recent and in a relevant industry
   count more), authored corpus documents, and skill claims, with
   self-reported skills weighted far below evidenced ones. Profile
   similarity alone can't tell a person who ran three pricing engagements
   from one who ticked a "pricing" box.

"Relevant" is decided by the same Embedder that built the profiles:
topic names, industries and authored-document names in the candidate pool
are embedded and compared to the query, mapped through a smooth ramp so
a borderline topic contributes a little rather than flipping on or off.

Pure with respect to infrastructure per CLAUDE.md constraint #6: Embedder
and ExpertiseStore are injected, nothing is read from disk or env, and
the result is plain data.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import PurePosixPath

from tessera.embedding.base import Embedder
from tessera.ingestion.expertise_loader import Person
from tessera.store.base import Evidence, ExpertiseStore, PersonMatch

# Candidate pool wide enough that a strong-evidence person sitting below
# many weak semantic matches still gets re-ranked up (the dataset has ~17%
# thin profiles that embed similarly to real experts).
CANDIDATE_K = 50
TOP_K = 5
MAX_EVIDENCE_PER_PERSON = 5

# Query↔topic cosine mapped to a [0, 1] relevance: at or below FLOOR a
# topic is noise (unrelated topics sit around 0.3–0.4 with MiniLM), at or
# above FULL it is a clear match (an exact topic name scores ~0.9).
TOPIC_SIM_FLOOR = 0.35
TOPIC_SIM_FULL = 0.65
# Industries are single common words and embed noisily, so a higher bar.
INDUSTRY_SIM_FLOOR = 0.50
INDUSTRY_SIM_FULL = 0.75

# Evidence weights. Authorship of a firm document on the topic is the
# strongest single signal; a project is next, scaled by recency and by an
# industry match; skills are a weak supplement, and a self-reported one
# barely registers — one evidenced project always outweighs any number of
# unsupported claims (test-enforced).
AUTHORED_WEIGHT = 1.5
PROJECT_WEIGHT = 1.0
INDUSTRY_BONUS = 0.5
SKILL_WEIGHT = 0.5
SELF_REPORTED_FACTOR = 0.25
# Added to the evidence score so semantic similarity still orders people
# with no topical evidence and breaks ties among equals.
SEMANTIC_WEIGHT = 1.0
# Projects lose half their weight every RECENCY_HALF_LIFE years.
RECENCY_HALF_LIFE = 5.0
# The dataset snapshot year; recency is measured against it, passed in
# rather than read from the clock so results are reproducible.
DATASET_REFERENCE_YEAR = 2026


# --- Query intent -----------------------------------------------------------
#
# "Who knows X" and "who LED X recently" are different questions. The
# first is served by topical evidence of any kind; the second is not
# served by someone who merely authored a document or supported an
# engagement, nor by a lead role from a decade ago. Two intents are read
# from the query with a small closed lexicon — deliberately lexical (not
# an LLM or embedding call) so it is transparent, free and testable, and
# so a query without these words scores exactly as it always did.
LEAD_CUES = frozenset(
    {"led", "lead", "leads", "leading", "ran", "run", "runs", "managed", "headed", "owned"}
)
RECENT_CUES = frozenset({"recent", "recently", "lately", "latest", "currently", "newest"})

# Roles that count as having led an engagement (dataset vocabulary,
# ingestion/expertise_loader.PROJECT_ROLES). Advisors and members
# supported one; they did not lead it.
LEAD_ROLES = frozenset(
    {"engagement partner", "engagement lead", "engagement manager", "workstream lead"}
)
# With lead intent, evidence that is not "leading" is discounted, not
# dropped: an author or a supporting-role holder is still relevant, just
# not what was asked for.
NON_LEAD_PROJECT_FACTOR = 0.25
NON_PROJECT_FACTOR = 0.25  # authored docs and skills under lead intent
# With recent intent, projects lose half their weight every 1.5 years
# instead of every RECENCY_HALF_LIFE.
RECENT_HALF_LIFE = 1.5


@dataclass(frozen=True)
class QueryIntent:
    """Modifiers the query asks for beyond the topic itself."""

    lead: bool = False
    recent: bool = False


def parse_intent(query: str) -> QueryIntent:
    """Read lead / recency intent from a query's words. Pure and cheap."""
    words = set(re.findall(r"[a-z]+", query.lower()))
    return QueryIntent(lead=bool(words & LEAD_CUES), recent=bool(words & RECENT_CUES))


@dataclass(frozen=True)
class ExpertiseResult:
    """Ranked people for one expertise query, best first, with the
    evidence behind each — ready for grounded generation (P3-4).
    """

    query: str
    matches: list[PersonMatch]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _ramp(sim: float, floor: float, full: float) -> float:
    return min(1.0, max(0.0, (sim - floor) / (full - floor)))


def _humanize(slug: str) -> str:
    return slug.replace("-", " ").replace("_", " ")


def _doc_label(path: str) -> str:
    return PurePosixPath(path).stem


def _relevance_tables(
    query_embedding: list[float], embedder: Embedder, people: list[Person]
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """(topic, industry, authored-path) → relevance in [0, 1], computed with
    one embed_documents call over the distinct strings in the pool.
    """
    topics = sorted(
        {s.topic for p in people for s in p.skills}
        | {e.topic for p in people for e in p.project_history}
    )
    industries = sorted({e.industry for p in people for e in p.project_history})
    docs = sorted({a for p in people for a in p.authored})

    texts = (
        [_humanize(t) for t in topics]
        + [_humanize(i) for i in industries]
        + [_humanize(_doc_label(d)) for d in docs]
    )
    if not texts:
        return {}, {}, {}
    vectors = embedder.embed_documents(texts)
    sims = [_cosine(query_embedding, v) for v in vectors]

    n_t, n_i = len(topics), len(industries)
    topic_rel = {
        t: _ramp(s, TOPIC_SIM_FLOOR, TOPIC_SIM_FULL) for t, s in zip(topics, sims[:n_t])
    }
    industry_rel = {
        i: _ramp(s, INDUSTRY_SIM_FLOOR, INDUSTRY_SIM_FULL)
        for i, s in zip(industries, sims[n_t : n_t + n_i])
    }
    doc_rel = {
        d: _ramp(s, TOPIC_SIM_FLOOR, TOPIC_SIM_FULL)
        for d, s in zip(docs, sims[n_t + n_i :])
    }
    return topic_rel, industry_rel, doc_rel


def _gather_evidence(
    person: Person,
    topic_rel: dict[str, float],
    industry_rel: dict[str, float],
    doc_rel: dict[str, float],
    reference_year: int,
    intent: QueryIntent = QueryIntent(),
) -> list[Evidence]:
    found: list[Evidence] = []
    non_project = NON_PROJECT_FACTOR if intent.lead else 1.0
    half_life = RECENT_HALF_LIFE if intent.recent else RECENCY_HALF_LIFE

    for path in person.authored:
        rel = doc_rel.get(path, 0.0)
        if rel > 0:
            found.append(
                Evidence(
                    "authored", f"authored {path}", AUTHORED_WEIGHT * rel * non_project
                )
            )

    for entry in person.project_history:
        rel = topic_rel.get(entry.topic, 0.0)
        if rel <= 0:
            continue
        age = max(0, reference_year - entry.year)
        recency = 0.5 ** (age / half_life)
        boost = 1.0 + INDUSTRY_BONUS * industry_rel.get(entry.industry, 0.0)
        role = (
            NON_LEAD_PROJECT_FACTOR
            if intent.lead and entry.role not in LEAD_ROLES
            else 1.0
        )
        found.append(
            Evidence(
                "project",
                f"{entry.topic} project in {entry.industry}, "
                f"{entry.year} ({entry.role})",
                PROJECT_WEIGHT * rel * recency * boost * role,
            )
        )

    for skill in person.skills:
        rel = topic_rel.get(skill.topic, 0.0)
        if rel <= 0:
            continue
        self_reported = skill.basis == "self_reported"
        factor = SELF_REPORTED_FACTOR if self_reported else 1.0
        label = "self-reported" if self_reported else "evidenced"
        found.append(
            Evidence(
                "skill",
                f"{skill.topic} skill, level {skill.level} ({label})",
                SKILL_WEIGHT * (skill.level / 5) * rel * factor * non_project,
                self_reported=self_reported,
            )
        )
    return found


def find_experts(
    query: str,
    embedder: Embedder,
    store: ExpertiseStore,
    where: dict[str, object] | None = None,
    k: int = TOP_K,
    reference_year: int = DATASET_REFERENCE_YEAR,
) -> ExpertiseResult:
    """Return the top-k people for query, ranked by evidence strength.

    where pre-filters the candidate pool on structured fields the store
    supports (practice, office, title) — the B-path analogue of A's
    metadata filtering, for queries that name a practice or location.
    Each match carries the specific Evidence that surfaced it and an
    evidence_score (topical strength, intent-independent) and rank_score
    (what ordered them); ``score`` stays the raw semantic similarity.

    The query's words can shift the re-rank (parse_intent): "led/ran/
    managed" discounts everything that is not a lead-level role, and
    "recently/latest" makes recency decay faster. A query with none of
    those words is scored exactly as before.
    """
    intent = parse_intent(query)
    query_embedding = embedder.embed_query(query)
    candidates = store.search(query_embedding, CANDIDATE_K, where=where)
    if not candidates:
        return ExpertiseResult(query=query, matches=[])

    topic_rel, industry_rel, doc_rel = _relevance_tables(
        query_embedding, embedder, [c.person for c in candidates]
    )

    ranked: list[tuple[float, PersonMatch]] = []
    neutral = QueryIntent()
    for cand in candidates:
        evidence = sorted(
            _gather_evidence(
                cand.person, topic_rel, industry_rel, doc_rel, reference_year, intent
            ),
            key=lambda e: e.strength,
            reverse=True,
        )
        rank_evidence = sum(e.strength for e in evidence)
        # The intent-adjusted total orders the shortlist, but "is there
        # expertise here at all" must not depend on how the question is
        # phrased: discounting for "led" shrinks every score, and the
        # generation floors were calibrated on the un-discounted scale.
        topical = (
            rank_evidence
            if intent == neutral
            else sum(
                e.strength
                for e in _gather_evidence(
                    cand.person, topic_rel, industry_rel, doc_rel, reference_year, neutral
                )
            )
        )
        rank_score = rank_evidence + SEMANTIC_WEIGHT * cand.score
        match = PersonMatch(
            person=cand.person,
            score=cand.score,
            evidence=tuple(evidence[:MAX_EVIDENCE_PER_PERSON]),
            evidence_score=topical,
            rank_score=rank_score,
        )
        ranked.append((rank_score, match))

    ranked.sort(key=lambda r: (-r[0], r[1].person.person_id))
    return ExpertiseResult(query=query, matches=[m for _, m in ranked[:k]])
