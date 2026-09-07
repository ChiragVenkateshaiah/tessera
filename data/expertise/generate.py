#!/usr/bin/env python3
"""Generate the synthesized firm expertise dataset for Tessera archetype B.

Deterministic: a fixed seed plus sorted iteration over the pilot corpus,
so re-running produces byte-identical YAML. Committed alongside its output
(``data/expertise/people/*.yaml``) so the dataset is reproducible and its
shape is inspectable — see ``data/expertise/README.md`` and
``docs/Tessera_Phase3_Plan.md`` §2.

Run from the repo root:  ``python data/expertise/generate.py``

What it does NOT do: emit any client name. ``project_history`` entries
carry industry + topic + role + year and nothing else — there is no
free-text field a client name could occupy (plan §2.3). The loader
(``src/tessera/ingestion/expertise_loader.py``) enforces that structurally.
"""

from __future__ import annotations

import random
from datetime import date
from pathlib import Path

import frontmatter
import yaml

SEED = 20260907
TOTAL_PEOPLE = 600
SNAPSHOT_CEILING = date(2026, 6, 30)  # newest possible last_updated

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_DIR = REPO_ROOT / "data" / "corpus"
OUT_DIR = Path(__file__).resolve().parent / "people"

# --- firm structure -------------------------------------------------------

# (title, weight) — a seniority pyramid, junior-heavy.
TITLE_WEIGHTS: list[tuple[str, float]] = [
    ("Analyst", 0.34),
    ("Consultant", 0.30),
    ("Engagement Manager", 0.20),
    ("Principal", 0.11),
    ("Partner", 0.05),
]

# Rough years at the firm by title — bounds project-history year spread.
TENURE_YEARS: dict[str, int] = {
    "Analyst": 2,
    "Consultant": 4,
    "Engagement Manager": 7,
    "Principal": 10,
    "Partner": 11,
}

ROLE_BY_TITLE: dict[str, list[str]] = {
    "Analyst": ["analyst", "modeling support"],
    "Consultant": ["workstream member", "analyst"],
    "Engagement Manager": ["workstream lead", "engagement manager"],
    "Principal": ["engagement lead", "engagement manager"],
    "Partner": ["engagement partner", "advisor"],
}

OFFICE_WEIGHTS: list[tuple[str, float]] = [
    ("London", 0.22),
    ("New York", 0.20),
    ("Singapore", 0.12),
    ("Frankfurt", 0.10),
    ("Chicago", 0.09),
    ("Sydney", 0.08),
    ("Dubai", 0.07),
    ("Sao Paulo", 0.06),
    ("Toronto", 0.06),
]

OFFICE_LANGUAGES: dict[str, list[str]] = {
    "London": ["French", "Spanish"],
    "New York": ["Spanish", "French"],
    "Singapore": ["Mandarin", "Malay", "Tamil"],
    "Frankfurt": ["German", "French"],
    "Chicago": ["Spanish"],
    "Sydney": ["Mandarin"],
    "Dubai": ["Arabic", "Hindi", "French"],
    "Sao Paulo": ["Portuguese", "Spanish"],
    "Toronto": ["French", "Mandarin"],
}

# --- practices ----------------------------------------------------------
#
# Each practice: how many consultants sit in it, its core topics (a query
# for one of these should find this practice's people), its adjacent
# topics (weaker signal), and the filename stems of the corpus documents
# it owns (for `authored`). Every topic here is asserted against the real
# corpus topic vocabulary at run time.

PRACTICES: dict[str, dict] = {
    "pricing": {
        "count": 68,
        "core": [
            "pricing-strategy",
            "value-based-pricing",
            "price-elasticity",
            "competitive-response",
            "pharma-pricing",
        ],
        "adjacent": ["benchmarking", "analytics", "market-access", "private-label"],
        "doc_stems": [
            "pricing-strategy-overview",
            "pricing-strategy-value-based-pricing",
            "pricing-strategy-price-elasticity",
            "pricing-strategy-elasticity-calculation-reference",
            "pricing-strategy-competitive-response",
            "pharma-value-based-contracting",
            "private-label-strategy-inflation",
        ],
    },
    "cost-transformation": {
        "count": 62,
        "core": [
            "cost-transformation",
            "sg-and-a",
            "zero-based-budgeting",
            "procurement",
            "margin-improvement",
        ],
        "adjacent": ["benchmarking", "shared-services", "supply-chain"],
        "doc_stems": [
            "cost-transformation-overview",
            "cost-transformation-procurement",
            "cost-transformation-sga-benchmarking",
            "cost-transformation-sga-cost-ratio-benchmarks",
            "cost-transformation-zero-based-budgeting",
        ],
    },
    "mergers-and-acquisitions": {
        "count": 78,
        "core": [
            "due-diligence",
            "financial-dd",
            "commercial-dd",
            "synergies",
            "ma",
            "ma-integration",
            "day-1",
        ],
        "adjacent": ["integration-management", "imo", "red-flags", "risk", "governance"],
        "doc_stems": [
            "due-diligence-commercial-overview",
            "due-diligence-financial-checklist",
            "due-diligence-financial-dd-information-request-list",
            "due-diligence-red-flag-reporting",
            "due-diligence-red-flag-severity-rubric",
            "due-diligence-synergy-quantification",
            "ma-integration-day-1-readiness",
            "ma-integration-day-1-runbook",
            "ma-integration-imo-setup",
        ],
    },
    "operating-model": {
        "count": 58,
        "core": [
            "operating-model",
            "org-design",
            "raci",
            "decision-rights",
            "shared-services",
        ],
        "adjacent": ["governance", "change-management", "cost-transformation"],
        "doc_stems": [
            "operating-model-overview",
            "operating-model-org-design-principles",
            "operating-model-raci-governance",
            "operating-model-decision-rights-matrix",
            "operating-model-shared-services",
        ],
    },
    "market-entry": {
        "count": 48,
        "core": [
            "market-entry",
            "market-sizing",
            "competitive-analysis",
            "entry-mode",
            "tam-sam-som",
        ],
        "adjacent": ["benchmarking", "joint-venture", "strategy"],
        "doc_stems": [
            "market-entry-overview",
            "market-entry-market-sizing",
            "market-entry-competitive-landscape",
            "market-entry-entry-mode-selection",
            "market-entry-go-no-go-quick-reference",
        ],
    },
    "digital-and-technology": {
        "count": 82,
        "core": [
            "digital-transformation",
            "core-banking",
            "genai",
            "artificial-intelligence",
            "technology-strategy",
            "tech-capability",
        ],
        "adjacent": ["roadmap", "maturity-model", "risk-management", "enterprise-strategy"],
        "doc_stems": [
            "digital-transformation-capability-assessment",
            "digital-transformation-roadmap-sequencing",
            "core-banking-modernization-build-vs-buy",
            "genai-adoption-maturity-model",
            "genai-in-retail-banking-risk",
            "economics-of-enterprise-ai-adoption",
        ],
    },
    "strategy": {
        "count": 44,
        "core": [
            "strategic-planning",
            "scenario-planning",
            "three-horizons",
            "innovation",
            "enterprise-strategy",
        ],
        "adjacent": ["scenario-analysis", "market-entry", "framework"],
        "doc_stems": [
            "strategic-planning-scenario-planning",
            "strategic-planning-three-horizons",
            "open-banking-2030-scenarios",
        ],
    },
    "supply-chain": {
        "count": 56,
        "core": [
            "supply-chain",
            "network-optimization",
            "logistics",
            "reshoring",
            "inventory-optimization",
        ],
        "adjacent": ["procurement", "omnichannel", "modeling", "resilience"],
        "doc_stems": [
            "supply-chain-network-optimization",
            "supply-chain-network-optimization-model-specification",
            "reshoring-supply-chain-resilience",
            "omnichannel-inventory-optimization",
        ],
    },
    "organization-and-change": {
        "count": 60,
        "core": [
            "change-management",
            "communication-planning",
            "stakeholder-mapping",
            "talent-strategy",
            "workforce-planning",
        ],
        "adjacent": ["org-design", "governance"],
        "doc_stems": [
            "change-management-communication-planning",
            "change-management-stakeholder-mapping",
            "talent-strategy-workforce-planning",
        ],
    },
    "sustainability": {
        "count": 44,
        "core": ["decarbonization", "sustainability", "capital-allocation"],
        "adjacent": ["manufacturing", "scenario-planning", "risk"],
        "doc_stems": [
            "decarbonization-heavy-manufacturing",
            "decarbonization-lever-economics-faq",
        ],
    },
}

# Which industry a topic tends to sit in, when the topic is sector-bound.
# Topics not listed here draw an industry from GENERAL_INDUSTRY_WEIGHTS.
TOPIC_INDUSTRY_HINT: dict[str, list[tuple[str, float]]] = {
    "pharma-pricing": [("pharma", 0.85), ("healthcare-providers", 0.15)],
    "value-based-pricing": [("pharma", 0.45), ("healthcare-providers", 0.2), ("financial-services", 0.15), ("consumer-goods", 0.2)],
    "market-access": [("pharma", 0.7), ("healthcare-providers", 0.3)],
    "clinical-trials": [("pharma", 1.0)],
    "core-banking": [("financial-services", 1.0)],
    "open-banking": [("financial-services", 1.0)],
    "payments": [("financial-services", 0.8), ("technology", 0.2)],
    "retail-banking": [("financial-services", 1.0)],
    "private-label": [("retail", 0.8), ("consumer-goods", 0.2)],
    "omnichannel": [("retail", 0.8), ("consumer-goods", 0.2)],
    "inventory-optimization": [("retail", 0.55), ("consumer-goods", 0.25), ("industrials", 0.2)],
    "retail-real-estate": [("retail", 1.0)],
    "store-strategy": [("retail", 1.0)],
    "reshoring": [("industrials", 0.7), ("technology", 0.15), ("consumer-goods", 0.15)],
    "decarbonization": [("industrials", 0.5), ("energy", 0.35), ("consumer-goods", 0.15)],
    "sustainability": [("industrials", 0.4), ("energy", 0.3), ("consumer-goods", 0.2), ("financial-services", 0.1)],
    "manufacturing": [("industrials", 1.0)],
    "network-optimization": [("industrials", 0.5), ("consumer-goods", 0.3), ("retail", 0.2)],
    "genai": [("financial-services", 0.3), ("technology", 0.3), ("consumer-goods", 0.15), ("industrials", 0.15), ("public-sector", 0.1)],
}

GENERAL_INDUSTRY_WEIGHTS: list[tuple[str, float]] = [
    ("financial-services", 0.22),
    ("industrials", 0.18),
    ("retail", 0.14),
    ("consumer-goods", 0.12),
    ("technology", 0.11),
    ("pharma", 0.08),
    ("energy", 0.06),
    ("healthcare-providers", 0.04),
    ("telecom", 0.03),
    ("public-sector", 0.02),
]

# --- name pools -------------------------------------------------------

FIRST_NAMES = [
    "Aditi", "Ahmed", "Aisha", "Akira", "Alejandro", "Amara", "Amelia", "Ananya",
    "Andre", "Anika", "Antoine", "Arjun", "Bianca", "Camila", "Carlos", "Chen",
    "Chloe", "Daniel", "Deepa", "Diego", "Elena", "Emeka", "Emma", "Fatima",
    "Felix", "Freya", "Gabriel", "Grace", "Hana", "Hassan", "Hiroshi", "Ingrid",
    "Isabella", "Ivan", "Jamal", "Javier", "Jia", "Johan", "Julia", "Kai",
    "Kenji", "Kiran", "Lars", "Layla", "Leila", "Liam", "Lucas", "Maria",
    "Marta", "Mateo", "Maya", "Mei", "Mohammed", "Nadia", "Naomi", "Nikhil",
    "Noah", "Nora", "Olga", "Omar", "Priya", "Rafael", "Rania", "Ravi",
    "Rebecca", "Ricardo", "Rohan", "Rosa", "Sana", "Santiago", "Sara", "Sofia",
    "Sven", "Tara", "Thomas", "Valentina", "Victor", "Wei", "Yara", "Yuki",
    "Yusuf", "Zainab", "Zara", "Zoe",
]

LAST_NAMES = [
    "Abbas", "Adeyemi", "Almeida", "Andersson", "Bauer", "Bianchi", "Bouchard",
    "Chen", "Costa", "Dubois", "Eriksson", "Fernandez", "Fischer", "Fitzgerald",
    "Gallo", "Gupta", "Haddad", "Hansen", "Hoffmann", "Ibrahim", "Iyer",
    "Jensen", "Johansson", "Kaur", "Keller", "Khan", "Kim", "Kowalski",
    "Kumar", "Larsen", "Lindqvist", "Lopez", "Mahmoud", "Marchetti", "Martin",
    "Mbeki", "Mehta", "Meyer", "Moreau", "Muller", "Nakamura", "Ndiaye",
    "Nguyen", "Novak", "Okafor", "Oliveira", "Osei", "Park", "Patel",
    "Pereira", "Petrov", "Rahman", "Reyes", "Romano", "Rossi", "Saito",
    "Santos", "Schmidt", "Sharma", "Silva", "Singh", "Soto", "Suzuki",
    "Tanaka", "Torres", "Tremblay", "Vasquez", "Wagner", "Wang", "Weber",
    "Yamamoto", "Yilmaz", "Zhang", "Zimmermann",
]

THIN_PROFILE_RATE = 0.17  # share of people with a deliberately weak signal


def weighted_choice(rng: random.Random, weighted: list[tuple[str, float]]) -> str:
    return rng.choices([v for v, _ in weighted], weights=[w for _, w in weighted])[0]


def load_corpus_facts() -> tuple[frozenset[str], dict[str, list[str]]]:
    """(topic vocabulary, {corpus-relative path: its topics}) from disk."""
    topics: set[str] = set()
    doc_topics: dict[str, list[str]] = {}
    for md in sorted(CORPUS_DIR.rglob("*.md")):
        rel = md.relative_to(CORPUS_DIR).as_posix()
        post = frontmatter.load(md)
        tlist = list(post.get("topics") or [])
        doc_topics[rel] = tlist
        topics.update(tlist)
    return frozenset(topics), doc_topics


def stem_to_path(stem: str, doc_topics: dict[str, list[str]]) -> str:
    for rel in doc_topics:
        if Path(rel).stem == stem:
            return rel
    raise SystemExit(f"generate.py: corpus doc stem {stem!r} not found on disk")


def pick_industry(rng: random.Random, topic: str) -> str:
    hint = TOPIC_INDUSTRY_HINT.get(topic)
    if hint:
        return weighted_choice(rng, hint)
    return weighted_choice(rng, GENERAL_INDUSTRY_WEIGHTS)


def make_last_updated(rng: random.Random) -> date:
    year = rng.choices([2026, 2025, 2024, 2023], weights=[0.62, 0.24, 0.10, 0.04])[0]
    month = rng.randint(1, 12)
    day = rng.randint(1, 28)
    d = date(year, month, day)
    return min(d, SNAPSHOT_CEILING)


def build_person(
    rng: random.Random,
    person_id: str,
    practice: str,
    spec: dict,
    doc_paths: dict[str, list[str]],
    name: str,
) -> dict:
    title = weighted_choice(rng, TITLE_WEIGHTS)
    office = weighted_choice(rng, OFFICE_WEIGHTS)
    thin = rng.random() < THIN_PROFILE_RATE

    core = spec["core"]
    adjacent = spec["adjacent"]

    # --- skills (topics + levels; basis filled in after evidence is known)
    if thin:
        n_skills = 2
        skill_topics = rng.sample(core, 1) + rng.sample(adjacent, 1)
        skill_levels = [rng.choice([1, 2, 2, 3]) for _ in skill_topics]
    else:
        n_skills = rng.choice([2, 3, 3, 4, 4, 5])
        n_core = min(n_skills, rng.choice([1, 2, 2, 3]))
        topics = rng.sample(core, min(n_core, len(core)))
        pool = adjacent + [t for t in core if t not in topics]
        while len(topics) < n_skills and pool:
            pick = rng.choice(pool)
            pool.remove(pick)
            topics.append(pick)
        # small chance of a cross-practice core skill
        if not thin and rng.random() < 0.18:
            other = rng.choice([p for p in PRACTICES if p != practice])
            topics.append(rng.choice(PRACTICES[other]["core"]))
        skill_topics = topics
        skill_levels = []
        for t in skill_topics:
            if t in core:
                skill_levels.append(rng.choice([2, 3, 3, 4, 4, 5]))
            else:
                skill_levels.append(rng.choice([1, 2, 2, 3, 3, 4]))

    # dedupe topics, keep first level seen
    seen: dict[str, int] = {}
    for t, lvl in zip(skill_topics, skill_levels):
        seen.setdefault(t, lvl)
    skill_topics = list(seen)
    skill_levels = list(seen.values())

    # --- project history
    if thin:
        n_projects = rng.choice([0, 0, 1])
    else:
        n_projects = {
            "Analyst": rng.randint(0, 3),
            "Consultant": rng.randint(1, 4),
            "Engagement Manager": rng.randint(2, 6),
            "Principal": rng.randint(3, 7),
            "Partner": rng.randint(4, 8),
        }[title]

    earliest = max(2015, SNAPSHOT_CEILING.year - TENURE_YEARS[title])
    projects: list[dict] = []
    for _ in range(n_projects):
        r = rng.random()
        if thin:
            topic = rng.choice(adjacent)
        elif r < 0.68 and skill_topics:
            topic = rng.choice(skill_topics)
        elif r < 0.88:
            topic = rng.choice(core)
        else:
            topic = rng.choice(adjacent)
        projects.append(
            {
                "industry": pick_industry(rng, topic),
                "topic": topic,
                "role": rng.choice(ROLE_BY_TITLE[title]),
                "year": rng.randint(earliest, SNAPSHOT_CEILING.year),
            }
        )
    projects.sort(key=lambda p: (-p["year"], p["topic"]))

    # --- authored corpus docs
    authored: list[str] = []
    if not thin:
        if title in ("Engagement Manager", "Principal", "Partner"):
            base_p = 0.16
        elif title == "Consultant":
            base_p = 0.035
        else:
            base_p = 0.008
        strong = {t for t, lvl in zip(skill_topics, skill_levels) if lvl >= 3}
        for stem in spec["doc_stems"]:
            rel = stem_to_path(stem, doc_paths)
            if strong.intersection(doc_paths[rel]) and rng.random() < base_p:
                authored.append(rel)
            if len(authored) >= 3:
                break
    authored.sort()

    # --- basis: evidenced iff a project or an authored doc backs the topic
    authored_topics: set[str] = set()
    for rel in authored:
        authored_topics.update(doc_paths[rel])
    project_topics = {p["topic"] for p in projects}
    skills = []
    for t, lvl in zip(skill_topics, skill_levels):
        evidenced = t in project_topics or t in authored_topics
        skills.append(
            {"topic": t, "level": lvl, "basis": "evidenced" if evidenced else "self_reported"}
        )

    # --- languages
    langs = ["English"]
    for second in OFFICE_LANGUAGES.get(office, []):
        if rng.random() < 0.5 and len(langs) < 3:
            langs.append(second)
    if len(langs) == 1 and rng.random() < 0.25:
        langs.append(rng.choice(["French", "Spanish", "Mandarin", "German", "Arabic"]))

    return {
        "person_id": person_id,
        "name": name,
        "title": title,
        "office": office,
        "practice": practice,
        "skills": skills,
        "project_history": projects,
        "authored": authored,
        "languages": langs,
        "last_updated": make_last_updated(rng),
    }


def main() -> None:
    rng = random.Random(SEED)
    corpus_topics, doc_topics = load_corpus_facts()

    # fail loudly on any typo'd practice topic
    for practice, spec in PRACTICES.items():
        for t in spec["core"] + spec["adjacent"]:
            if t not in corpus_topics:
                raise SystemExit(
                    f"generate.py: practice {practice!r} topic {t!r} not in the corpus vocabulary"
                )
    if sum(s["count"] for s in PRACTICES.values()) != TOTAL_PEOPLE:
        raise SystemExit("generate.py: practice counts do not sum to TOTAL_PEOPLE")

    # a deterministic pool of unique names
    all_names = sorted(f"{f} {l}" for f in FIRST_NAMES for l in LAST_NAMES)
    names = rng.sample(all_names, TOTAL_PEOPLE)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for existing in OUT_DIR.glob("*.yaml"):
        existing.unlink()

    next_id = 1
    name_idx = 0
    for practice, spec in PRACTICES.items():
        records = []
        for _ in range(spec["count"]):
            pid = f"c{next_id:04d}"
            records.append(
                build_person(rng, pid, practice, spec, doc_topics, names[name_idx])
            )
            next_id += 1
            name_idx += 1
        out = OUT_DIR / f"{practice}.yaml"
        out.write_text(
            "# Generated by data/expertise/generate.py — do not edit by hand.\n"
            f"# Practice: {practice}. {len(records)} consultants.\n"
            + yaml.safe_dump(
                records,
                sort_keys=False,
                default_flow_style=False,
                allow_unicode=True,
                width=100,
            ),
            encoding="utf-8",
        )
        print(f"{practice:26} {len(records):3d} -> {out.name}")

    print(f"\n{next_id - 1} consultants written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
