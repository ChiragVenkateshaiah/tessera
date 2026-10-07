"""Taint-based redaction for outbound traces (Phase 5 plan §3.9.3).

Keying redaction on ``sensitivity: restricted`` isn't enough: quarantined
case studies resolve as internal, restricted text sits inside rendered
prompt strings with no label on it, and for a cleared partner the answer
*is* the restricted findings. So redaction works from a **taint set** —
the strings that must never leave — and from whether a request's
retrieval **touched** restricted or quarantined content:

- ``TaintTerms`` is the corpus-wide set, built once by a composition root:
  every restricted and quarantined document's chunk texts, the sentences
  of their bodies (so a partial quote is still caught), their titles and
  paths (a path or chunk id like ``halcyon-grocer-price-architecture::3``
  names the client), and the engagement codenames. Extra terms (the eval
  set's engagement markers) can be added.
- ``RequestTaint`` is one request's state, held in a contextvar: the
  corpus terms plus the request's own (the asker's person_id, the text of
  every restricted chunk any retrieval attempt returned), whether the
  request touched tainted content, and the placeholder numbering.
- ``Redactor`` replaces every tainted substring, case-insensitively, with
  an opaque ``[withheld-N]`` — never a chunk id or path.

Framework-free: ``langsmith_tracing`` applies this to LangSmith runs, but
nothing here knows about LangSmith.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Iterable, Mapping
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from tessera.ingestion.chunker import chunk_document
from tessera.ingestion.loader import SENSITIVITY_INTERNAL, Document
from tessera.store.base import SearchResult

# Body sentences shorter than this are left out of the taint set: short
# fragments ("## Scope", "Outcome") also occur in internal documents, and
# the full chunk texts and longer sentences already cover the content.
MIN_SENTENCE_CHARS = 24
# A run of this many consecutive words shared with a tainted text is
# redacted wherever it appears (Redactor's second pass).
SHINGLE_WORDS = 6

_WORD = re.compile(r"\w+(?:[.,'’]\w+)*")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?;:])\s+|\n+")
_LIST_MARK = re.compile(r"^[\s#>*\-|]+")


@dataclass(frozen=True)
class TaintTerms:
    """The corpus-wide taint set.

    ``texts`` match as substrings; ``words`` (codenames, person ids) match
    only as whole words, so "aurora" doesn't hit inside another word.
    ``tainted_paths`` are the document paths (as chunks and search results
    carry them) whose retrieval taints a request; restricted results taint
    it by their label as well, so a document added after this set was
    built still counts.
    """

    texts: frozenset[str] = frozenset()
    words: frozenset[str] = frozenset()
    tainted_paths: frozenset[str] = frozenset()

    def with_terms(self, texts: Iterable[str] = (), words: Iterable[str] = ()) -> TaintTerms:
        return TaintTerms(
            texts=self.texts | {t for t in texts if t.strip()},
            words=self.words | {w for w in words if w.strip()},
            tainted_paths=self.tainted_paths,
        )

    def touches(self, result: SearchResult) -> bool:
        """Whether retrieving this chunk taints the request."""
        return result.sensitivity != SENSITIVITY_INTERNAL or result.document_path in self.tainted_paths


def _sentences(body: str) -> set[str]:
    out: set[str] = set()
    for piece in _SENTENCE_SPLIT.split(body):
        piece = _LIST_MARK.sub("", piece).strip()
        if len(piece) >= MIN_SENTENCE_CHARS:
            out.add(piece)
    return out


def corpus_taint(
    documents: Iterable[Document],
    corpus_dir: Path,
    *,
    extra_texts: Iterable[str] = (),
) -> TaintTerms:
    """The taint set of a loaded corpus: every restricted or quarantined
    document's chunk texts, body sentences, title and path forms, and every
    engagement codename. ``extra_texts`` adds further substrings (e.g. the
    access eval set's engagement markers).
    """
    texts: set[str] = set(t for t in extra_texts if t.strip())
    words: set[str] = set()
    paths: set[str] = set()
    for doc in documents:
        if doc.engagement:
            words.add(doc.engagement)
        if not (doc.is_restricted or doc.is_quarantined):
            continue
        texts.update(chunk.text for chunk in chunk_document(doc) if chunk.text.strip())
        texts.update(_sentences(doc.body))
        texts.add(doc.title)
        texts.add(doc.path.stem)
        texts.add(doc.path.name)
        try:
            texts.add(doc.path.relative_to(corpus_dir).as_posix())
        except ValueError:
            pass
        paths.add(str(doc.path))
        texts.add(str(doc.path))
    return TaintTerms(texts=frozenset(texts), words=frozenset(words), tainted_paths=frozenset(paths))


class Placeholders:
    """Numbers each distinct withheld string, in order of first sight: the
    same secret gets the same ``[withheld-N]`` throughout one request."""

    def __init__(self) -> None:
        self._numbers: dict[str, int] = {}
        self._lock = threading.Lock()

    def __call__(self, secret: str) -> str:
        key = secret.lower()
        with self._lock:
            number = self._numbers.setdefault(key, len(self._numbers) + 1)
        return f"[withheld-{number}]"


@lru_cache(maxsize=8192)
def _shingles(text: str) -> frozenset[tuple[str, ...]]:
    words = [w.lower() for w in _WORD.findall(text)]
    return frozenset(
        tuple(words[i : i + SHINGLE_WORDS]) for i in range(len(words) - SHINGLE_WORDS + 1)
    )


class Redactor:
    """Replaces every tainted substring of every string — dict keys
    included — with a placeholder, in two passes:

    1. **Exact terms**, case-insensitively: chunk texts, sentences, titles,
       paths; codenames and person ids as whole words.
    2. **Fragments**: any run of ``SHINGLE_WORDS`` consecutive words that
       also occurs in a tainted text. This catches what exact matching
       can't — a quote that starts mid-sentence, or a prompt cut off at a
       length limit (a framework truncating a run name, say).
    """

    def __init__(self, terms: TaintTerms) -> None:
        self.terms = terms
        alternatives = [(t, re.escape(t)) for t in terms.texts] + [
            (w, rf"(?<![A-Za-z0-9]){re.escape(w)}(?![A-Za-z0-9])") for w in terms.words
        ]
        # Longest first, so a chunk text wins over a codename inside it.
        alternatives.sort(key=lambda a: len(a[0]), reverse=True)
        self._pattern = (
            re.compile("|".join(p for _, p in alternatives), re.IGNORECASE)
            if alternatives
            else None
        )
        self._fragments: frozenset[tuple[str, ...]] = frozenset().union(
            *(_shingles(t) for t in terms.texts)
        )

    def text(self, value: str, placeholders: Placeholders) -> str:
        if not value:
            return value
        if self._pattern is not None:
            value = self._pattern.sub(lambda m: placeholders(m.group(0)), value)
        return self._without_fragments(value, placeholders) if self._fragments else value

    def _without_fragments(self, value: str, placeholders: Placeholders) -> str:
        words = list(_WORD.finditer(value))
        spans: list[list[int]] = []
        for i in range(len(words) - SHINGLE_WORDS + 1):
            window = tuple(w.group(0).lower() for w in words[i : i + SHINGLE_WORDS])
            if window not in self._fragments:
                continue
            start, end = words[i].start(), words[i + SHINGLE_WORDS - 1].end()
            if spans and start <= spans[-1][1]:
                spans[-1][1] = max(spans[-1][1], end)
            else:
                spans.append([start, end])
        if not spans:
            return value
        out, last = [], 0
        for start, end in spans:
            out.append(value[last:start])
            out.append(placeholders(value[start:end]))
            last = end
        out.append(value[last:])
        return "".join(out)

    def value(self, value: Any, placeholders: Placeholders) -> Any:
        """``value`` with every string in it redacted (JSON-shaped data:
        dicts, lists, tuples, strings; anything else passes through)."""
        if isinstance(value, str):
            return self.text(value, placeholders)
        if isinstance(value, Mapping):
            return {
                (self.text(k, placeholders) if isinstance(k, str) else k): self.value(v, placeholders)
                for k, v in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [self.value(v, placeholders) for v in value]
        return value


@dataclass
class RequestTaint:
    """One request's taint: the corpus terms plus its own, and whether its
    retrieval touched restricted or quarantined content."""

    terms: TaintTerms
    placeholders: Placeholders = field(default_factory=Placeholders)
    touched: bool = False
    _texts: set[str] = field(default_factory=set)
    _words: set[str] = field(default_factory=set)
    _redactor: Redactor | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def add_words(self, *words: str) -> None:
        with self._lock:
            self._words.update(w for w in words if w)
            self._redactor = None

    def record_results(self, results: Iterable[SearchResult]) -> None:
        """Note what a retrieval attempt returned: any tainted chunk marks
        the request as touched, and its text joins the request's terms."""
        tainted = [r for r in results if self.terms.touches(r)]
        if not tainted:
            return
        with self._lock:
            self.touched = True
            self._texts.update(r.text for r in tainted if r.text.strip())
            self._words.update(r.engagement for r in tainted if r.engagement)
            self._redactor = None

    def redactor(self) -> Redactor:
        with self._lock:
            if self._redactor is None:
                self._redactor = Redactor(self.terms.with_terms(self._texts, self._words))
            return self._redactor


# The request being answered, set by the tracing composition-root wrapper
# for the duration of one run.
CURRENT_REQUEST: ContextVar[RequestTaint | None] = ContextVar("tessera_request_taint", default=None)
