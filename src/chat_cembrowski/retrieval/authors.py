"""
Author-name recognition.

Author names are never part of what gets embedded (see chunker.py's
_build_*_embed_text — only title/year/publication are), so a vector search
has nothing to match "who is X" against. This module answers that question
with a metadata lookup instead: scroll every distinct author name out of
Qdrant, fuzzy-match the question against that list, and (on a hit) fetch
every chunk crediting that person via a payload filter — exhaustive across
the whole corpus, not just whatever lands in a top-k similarity search.
"""

from __future__ import annotations

import time
from typing import Iterator

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue
from rapidfuzz import fuzz, process, utils

# Long enough to avoid re-scrolling the collection on every question; short
# enough that a poster ingested straight into Qdrant (no backend redeploy —
# see the release flow notes) is recognized within the hour.
CACHE_TTL_SECONDS = 60 * 60

# rapidfuzz partial_ratio, 0-100. High enough to require most of a full name
# to line up (tolerating minor typos/case/spacing), so a bare common word in
# an unrelated question doesn't spuriously match someone's surname.
MATCH_THRESHOLD = 90

_cache: dict[str, tuple[float, list[str]]] = {}


def get_known_authors(client: QdrantClient, collection_name: str) -> list[str]:
    """Every distinct author name in the corpus, cached for CACHE_TTL_SECONDS."""
    now = time.time()
    cached = _cache.get(collection_name)
    if cached and now - cached[0] < CACHE_TTL_SECONDS:
        return cached[1]

    names: set[str] = set()
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection_name,
            limit=256,
            with_payload=["authors"],
            with_vectors=False,
            offset=offset,
        )
        for point in points:
            for name in (point.payload or {}).get("authors") or []:
                # SerpAPI truncates long author lists with "..." — not a name.
                if name and name != "...":
                    names.add(name)
        if offset is None:
            break

    result = sorted(names)
    _cache[collection_name] = (now, result)
    return result


def match_author(question: str, known_authors: list[str]) -> str | None:
    """
    Best fuzzy match for `question` among `known_authors`, or None if nothing
    clears MATCH_THRESHOLD. partial_ratio so a full name can match against a
    substring of a longer question; `default_process` lowercases and strips
    punctuation on both sides first, since questions arrive lowercase but
    corpus names are stored title-case ("Mark Cervinski") — without this, the
    case mismatch alone was enough to push a genuine match below threshold.
    """
    if not known_authors:
        return None

    best = process.extractOne(
        question,
        known_authors,
        scorer=fuzz.partial_ratio,
        processor=utils.default_process,
    )
    if best is None:
        return None

    name, score, _ = best
    return name if score >= MATCH_THRESHOLD else None


def fetch_chunks_by_author(
    client: QdrantClient, collection_name: str, author_name: str
) -> Iterator:
    """Yield every point whose `authors` payload list contains `author_name` exactly."""
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection_name,
            scroll_filter=Filter(
                must=[FieldCondition(key="authors", match=MatchValue(value=author_name))]
            ),
            limit=256,
            with_payload=True,
            with_vectors=False,
            offset=offset,
        )
        yield from points
        if offset is None:
            break
