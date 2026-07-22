"""
Backfill `site_path` and `poster_id` onto poster chunks in Qdrant.

The RAG corpus embeds each poster as a page image whose payload carries a title
and authors but nothing that ties it to a page on the website. This script reads
the website's poster manifest, matches each poster chunk to a manifest entry by
normalized title, and writes two payload fields:

  site_path  e.g. "/presentation/gem-4000-cartridge-instability"
  poster_id  e.g. "pos-gem-4000-cartridge-instability"

`query_engine._search` reads these back and puts them on every SourceRef, so a
`[n]` citation in an answer links to the right poster. Chunks without these
fields (documents, or posters not yet linked) degrade to an unlinked citation —
so running this is safe and idempotent, and re-running after any future
ingestion just re-links what's there.

This uses Qdrant's `set_payload`, a payload-only write: vectors are never
touched and no embeddings are recomputed, so there is no Voyage cost.

Matching mirrors `frontend/lib/citations.ts::normalize` exactly (NFKD →
lowercase → strip a trailing ".pdf" → collapse non-alphanumerics to spaces), so
backend and frontend agree on what "the same title" means.

The website and the corpus were built from the same source PDFs but titled
independently: for 7 posters the titles are identical, but for 8 the website
uses a shorter submission-style title. Automatic title matching handles the 7;
`TITLE_ALIASES` below is the hand-curated bridge for the other 8, each pairing
verified against the poster's author list. One corpus poster ("Using serial
patient data…") has no page on the site and stays unlinked by design.

Usage:
    # dry run (default): report matches without writing
    python scripts/link_posters.py

    # apply the payload updates
    python scripts/link_posters.py --apply

    # against a deployed manifest instead of the local file
    python scripts/link_posters.py --manifest https://<site>/posters/manifest.json --apply

Reads QDRANT_CLUSTER_ENDPOINT / QDRANT_API_KEY (and optionally
QDRANT_COLLECTION_NAME) from the environment or the repo's .env.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import unicodedata
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from qdrant_client import QdrantClient

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Default collection when QDRANT_COLLECTION_NAME is unset. This is the live
# production collection the website's backend points at.
DEFAULT_COLLECTION = "BAPa-V1"

# The manifest the website publishes, resolved relative to the sibling repo when
# both live under the same PAAN/ parent. Override with --manifest (path or URL).
DEFAULT_MANIFEST = (
    PROJECT_ROOT.parent
    / "PAAN-cembrowski"
    / "frontend"
    / "public"
    / "posters"
    / "manifest.json"
)

# Corpus title (as stored in Qdrant) → website slug, for the posters the site
# re-titled. Each pair was confirmed by matching the poster's author list across
# both sources, since the titles themselves diverge. Keyed by the raw Qdrant
# title; normalized at load. Update this if a title changes on either side —
# a --dry-run will flag any that stop matching.
TITLE_ALIASES: dict[str, str] = {
    "Barricor Blood Drawing Tube reduces preanalytic variation in Beckman Coulter AccuTnI+3 assay: a natural experiment": "barricor-beckman-accutni-preanalytic-variation",
    "Preanalytical/Biologic/Analytical Variation of patients tested with the iSTAT compared to patients tested with Gem 4000, Radiometer and Siemens for electrolytes, glucose and blood gases": "istat-vs-gem-radiometer-siemens-variation",
    "Data mining of 2 years of stat and emergency testing in multiple institutions demonstrates significant presumptive blood drawing tube-dependent preanalytic errors in hemoglobin or hematocrit measurements": "tube-preanalytic-error-hemoglobin-hematocrit",
    "Automated haematology QC survey indicates that the actionable tests (WBC, neutrophils, platelets, lymphocytes and haemoglobin) are minimally 3 sigma and that haemoglobin should be reported instead of (low sigma) haematocrit": "haematology-qc-sigma-survey",
    "Biologic Variation in platelet counts of inpatients with thrombocytopenia when monitored by the Sysmex XN 9000": "platelet-biologic-variation-thrombocytopenia",
    "Comparison of long term variation of Siemens Vista A1c patient data to long term patient variation of Sebia Capillarys 2 Flex Piercing and Roche Tina Quant Gen II A1c": "siemens-vista-a1c-long-term-variation",
    "The Average of Delta: Monitoring Assay Performance Through the Use of the Mean Intra-Individual Delta": "average-of-delta-assay-monitoring",
    "Use of machine learning to determine relevant metabolic syndrome chemistry and hematology tests associated with increased waist circumference in NHANES data(1999-2014)": "machine-learning-waist-circumference-nhanes",
}


def normalize(value: str) -> str:
    """
    Mirror of frontend/lib/citations.ts::normalize. Case, punctuation and
    whitespace differences are noise; everything else matters.
    """
    value = unicodedata.normalize("NFKD", value).lower()
    value = re.sub(r"\.pdf$", "", value)
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return value.strip()


def load_manifest(source: str) -> list[dict[str, Any]]:
    """Load the manifest from an http(s) URL or a filesystem path."""
    if source.startswith(("http://", "https://")):
        logger.info("Reading manifest: %s", source)
        resp = requests.get(source, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    else:
        path = Path(source)
        logger.info("Reading manifest: %s", path)
        data = json.loads(path.read_text(encoding="utf-8"))
    posters = data.get("posters", [])
    if not posters:
        raise SystemExit("Manifest has no posters — wrong file?")
    return posters


def slug_from_url(url: str) -> str:
    """`/presentation/<slug>` → `<slug>`."""
    return url.rstrip("/").rsplit("/", 1)[-1]


def build_manifest_indexes(
    posters: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    """
    Return (by_title, by_slug), each mapping to {poster_id, site_path, title}.
    `by_title` is keyed by normalized manifest title (the automatic match);
    `by_slug` is keyed by slug (the target of a TITLE_ALIASES lookup).
    """
    by_title: dict[str, dict[str, str]] = {}
    by_slug: dict[str, dict[str, str]] = {}
    for poster in posters:
        entry = {
            "poster_id": poster["id"],
            "site_path": poster["url"],
            "title": poster.get("title", ""),
        }
        key = normalize(entry["title"])
        if key:
            by_title[key] = entry
        by_slug[slug_from_url(poster["url"])] = entry
    return by_title, by_slug


def build_alias_index(
    by_slug: dict[str, dict[str, str]],
) -> dict[str, dict[str, str]]:
    """Map normalized corpus title → manifest entry, via the curated aliases."""
    alias_index: dict[str, dict[str, str]] = {}
    for raw_title, slug in TITLE_ALIASES.items():
        entry = by_slug.get(slug)
        if entry is None:
            raise SystemExit(
                f"TITLE_ALIASES points at slug '{slug}', which is not in the "
                "manifest. Fix the alias or rebuild the manifest."
            )
        alias_index[normalize(raw_title)] = entry
    return alias_index


def build_client() -> QdrantClient:
    endpoint = os.getenv("QDRANT_CLUSTER_ENDPOINT")
    api_key = os.getenv("QDRANT_API_KEY")
    if not endpoint:
        raise SystemExit(
            "QDRANT_CLUSTER_ENDPOINT is not set. Point --env-file at a .env with "
            "cluster credentials, or export them."
        )
    return QdrantClient(url=endpoint, api_key=api_key or None)


def iter_poster_points(client: QdrantClient, collection: str):
    """
    Yield every non-document point that carries a title. Documents (internal
    notes/code) have no public page and are skipped so we never mislabel one
    that happens to share a title with a poster.
    """
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            limit=128,
            with_payload=True,
            with_vectors=False,
            offset=offset,
        )
        for point in points:
            payload = point.payload or {}
            if payload.get("source_type") == "document":
                continue
            if not payload.get("title"):
                continue
            yield point
        if offset is None:
            break


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill site_path/poster_id onto poster chunks in Qdrant."
    )
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST),
        help="Poster manifest: a filesystem path or an http(s) URL "
        f"(default: {DEFAULT_MANIFEST}).",
    )
    parser.add_argument(
        "--collection",
        default=os.getenv("QDRANT_COLLECTION_NAME") or DEFAULT_COLLECTION,
        help="Qdrant collection to update "
        f"(default: $QDRANT_COLLECTION_NAME or '{DEFAULT_COLLECTION}').",
    )
    parser.add_argument(
        "--env-file",
        default=None,
        help="Path to a .env with Qdrant credentials. Defaults to the repo's "
        ".env, then the sibling backend's .env.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write the payload updates. Without this, runs a dry run.",
    )
    args = parser.parse_args()

    # Load credentials: explicit --env-file wins, else repo .env, else the
    # sibling backend .env (which is where the production creds live).
    if args.env_file:
        load_dotenv(args.env_file, override=True)
    else:
        load_dotenv(PROJECT_ROOT / ".env")
        load_dotenv(
            PROJECT_ROOT.parent / "PAAN-cembrowski" / "backend" / ".env",
        )

    manifest = load_manifest(args.manifest)
    by_title, by_slug = build_manifest_indexes(manifest)
    alias_index = build_alias_index(by_slug)
    logger.info("Manifest posters: %d (%d curated aliases)", len(by_slug), len(alias_index))

    client = build_client()
    logger.info("Collection: %s", args.collection)

    matched_poster_ids: set[str] = set()
    unmatched_titles: list[str] = []
    updates: list[tuple[str, dict[str, str]]] = []  # (point_id, payload)

    for point in iter_poster_points(client, args.collection):
        payload = point.payload or {}
        title = payload["title"]
        key = normalize(title)
        entry = by_title.get(key) or alias_index.get(key)
        if entry is None:
            unmatched_titles.append(title)
            continue
        matched_poster_ids.add(entry["poster_id"])
        updates.append(
            (
                str(point.id),
                {
                    "site_path": entry["site_path"],
                    "poster_id": entry["poster_id"],
                },
            )
        )

    # Report both directions of the gap.
    logger.info("")
    logger.info("Matched poster chunks: %d", len(updates))
    logger.info("Distinct posters matched: %d", len(matched_poster_ids))

    if unmatched_titles:
        logger.info("")
        logger.info("Qdrant poster chunks with NO manifest match (%d):", len(unmatched_titles))
        for title in unmatched_titles:
            logger.info("  - %s", title)

    missing = [
        entry["title"]
        for entry in by_slug.values()
        if entry["poster_id"] not in matched_poster_ids
    ]
    if missing:
        logger.info("")
        logger.info("Manifest posters NOT found in Qdrant (%d):", len(missing))
        for title in missing:
            logger.info("  - %s", title)

    if not args.apply:
        logger.info("")
        logger.info("Dry run — no changes written. Re-run with --apply to write.")
        return

    logger.info("")
    logger.info("Applying %d payload updates...", len(updates))
    for point_id, payload in updates:
        client.set_payload(
            collection_name=args.collection,
            payload=payload,
            points=[point_id],
        )
    logger.info("Done. %d chunks linked.", len(updates))


if __name__ == "__main__":
    main()
