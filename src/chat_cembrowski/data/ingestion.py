'''Ingestion module for research papers.
    Fetches papers via SerpAPI Google Scholar Author API, or creates Paper objects
    from locally-sourced PDFs by extracting metadata from the first page via LLM.
'''

import csv
import json
import os
import re
from dotenv import load_dotenv
import time
import logging
from pathlib import Path
from typing import Optional
import uuid

import fitz  # PyMuPDF
import serpapi
import requests
from openai import OpenAI
from rapidfuzz import fuzz, process, utils
from . import ocr
from .models import Paper
from .serialization import save_paper, save_papers_to_json, load_papers_from_json

load_dotenv()

logger = logging.getLogger(__name__)

SERPAPI_KEY = str(os.getenv("SERPAPI_KEY"))
DEFAULT_AUTHOR_ID = "j8iA0kAAAAAJ"  # George Cembrowski's Google Scholar Author ID
DATA_DIR = Path(__file__).resolve().parents[3] / "data/papers"
SERPAPI_BASE_URL = "https://serpapi.com/search"
REQUEST_DELAY = 1  # Delay between requests in seconds
SERPAPI_AUTHOR_PAGE_SIZE = 100  # Max articles per Google Scholar Author API call
AUTHOR_TRUNCATION_MARKERS = {"...", "…"}  # SerpAPI's "and more authors" sentinels
CATALOG_PATH = Path(__file__).resolve().parents[3] / "data" / "catalog.csv"

def _get_author_articles(
        api_key: str,
        author_id: str = DEFAULT_AUTHOR_ID,
        num_articles: int = 10,
) -> list[dict]:
    """
    Calls Google Scholar Author API to get the author's articles, paginating
    until num_articles are collected or the author's list is exhausted.

    The API returns SERPAPI_AUTHOR_PAGE_SIZE articles per call at most and
    defaults to 20, so a single un-paginated call silently caps the corpus at
    20 papers no matter how large num_articles is. Requesting more than the
    author has published is safe: the loop stops when a page comes back empty
    or SerpAPI reports no next page.

    Args:
        api_key: SerpAPI API key
        author_id: Google Scholar Author ID
        num_articles: Maximum number of articles to retrieve

    Returns:
            List of dictionaries representing the author's articles.
    """

    client = serpapi.Client(api_key=api_key)
    articles: list[dict] = []
    start = 0

    while len(articles) < num_articles:
        page_size = min(SERPAPI_AUTHOR_PAGE_SIZE, num_articles - len(articles))
        results = client.search({
            "engine": "google_scholar_author",
            "author_id": author_id,
            "num": page_size,
            "start": start,
        })
        page = results.get("articles", [])
        if not page:
            break

        articles.extend(page)
        logger.info(
            f"Retrieved {len(page)} articles at offset {start} "
            f"({len(articles)} total) for author {author_id}"
        )

        if not results.get("serpapi_pagination", {}).get("next"):
            break

        start += len(page)
        time.sleep(REQUEST_DELAY)

    logger.info(f"Retrieved {len(articles)} articles for author {author_id}")
    return articles[:num_articles]

def _find_public_resource(
        article_title: str,
        api_key: str,
) -> tuple[Optional[str], Optional[str]]:
    """
    Searches for a specific paper using Google Scholar API and looks for a publicly available PDF link.

    PDF only: every downstream stage (parser, image_extractor, vectordb) opens
    the source file with PyMuPDF, so an HTML resource saved here would be
    registered as a Paper and then crash or be skipped further down the
    pipeline. Articles whose only public resource is HTML are reported and
    left for manual collection into data/papers/.

    Args:
        article_title: Title of the article to search for
        api_key: SerpAPI API key
    """

    time.sleep(REQUEST_DELAY)
    logger.info(f"Searching for public resources for article: {article_title}")
    client = serpapi.Client(api_key=api_key)
    results = client.search({
        "engine": "google_scholar",
        "q": article_title,
        "num": 1,
    })

    organic_results = results.get("organic_results", [])
    if not organic_results:
        logger.debug(f"No search results found for article: {article_title}")
        return (None, None)

    article = organic_results[0]
    resources: list[dict] = article.get("resources", [])
    result_id = article.get("result_id", "")
    saw_html = False
    for resource in resources:
        file_format: str = resource.get("file_format", "").lower()
        link: str = resource.get("link", "")
        if not link:
            continue

        if file_format == "pdf":
            logger.info(f"Found public PDF for article '{article_title}': {link}")
            return link, result_id
        if file_format == "html":
            saw_html = True

    if saw_html:
        logger.info(
            f"Only an HTML resource is available for '{article_title}' — skipping. "
            "Save a PDF into data/papers/ and run 'ingest_local' to include it."
        )
    else:
        logger.debug(f"No public resources found for article: {article_title}")
    return (None, None)

def _download_file(url: str, dest_path: Path) -> bool:
    """
    Downloads a file from *url* to *dest_path*.
    Returns True on success, False on failure.
    Skips download if the file already exists (idempotent re-runs).
    """
    if dest_path.exists():
        logger.info(f"File already exists, skipping download: {dest_path}")
        return True

    try:
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            with open(dest_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        logger.info("Saved: %s (%.1f MB)", dest_path.name, dest_path.stat().st_size / (1024 * 1024))
        return True
    except requests.RequestException as e:
        logger.warning(f"Error downloading file")
        if e.response is not None and getattr(e.response, 'status_code', None) == 403:
            logger.warning(f"Access forbidden for chosen URL")
        if dest_path.exists():
            dest_path.unlink()  # Remove incomplete file
        return False

_UNSAFE_ID_CHARS = re.compile(r"[^A-Za-z0-9._-]")


def _safe_id(raw: str) -> str:
    """
    Turns a Scholar citation_id into a filesystem-safe paper ID.

    citation_ids look like 'j8iA0kAAAAAJ:2osOgNQ5qMEC'. The colon is illegal in
    Windows filenames and paper IDs become '<id>.json', so it has to go before
    the ID is ever used as a path component.
    """
    cleaned = _UNSAFE_ID_CHARS.sub("_", raw).strip("_")
    return cleaned or str(uuid.uuid7())


def _parse_article(article: dict) -> Optional[Paper]:
    """
    Builds a catalog Paper (no PDF yet) from one Google Scholar Author entry.

    Returns None when the entry has no title, which is the only field there is
    no sensible fallback for.
    """
    title: str = article.get("title", "").strip()
    if not title:
        logger.warning(f"Article missing title, skipping: {article}")
        return None

    citation_id: str = article.get("citation_id", "").strip()
    paper_id = _safe_id(citation_id) if citation_id else str(uuid.uuid7())

    authors_raw: str = article.get("authors", "")
    authors = [a.strip() for a in authors_raw.split(",") if a.strip()]

    year_raw = str(article.get("year", "")).strip()
    try:
        year = int(year_raw) if year_raw else None
    except (TypeError, ValueError):
        year = None

    cited_by = article.get("cited_by") or {}
    citations = cited_by.get("value") if isinstance(cited_by, dict) else None

    return Paper(
        id=paper_id,
        source_file="",  # catalog entry: PDF not acquired yet
        title=title,
        authors=authors or None,
        year=year,
        publication=article.get("publication", "").strip() or None,
        scholar_link=article.get("link") or None,
        cited_by=citations,
        processed=False,
        text="",
    )


def _write_catalog(papers: list[Paper], catalog_path: Path = CATALOG_PATH) -> Path:
    """
    Writes the sourcing checklist: one row per known work, most-cited first.

    This is the artifact the manual acquisition workflow runs on, so it is
    ordered by citation count rather than by Scholar's ordering — if only part
    of the corpus can be sourced by hand, the most-cited works are worth doing
    first.
    """
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(papers, key=lambda p: (p.cited_by or 0), reverse=True)

    with open(catalog_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "id", "title", "authors", "year", "publication",
            "cited_by", "have_pdf", "pdf_url", "scholar_link",
        ])
        for p in rows:
            writer.writerow([
                p.id,
                p.title or "",
                "; ".join(p.authors or []),
                p.year or "",
                p.publication or "",
                p.cited_by if p.cited_by is not None else "",
                "yes" if p.has_pdf else "no",
                p.pdf_url or "",
                p.scholar_link or "",
            ])

    logger.info(f"Wrote catalog of {len(rows)} works to {catalog_path}")
    return catalog_path


def fetch_author_catalog(
    author_id: str = DEFAULT_AUTHOR_ID,
    api_key: str = SERPAPI_KEY,
    num_articles: int = 1000,
    json_dir: Optional[Path] = None,
    catalog_path: Path = CATALOG_PATH,
    with_pdf_links: bool = False,
    max_lookups: Optional[int] = None,
) -> list[Paper]:
    """
    Discovery without acquisition: records every work the author has, with no
    PDF download.

    This is the cheap half of ingestion. Paginating the author's full list
    costs ceil(N/100) SerpAPI searches — 4 for a 332-work profile — whereas
    resolving a public PDF link costs one search *per article*. Since publisher
    bot protection blocks nearly all of those downloads anyway, the catalog is
    usually the only part worth paying for; PDFs are then collected by hand
    into data/papers/ and attached with ingest_local_pdfs().

    Existing Papers are merged, never clobbered: a catalog re-run refreshes
    metadata but leaves source_file, text and processed intact on works whose
    PDF has already been acquired.

    Args:
        author_id: Google Scholar Author ID
        api_key: SerpAPI API key
        num_articles: Upper bound on works to fetch; overshooting is safe
        json_dir: Where Paper JSONs live (default: data/json)
        catalog_path: Where to write the CSV checklist (default: data/catalog.csv)
        with_pdf_links: Also resolve a public PDF URL per work. Costs one
            SerpAPI search each — off by default for that reason.
        max_lookups: Cap on those paid lookups, applied to the most-cited works
            first. None means no cap.

    Returns:
        The full list of catalog Paper objects.
    """
    if not api_key:
        logger.error("SERPAPI_KEY not set in environment variables.")
        return []

    if json_dir is None:
        json_dir = Path(__file__).resolve().parents[3] / "data" / "json"

    articles = _get_author_articles(api_key, author_id, num_articles)
    existing = {p.id: p for p in load_papers_from_json(json_dir)}

    papers: list[Paper] = []
    seen_titles: dict[str, str] = {}
    duplicates = 0

    for article in articles:
        paper = _parse_article(article)
        if paper is None:
            continue

        # Scholar keeps separate records for the same work (abstract vs journal
        # version), which would otherwise become duplicate Papers and duplicate
        # chunks in Qdrant.
        title_key = (paper.title or "").strip().lower()
        if title_key in seen_titles:
            duplicates += 1
            logger.info(f"Duplicate title, skipping: {paper.title}")
            continue
        seen_titles[title_key] = paper.id

        # Merge rather than overwrite: never drop an already-acquired PDF.
        prior = existing.get(paper.id)
        if prior is not None:
            paper.source_file = prior.source_file
            paper.text = prior.text
            paper.processed = prior.processed
            paper.first_page_number = prior.first_page_number
            paper.pdf_url = prior.pdf_url or paper.pdf_url
            if not _authors_incomplete(prior.authors):
                paper.authors = prior.authors

        papers.append(paper)

    if with_pdf_links:
        targets = [p for p in sorted(papers, key=lambda p: (p.cited_by or 0), reverse=True)
                   if not p.has_pdf and not p.pdf_url]
        if max_lookups is not None:
            targets = targets[:max_lookups]
        logger.info(
            f"Resolving public PDF links for {len(targets)} work(s) "
            f"— {len(targets)} SerpAPI searches."
        )
        for paper in targets:
            url, _ = _find_public_resource(paper.title or "", api_key)
            if url:
                paper.pdf_url = url

    for paper in papers:
        save_paper(paper, json_dir)

    _write_catalog(papers, catalog_path)

    with_links = sum(1 for p in papers if p.pdf_url)
    have_pdf = sum(1 for p in papers if p.has_pdf)
    logger.info(
        f"Catalog complete: {len(papers)} works "
        f"({duplicates} duplicate title(s) skipped), "
        f"{have_pdf} with a PDF already acquired, {with_links} with a public PDF link."
    )
    return papers


def fetch_author_papers(
    author_id: str = DEFAULT_AUTHOR_ID,
    api_key: str = SERPAPI_KEY,
    data_dir: Path = DATA_DIR,
    num_articles: int = 25,
    interactive: bool = False,
) -> list[Paper]:
    """
    Main function to fetch papers for a given author.

    Each article costs one extra SerpAPI search (the public-resource lookup),
    so a run over N articles spends roughly N+1 credits and takes at least
    2*N seconds from REQUEST_DELAY alone.

    Args:
        author_id: Google Scholar Author ID
        api_key: SerpAPI API key
        num_articles: Maximum number of articles to fetch
        data_dir: Directory to save downloaded papers
        interactive: Pause on each failed download so the PDF can be placed by
            hand. Off by default — publisher links 403 often enough that a
            large run would otherwise stall indefinitely on stdin.

    Returns:
        List of Paper objects.
    """

    if not api_key:
        logger.error("SERPAPI_KEY not set in environment variables.")
        return []

    data_dir.mkdir(parents=True, exist_ok=True)

    articles = _get_author_articles(api_key, author_id, num_articles)
    papers: list[Paper] = []
    failed_downloads: list[tuple[str, str]] = []

    for article in articles:

        # Parse Title
        title: str = article.get("title","").strip()
        if not title:
            logger.warning("Article missing title, skipping: %s", article)
            continue

        # Check if paper is publicly available and get download link
        resource_url, result_id = _find_public_resource(title, api_key)
        if not resource_url:
            logger.info(f"No public resource found for article '{title}', skipping.")
            continue

        # Download the paper
        filename_id = result_id or title
        filename = f"{filename_id}.pdf"
        dest_path = data_dir / filename

        success = _download_file(resource_url, dest_path)
        if not success:
            # No Paper is created: a JSON pointing at a file that was never
            # written breaks vectordb, which opens source_file unconditionally.
            logger.error(f"Failed to download paper '{title}' — not registering it.")
            logger.info(
                f"To include it, download {resource_url} manually into {data_dir} "
                "and run 'ingest_local'."
            )
            failed_downloads.append((title, resource_url))
            if interactive:
                input("Press Enter to continue")
            time.sleep(REQUEST_DELAY)
            continue

        paper_id = _safe_id(result_id.strip()) if result_id and result_id.strip() else str(uuid.uuid7())

        # Parse Authors (SerpAPI returns authors as comma-separated string)
        authors_raw: str = article.get("authors", "")
        authors: list[str] = [a.strip() for a in authors_raw.split(",") if a.strip()]

        # Parse Year
        year_raw: str = article.get("year", "").strip()
        try:
            year = int(year_raw) if year_raw else None
        except (TypeError, ValueError):
            year = None

        # Parse Publication
        publication: Optional[str] = article.get("publication", "").strip() or None

        # Create Paper object
        paper = Paper(
            id=paper_id,
            source_file=filename,
            text="",  # Placeholder, will be filled after parsing
            title=title,
            authors=authors,
            year=year,
            publication=publication,
            processed=False
        )
        papers.append(paper)
        logger.info(f"Added paper: {title}")

        time.sleep(REQUEST_DELAY)

    logger.info(f"Finished fetching papers. Total papers fetched: {len(papers)}")
    if failed_downloads:
        logger.warning(
            f"{len(failed_downloads)} article(s) had a public PDF that could not be "
            f"downloaded. Collect these by hand into {data_dir}, then run 'ingest_local':"
        )
        for title, url in failed_downloads:
            logger.warning(f"  - {title}: {url}")
    save_papers_to_json(papers)
    return papers


def _extract_leading_text(
    pdf_path: Path,
    max_pages: int = 3,
    client: Optional[OpenAI] = None,
) -> str:
    """
    Returns concatenated text from the first few pages.

    Falls back to OCR when the PDF has no text layer. A scanned cover is an
    image, so without this the document would be filed with no title, authors
    or year — and metadata is what every citation to it is built from.

    Args:
        pdf_path:  Path to the PDF.
        max_pages: How many leading pages to read.
        client:    OpenAI client used for the OCR fallback. Without one, a
                   scanned PDF simply returns "".
    """
    doc = fitz.open(str(pdf_path))
    if not doc:
        return ""
    pages = doc[:max_pages]
    res = ""
    for page in pages:
        try:
            text = str(page.get_text())
            if not text:
                logger.warning(f"No text extracted from page {page.number} of {pdf_path.name}")
                continue
            res += text
        except Exception as e:
            logger.warning(f"Error extracting text from page {page.number} of {pdf_path.name}: {e}")

    if res.strip() or client is None:
        return res

    logger.info(f"No text layer in {pdf_path.name} — reading metadata via OCR.")
    try:
        return ocr.ocr_leading_text(pdf_path, client)
    except Exception as e:
        logger.warning(f"OCR metadata extraction failed for {pdf_path.name}: {e}")
        return ""


def _extract_metadata_with_llm(first_page_text: str, client: OpenAI) -> dict:
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You extract academic paper metadata from first-page text. "
                    "Return a JSON object with keys: title (string or null), "
                    "authors (list of strings or null), year (integer or null), "
                    "publication (string or null), "
                    "first_page_number (integer or 1 — the printed page number on the first page, "
                    "e.g. 471 if the article begins on page 471 of a journal). You may have to infer the first page number from subsequent page numbers if it's not explicitly printed on the first page."
                    "Return all authors — do not truncate. "
                    "If a field cannot be determined, use null."
                ),
            },
            {"role": "user", "content": first_page_text},
        ],
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content
    if content is None:
        logger.warning("LLM metadata response content is missing.")
        return {}
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        logger.warning("Failed to parse LLM metadata response as JSON.")
        return {}


def _iter_pdfs(papers_dir: Path) -> list[Path]:
    """
    Every PDF in papers_dir, matched case-insensitively on all platforms.

    glob("*.pdf") is case-insensitive on Windows because the filesystem is, but
    case-sensitive on Linux — so a browser-saved "Paper.PDF" is picked up
    locally and silently ignored once this runs in a container. Publisher sites
    hand out uppercase extensions often enough that a hand-collected batch
    would half-ingest with no error at all.
    """
    return sorted(
        p for p in papers_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ".pdf"
    )


# rapidfuzz token_set_ratio, 0-100. Titles come from two independent sources
# (Scholar's listing and the PDF's own first page), which differ in casing,
# punctuation and trailing subtitles, but a genuine match still shares nearly
# all its tokens. Set high: a wrong attachment silently files a PDF under
# another work's citation.
TITLE_MATCH_THRESHOLD = 92


def _match_catalog_entry(title: Optional[str], candidates: list[Paper]) -> Optional[Paper]:
    """
    Finds the catalog entry a freshly-downloaded PDF belongs to, by title.

    Returns None when there is no confident match, in which case the caller
    creates a standalone Paper — an unmatched PDF is a minor annoyance, a
    mismatched one corrupts a citation.
    """
    if not title or not candidates:
        return None

    titled = [p for p in candidates if p.title]
    if not titled:
        return None

    match = process.extractOne(
        title,
        [p.title for p in titled],
        scorer=fuzz.token_set_ratio,
        processor=utils.default_process,
        score_cutoff=TITLE_MATCH_THRESHOLD,
    )
    if match is None:
        return None

    _, score, idx = match
    logger.info(f"Title matched catalog entry (score {score:.0f}): {titled[idx].title}")
    return titled[idx]


def _authors_incomplete(authors: Optional[list[str]]) -> bool:
    """
    True when an author list is missing or visibly truncated.

    SerpAPI reports "and more authors" with a trailing ellipsis, which arrives
    as its own list element after the comma split. It emits both the three-dot
    and the single-character form, hence the marker set.
    """
    if not authors:
        return True
    return any(a.strip() in AUTHOR_TRUNCATION_MARKERS for a in authors)


def ingest_local_pdfs(
    papers_dir: Path = DATA_DIR,
    json_dir: Optional[Path] = None,
    reextract_authors: bool = False,
) -> list[Paper]:
    """
    Creates Paper objects for PDFs in papers_dir that don't have a JSON entry,
    and patches missing/truncated author lists and missing first_page_number
    using first-page LLM extraction.

    Args:
        papers_dir: Directory containing PDF files (default: data/papers)
        json_dir: Directory containing JSON metadata files (default: data/json)
        reextract_authors: Re-derive authors from the PDF for every registered
            paper, not just visibly truncated ones. SerpAPI sometimes returns a
            short author list with no ellipsis to mark it, which no sentinel can
            detect; the PDF is authoritative. Costs one LLM call per paper.

    Returns:
        List of newly created Paper objects.
    """
    if json_dir is None:
        json_dir = Path(__file__).resolve().parents[3] / "data" / "json"

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        logger.error("OPENAI_API_KEY not set in environment.")
        return []

    client = OpenAI(api_key=openai_api_key)

    existing_papers = list(load_papers_from_json(json_dir))
    known_source_files = {p.source_file for p in existing_papers if p.source_file}
    # Catalog rows awaiting a PDF — a hand-downloaded file should attach to one
    # of these rather than becoming a second record for the same work.
    awaiting_pdf = [p for p in existing_papers if not p.has_pdf]

    new_papers: list[Paper] = []

    for pdf_path in _iter_pdfs(papers_dir):
        if pdf_path.name in known_source_files:
            continue

        logger.info(f"Creating Paper for unregistered PDF: {pdf_path.name}")
        first_page_text = _extract_leading_text(pdf_path, client=client)

        if not first_page_text.strip():
            logger.warning(f"No extractable text in leading pages of {pdf_path.name}, skipping metadata extraction.")
            metadata: dict = {}
        else:
            metadata = _extract_metadata_with_llm(first_page_text, client)

        matched = _match_catalog_entry(metadata.get("title"), awaiting_pdf)
        if matched is not None:
            new_filename = f"{matched.id}.pdf"
            pdf_path.rename(papers_dir / new_filename)
            logger.info(
                f"Attached {pdf_path.name} → catalog entry '{matched.title}' ({new_filename})"
            )
            matched.source_file = new_filename
            # Trust the PDF over Scholar for these two, per _extract_metadata_with_llm.
            if metadata.get("authors"):
                matched.authors = metadata["authors"]
            if metadata.get("first_page_number") is not None:
                matched.first_page_number = metadata["first_page_number"]
            save_paper(matched, json_dir)
            awaiting_pdf.remove(matched)
            known_source_files.add(new_filename)
            new_papers.append(matched)
            continue

        paper_id = str(uuid.uuid7())
        new_filename = f"{paper_id}.pdf"
        pdf_path.rename(papers_dir / new_filename)
        logger.info(f"Renamed {pdf_path.name} → {new_filename}")

        paper = Paper(
            id=paper_id,
            source_file=new_filename,
            title=metadata.get("title"),
            authors=metadata.get("authors"),
            year=metadata.get("year"),
            publication=metadata.get("publication"),
            first_page_number=metadata.get("first_page_number"),
            processed=False,
            text="",
        )
        save_paper(paper, json_dir)
        new_papers.append(paper)
        known_source_files.add(new_filename)
        logger.info(f"Created Paper: {paper.title}")

    # Patch existing papers missing first_page_number or with truncated authors
    patched_count = 0
    for paper in existing_papers:
        needs_authors = reextract_authors or _authors_incomplete(paper.authors)
        needs_first_page = paper.first_page_number is None

        if not (needs_authors or needs_first_page):
            continue

        pdf_path = papers_dir / paper.source_file
        if not pdf_path.exists():
            logger.warning(f"PDF not found for patch: {pdf_path.name}")
            continue

        first_page_text = _extract_leading_text(pdf_path, client=client)
        if not first_page_text.strip():
            logger.warning(f"No extractable text in leading pages of {pdf_path.name}, cannot patch.")
            continue

        metadata = _extract_metadata_with_llm(first_page_text, client)
        patched = False

        if needs_authors and metadata.get("authors"):
            paper.authors = metadata["authors"]
            patched = True
            logger.info(f"Patched authors for: {paper.title}")

        if needs_first_page and metadata.get("first_page_number") is not None:
            paper.first_page_number = metadata["first_page_number"]
            patched = True
            logger.info(f"Patched first_page_number for: {paper.title}")

        if patched:
            save_paper(paper, json_dir)
            patched_count += 1

    logger.info(
        f"Ingestion complete. New papers created: {len(new_papers)}. "
        f"Existing papers patched: {patched_count}."
    )
    return new_papers


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(description="Ingest research papers.")
    parser.add_argument(
        "mode",
        nargs="?",
        choices=["ingest_local", "catalog"],
        help=(
            "'ingest_local' creates Paper objects from locally-sourced PDFs. "
            "'catalog' records every work the author has without downloading "
            "anything (ceil(N/100) SerpAPI searches total)."
        ),
    )
    parser.add_argument(
        "--author-id",
        default=DEFAULT_AUTHOR_ID,
        help=f"Google Scholar Author ID to fetch papers for (default: {DEFAULT_AUTHOR_ID}).",
    )
    parser.add_argument(
        "--num-articles",
        type=int,
        default=None,
        help=(
            "Maximum number of articles to fetch (default: 25 when downloading, "
            "1000 in catalog mode — i.e. everything). Results are paginated, so "
            "a value larger than the author's publication count simply fetches "
            "everything."
        ),
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Pause on each failed download to allow placing the PDF by hand.",
    )
    parser.add_argument(
        "--reextract-authors",
        action="store_true",
        help=(
            "ingest_local only: re-derive authors from the PDF for every "
            "registered paper, not just visibly truncated ones."
        ),
    )
    parser.add_argument(
        "--with-pdf-links",
        action="store_true",
        help=(
            "catalog only: also resolve a public PDF URL per work. Costs ONE "
            "SerpAPI search per article — use --max-lookups to bound it."
        ),
    )
    parser.add_argument(
        "--max-lookups",
        type=int,
        default=None,
        help=(
            "catalog only: cap --with-pdf-links searches, spent on the "
            "most-cited works first."
        ),
    )
    args = parser.parse_args()

    # Catalog mode paginates for ceil(N/100) searches regardless of N, so the
    # default is "everything"; the download path costs a search per article and
    # keeps its conservative default.
    num_articles = args.num_articles
    if num_articles is None:
        num_articles = 1000 if args.mode == "catalog" else 25

    if args.mode == "ingest_local":
        papers = ingest_local_pdfs(reextract_authors=args.reextract_authors)
        print(f"Ingested {len(papers)} papers from local PDFs.")
    elif args.mode == "catalog":
        papers = fetch_author_catalog(
            author_id=args.author_id,
            num_articles=num_articles,
            with_pdf_links=args.with_pdf_links,
            max_lookups=args.max_lookups,
        )
        have = sum(1 for p in papers if p.has_pdf)
        links = sum(1 for p in papers if p.pdf_url)
        print(
            f"Cataloged {len(papers)} works for author ID {args.author_id} "
            f"({have} with a PDF, {links} with a public PDF link). "
            f"Checklist: {CATALOG_PATH}"
        )
    else:
        papers = fetch_author_papers(
            author_id=args.author_id,
            num_articles=num_articles,
            interactive=args.interactive,
        )
        print(f"Fetched {len(papers)} papers for author ID {args.author_id}.")

    for paper in papers:
        print(f"- {paper.title} ({paper.source_file}, {len(paper.text)} characters)")
