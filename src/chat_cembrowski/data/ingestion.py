'''Ingestion module for research papers.
    Fetches papers via SerpAPI Google Scholar Author API, or creates Paper objects
    from locally-sourced PDFs by extracting metadata from the first page via LLM.
'''

import json
import os
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

        paper_id = result_id.strip() if result_id and result_id.strip() else str(uuid.uuid7())

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
    known_source_files = {p.source_file for p in existing_papers}

    new_papers: list[Paper] = []

    for pdf_path in sorted(papers_dir.glob("*.pdf")):
        if pdf_path.name in known_source_files:
            continue

        logger.info(f"Creating Paper for unregistered PDF: {pdf_path.name}")
        first_page_text = _extract_leading_text(pdf_path, client=client)

        if not first_page_text.strip():
            logger.warning(f"No extractable text in leading pages of {pdf_path.name}, skipping metadata extraction.")
            metadata: dict = {}
        else:
            metadata = _extract_metadata_with_llm(first_page_text, client)

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
        choices=["ingest_local"],
        help="Use 'ingest_local' to create Paper objects from locally-sourced PDFs.",
    )
    parser.add_argument(
        "--author-id",
        default=DEFAULT_AUTHOR_ID,
        help=f"Google Scholar Author ID to fetch papers for (default: {DEFAULT_AUTHOR_ID}).",
    )
    parser.add_argument(
        "--num-articles",
        type=int,
        default=25,
        help=(
            "Maximum number of articles to fetch (default: 25). Results are "
            "paginated, so a value larger than the author's publication count "
            "simply fetches everything. Costs ~1 SerpAPI credit per article."
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
    args = parser.parse_args()

    if args.mode == "ingest_local":
        papers = ingest_local_pdfs(reextract_authors=args.reextract_authors)
        print(f"Ingested {len(papers)} papers from local PDFs.")
    else:
        papers = fetch_author_papers(
            author_id=args.author_id,
            num_articles=args.num_articles,
            interactive=args.interactive,
        )
        print(f"Fetched {len(papers)} papers for author ID {args.author_id}.")

    for paper in papers:
        print(f"- {paper.title} ({paper.source_file}, {len(paper.text)} characters)")
