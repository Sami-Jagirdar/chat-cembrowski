'''Fetcher for downloading research papers. 
    Uses SerpAPI's Google Scholar Author API to search for an authors papers, 
    Finds he papers that are publicly available as a PDF/HTML, downloads them,
    and saves a list of Papers objects ready for downstream processing.  
'''

import os
from dotenv import load_dotenv
import time
import logging
from pathlib import Path
from typing import Optional

import serpapi
import requests
from .models import Paper
from .serialization import save_papers_to_json

load_dotenv()

logger = logging.getLogger(__name__)

SERPAPI_KEY = os.getenv("SERPAPI_KEY")
DEFAULT_AUTHOR_ID = "j8iA0kAAAAAJ"  # George Cembrowski's Google Scholar Author ID
DATA_DIR = Path(__file__).resolve().parents[3] / "data/papers"
SERPAPI_BASE_URL = "https://serpapi.com/search"
REQUEST_DELAY = 1  # Delay between requests in seconds

def _get_author_articles(
        api_key: str,
        author_id: str = DEFAULT_AUTHOR_ID,
        num_articles: int = 10,
) -> list[dict]:
    """
    Calls Google Scholar Author API to get the author's top articles.

    Args:
        api_key: SerpAPI API key
        author_id: Google Scholar Author ID
        num_articles: Number of articles to retrieve

    Returns:
            List of dictionaries representing the author's articles.
    """

    client = serpapi.Client(api_key=api_key)
    results = client.search({
        "engine": "google_scholar_author",
        "author_id": author_id,
    })
    articles = results.get("articles", [])
    logger.info(f"Retrieved {len(articles)} articles for author {author_id}")
    return articles[:num_articles]

def _find_public_resource(
        article_title: str,
        api_key: str,
) -> tuple[Optional[str], Optional[str]]:
    """ 
    Searches for a specific paper using Google Scholar API and looks for a publicly available PDF or HTML link.
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
    for resource in resources:
        file_format: str = resource.get("file_format", "").lower()
        link: str = resource.get("link", "")
        

        if file_format in ["pdf", "html"] and link:
            logger.info(f"Found public resource for article '{article_title}': {link} ({file_format})")
            return link, result_id
        
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
        if e.response.status_code == 403:
            logger.warning(f"Access forbidden for chosen URL")
        if dest_path.exists():
            dest_path.unlink()  # Remove incomplete file
        return False

def fetch_author_papers(
    author_id: str = DEFAULT_AUTHOR_ID,
    api_key: str = SERPAPI_KEY,
    data_dir: Path = DATA_DIR,
    num_articles: int = 25,
) -> list[Paper]:
    """
    Main function to fetch papers for a given author.
    Args:
        author_id: Google Scholar Author ID
        api_key: SerpAPI API key
        num_articles: Number of articles to fetch
        data_dir: Directory to save downloaded papers

    Returns:
        List of Paper objects.
    """

    if not api_key:
        logger.error("SERPAPI_KEY not set in environment variables.")
        return []
    
    data_dir.mkdir(parents=True, exist_ok=True)

    articles = _get_author_articles(api_key, author_id, num_articles)
    papers: list[Paper] = []

    for article in articles:        

        # Parse Title
        title: str = article.get("title","").strip()
        if not title:
            logger.warning("Article missing title, skipping: %s", article)
            continue
            
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

        # Check if paper is publicly available and get download link
        resource_url, result_id = _find_public_resource(title, api_key)
        if not resource_url:
            logger.info(f"No public resource found for article '{title}', skipping.")
            continue

        # Download the paper
        ext = "html" if resource_url.lower().endswith(".html") else "pdf"
        filename = result_id + f".{ext}"
        dest_path = data_dir / filename

        success = _download_file(resource_url, dest_path)
        if not success:
            logger.error(f"Failed to download paper '{title}'.")
            logger.info(f"Please manually download the article at url: {resource_url}. Rename it to {filename} and save it in {data_dir} to include it in the dataset.")
            input("Press Enter to continue")
            
        # Create Paper object
        paper = Paper(
            id=result_id,
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
    save_papers_to_json(papers) # TODO: should maybe be stored in a db later instead
    return papers
        

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    papers = fetch_author_papers()
    print(f"Fetched {len(papers)} papers for author ID {DEFAULT_AUTHOR_ID}.")

    for paper in papers:
        print(f"- {paper.title} ({paper.source_file}, {len(paper.text)} characters)")
        

