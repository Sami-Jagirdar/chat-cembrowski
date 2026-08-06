"""
OCR support for scanned PDFs that carry no text layer.

`pymupdf4llm` returns nothing for a page that is a photograph of paper, so a
scanned source would otherwise reach the chunker with empty text and be skipped
outright. This module fills that gap: it renders each page to an image and
transcribes it with GPT-4.1 vision, returning markdown the rest of the pipeline
can treat exactly like parser output.

Two wrinkles this handles that a naive "render and OCR" would not:

**Two-up spreads.** Scanned books are commonly captured a facing pair at a time,
so one PDF page holds two book pages side by side. Splitting at the gutter
before OCR doubles the effective resolution the model sees, keeps it from
reading across the fold into the facing column, and makes each unit a real book
page that can carry its own citation.

**Printed page numbers.** The folio printed on the page is what a citation
should name, and it rarely matches the PDF index (front matter, two-up capture).
The transcription reports it, and it is preferred over the sequential position.

Results are cached to disk per rendered image, so a re-run costs nothing and an
interrupted run resumes where it stopped.
"""

from __future__ import annotations

import base64
import json
import logging
import re
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import fitz
from openai import OpenAI

from .parser import MAX_PAGE_IMAGE_PIXELS, PAGE_IMAGE_PIXEL_SAFETY_MARGIN

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OCR_CACHE_DIR = PROJECT_ROOT / "data" / "ocr_cache"

OCR_MODEL = "gpt-4.1"

# ~216 DPI (72 * zoom). Higher than the 2.0 used for born-digital page renders:
# the model is reading these glyphs rather than being handed a text layer
# alongside them, so legibility matters more than render cost.
DEFAULT_OCR_ZOOM = 3.0

# A page wider than this ratio is treated as a two-up spread and split at the
# gutter. Portrait book pages sit near 0.7 and square posters at 1.0, so a
# landscape sheet holding a facing pair is unambiguous.
SPREAD_ASPECT_RATIO = 1.15

# Each half extends this fraction of the sheet width past the midpoint, so a
# scan whose fold sits slightly off-centre still captures its full text block.
# Well inside a book's inner margin, so it only ever picks up white space.
GUTTER_OVERLAP_FRACTION = 0.01

# Below this fraction of non-white pixels a rendered half is blank (the empty
# side of a title page, a chapter verso) and not worth an API call.
BLANK_INK_FRACTION = 0.001

# How many leading sheets to transcribe when recovering metadata off a scanned
# cover. Two spreads covers title page plus copyright page.
METADATA_SHEETS = 2

# Generous, because the failure this absorbs is a token-per-minute rate limit
# rather than a genuine error: on a low tier a book-length run spends most of
# its wall clock waiting for quota, and a page that exhausts its attempts is
# dropped from the document entirely.
OCR_RETRIES = 8
OCR_RETRY_BACKOFF_SECONDS = 4
OCR_RETRY_BACKOFF_CAP_SECONDS = 60

# Pages are transcribed concurrently — each is an independent request, and a
# book-length document is otherwise bounded by round-trip latency alone. Low by
# default because throughput here is capped by the account's tokens-per-minute
# allowance, not by latency: past that ceiling, more workers only convert
# successful requests into 429s. Raise it if the key has TPM headroom.
DEFAULT_OCR_WORKERS = 3


OCR_SYSTEM_PROMPT = """\
You transcribe a scanned page of a printed book or document into markdown.

Transcribe VERBATIM. Never summarise, correct, modernise, or omit content.

Rules:
- Reproduce the body text as flowing markdown paragraphs. Rejoin words broken by
  a hyphen at a line break; do not preserve the original line wrapping.
- Use markdown headings for section headings, and preserve numbered or bulleted
  lists (including reference lists) with their original numbering.
- Render tables as markdown tables. If a table is too irregular for that, lay it
  out as text and keep the column relationships legible.
- Keep mathematical and statistical notation readable in plain text: write
  subscripts and superscripts inline (1_2s, CS^1.0s_2.7s, s_a), and keep Greek
  letters as their characters (alpha as α, chi as χ, delta as Δ).
- For each figure, chart, or photograph, emit a block that begins with its
  verbatim caption in bold, then one or two factual sentences describing what it
  shows — axis labels and units, the series plotted, the trend or comparison it
  demonstrates. Describe only what is visible. This makes the figure findable by
  search; do not blur it into the surrounding body text.
- OMIT the running head and the printed page number from the markdown body.
  They repeat on every page and add nothing to a retrieved passage. Report the
  page number in the `printed_page` field instead.
- If the page is blank, or holds only a running head, return an empty markdown
  string.

Return a JSON object with exactly these keys:
  "markdown"     - the transcription, as a string. Empty string if blank.
  "printed_page" - the page number printed on the page, as an integer. Use null
                   if none is printed. Roman numerals in front matter should be
                   converted to their integer value only if unambiguous;
                   otherwise use null.
"""

OCR_USER_PROMPT = (
    "Transcribe this page. Return only the JSON object described in your instructions."
)


def _clamped_zoom(width: float, height: float, zoom: float) -> float:
    """
    Reduce `zoom` if it would render an area past the embedding pixel cap.

    Mirrors the per-page clamp in `parser.render_pdf_pages_as_images`: these
    renders are also fed to Voyage, which rejects oversized images, and page
    dimensions vary enough across sources that a fixed zoom is not safe.
    """
    area = width * height
    if area <= 0:
        return zoom
    budget = MAX_PAGE_IMAGE_PIXELS * PAGE_IMAGE_PIXEL_SAFETY_MARGIN
    if area * (zoom**2) <= budget:
        return zoom
    return (budget / area) ** 0.5


def _is_blank(image_path: Path) -> bool:
    """
    True when a rendered half holds essentially no ink.

    Only used to skip an API call. A false negative costs one wasted request and
    is caught downstream by the empty-transcription check, so the threshold is
    deliberately conservative — scanner edge shadow alone should not clear it.
    """
    try:
        import PIL.Image

        with PIL.Image.open(image_path) as img:
            thumb = img.convert("L").resize((120, 120))
        dark = sum(1 for pixel in thumb.getdata() if pixel < 200)
        return dark / (120 * 120) < BLANK_INK_FRACTION
    except Exception as e:
        logger.warning(f"Blank check failed for {image_path.name}: {e}")
        return False


def has_text_layer(pdf_path: Path, sample_size: int = 8, min_chars: int = 30) -> bool:
    """
    Report whether a PDF carries an extractable text layer.

    Samples pages spread across the document rather than reading all of them, so
    this stays cheap on a large book. A scanned source returns no text on every
    sampled page; a born-digital one returns text on essentially all of them.

    Args:
        pdf_path:   Path to the PDF.
        sample_size: How many pages to sample across the document.
        min_chars:  Characters on a page below which it counts as textless.

    Returns:
        True if any sampled page yields text, False for a scanned document.
        True on failure, so an unreadable file falls through to the normal path
        rather than silently triggering an expensive OCR run.
    """
    try:
        doc = fitz.open(str(pdf_path))
    except Exception as e:
        logger.warning(f"Could not open {pdf_path.name} to check for text: {e}")
        return True

    try:
        count = doc.page_count
        if count == 0:
            return True
        step = max(1, count // sample_size)
        for index in range(0, count, step):
            if len(doc[index].get_text().strip()) >= min_chars:
                return True
        return False
    finally:
        doc.close()


def unit_key(sheet: int, half: str) -> str:
    """
    Stable human-readable name for one logical page, e.g. "144R" or "145".

    This is what `--exclude-units` matches on, so it has to be something a
    person can read off a contact sheet and type.
    """
    return f"{sheet}{half}"


def render_page_units(
    pdf_path: Path,
    out_dir: Path,
    paper_id: str,
    zoom: float = DEFAULT_OCR_ZOOM,
    split_spreads: bool | None = None,
    first_sheet: int | None = None,
    last_sheet: int | None = None,
    exclude_units: set[str] | None = None,
) -> list[dict]:
    """
    Render a scanned PDF into one image per logical page.

    When a sheet holds a two-up spread it is split at the gutter, so one sheet
    yields two images. Blank halves are dropped.

    `first_sheet`/`last_sheet`/`exclude_units` exist because a scanned book is
    rarely just the book: loose inserts, product literature and clippings get
    captured along with it. Anything not part of the work has to be cut, or it
    ends up cited under the book's title and page numbering.

    Args:
        pdf_path:      Path to the source PDF.
        out_dir:       Directory to write PNGs into (created if missing).
        paper_id:      Used to build stable, unique filenames.
        zoom:          Render scale factor (72 DPI * zoom), clamped per page.
        split_spreads: Force spread splitting on or off. None auto-detects from
                       the page aspect ratio.
        first_sheet:   First 1-based PDF sheet to render (inclusive).
        last_sheet:    Last 1-based PDF sheet to render (inclusive).
        exclude_units: Unit keys to skip, e.g. {"144R"}. See `unit_key`.

    Returns:
        List of dicts ordered by reading order, each with: 'sheet' (1-based PDF
        page), 'half' ("L", "R", or "" when not split), 'sequence' (1-based
        position among logical pages), 'image_file' (filename within out_dir).
    """
    doc = fitz.open(str(pdf_path))
    out_dir.mkdir(parents=True, exist_ok=True)

    sheet_count = doc.page_count
    excluded = {key.upper() for key in (exclude_units or set())}
    units: list[dict] = []
    sequence = 0
    skipped: list[str] = []

    try:
        for page in doc:
            sheet = page.number + 1
            if first_sheet is not None and sheet < first_sheet:
                continue
            if last_sheet is not None and sheet > last_sheet:
                continue
            rect = page.rect

            if split_spreads is None:
                is_spread = rect.height > 0 and (rect.width / rect.height) > SPREAD_ASPECT_RATIO
            else:
                is_spread = split_spreads

            if is_spread:
                midpoint = (rect.x0 + rect.x1) / 2
                margin = rect.width * GUTTER_OVERLAP_FRACTION
                clips = [
                    ("L", fitz.Rect(rect.x0, rect.y0, midpoint + margin, rect.y1)),
                    ("R", fitz.Rect(midpoint - margin, rect.y0, rect.x1, rect.y1)),
                ]
            else:
                clips = [("", rect)]

            for half, clip in clips:
                if unit_key(sheet, half).upper() in excluded:
                    skipped.append(unit_key(sheet, half))
                    continue

                effective_zoom = _clamped_zoom(clip.width, clip.height, zoom)
                pix = page.get_pixmap(
                    matrix=fitz.Matrix(effective_zoom, effective_zoom), clip=clip
                )
                filename = f"{paper_id}_s{sheet:04d}{half}.png"
                image_path = out_dir / filename
                pix.save(str(image_path))

                if _is_blank(image_path):
                    logger.debug(f"Blank page unit, dropping: {filename}")
                    image_path.unlink(missing_ok=True)
                    continue

                sequence += 1
                units.append(
                    {
                        "sheet": sheet,
                        "half": half,
                        "sequence": sequence,
                        "image_file": filename,
                    }
                )
    finally:
        doc.close()

    scope = ""
    if first_sheet is not None or last_sheet is not None:
        scope = f" (sheets {first_sheet or 1}-{last_sheet or sheet_count})"
    if skipped:
        scope += f", excluding {', '.join(skipped)}"
    logger.info(
        f"Rendered {len(units)} page units from {sheet_count} sheets "
        f"of {pdf_path.name}{scope}."
    )
    return units


def _encode_image(image_path: Path) -> str:
    return base64.b64encode(image_path.read_bytes()).decode("ascii")


def transcribe_image(
    image_path: Path,
    client: OpenAI,
    cache_dir: Path | None = OCR_CACHE_DIR,
    model: str = OCR_MODEL,
) -> dict:
    """
    Transcribe one rendered page image to markdown via GPT-4.1 vision.

    Args:
        image_path: PNG to transcribe.
        client:     Active OpenAI client.
        cache_dir:  Directory for cached transcriptions, keyed by image
                    filename. None disables caching.
        model:      Vision-capable chat model.

    Returns:
        Dict with 'markdown' (str) and 'printed_page' (int or None). On failure
        after retries, 'markdown' is empty and 'error' carries the reason —
        failures are never cached, so a later run retries them.
    """
    cache_path = cache_dir / f"{image_path.stem}.json" if cache_dir else None
    if cache_path and cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning(f"Corrupt OCR cache entry, re-transcribing: {cache_path.name}")

    data_url = f"data:image/png;base64,{_encode_image(image_path)}"
    last_error = ""

    for attempt in range(1, OCR_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                max_tokens=4096,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": OCR_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": OCR_USER_PROMPT},
                            {
                                "type": "image_url",
                                # "high" detail is what makes body text legible;
                                # the default downsamples past readability.
                                "image_url": {"url": data_url, "detail": "high"},
                            },
                        ],
                    },
                ],
            )
            payload = json.loads(response.choices[0].message.content or "{}")
            result = {
                "markdown": (payload.get("markdown") or "").strip(),
                "printed_page": _coerce_page(payload.get("printed_page")),
            }
            if cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(
                    json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            return result

        except Exception as e:
            last_error = str(e)
            if _is_rate_limit(e):
                logger.debug(
                    f"Rate limited on {image_path.name} "
                    f"(attempt {attempt}/{OCR_RETRIES}); backing off."
                )
            else:
                logger.warning(
                    f"OCR attempt {attempt}/{OCR_RETRIES} failed for {image_path.name}: {e}"
                )
            if attempt < OCR_RETRIES:
                time.sleep(_retry_delay(e, attempt))

    logger.error(f"OCR failed for {image_path.name} after {OCR_RETRIES} attempts.")
    return {"markdown": "", "printed_page": None, "error": last_error}


def _retry_delay(error: Exception, attempt: int) -> float:
    """
    Seconds to wait before retrying a failed transcription.

    A rate-limit response states how long to wait ("Please try again in 2.52s");
    honouring that is both faster and gentler than guessing. Everything else
    backs off exponentially. A little padding on the server's figure keeps a
    pool of workers from all waking at the same instant and colliding again.
    """
    hint = re.search(r"try again in ([\d.]+)s", str(error))
    if hint:
        return min(float(hint.group(1)) + 0.5, OCR_RETRY_BACKOFF_CAP_SECONDS)
    return min(
        OCR_RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1)), OCR_RETRY_BACKOFF_CAP_SECONDS
    )


def _is_rate_limit(error: Exception) -> bool:
    """Rate limits are expected traffic shaping on a long run, not a fault."""
    text = str(error)
    return "rate_limit_exceeded" in text or "429" in text


def _coerce_page(value) -> int | None:
    """Accept an int, a numeric string, or nothing for the printed page number."""
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def repair_folios(pages: list[dict]) -> int:
    """
    Fix printed page numbers that contradict their neighbours, in place.

    Transcription reads the folio off the page, and on a page dominated by a
    figure or a table it occasionally picks up the wrong number — a figure
    number, an axis value, a table cell. Roughly 1% of pages in practice, but
    each one is a citation pointing at the wrong page of a real book.

    The repair only fires where the answer is unambiguous: the pages either
    side are numbered and differ by exactly two, so the page between them can
    only be the value in the middle. That also fills in a page whose folio is
    genuinely absent, as chapter openers often are.

    Deliberately narrow. It will not touch a numbering discontinuity (front
    matter meeting the body, say), because there the neighbours do not bracket
    cleanly — and inventing a number there would be worse than having none.

    Args:
        pages: Page dicts from `ocr_pdf`, in reading order. Mutated in place.

    Returns:
        How many folios were repaired.
    """
    repaired = 0
    for i in range(1, len(pages) - 1):
        previous = pages[i - 1]["printed_page"]
        following = pages[i + 1]["printed_page"]
        if previous is None or following is None or following - previous != 2:
            continue

        implied = previous + 1
        if pages[i]["printed_page"] == implied:
            continue

        logger.warning(
            f"Folio on sheet {pages[i]['sheet']}{pages[i]['half']} reads "
            f"{pages[i]['printed_page']}, but sits between {previous} and "
            f"{following} — correcting to {implied}."
        )
        pages[i]["printed_page"] = implied
        repaired += 1

    return repaired


def ocr_pdf(
    pdf_path: Path,
    paper_id: str,
    images_dir: Path,
    client: OpenAI,
    cache_dir: Path | None = OCR_CACHE_DIR,
    zoom: float = DEFAULT_OCR_ZOOM,
    split_spreads: bool | None = None,
    first_sheet: int | None = None,
    last_sheet: int | None = None,
    exclude_units: set[str] | None = None,
    model: str = OCR_MODEL,
    max_workers: int = DEFAULT_OCR_WORKERS,
) -> list[dict]:
    """
    Render and transcribe an entire scanned PDF.

    Args:
        pdf_path:      Path to the scanned PDF.
        paper_id:      Paper ID, used for image filenames and the cache key.
        images_dir:    Directory to render page images into. These are also what
                       the multimodal embed step reads, so they are kept.
        client:        Active OpenAI client.
        cache_dir:     Transcription cache directory; None disables caching.
        zoom:          Render scale factor.
        split_spreads: Force spread splitting on or off; None auto-detects.
        first_sheet:   First 1-based PDF sheet to ingest (inclusive).
        last_sheet:    Last 1-based PDF sheet to ingest (inclusive).
        exclude_units: Unit keys to skip, e.g. {"144R"}.
        model:         Vision-capable chat model.
        max_workers:   Pages transcribed concurrently.

    Returns:
        List of page dicts in reading order, each with 'sequence', 'sheet',
        'half', 'image_file', 'text' (markdown) and 'printed_page'. Pages that
        transcribed to nothing are dropped; their rendered images are removed so
        no blank page reaches the embed step. 'sequence' is renumbered
        contiguously from 1 over what survives.
    """
    if cache_dir:
        cache_dir = cache_dir / paper_id

    units = render_page_units(
        pdf_path,
        images_dir,
        paper_id,
        zoom=zoom,
        split_spreads=split_spreads,
        first_sheet=first_sheet,
        last_sheet=last_sheet,
        exclude_units=exclude_units,
    )
    if not units:
        logger.warning(f"No renderable page units in {pdf_path.name}.")
        return []

    # Transcribe concurrently, then reassemble in reading order. Slots start as
    # a failure so a page whose future never lands is reported, not silently
    # treated as a blank and dropped from the document.
    results: list[dict] = [
        {"markdown": "", "printed_page": None, "error": "not transcribed"}
        for _ in units
    ]
    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                transcribe_image,
                images_dir / unit["image_file"],
                client,
                cache_dir,
                model,
            ): index
            for index, unit in enumerate(units)
        }
        for future in as_completed(futures):
            index = futures[future]
            try:
                results[index] = future.result()
            except Exception as e:  # transcribe_image handles its own errors
                logger.error(f"Unexpected OCR failure for {units[index]['image_file']}: {e}")
                results[index] = {"markdown": "", "printed_page": None, "error": str(e)}
            completed += 1
            if completed % 25 == 0:
                logger.info(f"Transcribed {completed}/{len(units)} page units...")

    pages: list[dict] = []
    failures: list[str] = []

    for unit, result in zip(units, results):
        image_path = images_dir / unit["image_file"]

        if result.get("error"):
            failures.append(unit["image_file"])
            continue
        if not result["markdown"]:
            logger.debug(f"Empty transcription, dropping: {unit['image_file']}")
            image_path.unlink(missing_ok=True)
            continue

        # Renumber over surviving pages only: blanks and failures leave gaps in
        # the render-time sequence, and this is the fallback page number when a
        # page has no printed folio.
        pages.append(
            {
                **unit,
                "sequence": len(pages) + 1,
                "text": result["markdown"],
                "printed_page": result["printed_page"],
            }
        )

    # Runs on every load rather than being folded into the cache, which stays a
    # verbatim record of what the model returned.
    repair_folios(pages)

    if failures:
        logger.error(
            f"{len(failures)} page unit(s) failed to transcribe and were skipped: "
            f"{', '.join(failures[:10])}{' ...' if len(failures) > 10 else ''}. "
            "Re-run to retry them — successful pages are cached."
        )

    logger.info(f"OCR complete for {pdf_path.name}: {len(pages)} page(s) transcribed.")
    return pages


def ocr_leading_text(
    pdf_path: Path,
    client: OpenAI,
    max_sheets: int = METADATA_SHEETS,
    model: str = OCR_MODEL,
) -> str:
    """
    Transcribe the opening sheets of a scanned PDF, for metadata extraction.

    A scanned cover yields nothing to `fitz.get_text()`, so title, authors and
    publication have to be read off the image. Renders to a temporary directory
    and does not cache: this runs once per document, at ingestion.

    Args:
        pdf_path:   Path to the scanned PDF.
        client:     Active OpenAI client.
        max_sheets: How many leading sheets to transcribe.
        model:      Vision-capable chat model.

    Returns:
        Concatenated markdown from the leading pages, or "" if nothing was read.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        units = render_page_units(pdf_path, tmp_dir, "probe", last_sheet=max_sheets)
        parts = [
            result["markdown"]
            for unit in units
            if (
                result := transcribe_image(
                    tmp_dir / unit["image_file"], client, cache_dir=None, model=model
                )
            )["markdown"]
        ]

    return "\n\n".join(parts)
