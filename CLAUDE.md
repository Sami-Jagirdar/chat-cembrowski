# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

```bash
# Install dependencies and package in editable mode
uv sync

# Run data pipeline stages (order matters)
uv run -m chat_cembrowski.data.ingestion            # 1a. Fetch papers via SerpAPI Google Scholar
uv run -m chat_cembrowski.data.ingestion ingest_local  # 1b. Create Paper objects from locally-sourced PDFs
uv run -m chat_cembrowski.data.parser               # 2. Parse PDFs → markdown, store in data/json/
uv run -m chat_cembrowski.data.image_extractor      # 3. Extract images → data/images/ + data/image_json/
uv run -m chat_cembrowski.data.doc_ingestion        # 4. Ingest docs from data/docs/ → data/doc_json/
uv run -m chat_cembrowski.data.vectordb             # 5. Chunk + embed + upsert papers, images, and docs to Qdrant

# Reset paper processed flags (before re-indexing)
uv run scripts/reset_paper_processed.py

# Run queries
uv run scripts/ask.py             # Edit questions in the file first

# Link poster chunks to their pages on the website (payload-only, no re-embed)
uv run scripts/link_posters.py            # dry run: report matches
uv run scripts/link_posters.py --apply    # write site_path + poster_id
```

## Architecture

This is a RAG (Retrieval-Augmented Generation) system that answers questions about George Cembrowski's research papers and related documents. It uses **Qdrant** for vector storage, **Voyage AI voyage-multimodal-3.5** (1024-dim) for embeddings, and **gpt-4.1** for answer synthesis (gpt-4.1-mini for query classification/routing).

General medical questions that fall outside Cembrowski's corpus are routed to NIH/NLM sources (MedlinePlus + PubMed) instead of the Cembrowski papers — see "Query routing" below.

### Package layout

```
src/chat_cembrowski/
  data/              # Data ingestion pipeline
    models.py        # Paper, Document, Chunk, ImageRecord dataclasses
    ingestion.py     # SerpAPI fetch + local PDF bootstrap → creates Paper JSON objects
    doc_ingestion.py # Ingests txt/md/docx/code files from data/docs/ → Document JSON objects
    image_extractor.py  # Extracts images per-page via fitz, finds captions, writes ImageRecord JSONs
    parser.py        # pymupdf4llm PDF → markdown, per-page extraction
    chunker.py       # Language-aware RecursiveCharacterTextSplitter (1024 chars, 128 overlap)
    serialization.py # JSON read/write for Paper and Document objects
    vectordb.py      # Qdrant client, Voyage AI embedding, batch upsert
  retrieval/         # Query interface
    query_engine.py  # Classify → route to Cembrowski (Qdrant) or NIH → build context → generate with GPT-4.1
    prompts.py       # System prompts (Cembrowski, NIH) + classifier prompt + citation formatting rules
    nih.py           # MedlinePlus + PubMed (NCBI E-utilities) search clients for general medical questions
scripts/
  ask.py                    # CLI entry point for asking questions
  reset_paper_processed.py  # Resets Paper.processed to False
data/
  papers/     # PDF source files (gitignored, kept locally)
  json/       # Paper metadata + extracted text as JSON (gitignored)
  docs/       # Miscellaneous context documents: txt, md, docx, code files (gitignored)
  doc_json/   # Document metadata + extracted text as JSON (gitignored)
  images/     # Extracted image files (gitignored)
  image_json/ # ImageRecord metadata as JSON (gitignored)
  vectors/    # Local Qdrant storage (gitignored)
extras/       # Miscellaneous files not part of the pipeline (gitignored)
```

### Data flow

1. **ingestion.py** — Two entry points: `fetch_author_papers()` calls SerpAPI's Google Scholar Author API, downloads PDFs, and saves Paper JSON objects (authors truncated by SerpAPI are stored with `"..."` as a sentinel). `ingest_local_pdfs()` scans `data/papers/` for PDFs without a JSON entry and creates Paper objects by extracting first-page text via PyMuPDF and calling GPT-4.1-mini for structured metadata; it also patches any existing Paper whose authors list contains `"..."` using the same first-page extraction.
2. **parser.py** — Parses each PDF via `pymupdf4llm.to_markdown()` and writes the extracted markdown into the corresponding JSON file in `data/json/`. Also provides `parse_pdf_for_pages()` for per-page extraction used by the chunker.
3. **image_extractor.py** — Extracts images from each parsed PDF page via PyMuPDF, finds captions in the surrounding text, and writes `ImageRecord` JSON files to `data/image_json/`.
4. **doc_ingestion.py** — Scans `data/docs/` for supported file types and creates `Document` objects with structured extracted text. `.docx` files are converted to markdown (headings → `#`, lists → `-`, tables → markdown tables). Code files are read as-is. Saves JSON to `data/doc_json/`. Idempotent.
5. **vectordb.py** — Embeds and upserts all unprocessed content into Qdrant: paper text chunks (text batches of 64), image chunks (multimodal image+text pairs, batches of 16), and document text chunks. Uses Voyage AI `voyage-multimodal-3.5`. Sets `processed = True` on success.

### Qdrant configuration

- **Collection**: `jenna_rimkus_papers`
- **Vector dims**: 1024 (Cosine distance)
- **Local mode**: If no `QDRANT_CLUSTER_ENDPOINT` is set, uses embedded local DB at `data/vectors/`
- **Cloud mode**: Set `QDRANT_CLUSTER_ENDPOINT` + `QDRANT_API_KEY` in `.env`

### Chunk payload structure

All points share `chunk_category` ("text" or "image"), `chunk_index`, and `text`.

**Paper text chunks** additionally store: `source_type="paper"` (implicit — field absent), `paper_id`, `title`, `authors`, `year`, `publication`, `page_start`, `page_end`, `page_label`.

**Image chunks** additionally store: `chunk_category="image"`, `paper_id`, `title`, `authors`, `year`, `publication`, `page`, `page_label`, `source_file`, `bbox`, `caption`, `image_type`.

**Document text chunks** additionally store: `source_type="document"`, `doc_id`, `title`, `file_type`.

**Website link fields** (`site_path`, `poster_id`) are written onto poster chunks after ingestion by `scripts/link_posters.py`, not by `vectordb.py` — see below. They are absent until that script runs and are always optional; `_search` reads them when present and a chunk without them degrades to an unlinked citation.

### Linking poster chunks to the website (`scripts/link_posters.py`)

The website (`PAAN-cembrowski/frontend`) hosts a page per poster and publishes `public/posters/manifest.json` mapping each poster's stable `id`/`slug` to its `/presentation/<slug>` URL. To turn a retrieved chunk into a clickable citation, each poster chunk needs to know its page on the site. `link_posters.py` backfills two payload fields via Qdrant `set_payload` (**payload-only — vectors and embeddings are untouched, so there is no Voyage cost**):

- `site_path` — e.g. `/presentation/gem-4000-cartridge-instability`
- `poster_id` — e.g. `pos-gem-4000-cartridge-instability`

Matching uses the same `normalize()` as `frontend/lib/citations.ts`. The corpus and the website were titled independently: 7 posters match on title automatically; the other 8 were re-titled on the site and are bridged by a hand-curated `TITLE_ALIASES` table in the script (each pair verified by author list). A healthy run reports **15 matched, 1 unmatched each way** — the one corpus poster with no site page ("Using serial patient data…") and the one site poster not in the corpus ("Extending accurate patient-based QA…"). Re-run (dry-run first) after any re-ingestion; if a title drifts, the dry-run surfaces it as unmatched and the alias table needs a one-line update.

### QueryEngine

- `query(question)` returns a plain answer string; `query_with_route(question)` returns `(answer, route)`; `query_structured(question)` returns a `QueryResult(answer, route, sources)` where `sources` is a list of `SourceRef` numbered 1-based in the same order as the `SOURCE {i}` blocks shown to the model — so a `[i]` citation in the answer maps to `sources[i - 1]`. `route` is `"cembrowski"` or `"nih"`.
- Embeds the question via Voyage AI `multimodal_embed` with `input_type="query"`
- Searches Qdrant for top-10 chunks (`query_points`)
- Routes retrieved points by `chunk_category` (image) and `source_type` (document vs paper)
- Builds a numbered context block (`SOURCE 1`, `SOURCE 2`, …). The model cites by bracketed number only — `[1]`, `[2]` — matching those blocks; the frontend resolves each number to a link via the aligned `sources` list. The model never writes titles or URLs as citations.
- Calls `gpt-4.1` with a system prompt enforcing ground-in-context answers and markdown output

### Query routing (Cembrowski vs. NIH)

Non-technical site visitors may ask general medical questions unrelated to Cembrowski's own research; per the client's direction, those are answered from NIH sources instead of being refused.

`QueryEngine.query_with_route()` decides per-question:
1. **Classify** — a cheap `gpt-4.1-mini` call (`_classify`, prompt in `prompts.CLASSIFIER_PROMPT`) labels the question `"cembrowski"` or `"general"`. Defaults to `"cembrowski"` on API failure.
2. **Retrieve + score-check** — if labeled `"cembrowski"`, the question is embedded and searched against Qdrant as usual. The top hit's cosine score must clear `SCORE_THRESHOLD` (0.30, in `query_engine.py`) to be trusted; this catches questions that were misclassified or simply aren't covered by the corpus.
3. **Route** — a strong Cembrowski match is answered via `_answer_cembrowski` (existing `SYSTEM_PROMPT`, cites Cembrowski's papers/documents/images). Everything else — `"general"`-labeled questions, or weak/no Cembrowski matches — is answered via `_answer_nih`.

`_answer_nih` calls `nih.search_nih(question)` (`src/chat_cembrowski/retrieval/nih.py`), which queries:
- **MedlinePlus Web Service** (`wsearch.nlm.nih.gov`) — primary source; plain-language consumer health topics, no API key needed.
- **PubMed via NCBI E-utilities** (`esearch`/`efetch`) — supplementary fallback when MedlinePlus returns few/no results; technical literature abstracts. Optional `NCBI_API_KEY` / `NCBI_EMAIL` env vars raise the free rate limit (3→10 req/sec) and follow NCBI etiquette, but both services work without any key.

Results are rendered into a context block (`_build_nih_context`, same `SOURCE {i}` format as the Cembrowski path) and answered with a separate `NIH_SYSTEM_PROMPT` (in `prompts.py`) that requires plain language, the same bracketed-number citations (`[1]`, `[2]`) as the Cembrowski path — resolved to the source URLs by the frontend — and a not-medical-advice disclaimer, kept deliberately distinct from `SYSTEM_PROMPT` so NIH answers are never confused with Cembrowski's own research findings. Network failures in `nih.py` return `[]` rather than raising, so `_answer_nih` degrades to a graceful "couldn't find NIH information" message instead of crashing the request.

### Page number note

Page numbers in chunk payloads are journal/article page numbers (offset from `first_page_number` extracted during ingestion). If `first_page_number` is unavailable, PDF page numbers are used as a fallback.
