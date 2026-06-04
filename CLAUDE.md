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
```

## Architecture

This is a RAG (Retrieval-Augmented Generation) system that answers questions about George Cembrowski's research papers and related documents. It uses **Qdrant** for vector storage, **Voyage AI voyage-multimodal-3.5** (1024-dim) for embeddings, and **gpt-4.1-mini** for answer synthesis.

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
    query_engine.py  # Embed → search Qdrant → build context → generate with GPT-4.1-mini
    prompts.py       # System prompt with citation formatting rules
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

### QueryEngine

- Embeds the question via Voyage AI `multimodal_embed` with `input_type="query"`
- Searches Qdrant for top-10 chunks (`query_points`)
- Routes retrieved points by `chunk_category` (image) and `source_type` (document vs paper)
- Builds a numbered context block; citation format varies by type:
  - Papers: `[Title, Publication, p. X]`
  - Images: `[Title, Publication, p. X, fig.]`
  - Documents: `[Title]`
- Calls `gpt-4.1-mini` with a system prompt enforcing ground-in-context answers

### Page number note

Page numbers in chunk payloads are journal/article page numbers (offset from `first_page_number` extracted during ingestion). If `first_page_number` is unavailable, PDF page numbers are used as a fallback.
