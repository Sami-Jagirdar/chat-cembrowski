# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

```bash
# Install dependencies and package in editable mode
uv sync

# Run data pipeline stages (order matters)
uv run -m chat_cembrowski.data.fetcher        # 1. Fetch papers via SerpAPI Google Scholar
uv run -m chat_cembrowski.data.parser         # 2. Parse PDFs → markdown, store in data/json/
uv run -m chat_cembrowski.data.vectordb       # 3. Chunk + embed + upsert to Qdrant

# Reset paper processed flags (before re-indexing)
uv run scripts/reset_paper_processed.py

# Run queries
uv run scripts/ask.py             # Edit questions in the file first
```

## Architecture

This is a RAG (Retrieval-Augmented Generation) system that answers questions about George Cembrowski's research papers. It uses **Qdrant** for vector storage, **OpenAI text-embedding-3-large** (1024-dim Matryoshka-truncated) for embeddings, and **gpt-4.1-mini** for answer synthesis.

### Package layout

```
src/chat_cembrowski/
  data/              # Data ingestion pipeline
    models.py        # Paper and Chunk dataclasses
    fetcher.py       # SerpAPI Google Scholar Author API → downloads PDFs
    parser.py        # pymupdf4llm PDF → markdown, per-page extraction
    chunker.py       # LangChain RecursiveCharacterTextSplitter (1024 chars, 128 overlap)
    serialization.py # JSON read/write for Paper objects
    vectordb.py      # Qdrant client, OpenAI embedding, batch upsert (64/batch)
  retrieval/          # Query interface
    query_engine.py  # Embed → search Qdrant → build context → generate with GPT-4.1-mini
    prompts.py       # System prompt with citation formatting rules
scripts/
  ask.py              # CLI entry point for asking questions
  reset_paper_processed.py  # Resets Paper.processed to False
data/
  papers/             # PDF source files (gitignored, kept locally)
  json/               # Paper metadata + extracted text as JSON (gitignored)
  vectors/            # Local Qdrant storage (gitignored)
extras/               # Miscellaneous files not part of the pipeline (gitignored)
```

### Data flow

1. **fetcher.py** — Calls SerpAPI's Google Scholar Author API to get Cembrowski's articles, searches each for public PDF/HTML downloads, downloads them to `data/papers/`. If a download fails, prompts the user to manually download and rename it.
2. **parser.py** — Parses each PDF via `pymupdf4llm.to_markdown()` and writes the extracted markdown into the corresponding JSON file in `data/json/`. Also provides `parse_pdf_for_pages()` for per-page extraction used by the chunker.
3. **chunker.py** — Stitches per-page text with overlap, splits with LangChain's `RecursiveCharacterTextSplitter` (markdown-aware), maps each chunk back to source page numbers via character offsets. Uses `chunk_size=1024`, `chunk_overlap=128`.
4. **vectordb.py** — Embeds chunks via OpenAI (batching 64 texts per call) and upserts them as `PointStruct`s into Qdrant collection `cembrowski_papers_v3`. Sets `Paper.processed = True` on success.

### Qdrant configuration

- **Collection**: `cembrowski_papers_v3`
- **Vector dims**: 1024 (Cosine distance)
- **Local mode**: If no `QDRANT_CLUSTER_ENDPOINT` is set, uses embedded local DB at `data/vectors/`
- **Cloud mode**: Set `QDRANT_CLUSTER_ENDPOINT` + `QDRANT_API_KEY` in `.env`

### Chunk payload structure

Each Qdrant point payload stores: `paper_id`, `title`, `authors`, `year`, `publication`, `chunk_index`, `page_start`, `page_end`, `page_label` (formatted as "p. X" or "pp. X–Y"), and `text`.

### QueryEngine

- Embeds the question with the same model/dims as indexing
- Searches Qdrant for top-10 chunks (`query_points`)
- Builds a numbered context block with title, publication, year, and page citations
- Calls `gpt-4.1-mini` with a system prompt that enforces ground-in-context answers and inline citations in `[Title, Publication, p. X]` format

### Page number note

Page numbers in chunk payloads are PDF page numbers, not journal/article page numbers. Real journal page numbers would require extracting the first page's printed number and applying an offset (listed as a future improvement).
