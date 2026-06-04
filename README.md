RAG Query System for answering questions based on George Cembrowski's publications and related documents.

## Installation

Install uv if not already available:
```
python -m pip install uv
```
Install project dependencies and the package in editable mode:
```
uv sync
```

## Environment

Create a `.env` file in the project root with the following variables:
```
SERPAPI_KEY=...
OPENAI_API_KEY=...
VOYAGE_API_KEY=...
QDRANT_API_KEY=...           # only needed for cloud mode
QDRANT_CLUSTER_ENDPOINT=...  # only needed for cloud mode
```

Accounts needed: SerpAPI (paper fetching), OpenAI (metadata extraction + answer generation), Voyage AI (embeddings), Qdrant (vector store — local mode works without an account).

## Data Structure

```
data/
  papers/     # PDF source files
  json/       # Paper metadata + extracted text as JSON
  docs/       # Miscellaneous context documents (txt, md, docx, code files)
  doc_json/   # Document metadata + extracted text as JSON
  images/     # Extracted image files
  image_json/ # Image metadata as JSON
  vectors/    # Local Qdrant storage (when not using cloud)
```

## Design Decisions and Models Used

- **Vector Embeddings**: Voyage AI `voyage-multimodal-3.5` (1024 dimensions) — handles text and image+text pairs in the same vector space
- **Answer Generation**: OpenAI `gpt-4.1-mini`
- **Chunking**: Language-aware recursive chunking — code files use language-specific splitters (Python, JS, TS, C++, Java, etc.), prose and docx use a markdown-aware splitter. `CHUNK_SIZE=1024`, `CHUNK_OVERLAP=128`
- **Retrieval**: Top 10 chunks from Qdrant vector search
- **PDF Parsing**: `pymupdf4llm` — extracts text, tables, and charts as structured markdown with automatic OCR if needed

## Running the Data Pipeline

### Research Papers

Run from the root of the repository in order:

1. `uv run -m chat_cembrowski.data.ingestion`
   Fetches paper metadata via SerpAPI Google Scholar. You may need to manually download some PDFs and place them in `data/papers/` as instructed by the script.

2. `uv run -m chat_cembrowski.data.ingestion ingest_local`
   Creates Paper objects for any PDFs in `data/papers/` not yet registered, extracting metadata from the first page via GPT-4.1-mini.

3. `uv run -m chat_cembrowski.data.parser`
   Parses each PDF to markdown and stores it in the Paper JSON.

4. `uv run -m chat_cembrowski.data.image_extractor`
   Extracts images from each PDF, finds captions, and writes ImageRecord JSONs to `data/image_json/`.

5. `uv run -m chat_cembrowski.data.vectordb`
   Chunks, embeds, and upserts everything to Qdrant.

To re-index papers from scratch, reset the processed flag first:
```
uv run scripts/reset_paper_processed.py
```

### Miscellaneous Documents

Place any `.txt`, `.md`, `.docx`, or code files in `data/docs/`, then run:

1. `uv run -m chat_cembrowski.data.doc_ingestion`
   Extracts structured text from each file. `.docx` files are converted to markdown (headings, lists, tables preserved). Code files are split with a language-aware splitter. Saves Document JSONs to `data/doc_json/`.

2. `uv run -m chat_cembrowski.data.vectordb`
   Embeds and upserts the new document chunks alongside any unprocessed papers.

Both steps are idempotent — already-processed files are skipped.

## Querying the System

Edit the questions in `scripts/ask.py`, then run:
```
uv run scripts/ask.py
```

The system retrieves across papers, images, and documents in a single search. Citations are formatted by source type:
- Papers: `[Title, Publication, p. X]`
- Images/figures: `[Title, Publication, p. X, fig.]`
- Documents/notes: `[Title]`

## Future Steps

- Metadata filtering for retrieval (filter by year, publication, source type)
- Reranker: retrieve top 30, rerank to top 10 with a model like `bge-reranker`
