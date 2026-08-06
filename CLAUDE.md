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

# Target a non-default collection (the code default is the production 'BAPa-V1')
uv run -m chat_cembrowski.data.vectordb --collection some-other-collection

# Routing eval — run after any change to CLASSIFIER_PROMPT, the corpus,
# SCORE_THRESHOLD, the embedding model, or the chunking strategy
uv run scripts/eval_routing.py                      # routing only: cheap and fast
uv run scripts/eval_routing.py --provider openai    # A/B the two providers
uv run scripts/eval_routing.py --full               # + answers and citation integrity

# Ingest a scanned (image-only) PDF. --method auto detects the missing text
# layer; the flags below trim loose material captured alongside the work.
uv run -m chat_cembrowski.data.vectordb --collection BAPa-V1 \
    --paper-id <id> --first-sheet 1 --last-sheet 144 --exclude-units 144R

# Reset paper processed flags (before re-indexing)
uv run scripts/reset_paper_processed.py

# Run queries
uv run scripts/ask.py             # Edit questions in the file first

# Link poster chunks to their pages on the website (payload-only, no re-embed)
uv run scripts/link_posters.py            # dry run: report matches
uv run scripts/link_posters.py --apply    # write site_path + poster_id
```

## Architecture

This is a RAG (Retrieval-Augmented Generation) system that answers questions about George Cembrowski's research papers and related documents. It uses **Qdrant** for vector storage, **Voyage AI voyage-multimodal-3.5** (1024-dim) for embeddings, and **Gemini via OpenRouter** for answer synthesis — `gemini-3.6-flash` for answers, `gemini-3.5-flash-lite` for classification/condensing. See "LLM provider" below and the README section "Which models the assistant uses, and why".

The **data pipeline still calls OpenAI directly** (OCR in `data/ocr.py`, metadata extraction in `data/ingestion.py`), so `OPENAI_API_KEY` is required regardless of `LLM_PROVIDER`. Only the retrieval path is switchable.

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
    ocr.py           # Scanned-PDF fallback: spread splitting + GPT-4.1 vision transcription
    chunker.py       # Language-aware RecursiveCharacterTextSplitter (1024 chars, 128 overlap)
    serialization.py # JSON read/write for Paper and Document objects
    vectordb.py      # Qdrant client, Voyage AI embedding, batch upsert
  retrieval/         # Query interface
    llm.py           # Provider factory (OpenRouter/OpenAI), model config, reasoning-effort control
    query_engine.py  # Classify → route to Cembrowski (Qdrant) or NIH → build context → generate
    prompts.py       # System prompts (Cembrowski, NIH) + classifier prompt + citation formatting rules
    nih.py           # MedlinePlus + PubMed (NCBI E-utilities) search clients for general medical questions
    authors.py       # Fuzzy author-name matching + payload-filtered chunk fetch (no LLM calls)
scripts/
  ask.py                    # CLI entry point for asking questions
  chat.py                   # Interactive multi-turn REPL
  eval_routing.py           # Routing eval — see "Routing eval" below
  eval_questions.json       # Labeled question set backing eval_routing.py
  reset_paper_processed.py  # Resets Paper.processed to False
data/
  papers/     # PDF source files (gitignored, kept locally)
  json/       # Paper metadata + extracted text as JSON (gitignored)
  docs/       # Miscellaneous context documents: txt, md, docx, code files (gitignored)
  doc_json/   # Document metadata + extracted text as JSON (gitignored)
  images/     # Extracted image files (gitignored)
  image_json/ # ImageRecord metadata as JSON (gitignored)
  page_images/# Full-page renders used by page-based and scanned chunking (gitignored)
  ocr_cache/  # Per-page OCR transcriptions, keyed by paper ID (gitignored)
  vectors/    # Local Qdrant storage (gitignored)
extras/       # Miscellaneous files not part of the pipeline (gitignored)
```

### Data flow

1. **ingestion.py** — Two entry points: `fetch_author_papers()` calls SerpAPI's Google Scholar Author API, downloads PDFs, and saves Paper JSON objects (authors truncated by SerpAPI are stored with `"..."` as a sentinel). `ingest_local_pdfs()` scans `data/papers/` for PDFs without a JSON entry and creates Paper objects by extracting first-page text via PyMuPDF and calling GPT-4.1-mini for structured metadata; it also patches any existing Paper whose authors list contains `"..."` using the same first-page extraction.
2. **parser.py** — Parses each PDF via `pymupdf4llm.to_markdown()` and writes the extracted markdown into the corresponding JSON file in `data/json/`. Also provides `parse_pdf_for_pages()` for per-page extraction used by the chunker.
3. **image_extractor.py** — Extracts images from each parsed PDF page via PyMuPDF, finds captions in the surrounding text, and writes `ImageRecord` JSON files to `data/image_json/`.
4. **doc_ingestion.py** — Scans `data/docs/` for supported file types and creates `Document` objects with structured extracted text. `.docx` files are converted to markdown (headings → `#`, lists → `-`, tables → markdown tables). Code files are read as-is. Saves JSON to `data/doc_json/`. Idempotent.
5. **ocr.py** — Fallback for PDFs with no text layer. `has_text_layer()` samples pages to detect a scan; `ocr_pdf()` renders each page and transcribes it with GPT-4.1 vision, returning markdown in the same per-page shape `parser.parse_pdf_for_pages()` produces. `ocr_leading_text()` is the same treatment for a scanned cover, so `ingest_local_pdfs` can still recover title/authors/year. See "Scanned PDFs" below.
6. **vectordb.py** — Embeds and upserts all unprocessed content into Qdrant: paper text chunks (text batches of 64), image chunks (multimodal image+text pairs, batches of 16), and document text chunks. Uses Voyage AI `voyage-multimodal-3.5`. Sets `processed = True` on success.

### LLM provider (`retrieval/llm.py`)

Both providers speak the OpenAI chat-completions schema — OpenRouter natively — so the OpenAI
SDK is already the abstraction layer and swapping providers is a `base_url` plus a model string.
`llm.get_config()` reads the environment; `llm.get_llm_client()` builds the client.

- `LLM_PROVIDER` — `openrouter` (default) or `openai`. Setting `openai` is the one-line rollback
  to `gpt-4.1` / `gpt-4.1-mini`. An unrecognized value raises rather than falling back, so a typo
  fails at startup instead of sending traffic somewhere unintended.
- `CHAT_MODEL` / `CLASSIFIER_MODEL` — override the per-provider defaults.
- `LLM_REASONING_EFFORT` — thinking budget for synthesis (`none`…`xhigh`, default `minimal`).
  The classifier and condenser are pinned to `llm.CLASSIFIER_REASONING_EFFORT` and ignore it, so
  raising the global setting can't quietly add seconds to every query's critical path.

`llm.completion_extras(config, effort=...)` returns the `extra_body` reasoning block on
OpenRouter and `{}` on OpenAI (which rejects unknown body fields), so call sites splat it
unconditionally.

**Thinking models make `max_tokens` a correctness concern, not just a cost knob.** Gemini 2.5+
and all of 3.x count reasoning tokens against `max_tokens` and bill them as output. A budget too
small to cover the thinking is spent entirely on it, and the call returns **empty content with
no exception** — a success, as far as the SDK is concerned. `_classify` capped at 5 tokens
originally, which is ample for the single word it emits and instantly fatal here: an empty label
matches neither `"meta"` nor `"general"` and falls through to the `"cembrowski"` default, sending
every question to the corpus with no error anywhere. Hence `CLASSIFIER_MAX_TOKENS = 512` and
friends (ceilings, not reservations — headroom is free), plus explicit empty-completion logging
in `_classify` and `_generate`.

### Qdrant configuration

- **Collection**: `BAPa-V1` in production — this is what the website's backend reads, what
  `scripts/link_posters.py` defaults to, and what `vectordb.COLLECTION_NAME` is set to, so
  the pipeline targets production by default. Pass `--collection <name>` to work against
  anything else.
- **Vector dims**: 1024 (Cosine distance)
- **Local mode**: If no `QDRANT_CLUSTER_ENDPOINT` is set, uses embedded local DB at `data/vectors/`
- **Cloud mode**: Set `QDRANT_CLUSTER_ENDPOINT` + `QDRANT_API_KEY` in `.env`

### Chunk payload structure

All points share `chunk_category` ("text" or "image"), `chunk_index`, and `text`.

**Paper text chunks** additionally store: `source_type="paper"` (implicit — field absent), `paper_id`, `title`, `authors`, `year`, `publication`, `page_start`, `page_end`, `page_label`.

**Image chunks** additionally store: `chunk_category="image"`, `paper_id`, `title`, `authors`, `year`, `publication`, `page`, `page_label`, `source_file`, `bbox`, `caption`, `image_type`.

**Document text chunks** additionally store: `source_type="document"`, `doc_id`, `title`, `file_type`.

**Website link fields** (`site_path`, `poster_id`) are written onto poster chunks after ingestion by `scripts/link_posters.py`, not by `vectordb.py` — see below. They are absent until that script runs and are always optional; `_search` reads them when present and a chunk without them degrades to an unlinked citation.

### Scanned PDFs (`data/ocr.py`)

`pymupdf4llm` returns nothing for a page that is a photograph of paper, so before this
existed a scanned source reached the chunker with empty text and was skipped outright —
silently, with only a log line. `vectordb --method auto` (the default) now checks for a text
layer per paper and routes scans through OCR instead.

- **Spread splitting.** Scanned books are commonly captured a facing pair at a time, so one
  PDF sheet holds two book pages. A sheet wider than `SPREAD_ASPECT_RATIO` (1.15) is split at
  the gutter, with a 1% overlap so an off-centre fold still captures the full text block.
  This doubles the resolution the model reads, stops it running columns together across the
  fold, and makes each chunk one real page. Portrait sources are left whole automatically.
- **Printed page numbers.** The transcription reports the folio printed on the page, and
  `chunk_scanned_pages` stores that as `page`. A scanned book's PDF index almost never matches
  its printed numbering (front matter is numbered separately; two-up capture halves the
  count), and the printed folio is the only value a reader can check against a physical copy.
  Pages with no folio fall back to their sequential position.
- **Figures.** The prompt has the model transcribe each figure's caption verbatim and then
  describe what it plots — axes, units, series, trend. Without this a scanned figure
  contributes nothing retrievable, which matters for a corpus this chart-heavy.
- **Caching.** Every transcription is written to `data/ocr_cache/<paper_id>/`, so re-running is
  free and an interrupted run resumes. Failures are never cached, so they retry.
- **Throughput is capped by tokens-per-minute, not latency.** A book-length run is bounded by
  the account's TPM allowance (30k for gpt-4.1 on the current key ≈ 13 pages/min, so ~25 min
  for a 290-page book). `DEFAULT_OCR_WORKERS` is therefore 3, not something larger: past the
  TPM ceiling, extra workers only convert successful requests into 429s. Retries honour the
  server's own "try again in Xs" hint. Raise `--ocr-workers` only if the key has headroom.
- **Trimming.** `--first-sheet` / `--last-sheet` / `--exclude-units` cut material captured
  alongside the work — loose inserts, product literature, clippings. This is not an edge case:
  the Cembrowski & Carey book carries a glucose meter insert, a BD Vacutainer technical sheet
  and a newspaper clipping after the index. Left in, they would be embedded under the book's
  title and cited with its page numbering. A unit key is a sheet number plus `L`/`R`
  (e.g. `144R`). Render a contact sheet of the tail before ingesting a new scan.

Scanned pages produce the same payload as born-digital page chunks — `chunk_category="image"`,
`image_type="page"` — so they retrieve and cite through the existing paths with no special
casing. Point IDs are `uuid5` over `{paper_id}:scan:{sheet}{half}`, so re-running after
adjusting the range or exclusions overwrites in place rather than duplicating.

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
- Calls the configured chat model with a system prompt enforcing ground-in-context answers and markdown output

**Only chunks with a `site_path` become numbered SOURCE blocks.** `_answer_cembrowski` splits
retrieval into `citable` (a linked poster) and `background` (documents, the textbook, and
posters `link_posters.py` hasn't matched), so the reader-facing source list is never handed
something unclickable. Background still informs the answer but carries no number.

Note the consequence: the textbook is unlinked and is most of the corpus, so a book-answered
question routinely has **zero** citable sources. When that happens the user message says so
explicitly instead of leaving an empty `Context:` heading — otherwise the model reads the gap
as an oversight and invents `[1]`, `[2]`, … that resolve to nothing. `SYSTEM_PROMPT` rule 9
states the same constraint. `scripts/eval_routing.py --full` checks for exactly this.

### Query routing (Cembrowski vs. NIH)

Non-technical site visitors may ask general medical questions unrelated to Cembrowski's own research; per the client's direction, those are answered from NIH sources instead of being refused.

`QueryEngine._route()` decides per-question (and is separated from answer generation precisely so `scripts/eval_routing.py` can measure it without paying for synthesis):

0. **Author match** — before any LLM call, the question is fuzzy-matched against every author name in the corpus (`authors.py`). A confident hit routes to `"author"` and answers from that person's works via a payload filter, skipping classification entirely.
1. **Classify** — a cheap `gemini-3.5-flash-lite` call (`_classify`, prompt in `prompts.CLASSIFIER_PROMPT`) labels the question `"cembrowski"`, `"general"`, or `"meta"`. Defaults to `"cembrowski"` on API failure or an empty completion — the latter is logged as a warning, because on a thinking model an empty label is indistinguishable from a real classification and would silently send every question to the corpus (see `CLASSIFIER_MAX_TOKENS`).
1b. **Meta** — a question about the site/assistant itself returns the static `META_ANSWER` with no retrieval at all. Deliberately not LLM-generated: a fixed string can't hallucinate, and before this route existed such questions fell through to a live NIH search that occasionally returned a barely-related health topic as "context".
2. **Retrieve + score-check** — if labeled `"cembrowski"`, the question is embedded and searched against Qdrant as usual. The top hit's cosine score must clear `SCORE_THRESHOLD` (0.30, in `query_engine.py`) to be trusted; this catches questions that were misclassified or simply aren't covered by the corpus.
3. **Route** — a strong Cembrowski match is answered via `_answer_cembrowski` (existing `SYSTEM_PROMPT`, cites Cembrowski's papers/documents/images). Everything else — `"general"`-labeled questions, or weak/no Cembrowski matches — is answered via `_answer_nih`.

**The classifier is the only gate on the corpus, so `CLASSIFIER_PROMPT` must be updated whenever
the corpus gains material.** Note the asymmetry in step 2: a `"cembrowski"` label still gets
score-checked and can fall back to NIH, but a `"general"` label **skips retrieval entirely** —
there is no safety net in that direction. A question the classifier misfiles as `"general"` is
unanswerable no matter how well it would have retrieved.

This is not hypothetical. When the Cembrowski & Carey textbook was ingested, the prompt still
described the corpus as research papers about "troponin, blood gas analyzers, sample tubes", so
questions on QC statistics, control rules, the testing process and test utilization were filed
as `"general"` and answered from NIH — one of them while sitting on a 0.692 top hit on exactly
the right page. Two *poster* questions were already failing this way beforehand.

Fixing it with a score threshold does not work, and it is worth knowing why: the two populations
overlap. Genuine consumer questions reach 0.516 ("what does a high ferritin level mean for my
health" — it matches the ferritin overdiagnosis poster), while real corpus questions drop to
0.501. No cutoff separates them. What does separate them is the subject: **is the question about
THE LABORATORY or about THE PATIENT?** The prompt now draws that line explicitly, with paired
examples on the same analyte.

**This is now measured rather than asserted.** `scripts/eval_routing.py` runs a labeled set
(`scripts/eval_questions.json`: 17 poster, 12 book, 10 consumer, 4 meta, 4 author) through
`_route` and reports per-route accuracy, a confusion table, top-hit scores, and the count of
empty classifier completions. Both providers measured on the same set: **47/47 routing and
47/47 citations clean on each**, 0 empty completions, corpus top-hit scores 0.473-0.769.
Routing is unaffected by the model swap (retrieval is Voyage/Qdrant); the difference is tail
latency — median/max 0.74s/1.49s on Gemini vs 0.88s/3.24s on GPT. Re-run after touching this prompt, the corpus, or
`SCORE_THRESHOLD` — the poster questions are written against the exact titles in `BAPa-V1`, so a
re-ingestion under a changed title should make them fail.

`_answer_nih` calls `nih.search_nih(question)` (`src/chat_cembrowski/retrieval/nih.py`), which queries:
- **MedlinePlus Web Service** (`wsearch.nlm.nih.gov`) — primary source; plain-language consumer health topics, no API key needed.
- **PubMed via NCBI E-utilities** (`esearch`/`efetch`) — supplementary fallback when MedlinePlus returns few/no results; technical literature abstracts. Optional `NCBI_API_KEY` / `NCBI_EMAIL` env vars raise the free rate limit (3→10 req/sec) and follow NCBI etiquette, but both services work without any key.

Results are rendered into a context block (`_build_nih_context`, same `SOURCE {i}` format as the Cembrowski path) and answered with a separate `NIH_SYSTEM_PROMPT` (in `prompts.py`) that requires plain language, the same bracketed-number citations (`[1]`, `[2]`) as the Cembrowski path — resolved to the source URLs by the frontend — and a not-medical-advice disclaimer, kept deliberately distinct from `SYSTEM_PROMPT` so NIH answers are never confused with Cembrowski's own research findings. Network failures in `nih.py` return `[]` rather than raising, so `_answer_nih` degrades to a graceful "couldn't find NIH information" message instead of crashing the request.

### Page number note

Page numbers in chunk payloads are journal/article page numbers (offset from `first_page_number` extracted during ingestion). If `first_page_number` is unavailable, PDF page numbers are used as a fallback.
