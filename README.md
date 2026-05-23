RAG Query System for answering questions based on George Cembrowski's publications

## USAGE
## Installation
- Install uv if not already available 
`python -m pip install uv`
- Install project dependencies and the package itself in editable mode
`uv sync`

## Environment
- Ensure you have a .env file with the SERPAPI_KEY, OPENAI_API_KEY, QDRANT_API_KEY and QDRANT_CLUSTER_ENPOINT variables
(You will need to create a free qdrant account, a serpapi account and an openAI account with some credits)

## Data Structure
- paper pdfs will be stored in data/papers folder
- paper json files with metadata will be stored in data/json folder
- If qdrant inference is done locally, then the vector embeddings collection will be stored in data/vectors folder

## Design Decisions and Models Used
- Vector Embeddings: OpenAI's text-embedding-3-large [Dimensions: 1024]
- Retrieval: OpenAI's GPT4-mini
- Chunking Method: Recursive Chunking with fallback to fixed CHUNK_SIZE=1024 characters and CHUNK_OVERLAP=128
- \# Chunks retrieved during vector search: Top 10
- PDF Parsing: Using Pymupdf4llm [Extracts texts,tables,charts as structured markdown with automatic OCR if needed. Images can directly be extracted, but no textual context of these images extracted]

## Running Data Pipeline
To Create a Vector store of George Cembrowski's publications you have to run fetcher -> parser -> vectordb present in src/data package
Run the following modules from the root of the repository

1. `uv run -m chat_cembrowski.data.fetcher` (You will have to manually download pdfs only if they don't already exist, then place them in the data/papers folder and rename them as instructed)
2. `uv run -m chat_cembrowski.data.parser`
3. `uv run -m chat_cembrowski.data.vectordb`

After these steps you should have a collection in your qdrant cluster with all the embeddings and their respective payloads

NOTE: In case you need to recreate chunks and embed them, run the following script first
`uv run scripts/reset_paper_processed.py`

## Querying the System
In scripts/ask.py, modify the questions you want to ask, then run it as follows:
`uv run scripts/ask.py`

# FUTURE STEPS
- As of May 17, 2026, not doing metadata filtering for retrieval just for time's sake. Will be added very soon
- Retrieving context from images as well in one of 2 ways:
-- Run the images extracted through a vision model to create detailed image captions with structural info, then add these to the chunks. (Similarly, we can even just use LlamaParse (paid) to parse both text and images in one shot to give us a single markdown to chunk)
-- We save the extracted images along with metadata separately, then search for relevant images during retrieval using the query and metadata. Then we include the images in the query to a multimodal model for retrieval.
- Adding a reranker. Instead of just top 10, we do top 30, then a reranker model like bge-reranker can get top 10 from those
- Currently page numbers stored in each chunk are simply based on pdf pages and not the actual article page numbers.
For example, an article can be part of a journal and start at page 777 of that journal. So for accurate citations, we need those page numbers. One method is to parse and extract the page number from the first page of the article, then simply add it as an offset to all the page numbers stored in the chunks.



