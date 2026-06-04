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

1. `uv run -m chat_cembrowski.data.ingestion` (You will have to manually download pdfs only if they don't already exist, then place them in the data/papers folder and rename them as instructed)
After this step, you can add additional pdf papers you may have gotten from other sources data/papers manuall if you want
2. `uv run -m chat_cembrowski.data.parser`
3. `uv run -m chat_cembrowski.data.image_extractor` 
4. `uv run -m chat_cembrowski.data.vectordb`

After these steps you should have a collection in your qdrant cluster with all the embeddings and their respective payloads

NOTE: In case you need to recreate chunks and embed them, run the following script first
`uv run scripts/reset_paper_processed.py`


If you have a local group of general documents that you would like to add to the vector store, you can store them in data/docs (Note that the general docs ingestion only supports extracting and embedding *text*)
Then the pipeline is as follows
1. `uv run -m chat_cembrowski.data.doc_ingestion`
2. `uv run -m chat_cembrowski.data.vectordb`

## Querying the System
In scripts/ask.py, modify the questions you want to ask, then run it as follows:
`uv run scripts/ask.py`

# FUTURE STEPS
- As of May 17, 2026, not doing metadata filtering for retrieval just for time's sake. Will be added very soon
- Adding a reranker. Instead of just top 10, we do top 30, then a reranker model like bge-reranker can get top 10 from those


