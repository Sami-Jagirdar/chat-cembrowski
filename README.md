RAG Query System for answering questions based on George Cembrowski's publications

## Usage Instructions

# Installation
- Install uv if not already available 
`python -m pip install uv`
- Install project dependencies from pyproject.toml
`uv install`

# Environment
- Ensure you have a .env file with the SERPAPI_KEY, QDRANT_API_KEY and QDRANT_CLUSTER_ENPOINT variables
(You will need to create a free qdrant account and a serpapi account)

# Running Data Pipeline
To Create a Vector store of George Cembrowski's publications you have to run fetcher -> parser -> vectordb present in src/data package
Run the following modules

`uv run src.data.fetcher`
`uv run src.data.parser`
`uv run src.data.vectordb`
