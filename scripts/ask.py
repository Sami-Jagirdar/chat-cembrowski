import logging
import os
from pathlib import Path
import sys

# This is a workaround for now to be able to import from src/ until we package this properly.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))) 

from src.data.vectordb import (
    get_openai_client,
    get_qdrant_client,
)

from src.retrieval.query_engine import QueryEngine


logging.basicConfig(level=logging.INFO)


def main() -> None:
    qdrant_client = get_qdrant_client()
    openai_client = get_openai_client()

    engine = QueryEngine(
        qdrant_client=qdrant_client,
        openai_client=openai_client,
        top_k=10,
    )

    questions = [
        "What are George Cembrowski's views on laboratory quality control?",
        "What did Cembrowski publish about diagnostic testing errors?",
        "How does Cembrowski discuss evidence-based laboratory medicine?",
    ]

    for question in questions:
        print("\n" + "=" * 80)
        print(f"QUESTION:\n{question}")
        print("=" * 80)

        answer = engine.query(question)

        print("\nANSWER:\n")
        print(answer)
        print()


if __name__ == "__main__":
    main()