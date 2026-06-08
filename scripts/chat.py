import argparse
import logging
import sys

from chat_cembrowski.data.vectordb import (
    COLLECTION_NAME,
    get_qdrant_client,
    get_openai_client,
    get_voyage_client,
)
from chat_cembrowski.retrieval.query_engine import QueryEngine

logging.basicConfig(level=logging.WARNING)


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive CLI for querying the RAG system.")
    parser.add_argument(
        "--collection",
        default=COLLECTION_NAME,
        help=f"Qdrant collection to query (default: {COLLECTION_NAME}).",
    )
    args = parser.parse_args()

    qdrant_client = get_qdrant_client()

    if not qdrant_client.collection_exists(args.collection):
        print(f"Error: collection '{args.collection}' does not exist.")
        available = [c.name for c in qdrant_client.get_collections().collections]
        if available:
            print(f"Available collections: {', '.join(available)}")
        else:
            print("No collections found. Run the vectordb pipeline first.")
        sys.exit(1)

    engine = QueryEngine(
        qdrant_client=qdrant_client,
        openai_client=get_openai_client(),
        voyage_client=get_voyage_client(),
        collection_name=args.collection,
    )

    print(f"Connected to collection '{args.collection}'. Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not question:
            continue

        if question.lower() in {"exit", "quit"}:
            print("Exiting.")
            break

        answer = engine.query(question)
        print(f"\nAssistant: {answer}\n")


if __name__ == "__main__":
    main()
