import argparse
import logging

from chat_cembrowski.data.vectordb import (
    get_openai_client,
    get_qdrant_client,
    get_voyage_client,
    COLLECTION_NAME,
)

from chat_cembrowski.retrieval.query_engine import QueryEngine


logging.basicConfig(level=logging.INFO)


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the RAG system.")
    parser.add_argument(
        "question",
        nargs="?",
        default=None,
        help="Question to ask. If omitted, runs a fixed demo list of sample questions.",
    )
    parser.add_argument(
        "--collection",
        default=COLLECTION_NAME,
        help=f"Qdrant collection name to query (default: {COLLECTION_NAME}).",
    )
    args = parser.parse_args()

    qdrant_client = get_qdrant_client()
    openai_client = get_openai_client()
    voyage_client = get_voyage_client()

    engine = QueryEngine(
        qdrant_client=qdrant_client,
        openai_client=openai_client,
        voyage_client=voyage_client,
        top_k=10,
        collection_name=args.collection,
    )

    if args.question:
        questions = [args.question]
    else:
        questions = [
            # Cembrowski corpus questions
            "In the Kingston 3-year dual analyzer study, what does Figure 2 show about how inter-analyzer variation changes in the hours immediately following a cartridge replacement?",
            "In the two-year Edmonton and Calgary hs-cTnT study, what mathematical method was used to separate the contributions of analytical, preanalytical, and biological variation from the sequential intrapatient troponin measurements?",
            "How long should a blood gas analyzer cartridge be left running before its results can be considered reliable for sequential patient monitoring?",
            "What is the practical clinical impact of using plasma separator tubes with gel (PST) versus Barricor tubes for high-sensitivity troponin testing in an emergency chest pain protocol?",
            # General medical questions, expected to route to NIH (MedlinePlus/PubMed)
            "What are the symptoms of type 2 diabetes?",
            "What is high blood pressure?",
        ]

    for question in questions:
        print("\n" + "=" * 80)
        print(f"QUESTION:\n{question}")
        print("=" * 80)

        answer, route = engine.query_with_route(question)

        print(f"\nANSWER (route: {route}):\n")
        print(answer)
        print()


if __name__ == "__main__":
    main()
