"""
Reset the `processed` flag so the next vectordb run re-indexes content.

Re-indexing is a true replacement now: vectordb deletes a source's existing
points (by paper_id / doc_id) before upserting, and chunk IDs are deterministic,
so this does not duplicate or leave orphans. Costs a re-embed of whatever is
reset, so scope it with --papers / --docs rather than resetting everything by
reflex.

    uv run scripts/reset_processed.py            # papers and documents
    uv run scripts/reset_processed.py --papers   # papers only
    uv run scripts/reset_processed.py --docs     # documents only
"""

import argparse

from chat_cembrowski.data.serialization import (
    load_documents_from_json,
    load_papers_from_json,
    save_document,
    save_paper,
)


def reset_papers() -> int:
    count = 0
    for paper in load_papers_from_json():
        if not paper.processed:
            continue
        paper.processed = False
        save_paper(paper)
        count += 1
    return count


def reset_documents() -> int:
    count = 0
    for doc in load_documents_from_json():
        if not doc.processed:
            continue
        doc.processed = False
        save_document(doc)
        count += 1
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reset processed flags for re-indexing.")
    parser.add_argument("--papers", action="store_true", help="Reset papers only.")
    parser.add_argument("--docs", action="store_true", help="Reset documents only.")
    args = parser.parse_args()

    # Neither flag means both, which is the common case.
    do_papers = args.papers or not args.docs
    do_docs = args.docs or not args.papers

    if do_papers:
        print(f"Reset {reset_papers()} paper(s).")
    if do_docs:
        print(f"Reset {reset_documents()} document(s).")
