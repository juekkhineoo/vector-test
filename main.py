"""
Semantic search over a dataset using HuggingFace sentence-transformers + PostgreSQL/pgvector.

Usage:
    python main.py                                      # default: TA13.csv
    python main.py --dataset data.csv
    python main.py --dataset data.json --top-k 5
    python main.py --dataset reviews.csv --no-cache    # clear DB and re-embed
    python main.py --model all-mpnet-base-v2            # change model
    python main.py --db-url postgresql://user:pass@localhost/mydb
"""

import argparse
import os
import sys

from embedder import TextEmbedder
from vector_store import VectorStore
from dataset_loader import load_dataset


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Semantic document search with HuggingFace embeddings + FAISS"
    )
    parser.add_argument(
        "--dataset", default="TA13.csv",
        help="Path to the dataset file (.csv, .json, .jsonl, or .txt). Default: TA13.csv",
    )
    parser.add_argument(
        "--text-column", default=None,
        help="Column / key containing the main text (auto-detected if omitted)",
    )
    parser.add_argument(
        "--model", default="all-MiniLM-L6-v2",
        help="HuggingFace sentence-transformers model name (default: all-MiniLM-L6-v2)",
    )
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="Number of results to return per query (default: 5)",
    )
    parser.add_argument(
        "--db-url",
        default=os.environ.get("DATABASE_URL", "postgresql://localhost/vectordb"),
        help=(
            "PostgreSQL connection string "
            "(default: $DATABASE_URL or postgresql://localhost/vectordb)"
        ),
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Clear the documents table and re-embed the dataset",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Embedding batch size (default: 64)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Store helpers
# ---------------------------------------------------------------------------

def _build_store(
    records: list[dict],
    embedder: TextEmbedder,
    batch_size: int,
    store: VectorStore,
) -> None:
    texts = [r["text"] for r in records]
    metadata = [r["metadata"] for r in records]

    print(f"Embedding {len(texts):,} documents …")
    vectors = embedder.embed(texts, batch_size=batch_size, show_progress=True)
    store.add(vectors=vectors, texts=texts, metadata=metadata)
    print(f"Inserted {len(texts):,} vectors into PostgreSQL.")


# ---------------------------------------------------------------------------
# Interactive search loop
# ---------------------------------------------------------------------------

def interactive_search(store: VectorStore, embedder: TextEmbedder, top_k: int) -> None:
    print("\n" + "=" * 62)
    print("  Semantic Search  |  type a query, or 'quit' to exit")
    print("=" * 62)

    while True:
        try:
            query = input("\nQuery> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not query:
            continue
        if query.lower() in {"quit", "exit", "q"}:
            print("Exiting.")
            break

        query_vec = embedder.embed(query)
        results = store.search(query_vec, top_k=top_k)

        if not results:
            print("No results found.")
            continue

        print(f"\nTop {len(results)} results for: {query!r}")
        print("-" * 62)
        for rank, r in enumerate(results, 1):
            # Truncate long texts to keep output readable
            snippet = r.text[:250].replace("\n", " ")
            if len(r.text) > 250:
                snippet += "…"

            # Show a few key metadata fields (skip bookkeeping keys)
            meta_pairs = [
                (k, v) for k, v in r.metadata.items()
                if k not in ("source", "row", "line")
            ][:4]
            meta_str = "  |  ".join(f"{k}: {v}" for k, v in meta_pairs)

            print(f"[{rank}] Score: {r.score:.4f}")
            print(f"    {snippet}")
            if meta_str:
                print(f"    ({meta_str})")
        print("-" * 62)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # 1. Load the embedding model
    embedder = TextEmbedder(model_name=args.model)

    # 2. Connect to PostgreSQL
    print(f"\nConnecting to PostgreSQL …")
    try:
        store = VectorStore(conn_str=args.db_url, embedding_dim=embedder.embedding_dim)
    except Exception as exc:
        print(f"ERROR: Could not connect to PostgreSQL: {exc}")
        sys.exit(1)

    # 3. Optionally clear existing data
    if args.no_cache and len(store) > 0:
        print("Clearing existing index …")
        store.reset()

    # 4. Embed and insert when the table was just created (first run) or wiped
    if store.is_new or (args.no_cache and len(store) == 0):
        print(f"\nLoading dataset from '{args.dataset}' …")
        try:
            records = load_dataset(args.dataset, text_column=args.text_column)
        except FileNotFoundError as exc:
            print(f"ERROR: {exc}")
            store.close()
            sys.exit(1)

        if not records:
            print("ERROR: No text records found in the dataset. "
                  "Check the file path and --text-column argument.")
            store.close()
            sys.exit(1)

        print(f"Loaded {len(records):,} documents.")
        _build_store(records, embedder, args.batch_size, store)
    else:
        print(f"Using existing index — {len(store):,} documents already in DB.")

    print(f"\nVector store ready — {len(store):,} documents indexed.")

    # 5. Interactive prompt loop
    try:
        interactive_search(store, embedder, top_k=args.top_k)
    finally:
        store.close()


if __name__ == "__main__":
    main()
