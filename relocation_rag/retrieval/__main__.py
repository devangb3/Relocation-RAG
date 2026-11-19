"""Vector and hybrid search over indexed relocation policy chunks."""

import argparse

from relocation_rag.embeddings import get_policy_collection
from relocation_rag.retrieval import HybridRetriever, VectorRetriever


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="?", default="What is the policy on relocation?")
    parser.add_argument("--vector-only", action="store_true")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()
    retriever = VectorRetriever if args.vector_only else HybridRetriever
    results = retriever(get_policy_collection()).search(args.query, args.limit)
    for rank, chunk_id in enumerate(results["ids"][0], 1):
        print(f"{rank}. {chunk_id}")


if __name__ == "__main__":
    main()
