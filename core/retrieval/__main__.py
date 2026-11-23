"""Vector and hybrid search over indexed relocation policy chunks."""

import argparse

from core.embeddings import get_policy_collection
from core.retrieval import HybridRetriever, RerankedRetriever, VectorRetriever


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="?", default="What is the policy on relocation?")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--vector-only", action="store_const", dest="retriever", const=VectorRetriever)
    mode.add_argument("--hybrid-only", action="store_const", dest="retriever", const=HybridRetriever)
    parser.set_defaults(retriever=RerankedRetriever)
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()
    results = args.retriever(get_policy_collection()).search(args.query, args.limit)
    for rank, chunk_id in enumerate(results["ids"][0], 1):
        print(f"{rank}. {chunk_id}")


if __name__ == "__main__":
    main()
