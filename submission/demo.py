"""Run from the repository root using python -m submission.demo."""
from pathlib import Path

from relocation_rag.embeddings import get_policy_collection
from relocation_rag.foundry import FoundryClient
from relocation_rag.index import index_policy
from relocation_rag.retrieval import HybridRetriever, VectorRetriever


def main():
    for path in sorted(Path("submission/handbooks").glob("*.md")):
        index_policy(path)
    collection = get_policy_collection()
    print(f"Indexed {collection.count()} policy passages.")
    query = "Which policy mentions ADV-4?"
    print(query)
    for retriever in (VectorRetriever, HybridRetriever):
        ids = retriever(collection).search(query, 5)["ids"][0]
        print(retriever.__name__, *ids, sep="\n  ")
    question = "What is the household shipment limit per transfer?"
    print("Question:", question)
    result = VectorRetriever(collection).search(question, 3)
    context = "\n\n".join(f"[{key}] {doc}" for key, doc in
                          zip(result["ids"][0], result["documents"][0], strict=True))
    print(FoundryClient.from_env().respond(
        "Answer from the passages and cite the supporting passage ID.",
        f"{question}\n\n{context}"))


if __name__ == "__main__":
    main()
