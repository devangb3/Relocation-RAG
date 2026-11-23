"""Vector and hybrid search over indexed relocation policy chunks."""

from datetime import date

from core.embeddings import get_policy_collection
from core.retrieval.hybrid_retriever import HybridRetriever
from core.retrieval.reranker import RerankedRetriever
from core.retrieval.vector_retriever import RetrievedChunks, VectorRetriever

__all__ = ["HybridRetriever", "RerankedRetriever", "VectorRetriever", "retrieve_policy"]


def retrieve_policy(query: str, as_of: date | None = None) -> RetrievedChunks:
    return RerankedRetriever(get_policy_collection(), as_of).search(query)
