"""Vector and hybrid search over indexed relocation policy chunks."""

from relocation_rag.embeddings import get_policy_collection
from relocation_rag.retrieval.hybrid_retriever import HybridRetriever
from relocation_rag.retrieval.reranker import RerankedRetriever
from relocation_rag.retrieval.vector_retriever import RetrievedChunks, VectorRetriever

__all__ = ["HybridRetriever", "RerankedRetriever", "VectorRetriever", "retrieve_policy"]


def retrieve_policy(query: str) -> RetrievedChunks:
    return RerankedRetriever(get_policy_collection()).search(query)
