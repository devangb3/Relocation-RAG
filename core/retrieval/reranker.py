"""Score hybrid candidates locally with a query-passage cross-encoder."""

from functools import cache

from sentence_transformers import CrossEncoder

from core.retrieval.hybrid_retriever import HybridRetriever
from core.retrieval.vector_retriever import RetrievedChunks


@cache
def get_reranker() -> CrossEncoder:
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")


class RerankedRetriever(HybridRetriever):
    def search(self, query: str, k: int = 10) -> RetrievedChunks:
        if k < 1:
            raise ValueError("Search requires a positive result limit.")
        candidates = super().search(query, max(30, k))
        passages = list(zip(
            candidates["ids"][0], candidates["documents"][0], candidates["metadatas"][0],
            strict=True,
        ))
        if not passages:
            return candidates
        pairs = [
            (query, f"{metadata['source']} | effective {metadata['effective_date']}\n{document}")
            for _, document, metadata in passages
        ]
        scores = get_reranker().predict(pairs, show_progress_bar=False)
        highest_first = sorted(
            range(len(passages)),
            key=lambda index: scores[index],
            reverse=True,
        )
        selected = [passages[index] for index in highest_first[:k]]
        return {
            "ids": [[key for key, _, _ in selected]],
            "documents": [[document for _, document, _ in selected]],
            "metadatas": [[metadata for _, _, metadata in selected]],
        }
