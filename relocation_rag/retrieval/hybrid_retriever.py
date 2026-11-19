from relocation_rag.retrieval.bm25 import rank_chunks_keyword
from relocation_rag.retrieval.vector_retriever import RetrievedChunks, VectorRetriever


class HybridRetriever(VectorRetriever):
    def search(self, query: str, k: int = 10) -> RetrievedChunks:
        results = super().search(query, k)

        chunks = self.collection.get(include=["documents", "metadatas"])
        documents = dict(zip(chunks["ids"], chunks["documents"], strict=True))
        metadata = dict(zip(chunks["ids"], chunks["metadatas"], strict=True))

        keyword_ids = rank_chunks_keyword(query, documents, k)
        keyword_hits = {key: (documents[key], metadata[key]) for key in keyword_ids}

        if not keyword_hits:
            return results

        vector_ids = results["ids"][0]
        vector_hits = dict(
            zip(
                vector_ids,
                zip(results["documents"][0], results["metadatas"][0], strict=True),
                strict=True,
            )
        )
        shared = [key for key in vector_ids if key in keyword_hits]
        keyword_only = [key for key in keyword_ids if key not in vector_hits]
        vector_only = [key for key in vector_ids if key not in keyword_hits]
        selected = (shared + keyword_only + vector_only)[:k]
        hits = vector_hits | keyword_hits
        return {
            "ids": [selected],
            "documents": [[hits[key][0] for key in selected]],
            "metadatas": [[hits[key][1] for key in selected]],
        }
