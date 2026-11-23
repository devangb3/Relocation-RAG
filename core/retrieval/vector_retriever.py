from datetime import date
from typing import TypedDict

from chromadb.api.models.Collection import Collection
from chromadb.api.types import Metadata

from core.embeddings import get_embedding_model


class RetrievedChunks(TypedDict):
    ids: list[list[str]]
    documents: list[list[str]]
    metadatas: list[list[Metadata]]


class VectorRetriever:
    def __init__(self, collection: Collection, as_of: date | None = None):
        self.collection = collection
        day = int((as_of or date.today()).isoformat().replace("-", ""))
        self.where = {"$and": [
            {"effective_from": {"$lte": day}},
            {"effective_to": {"$gt": day}},
        ]}

    def search(self, query: str, limit: int = 10) -> RetrievedChunks:
        if not query.strip() or limit < 1:
            raise ValueError("Search requires a nonblank query and a positive limit.")
        vector = get_embedding_model().encode(query, normalize_embeddings=True)
        results = self.collection.query(
            query_embeddings=[vector],
            n_results=limit,
            where=self.where,
            include=["documents", "metadatas"],
        )
        documents = results["documents"]
        metadatas = results["metadatas"]
        if documents is None or metadatas is None:
            raise RuntimeError("Policy search returned no documents or metadata.")
        return {"ids": results["ids"], "documents": documents, "metadatas": metadatas}
