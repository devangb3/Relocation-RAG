"""Local embedding model and its Chroma collection."""

from functools import cache

import chromadb
from chromadb.api.models.Collection import Collection
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@cache
def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)


def get_policy_collection() -> Collection:
    return chromadb.PersistentClient(path=".chroma").get_or_create_collection(
        "policy_chunks",
        configuration={"hnsw": {"space": "cosine"}},
        embedding_function=None,
    )
