from unittest.mock import Mock

import pytest

from relocation_rag import retrieval


@pytest.mark.parametrize(
    "question,expected",
    [
        ("What mentions adv-4?", ["shared", "keyword", "vector"]),
        ("REQUESTING CASH", ["shared", "keyword", "vector"]),
        ("Unrelated phrase", ["vector", "shared", "other"]),
    ],
)
def test_hybrid_merge_preserves_citations(monkeypatch, question, expected):
    docs = {
        "vector": "See ADV-40",
        "shared": "Requesting cash with ADV-4",
        "other": "See travel rules",
        "keyword": "Requesting cash requires ADV-4",
    }
    vector_ids = ["vector", "shared", "other"]
    collection = Mock()
    collection.query.return_value = {
        "ids": [vector_ids],
        "documents": [[docs[key] for key in vector_ids]],
        "metadatas": [[{"section": key} for key in vector_ids]],
    }
    collection.get.return_value = {
        "ids": list(docs),
        "documents": list(docs.values()),
        "metadatas": [{"section": key} for key in docs],
    }
    monkeypatch.setattr("relocation_rag.retrieval.vector_retriever.get_embedding_model", Mock())
    result = retrieval.HybridRetriever(collection).search(question, limit=3)
    assert result == {
        "ids": [expected],
        "documents": [[docs[key] for key in expected]],
        "metadatas": [[{"section": key} for key in expected]],
    }
