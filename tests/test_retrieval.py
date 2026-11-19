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
    result = retrieval.HybridRetriever(collection).search(question, k=3)
    assert result == {
        "ids": [expected],
        "documents": [[docs[key] for key in expected]],
        "metadatas": [[{"section": key} for key in expected]],
    }


def test_reranker_selects_aligned_citations_and_preserves_ties(monkeypatch):
    docs = ["first", "second", "third"]
    metadata = [{"source": doc, "effective_date": "2026-01-01"} for doc in docs]
    hybrid = Mock(return_value={"ids": [docs], "documents": [docs], "metadatas": [metadata]})
    monkeypatch.setattr(retrieval.HybridRetriever, "search", hybrid)
    model = Mock()
    model.predict.return_value = [0, 2, 2]
    monkeypatch.setattr("relocation_rag.retrieval.reranker.get_reranker", lambda: model)
    result = retrieval.RerankedRetriever(Mock()).search("question", 2)
    hybrid.assert_called_once_with("question", 30)
    model.predict.assert_called_once_with(
        [("question", f"{doc} | effective 2026-01-01\n{doc}") for doc in docs],
        show_progress_bar=False,
    )
    assert result == {"ids": [docs[1:]], "documents": [docs[1:]], "metadatas": [metadata[1:]]}


@pytest.mark.parametrize(("query", "limit"), [("  ", 1), ("policy", 0)])
def test_vector_search_rejects_invalid_input_before_embedding(monkeypatch, query, limit):
    model = Mock()
    monkeypatch.setattr("relocation_rag.retrieval.vector_retriever.get_embedding_model", model)
    with pytest.raises(ValueError, match="nonblank query and a positive limit"):
        retrieval.VectorRetriever(Mock()).search(query, limit)
    model.assert_not_called()


@pytest.mark.parametrize("missing_field", ["documents", "metadatas"])
def test_vector_search_requires_documents_and_citations(monkeypatch, missing_field):
    collection = Mock()
    response = {"ids": [["id"]], "documents": [["text"]], "metadatas": [[{"source": "policy"}]]}
    response[missing_field] = None
    collection.query.return_value = response
    monkeypatch.setattr("relocation_rag.retrieval.vector_retriever.get_embedding_model", Mock())
    with pytest.raises(RuntimeError, match="no documents or metadata"):
        retrieval.VectorRetriever(collection).search("question")


def test_empty_retrieval_skips_reranking(monkeypatch):
    empty = {"ids": [[]], "documents": [[]], "metadatas": [[]]}
    monkeypatch.setattr(retrieval.HybridRetriever, "search", Mock(return_value=empty))
    get_model = Mock()
    monkeypatch.setattr("relocation_rag.retrieval.reranker.get_reranker", get_model)
    assert retrieval.RerankedRetriever(Mock()).search("question") == empty
    get_model.assert_not_called()
