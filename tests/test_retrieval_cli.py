import runpy
import sys
from unittest.mock import Mock

import pytest

from relocation_rag import embeddings, retrieval


@pytest.mark.parametrize(
    ("flags", "backend"),
    [
        ([], "RerankedRetriever"),
        (["--hybrid-only"], "HybridRetriever"),
        (["--vector-only"], "VectorRetriever"),
    ],
)
def test_cli_uses_selected_backend_and_prints_ranked_ids(
    monkeypatch, capsys, flags, backend
):
    collection = Mock()
    monkeypatch.setattr(embeddings, "get_policy_collection", lambda: collection)
    retriever = Mock()
    retriever.return_value.search.return_value = {"ids": [["claims/B/1", "claims/A/1"]]}
    for name in ("RerankedRetriever", "HybridRetriever", "VectorRetriever"):
        constructor = (
            retriever if name == backend else Mock(side_effect=AssertionError(name))
        )
        monkeypatch.setattr(retrieval, name, constructor)
    monkeypatch.setattr(
        sys, "argv", ["retrieval", *flags, "When can I file?", "--limit", "2"]
    )

    runpy.run_module("relocation_rag.retrieval.__main__", run_name="__main__")

    retriever.assert_called_once_with(collection)
    retriever.return_value.search.assert_called_once_with("When can I file?", 2)
    assert capsys.readouterr().out == "1. claims/B/1\n2. claims/A/1\n"
