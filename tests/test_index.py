from datetime import date
from pathlib import Path
from unittest.mock import Mock

import chromadb
import pytest

from relocation_rag import index
from relocation_rag.retrieval.vector_retriever import VectorRetriever


@pytest.fixture
def collection(tmp_path, monkeypatch, tokenizer):
    collection = chromadb.PersistentClient(path=str(tmp_path)).create_collection(
        "policy_chunks", embedding_function=None
    )
    monkeypatch.setattr(index, "get_policy_collection", lambda: collection)
    model = Mock(tokenizer=tokenizer, max_seq_length=512)
    model.encode.side_effect = lambda texts, **kwargs: [[1.0, 0.0] for _ in texts]
    monkeypatch.setattr(index, "get_embedding_model", lambda: model)

    return collection


def test_reindex_updates_citations_and_removes_only_superseded_chunks(tmp_path, collection):
    policy = tmp_path / "claims.md"
    other = tmp_path / "travel.md"
    header = "# Claims\nCompany | Effective January 1, 2025\n"
    policy.write_text(
        header + "## A Filing\nFile in 30 days.\n## B Appeals\nContact Finance."
    )
    other.write_text(header + "## A Travel\nKeep receipts.")
    index.index_policy(policy)
    index.index_policy(other)
    assert set(collection.get()["ids"]) == {"claims/A/1", "claims/B/1", "travel/A/1"}

    policy.write_text(
        header.replace("2025", "2026") + "## A Filing\n| Days |\n| --- |\n| 45 |"
    )
    index.index_policy(policy)
    rows = collection.get(include=["documents", "metadatas"])
    records = dict(
        zip(
            rows["ids"],
            zip(rows["documents"], rows["metadatas"], strict=True),
            strict=True,
        )
    )
    assert records == {
        "claims/A/1": (
            "A Filing\n\nDays is 45.",
            {
                "source": "claims.md",
                "policy_id": "Claims",
                "effective_from": 20260101,
                "effective_to": 100000000,
                "section": "A",
                "effective_date": "2026-01-01",
                "table_number": 1,
                "row_start": 1,
                "row_end": 1,
            },
        ),
        "travel/A/1": (
            "A Travel\n\nKeep receipts.",
            {
                "source": "travel.md",
                "policy_id": "Claims",
                "effective_from": 20250101,
                "effective_to": 100000000,
                "section": "A",
                "effective_date": "2025-01-01",
            },
        ),
    }


def test_bundled_claims_policy_switches_on_january_1(collection):
    policies = Path(__file__).resolve().parents[1] / "data" / "policies"
    previous = policies / "rel-103-claims-2025.md"
    current = policies / "rel-103-claims-2026.md"
    index.index_policy(previous)
    index.index_policy(current)

    for day, edition in ((date(2025, 12, 31), previous), (date(2026, 1, 1), current)):
        where = VectorRetriever(collection, day).where
        rows = collection.get(where=where, include=["metadatas"])
        assert {metadata["source"] for metadata in rows["metadatas"]} == {edition.name}
