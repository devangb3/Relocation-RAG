from unittest.mock import Mock

from relocation_rag import embeddings, index


def test_reindex_updates_citations_and_removes_only_superseded_chunks(
    tmp_path, monkeypatch, tokenizer
):
    monkeypatch.chdir(tmp_path)  # Isolate the persistent Chroma database.
    monkeypatch.setattr(embeddings, "LoggingService", Mock())
    model = Mock(tokenizer=tokenizer, max_seq_length=512)
    model.encode.side_effect = lambda texts, **kwargs: [[1.0, 0.0] for _ in texts]
    monkeypatch.setattr(index, "get_embedding_model", lambda: model)

    policy = tmp_path / "claims.md"
    other = tmp_path / "travel.md"
    header = "# Claims\nCompany | Effective January 1, 2025\n"
    policy.write_text(
        header + "## A Filing\nFile in 30 days.\n## B Appeals\nContact Finance."
    )
    other.write_text(header + "## A Travel\nKeep receipts.")
    index.index_policy(policy)
    index.index_policy(other)
    collection = embeddings.get_policy_collection()
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
