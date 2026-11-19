from pathlib import Path

import pytest
from transformers import BertTokenizer

from relocation_rag.chunking import chunk_policy


@pytest.fixture
def tokenizer():
    tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "word", "##piece", "##s", "."]
    vocab = {token: index for index, token in enumerate(tokens)}
    return BertTokenizer(vocab=vocab, model_max_length=512)


def write_policy(path: Path, sections: str) -> Path:
    path.write_text(
        f"# Example policy\n\nCompany | Effective January 1, 2026\n\n{sections}\n",
        encoding="utf-8",
    )
    return path


def test_token_budget_and_overlap_stay_within_the_section(tmp_path, tokenizer):
    path = write_policy(
        tmp_path / "policy.md",
        "## REL-001.1 Scope\n\nWordpieces matter. Keep the receipt.\n\n\n\n"
        "File it within thirty days.\n\n## REL-001.2 Exceptions\n\nAsk Finance.",
    )
    chunks = chunk_policy(path, tokenizer, target_tokens=20, overlap_tokens=4)

    assert [chunk.text for chunk in chunks] == [
        "REL-001.1 Scope\n\nWordpieces matter. Keep the receipt.",
        "REL-001.1 Scope\n\nKeep the receipt.\n\nFile it within thirty days.",
        "REL-001.2 Exceptions\n\nAsk Finance.",
    ]
    assert all(len(tokenizer(chunk.text)["input_ids"]) <= 20 for chunk in chunks)
    assert [chunk.id for chunk in chunks] == [
        "policy/REL-001.1/1",
        "policy/REL-001.1/2",
        "policy/REL-001.2/1",
    ]
    assert all(chunk.effective_date == "2026-01-01" for chunk in chunks)


def test_single_row_becomes_a_sentence_and_checks_the_hard_limit(tmp_path, tokenizer):
    table = "| Cost | Eligible |\n| --- | --- |\n| Fare | Yes |"
    path = write_policy(tmp_path / "policy.md", f"## REL-001.1 Scope\n\n{table}")

    chunks = chunk_policy(path, tokenizer, target_tokens=16, overlap_tokens=0)

    assert [chunk.text for chunk in chunks] == ["REL-001.1 Scope\n\nCost is Fare and Eligible is Yes."]
    limit = len(tokenizer(chunks[0].text)["input_ids"]) - 1
    parts = chunk_policy(path, tokenizer, target_tokens=limit, overlap_tokens=0, max_tokens=limit)
    assert len(parts) > 1
    assert all(part.row_start == part.row_end == 1 for part in parts)
    assert all(len(tokenizer(part.text)["input_ids"]) <= limit for part in parts)
    assert " ".join(part.text.split("\n\n")[1] for part in parts) == "Cost is Fare and Eligible is Yes."


def test_chunker_splits_oversized_prose_without_dropping_words(tmp_path, tokenizer):
    words = [f"item{index}" for index in range(40)]
    path = write_policy(tmp_path / "policy.md", f"## REL-001.1 Scope\n\n{' '.join(words)}")

    chunks = chunk_policy(path, tokenizer, target_tokens=16, overlap_tokens=4, max_tokens=20)

    assert all(len(tokenizer(chunk.text)["input_ids"]) <= 20 for chunk in chunks)
    actual_words = " ".join(chunk.text.split("\n\n", 1)[1] for chunk in chunks).split()
    assert actual_words == words


def test_uncaptioned_table_splits_with_headers_and_source_rows(tmp_path, tokenizer):
    header = "| Portal label | Event recorded |\n| --- | --- |"
    rows = [f"| Case {i} | Filing accepted |" for i in range(50)]
    table = "\n".join([header, *rows])
    path = write_policy(tmp_path / "policy.md", f"## Register\n\n{table}\n\nEnd of register.")
    chunks = chunk_policy(path, tokenizer, target_tokens=40, overlap_tokens=4, max_tokens=50)

    fragments, prose = chunks[:-1], chunks[-1]
    assert 1 < len(fragments) < len(rows)
    next_row = 1
    for chunk in fragments:
        assert chunk.table_number == 1
        assert chunk.row_start == next_row
        sentences = [
            f"Portal label is Case {i} and Event recorded is Filing accepted."
            for i in range(next_row - 1, chunk.row_end)
        ]
        assert chunk.text == "Register\n\n" + "\n".join(sentences)
        assert len(tokenizer(chunk.text)["input_ids"]) <= 40
        next_row = chunk.row_end + 1
    assert next_row == len(rows) + 1
    assert prose.table_number is None
    assert prose.text == "Register\n\nEnd of register."
