from pathlib import Path

import pytest

from core.chunking import chunk_policy


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


def test_header_only_table_and_escaped_cells_keep_distinct_citations(tmp_path, tokenizer):
    path = write_policy(tmp_path / "claims.md", (
        "## Filing\n\n| Cost | Amount |\n| --- | --- |\n\n"
        "| Cost | Amount |\n| --- | --- |\n| Travel \\| lodging | |"
    ))
    header, row = chunk_policy(path, tokenizer)
    assert header.text == "Filing\n\nColumns include Cost, Amount."
    assert (header.table_number, header.row_start, header.row_end) == (1, None, None)
    assert row.text == "Filing\n\nCost is Travel | lodging and Amount is blank."
    assert (row.table_number, row.row_start, row.row_end) == (2, 1, 1)


@pytest.mark.parametrize(("markdown", "error"), [
    ("## Filing\nKeep receipts.", "expected an effective date"),
    ("Company | Effective January 1, 2026\n## Filing\n", "empty section"),
    ("Company | Effective January 1, 2026\n## Filing\n| A | B |\n|--|--|\n| one |",
     "Filing: table row 1 has 1 cells for 2 columns"),
])
def test_invalid_policy_identifies_source(tmp_path, tokenizer, markdown, error):
    path = tmp_path / "claims.md"
    path.write_text(markdown)
    with pytest.raises(ValueError, match=error) as exc:
        chunk_policy(path, tokenizer)
    assert path.name in str(exc.value)
