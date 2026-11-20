"""Build overlapping policy passages within numbered Markdown sections."""

import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from itertools import groupby
from pathlib import Path

from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

DEFAULT_TARGET_TOKENS = 160
DEFAULT_OVERLAP_TOKENS = 32
_BLOCK_BREAK = re.compile(r"\n(?:[ \t]*\n)+")
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")
_TABLE_SEPARATOR = re.compile(r"[ \t|:-]+")


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    source: str
    section: str
    effective_date: str
    policy_id: str
    effective_from: int
    effective_to: int
    table_number: int | None = None
    row_start: int | None = None
    row_end: int | None = None


def _token_count(
    tokenizer: PreTrainedTokenizerBase, text: str, *, special_tokens: bool = True
) -> int:
    return len(tokenizer(
        text, add_special_tokens=special_tokens, truncation=False, padding=False, verbose=False
    )["input_ids"])


def _is_table(block: str) -> bool:
    lines = block.splitlines()
    return (
        len(lines) >= 2
        and "|" in lines[0]
        and "-" in lines[1]
        and _TABLE_SEPARATOR.fullmatch(lines[1]) is not None
    )


def _passage(heading: str, *blocks: str) -> str:
    return "\n\n".join((heading, *blocks))


def _split_block(
    block: str, heading: str, tokenizer: PreTrainedTokenizerBase, max_tokens: int
) -> list[str]:
    if _token_count(tokenizer, _passage(heading, block)) <= max_tokens:
        return [block]

    parts = []
    part = ""
    for word in block.split():
        candidate = f"{part} {word}" if part else word
        if _token_count(tokenizer, _passage(heading, candidate)) <= max_tokens:
            part = candidate
            continue
        if _token_count(tokenizer, _passage(heading, word)) > max_tokens:
            raise ValueError("a word exceeds the embedding model's token limit")
        parts.append(part)
        part = word
    parts.append(part)
    return parts


def _last_sentence(block: str) -> str:
    if block.endswith((".", "!", "?")):
        return _SENTENCE_BREAK.split(block)[-1]
    return ""


def _table_cells(line: str) -> list[str]:
    line = line.strip().removeprefix("|")
    if line.endswith("|") and not line.endswith(r"\|"):
        line = line[:-1]
    return [cell.strip().replace(r"\|", "|") for cell in re.split(r"(?<!\\)\|", line)]


def _table_chunks(
    table: str, heading: str, tokenizer: PreTrainedTokenizerBase, target_tokens: int, max_tokens: int
) -> list[tuple[str, int | None, int | None]]:
    """Pack consecutive row sentences, keeping one-based row ranges for citations."""
    passages: list[tuple[str, int | None, int | None]] = []
    header, _, *rows = table.splitlines()
    columns = _table_cells(header)
    if not rows:
        passage = _passage(heading, "Columns include " + ", ".join(columns) + ".")
        if _token_count(tokenizer, passage) > max_tokens:
            raise ValueError("a Markdown table header exceeds the embedding model's token limit")
        passages.append((passage, None, None))
        return passages

    start = 1
    passage = ""
    for number, row in enumerate(rows, start=1):
        values = _table_cells(row)
        if len(values) != len(columns):
            raise ValueError(f"table row {number} has {len(values)} cells for {len(columns)} columns")
        fields = [
            f"{column} is {value or 'blank'}" for column, value in zip(columns, values, strict=True)
        ]
        sentence = fields[0]
        if len(fields) > 1:
            conjunction = ", and " if len(fields) > 2 else " and "
            sentence = ", ".join(fields[:-1]) + conjunction + fields[-1]
        if not sentence.endswith((".", "?", "!")):
            sentence += "."
        candidate = f"{passage}\n{sentence}" if passage else _passage(heading, sentence)
        size = _token_count(tokenizer, candidate)
        if passage and size > target_tokens:
            passages.append((passage, start, number - 1))
            start = number
            candidate = _passage(heading, sentence)
            size = _token_count(tokenizer, candidate)
        if size > max_tokens:
            for part in _split_block(sentence, heading, tokenizer, max_tokens):
                passages.append((_passage(heading, part), number, number))
            passage = ""
            start = number + 1
            continue
        passage = candidate
    if passage:
        passages.append((passage, start, len(rows)))
    return passages


def _prose_chunks(
    heading: str,
    paragraphs: Iterable[str],
    tokenizer: PreTrainedTokenizerBase,
    target_tokens: int,
    overlap_tokens: int,
    max_tokens: int,
) -> list[str]:
    passages = []
    group: list[str] = []
    previous_sentence = ""
    for paragraph in paragraphs:
        blocks = _split_block(paragraph, heading, tokenizer, max_tokens)
        for block in blocks:
            if group and _token_count(tokenizer, _passage(heading, *group, block)) > target_tokens:
                passages.append(_passage(heading, *group))
                group = []
                if (
                    previous_sentence
                    and _token_count(tokenizer, previous_sentence, special_tokens=False)
                    <= overlap_tokens
                    and _token_count(tokenizer, _passage(heading, previous_sentence, block))
                    <= target_tokens
                ):
                    group.append(previous_sentence)
            group.append(block)
            previous_sentence = _last_sentence(block) if len(blocks) == 1 else ""
    passages.append(_passage(heading, *group))
    return passages


def _section_chunks(
    body: str, heading: str, tokenizer: PreTrainedTokenizerBase,
    target_tokens: int, overlap_tokens: int, max_tokens: int,
) -> list[tuple[str, int | None, int | None, int | None]]:
    passages: list[tuple[str, int | None, int | None, int | None]] = []
    table_number = 0
    for is_table, paragraphs in groupby(_BLOCK_BREAK.split(body.strip()), _is_table):
        if is_table:
            for table in paragraphs:
                table_number += 1
                for passage, start, end in _table_chunks(
                    table, heading, tokenizer, target_tokens, max_tokens
                ):
                    passages.append((passage, table_number, start, end))
        else:
            for passage in _prose_chunks(
                heading, paragraphs, tokenizer, target_tokens, overlap_tokens, max_tokens
            ):
                passages.append((passage, None, None, None))
    return passages


def chunk_policy(
    path: Path,
    tokenizer: PreTrainedTokenizerBase | None = None,
    *,
    target_tokens: int = DEFAULT_TARGET_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    max_tokens: int | None = None,
) -> list[Chunk]:
    markdown = path.read_text(encoding="utf-8")
    header, *sections = re.split(r"^## ", markdown, flags=re.MULTILINE)
    date = re.search(r"\| Effective (.+)$", header, re.MULTILINE)
    if not date or not sections:
        logger.error(f"{path}: expected an effective date and numbered sections")
        raise ValueError(f"{path}: expected an effective date and numbered sections")
    effective_date = datetime.strptime(date[1].strip(), "%B %d, %Y").date().isoformat()

    title = header.splitlines()[0].removeprefix("# ").strip()
    policy_id = title.partition(" · ")[0]
    # Chroma range filters require numbers. Use YYYYMMDD and an open-ended upper bound.
    effective_from = int(effective_date.replace("-", ""))
    end_marker = re.search(r"^Valid until(?: \(exclusive\))?: (.+)$", header, re.MULTILINE)
    effective_to = 100000000
    if end_marker:
        end_date = datetime.strptime(end_marker[1].strip(), "%Y-%m-%d").date()
        effective_to = int(end_date.strftime("%Y%m%d"))
        if effective_to <= effective_from:
            raise ValueError(f"{path}: validity must end after the effective date")

    model_limit = tokenizer.model_max_length
    if max_tokens is not None:
        model_limit = min(model_limit, max_tokens)
    if not 0 <= overlap_tokens < target_tokens <= model_limit:
        raise ValueError(f"Expected 0 <= overlap_tokens < target_tokens <= {model_limit}")

    chunks = []
    for section in sections:
        heading, _, body = section.partition("\n")
        if not heading.strip() or not body.strip():
            logger.error(f"{path}: empty section {heading!r}")
            raise ValueError(f"{path}: empty section {heading!r}")
        section_id = heading.split()[0]
        if _token_count(tokenizer, heading) > model_limit:
            raise ValueError(f"{path}: {section_id}: the section heading exceeds the token limit")
        try:
            passages = _section_chunks(
                body, heading, tokenizer, target_tokens, overlap_tokens, model_limit
            )
            for index, (text, table, start, end) in enumerate(passages, start=1):
                chunks.append(Chunk(
                    id=f"{path.stem}/{section_id}/{index}", text=text, source=path.name,
                    section=section_id, effective_date=effective_date, table_number=table,
                    row_start=start, row_end=end, policy_id=policy_id,
                    effective_from=effective_from, effective_to=effective_to,
                ))
        except ValueError as error:
            raise ValueError(f"{path}: {section_id}: {error}") from error
    return chunks
