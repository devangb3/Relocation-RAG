"""Split the policy corpus at numbered sections and paragraph boundaries."""

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    source: str
    section: str
    effective_date: str


def chunk_policy(path: Path) -> list[Chunk]:
    markdown = path.read_text(encoding="utf-8")
    header, *sections = re.split(r"^## ", markdown, flags=re.MULTILINE)
    date = re.search(r"\| Effective (.+)$", header, re.MULTILINE)
    if not date or not sections:
        logger.error(f"{path}: expected an effective date and numbered sections")
        raise ValueError(f"{path}: expected an effective date and numbered sections")
    effective_date = datetime.strptime(date[1].strip(), "%B %d, %Y").date().isoformat()

    chunks = []
    for section in sections:
        heading, _, body = section.partition("\n")
        section_id = heading.split()[0]
        if not body.strip():
            logger.error(f"{path}: empty section {heading!r}")
            raise ValueError(f"{path}: empty section {heading!r}")
        # Keep contiguous Markdown blocks (including whole tables) together.
        paragraphs = re.split(r"\n[ \t]*\n", body.strip())
        groups = [paragraphs[0]]
        for paragraph in paragraphs[1:]:
            if len(groups[-1].split()) + len(paragraph.split()) <= 128:
                groups[-1] += "\n\n" + paragraph
            else:
                groups.append(paragraph)
        for index, group in enumerate(groups, start=1):
            chunks.append(Chunk(
                id=f"{path.stem}/{section_id}/{index}",
                text=f"{heading}\n\n{group}",
                source=path.name,
                section=section_id,
                effective_date=effective_date,
            ))
    return chunks
