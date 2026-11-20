"""Evidence supplied during one question and its follow-up searches."""

import json
from dataclasses import dataclass
from datetime import date

from relocation_rag.retrieval.vector_retriever import RetrievedChunks


@dataclass(frozen=True)
class SourcePassage:
    filename: str   
    section: str
    effective_date: str
    text: str

    @property
    def citation(self) -> str:
        return f"[{self.section} | {self.filename} | effective {self.effective_date}]"


class SourceMap:
    """Keep one snapshot per indexed chunk ID for the lifetime of an answer."""

    def __init__(self) -> None:
        self._passages: dict[str, SourcePassage] = {}

    def add_results(self, results: RetrievedChunks, as_of: date) -> str:
        documents = results["documents"]
        if not documents or not documents[0]:
            raise ValueError(
                "No policy passages found. Run python -m relocation_rag.index first."
            )
        if not results.get("metadatas") or not results.get("ids"):
            raise ValueError(
                "Policy passages are missing citation metadata. Rebuild the policy index."
            )

        blocks = []
        for chunk_id, text, meta in zip(
            results["ids"][0], documents[0], results["metadatas"][0], strict=True
        ):
            fields = (
                [meta.get(key) for key in ("source", "section", "effective_date")]
                if meta
                else []
            )
            if not fields or not all(
                isinstance(value, str) and value.strip()
                for value in [chunk_id, *fields]
            ):
                raise ValueError(
                    "Policy passages are missing citation metadata. Rebuild the policy index."
                )
            passage = SourcePassage(*fields, text)
            if self._passages.setdefault(chunk_id, passage) != passage:
                raise ValueError(
                    f"Source chunk changed during this question: {chunk_id}"
                )
            blocks.append(f"Source ID: {chunk_id}\n{passage.citation}\n{text}")
        return f"Policy editions selected for {as_of.isoformat()}.\n\n" + "\n\n".join(
            blocks
        )

    def answer_format(self) -> dict:
        claim = {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "source_ids": {
                    "type": "array",
                    "items": {"type": "string", "enum": list(self._passages)},
                },
            },
            "required": ["text", "source_ids"],
            "additionalProperties": False,
        }
        return {
            "type": "json_schema",
            "name": "cited_answer",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "claims": {"type": "array", "items": claim},
                    "missing_evidence": {"type": ["string", "null"]},
                },
                "required": ["claims", "missing_evidence"],
                "additionalProperties": False,
            },
        }

    def render(self, response: str) -> str:
        answer = json.loads(response)
        if (
            not isinstance(answer, dict)
            or answer.keys() != {"claims", "missing_evidence"}
            or not isinstance(answer["claims"], list)
        ):
            raise ValueError("Expected a structured answer with claims.")
        lines = []
        for claim in answer["claims"]:
            if (
                not isinstance(claim, dict)
                or not isinstance(claim.get("text"), str)
                or not claim["text"].strip()
            ):
                raise ValueError("Expected a nonblank claim.")
            ids = claim.get("source_ids")
            if not isinstance(ids, list) or not ids:
                raise ValueError("Each claim requires supporting source IDs.")
            if any(
                not isinstance(key, str) or key not in self._passages for key in ids
            ):
                raise ValueError("Answer cites an unknown source ID.")
            citations = dict.fromkeys(self._passages[key].citation for key in ids)
            lines.append(f"{claim['text'].strip()} {' '.join(citations)}")
        gap = answer.get("missing_evidence")
        if gap is not None:
            if not isinstance(gap, str) or not gap.strip():
                raise ValueError("Expected a description of the missing evidence.")
            lines.append(f"Missing evidence: {gap.strip()}")
        if not lines:
            raise ValueError(
                "Answer contains neither supported claims nor a description of missing evidence."
            )
        return "\n".join(lines)
